#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import PatchSpec, cache_multihead_outputs, run_with_multihead_patch
from medrag.causal_localization import build_metadata, load_medqa_causal_inputs, predict_letter_from_logits, read_jsonl
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint top-5 MedQA head patching.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument(
        "--head_importance_path",
        default="results/medqa_final_v3/localization_v2/head_importance.json",
    )
    parser.add_argument("--pair_path", default="results/medqa_final_v3/causal/patching_pairs.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/localization_v2/joint_patching")
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/localization_v2/figures")
    parser.add_argument(
        "--l27_summary_path",
        default="results/medqa_final_v3/causal/patching/summary.json",
    )
    parser.add_argument(
        "--l27_pair_index_path",
        default="results/medqa_final_v3/causal/patching/pair_index.json",
    )
    parser.add_argument(
        "--single_head_root",
        default="results/medqa_final_v3/localization_v2/patching_at_new_head",
    )
    parser.add_argument("--top_k_heads", type=int, default=5)
    parser.add_argument("--control_a_draws", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--substantive_flip_rate", type=float, default=0.30)
    parser.add_argument("--significance_threshold", type=float, default=0.01)
    return parser.parse_args()


def _cell_name(layer: int, head: int) -> str:
    return f"L{int(layer)}_H{int(head)}"


def _rate(flags: list[bool]) -> tuple[float, list[float], int, int]:
    k = int(sum(1 for flag in flags if bool(flag)))
    n = int(len(flags))
    rate = float(k / max(n, 1))
    ci = wilson_ci(k, n) if n > 0 else (0.0, 0.0)
    return float(rate), [float(ci[0]), float(ci[1])], int(k), int(n)


def _star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _majority_answer(answers: list[str | None]) -> str | None:
    valid = [str(x) for x in answers if x is not None]
    if not valid:
        return None
    counts = Counter(valid)
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return str(ranked[0][0])


def _load_top_heads(path: Path, top_k: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("top_5_heads", [])
    if not isinstance(rows, list) or len(rows) < int(top_k):
        raise RuntimeError(f"{path} does not contain at least {top_k} localized heads.")
    out: list[dict[str, Any]] = []
    for row in rows[: int(top_k)]:
        out.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "rank": int(row.get("rank", 0)),
                "mean_contribution": float(row.get("mean_contribution", 0.0)),
                "cell_name": _cell_name(int(row["layer"]), int(row["head"])),
            }
        )
    return out


def _control_specs_for_draw(
    *,
    top_heads: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[PatchSpec], list[dict[str, int]]]:
    specs: list[PatchSpec] = []
    picks: list[dict[str, int]] = []
    num_heads = 32
    for row in top_heads:
        layer = int(row["layer"])
        ref_head = int(row["head"])
        candidates = [head for head in range(num_heads) if int(head) != int(ref_head)]
        if not candidates:
            raise RuntimeError(f"No control heads available for layer={layer} ref_head={ref_head}.")
        control_head = int(rng.choice(candidates))
        specs.append(PatchSpec(layer=int(layer), head=int(control_head), position=-1))
        picks.append({"layer": int(layer), "reference_head": int(ref_head), "control_head": int(control_head)})
    return specs, picks


def _shared_nonfinal_position(source_len: int, dest_len: int, rng: random.Random) -> int:
    upper = min(int(source_len), int(dest_len)) - 1
    if upper <= 0:
        return 0
    return int(rng.randrange(0, upper))


def _shared_pairwise_mcnemar(
    *,
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, Any]:
    shared_ids = sorted(set(left_rows.keys()) & set(right_rows.keys()))
    return {
        "n_shared_pairs": int(len(shared_ids)),
        "mcnemar": mcnemar_from_paired(
            [bool(left_rows[pair_id].get(left_key, False)) for pair_id in shared_ids],
            [bool(right_rows[pair_id].get(right_key, False)) for pair_id in shared_ids],
        ),
        "shared_pair_ids": shared_ids,
    }


def _load_pair_rows(path: Path, *, main_key: str) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json":
        payload = read_json(path)
        rows = payload.get("pairs", [])
    else:
        rows = read_jsonl(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected rows list in {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            continue
        out[pair_id] = row
    if not out:
        raise RuntimeError(f"No pair rows found in {path}")
    return out


def _plot_joint_bars(
    *,
    l27_summary: dict[str, Any],
    top1_summary: dict[str, Any],
    joint_summary: dict[str, Any],
    out_path: Path,
) -> None:
    labels = [
        "L27/H20 main",
        f"{top1_summary['cell_name'].replace('_', '/')} main",
        "Joint top-5 main",
        "Joint controls pooled",
    ]
    values = [
        float(l27_summary["main_flip_rate"]),
        float(top1_summary["main_flip_rate"]),
        float(joint_summary["main_flip_rate"]),
        float(joint_summary["controls_pooled_flip_rate"]),
    ]
    cis = [
        l27_summary["main_flip_rate_ci95_wilson"],
        top1_summary["main_flip_rate_ci95_wilson"],
        joint_summary["main_flip_rate_ci95_wilson"],
        joint_summary["controls_pooled_flip_rate_ci95_wilson"],
    ]
    lowers = [max(0.0, values[i] - float(cis[i][0])) for i in range(len(values))]
    uppers = [max(0.0, float(cis[i][1]) - values[i]) for i in range(len(values))]
    err = np.asarray([lowers, uppers], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(labels))
    colors = ["#777777", "#9b59b6", "#1f77b4", "#8c8c8c"]
    bars = ax.bar(x, values, yerr=err, capsize=6.0, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Flip Rate To Correct")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("MedQA Joint Multi-Head Patching")
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    comparisons = [
        ("mcnemar_joint_vs_l27", 0, 2, float(joint_summary["mcnemar_joint_main_vs_l27_single"]["mcnemar"]["p_value"])),
        ("mcnemar_joint_vs_top1", 1, 2, float(joint_summary["mcnemar_joint_main_vs_top1_single"]["mcnemar"]["p_value"])),
    ]
    y_base = max(values) + 0.08
    for offset, (_, left, right, p_value) in enumerate(comparisons):
        y = y_base + (offset * 0.06)
        ax.plot([left, left, right, right], [y, y + 0.015, y + 0.015, y], color="black", lw=1.0)
        ax.text((left + right) / 2.0, y + 0.02, _star(p_value), ha="center", va="bottom")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    top_heads = _load_top_heads((PROJECT_ROOT / args.head_importance_path).resolve(), int(args.top_k_heads))
    pair_payload = read_json((PROJECT_ROOT / args.pair_path).resolve())
    pair_rows = pair_payload.get("pairs", [])
    if not isinstance(pair_rows, list) or not pair_rows:
        raise RuntimeError(f"No patching pairs found in {(PROJECT_ROOT / args.pair_path).resolve()}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    inputs = load_medqa_causal_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=PROJECT_ROOT / args.conflict_path,
        matched_subset_path=PROJECT_ROOT / args.matched_subset_path,
        label_audit_path=PROJECT_ROOT / args.label_audit_path,
        h4_results_path=PROJECT_ROOT / args.h4_results_path,
        cache_dir=PROJECT_ROOT / args.cache_dir,
        target_layer=max(int(row["layer"]) for row in top_heads),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )

    main_specs = [PatchSpec(layer=int(row["layer"]), head=int(row["head"]), position=-1) for row in top_heads]

    results: list[dict[str, Any]] = []
    main_flags: list[bool] = []
    control_a_flags: list[bool] = []
    control_b_flags: list[bool] = []
    control_a_raw_flags: list[bool] = []

    for pair_index, pair in enumerate(pair_rows):
        source = inputs.prepared_by_id[str(pair["source_id"])]
        dest = inputs.prepared_by_id[str(pair["dest_id"])]
        correct_letter = str(pair["shared_letter"])

        source_main_patches = cache_multihead_outputs(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            patch_specs=main_specs,
        )
        main_logits = run_with_multihead_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            patched_values=source_main_patches,
        )
        main_answer, _ = predict_letter_from_logits(main_logits, inputs.letter_token_sets)
        main_flip = bool(main_answer == correct_letter)

        control_a_draw_rows: list[dict[str, Any]] = []
        control_a_draw_answers: list[str | None] = []
        control_a_draw_flags: list[bool] = []
        for draw_idx in range(int(args.control_a_draws)):
            rng = random.Random(int(args.seed) + int(pair_index) * 1009 + int(draw_idx) * 9173 + 17)
            control_specs, control_picks = _control_specs_for_draw(top_heads=top_heads, rng=rng)
            source_control_patches = cache_multihead_outputs(
                model=model,
                prompt_ids=source.prompt_ids.to(device),
                patch_specs=control_specs,
            )
            control_logits = run_with_multihead_patch(
                model=model,
                prompt_ids=dest.prompt_ids.to(device),
                patched_values=source_control_patches,
            )
            control_answer, _ = predict_letter_from_logits(control_logits, inputs.letter_token_sets)
            control_flip = bool(control_answer == correct_letter)
            control_a_draw_rows.append(
                {
                    "draw_index": int(draw_idx),
                    "control_heads": control_picks,
                    "patched_answer": control_answer,
                    "flip_indicator": bool(control_flip),
                }
            )
            control_a_draw_answers.append(control_answer)
            control_a_draw_flags.append(bool(control_flip))
            control_a_raw_flags.append(bool(control_flip))

        control_a_answer = _majority_answer(control_a_draw_answers)
        control_a_flip = bool(sum(1 for flag in control_a_draw_flags if flag) >= ((len(control_a_draw_flags) // 2) + 1))

        pos_rng = random.Random(int(args.seed) + int(pair_index) * 31337 + 101)
        random_position = _shared_nonfinal_position(
            source_len=int(source.prompt_ids.shape[0]),
            dest_len=int(dest.prompt_ids.shape[0]),
            rng=pos_rng,
        )
        random_specs = [PatchSpec(layer=int(row["layer"]), head=int(row["head"]), position=int(random_position)) for row in top_heads]
        source_random_patches = cache_multihead_outputs(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            patch_specs=random_specs,
        )
        control_b_logits = run_with_multihead_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            patched_values=source_random_patches,
        )
        control_b_answer, _ = predict_letter_from_logits(control_b_logits, inputs.letter_token_sets)
        control_b_flip = bool(control_b_answer == correct_letter)

        results.append(
            {
                "pair_id": str(pair["pair_id"]),
                "source_id": str(pair["source_id"]),
                "dest_id": str(pair["dest_id"]),
                "correct_letter": str(correct_letter),
                "main_patched_answer": main_answer,
                "main_flip_indicator": bool(main_flip),
                "controlA_patched_answer": control_a_answer,
                "controlA_flip_indicator": bool(control_a_flip),
                "controlB_patched_answer": control_b_answer,
                "controlB_flip_indicator": bool(control_b_flip),
                "top5_heads_used": [
                    {"layer": int(row["layer"]), "head": int(row["head"]), "rank": int(row["rank"])}
                    for row in top_heads
                ],
                "controlA_draws": control_a_draw_rows,
                "controlB_random_position": int(random_position),
            }
        )
        main_flags.append(bool(main_flip))
        control_a_flags.append(bool(control_a_flip))
        control_b_flags.append(bool(control_b_flip))

    main_rate, main_ci, main_k, n_pairs = _rate(main_flags)
    control_a_rate, control_a_ci, control_a_k, _ = _rate(control_a_flags)
    control_b_rate, control_b_ci, control_b_k, _ = _rate(control_b_flags)
    control_a_raw_rate, control_a_raw_ci, control_a_raw_k, control_a_raw_n = _rate(control_a_raw_flags)
    controls_pooled_flags = list(control_a_raw_flags) + list(control_b_flags)
    controls_pooled_rate, controls_pooled_ci, controls_pooled_k, controls_pooled_n = _rate(controls_pooled_flags)

    mcnemar_a = mcnemar_from_paired(main_flags, control_a_flags)
    mcnemar_b = mcnemar_from_paired(main_flags, control_b_flags)
    causal_claim_supported = bool(
        float(main_rate) > float(args.substantive_flip_rate)
        and float(mcnemar_a["p_value"]) < float(args.significance_threshold)
        and float(mcnemar_b["p_value"]) < float(args.significance_threshold)
    )

    if causal_claim_supported:
        finding_text = (
            "MedQA letter-commitment is distributed across the top-5 localized heads; "
            "simultaneous patching flips answers significantly above control while single-head patching fails."
        )
    else:
        finding_text = (
            "Even joint patching of the top-5 localized heads does not causally flip letter commitment, "
            "indicating the decision locus on MedQA is not compactly representable by any small head subset. "
            "Margin-gated intervention remains empirically effective and operates independently of the causal-commitment structure."
        )

    joint_rows_by_pair = {str(row["pair_id"]): row for row in results}
    l27_summary = read_json((PROJECT_ROOT / args.l27_summary_path).resolve())
    l27_pair_rows = _load_pair_rows((PROJECT_ROOT / args.l27_pair_index_path).resolve(), main_key="flip_indicator_main")

    top1_cell = str(top_heads[0]["cell_name"])
    single_head_root = (PROJECT_ROOT / args.single_head_root).resolve()
    top1_results_path = single_head_root / top1_cell / "results.jsonl"
    top1_summary_root = read_json(single_head_root / "summary.json")
    top1_summary = next(
        row for row in top1_summary_root.get("candidate_summaries", []) if str(row.get("cell_name")) == str(top1_cell)
    )
    top1_pair_rows = _load_pair_rows(top1_results_path, main_key="flip_indicator_main")

    joint_vs_l27 = _shared_pairwise_mcnemar(
        left_rows=joint_rows_by_pair,
        right_rows=l27_pair_rows,
        left_key="main_flip_indicator",
        right_key="flip_indicator_main",
    )
    joint_vs_top1 = _shared_pairwise_mcnemar(
        left_rows=joint_rows_by_pair,
        right_rows=top1_pair_rows,
        left_key="main_flip_indicator",
        right_key="flip_indicator_main",
    )

    summary = {
        "n_pairs": int(n_pairs),
        "top5_heads": top_heads,
        "main_flip_rate": float(main_rate),
        "main_flip_rate_ci95_wilson": main_ci,
        "main_flips": int(main_k),
        "control_A_flip_rate": float(control_a_rate),
        "control_A_flip_rate_ci95_wilson": control_a_ci,
        "control_A_flips": int(control_a_k),
        "control_A_pooled_flip_rate_raw": float(control_a_raw_rate),
        "control_A_pooled_flip_rate_raw_ci95_wilson": control_a_raw_ci,
        "control_A_pooled_total_runs": int(control_a_raw_n),
        "control_A_pooled_total_flips": int(control_a_raw_k),
        "control_B_flip_rate": float(control_b_rate),
        "control_B_flip_rate_ci95_wilson": control_b_ci,
        "control_B_flips": int(control_b_k),
        "controls_pooled_flip_rate": float(controls_pooled_rate),
        "controls_pooled_flip_rate_ci95_wilson": controls_pooled_ci,
        "controls_pooled_total_runs": int(controls_pooled_n),
        "controls_pooled_total_flips": int(controls_pooled_k),
        "mcnemar_main_vs_control_A": mcnemar_a,
        "mcnemar_main_vs_control_B": mcnemar_b,
        "mcnemar_joint_main_vs_l27_single": joint_vs_l27,
        "mcnemar_joint_main_vs_top1_single": joint_vs_top1,
        "causal_claim_supported": bool(causal_claim_supported),
        "finding": str(finding_text),
    }

    metadata = build_metadata(
        args=args,
        input_files=[
            PROJECT_ROOT / args.conflict_path,
            PROJECT_ROOT / args.matched_subset_path,
            PROJECT_ROOT / args.label_audit_path,
            PROJECT_ROOT / args.h4_results_path,
            PROJECT_ROOT / args.head_importance_path,
            PROJECT_ROOT / args.pair_path,
            PROJECT_ROOT / args.l27_summary_path,
            PROJECT_ROOT / args.l27_pair_index_path,
            top1_results_path,
            single_head_root / "summary.json",
        ],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    summary["metadata"] = metadata

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "results.jsonl", results)
    (out_dir / "finding.txt").write_text(str(finding_text) + "\n", encoding="utf-8")

    _plot_joint_bars(
        l27_summary=l27_summary,
        top1_summary=top1_summary,
        joint_summary=summary,
        out_path=figures_dir / "joint_patching_bars.png",
    )

    print(
        {
            "main_flip_rate": float(main_rate),
            "control_A_flip_rate": float(control_a_rate),
            "control_B_flip_rate": float(control_b_rate),
            "causal_claim_supported": bool(causal_claim_supported),
        }
    )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"results={out_dir / 'results.jsonl'}")
    print(f"finding={out_dir / 'finding.txt'}")
    print(f"figure={figures_dir / 'joint_patching_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
