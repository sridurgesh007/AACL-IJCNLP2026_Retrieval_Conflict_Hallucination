#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import cache_head_output, run_with_patch
from medrag.causal_localization import build_metadata, load_medqa_causal_inputs, predict_letter_from_logits
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl
from medrag.utils.stats import mcnemar_from_paired, wilson_ci

RATE_FIELDS = [
    "main_flip_rate",
    "control_A_flip_rate",
    "control_A_all_heads_pooled_flip_rate",
    "control_B_flip_rate",
    "control_C_flip_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedQA causal patching at MedQA-localized candidate head(s).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument(
        "--localization_decision_path",
        default="results/medqa_final_v3/localization_v2/decision.json",
    )
    parser.add_argument("--pair_path", default="results/medqa_final_v3/causal/patching_pairs.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/localization_v2/patching_at_new_head")
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/localization_v2/figures")
    parser.add_argument(
        "--baseline_null_summary_path",
        default="results/medqa_final_v3/causal/patching/summary.json",
    )
    parser.add_argument(
        "--baseline_null_pair_index_path",
        default="results/medqa_final_v3/causal/patching/pair_index.json",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--sampled_control_head_offset", type=int, default=0)
    parser.add_argument("--sampled_control_layer_offset", type=int, default=0)
    parser.add_argument("--substantive_flip_rate", type=float, default=0.30)
    parser.add_argument("--significance_threshold", type=float, default=0.01)
    return parser.parse_args()


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


def _cell_name(layer: int, head: int) -> str:
    return f"L{int(layer)}_H{int(head)}"


def _candidate_heads(decision_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_payload.get("candidate_heads", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Localization decision file does not contain any candidate heads.")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "mean_contribution": float(row.get("mean_contribution", 0.0)),
                "rank": int(row.get("rank", 0)),
                "cell_name": _cell_name(int(row["layer"]), int(row["head"])),
            }
        )
    if not out:
        raise RuntimeError("Localization decision file contained no valid candidate head rows.")
    return out


def _load_existing_null_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("pairs", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected 'pairs' list in {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair_id = str(row.get("pair_id", "")).strip()
        if pair_id:
            out[pair_id] = row
    if not out:
        raise RuntimeError(f"No pair rows found in {path}")
    return out


def _run_candidate(
    *,
    candidate: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    inputs,
    model,
    device,
    out_dir: Path,
    seed: int,
    sampled_control_head_offset: int,
    sampled_control_layer_offset: int,
    substantive_flip_rate: float,
    significance_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer = int(candidate["layer"])
    head = int(candidate["head"])
    cell_name = str(candidate["cell_name"])
    cell_dir = out_dir / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)

    num_heads = int(model.config.num_attention_heads)
    num_layers = int(model.config.num_hidden_layers)
    control_heads = [int(h) for h in range(num_heads) if int(h) != int(head)]
    control_layers = [int(l) for l in range(num_layers) if int(l) != int(layer)]
    if not control_heads:
        raise RuntimeError(f"No control heads available for candidate {cell_name}.")
    if not control_layers:
        raise RuntimeError(f"No control layers available for candidate {cell_name}.")

    pair_results: list[dict[str, Any]] = []
    main_flags: list[bool] = []
    control_a_flags: list[bool] = []
    control_b_flags: list[bool] = []
    control_c_flags: list[bool] = []
    control_a_all_head_flags: list[bool] = []

    for pair_index, pair in enumerate(pair_rows):
        source = inputs.prepared_by_id[str(pair["source_id"])]
        dest = inputs.prepared_by_id[str(pair["dest_id"])]
        correct_letter = str(pair["shared_letter"])

        sampled_head = control_heads[(pair_index + int(sampled_control_head_offset)) % len(control_heads)]
        sampled_layer = control_layers[(pair_index + int(sampled_control_layer_offset)) % len(control_layers)]
        rng = random.Random(int(seed) + int(pair_index) + 17)
        random_position = rng.randrange(0, max(1, int(dest.prompt_ids.shape[0]) - 1))

        source_main = cache_head_output(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            layer=int(layer),
            head=int(head),
            position=-1,
        )
        patched_main_logits = run_with_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            layer=int(layer),
            head=int(head),
            position=-1,
            patched_value=source_main,
        )
        patched_main_answer, main_scores = predict_letter_from_logits(patched_main_logits, inputs.letter_token_sets)
        flip_main = bool(patched_main_answer == correct_letter)

        control_a_answers: dict[str, str | None] = {}
        control_a_flags_by_head: dict[str, bool] = {}
        for other_head in control_heads:
            source_head = cache_head_output(
                model=model,
                prompt_ids=source.prompt_ids.to(device),
                layer=int(layer),
                head=int(other_head),
                position=-1,
            )
            patched_logits = run_with_patch(
                model=model,
                prompt_ids=dest.prompt_ids.to(device),
                layer=int(layer),
                head=int(other_head),
                position=-1,
                patched_value=source_head,
            )
            answer, _ = predict_letter_from_logits(patched_logits, inputs.letter_token_sets)
            control_a_answers[str(other_head)] = answer
            control_a_flags_by_head[str(other_head)] = bool(answer == correct_letter)
            control_a_all_head_flags.append(bool(answer == correct_letter))

        sampled_a_answer = control_a_answers[str(sampled_head)]
        sampled_a_flip = bool(control_a_flags_by_head[str(sampled_head)])

        patched_b_logits = run_with_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            layer=int(layer),
            head=int(head),
            position=int(random_position),
            patched_value=source_main,
        )
        patched_b_answer, _ = predict_letter_from_logits(patched_b_logits, inputs.letter_token_sets)
        sampled_b_flip = bool(patched_b_answer == correct_letter)

        source_c = cache_head_output(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            layer=int(sampled_layer),
            head=int(head),
            position=-1,
        )
        patched_c_logits = run_with_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            layer=int(sampled_layer),
            head=int(head),
            position=-1,
            patched_value=source_c,
        )
        patched_c_answer, _ = predict_letter_from_logits(patched_c_logits, inputs.letter_token_sets)
        sampled_c_flip = bool(patched_c_answer == correct_letter)

        row = {
            "pair_id": str(pair["pair_id"]),
            "candidate_layer": int(layer),
            "candidate_head": int(head),
            "cell_name": str(cell_name),
            "source_id": str(pair["source_id"]),
            "dest_id": str(pair["dest_id"]),
            "correct_letter": str(correct_letter),
            "source_answer": source.base_letter,
            "baseline_dest_answer": pair.get("baseline_dest_answer"),
            "source_margin": float(pair["source_margin"]),
            "dest_margin": float(pair["dest_margin"]),
            "patched_answer_main": patched_main_answer,
            "patched_answer_main_letter_scores": main_scores,
            "patched_answers_control_A": control_a_answers,
            "patched_answer_control_B": patched_b_answer,
            "patched_answer_control_C": patched_c_answer,
            "sampled_control_A_head": int(sampled_head),
            "sampled_control_B_position": int(random_position),
            "sampled_control_C_layer": int(sampled_layer),
            "flip_indicator_main": bool(flip_main),
            "flip_indicator_controls": {
                "control_A_sampled": bool(sampled_a_flip),
                "control_A_all_heads": control_a_flags_by_head,
                "control_B": bool(sampled_b_flip),
                "control_C": bool(sampled_c_flip),
            },
        }
        pair_results.append(row)
        main_flags.append(bool(flip_main))
        control_a_flags.append(bool(sampled_a_flip))
        control_b_flags.append(bool(sampled_b_flip))
        control_c_flags.append(bool(sampled_c_flip))

    main_rate, main_ci, main_k, n_pairs = _rate(main_flags)
    control_a_rate, control_a_ci, control_a_k, _ = _rate(control_a_flags)
    control_b_rate, control_b_ci, control_b_k, _ = _rate(control_b_flags)
    control_c_rate, control_c_ci, control_c_k, _ = _rate(control_c_flags)
    control_a_all_rate, control_a_all_ci, control_a_all_k, control_a_all_n = _rate(control_a_all_head_flags)

    mcnemar_a = mcnemar_from_paired(main_flags, control_a_flags)
    mcnemar_b = mcnemar_from_paired(main_flags, control_b_flags)
    mcnemar_c = mcnemar_from_paired(main_flags, control_c_flags)

    control_max = max(control_a_rate, control_b_rate, control_c_rate)
    causal_claim_supported = bool(
        main_rate > float(substantive_flip_rate)
        and float(mcnemar_a["p_value"]) < float(significance_threshold)
        and float(mcnemar_b["p_value"]) < float(significance_threshold)
        and float(mcnemar_c["p_value"]) < float(significance_threshold)
    )
    partial_support = bool((main_rate > control_max) and (not causal_claim_supported))

    summary = {
        "cell_name": str(cell_name),
        "candidate_layer": int(layer),
        "candidate_head": int(head),
        "n_pairs": int(n_pairs),
        "main_flip_rate": float(main_rate),
        "main_flip_rate_ci95_wilson": main_ci,
        "main_flips": int(main_k),
        "control_A_flip_rate": float(control_a_rate),
        "control_A_flip_rate_ci95_wilson": control_a_ci,
        "control_A_flips": int(control_a_k),
        "control_A_all_heads_pooled_flip_rate": float(control_a_all_rate),
        "control_A_all_heads_pooled_flip_rate_ci95_wilson": control_a_all_ci,
        "control_A_all_heads_total_runs": int(control_a_all_n),
        "control_A_all_heads_total_flips": int(control_a_all_k),
        "control_B_flip_rate": float(control_b_rate),
        "control_B_flip_rate_ci95_wilson": control_b_ci,
        "control_B_flips": int(control_b_k),
        "control_C_flip_rate": float(control_c_rate),
        "control_C_flip_rate_ci95_wilson": control_c_ci,
        "control_C_flips": int(control_c_k),
        "effect_main_minus_control_A": float(main_rate - control_a_rate),
        "effect_main_minus_control_B": float(main_rate - control_b_rate),
        "effect_main_minus_control_C": float(main_rate - control_c_rate),
        "mcnemar_main_vs_control_A": mcnemar_a,
        "mcnemar_main_vs_control_B": mcnemar_b,
        "mcnemar_main_vs_control_C": mcnemar_c,
        "causal_claim_supported": bool(causal_claim_supported),
        "partial_support": bool(partial_support),
        "decision_rule": {
            "substantive_flip_rate_threshold": float(substantive_flip_rate),
            "significance_threshold": float(significance_threshold),
            "main_above_control_max": bool(main_rate > control_max),
        },
    }
    write_json(cell_dir / "summary.json", summary)
    write_jsonl(cell_dir / "results.jsonl", pair_results)
    return summary, pair_results


def _plot_comparison(
    *,
    baseline_summary: dict[str, Any] | None,
    candidate_summaries: list[dict[str, Any]],
    out_path: Path,
) -> None:
    groups: list[dict[str, Any]] = []
    if baseline_summary is not None:
        groups.append(
            {
                "label": "L27/H20 null",
                "summary": baseline_summary,
                "p_text": (
                    f"A:{_star(float(baseline_summary['mcnemar_main_vs_control_A']['p_value']))} "
                    f"B:{_star(float(baseline_summary['mcnemar_main_vs_control_B']['p_value']))} "
                    f"C:{_star(float(baseline_summary['mcnemar_main_vs_control_C']['p_value']))}"
                ),
            }
        )
    for summary in candidate_summaries:
        groups.append(
            {
                "label": str(summary["cell_name"]).replace("_", "/"),
                "summary": summary,
                "p_text": (
                    f"A:{_star(float(summary['mcnemar_main_vs_control_A']['p_value']))} "
                    f"B:{_star(float(summary['mcnemar_main_vs_control_B']['p_value']))} "
                    f"C:{_star(float(summary['mcnemar_main_vs_control_C']['p_value']))}"
                ),
            }
        )

    fields = [
        ("main_flip_rate", "Main"),
        ("control_A_flip_rate", "Ctrl A"),
        ("control_B_flip_rate", "Ctrl B"),
        ("control_C_flip_rate", "Ctrl C"),
    ]
    colors = ["#1f77b4", "#8c8c8c", "#bfa24a", "#5aa469"]
    width = 0.18
    x = np.arange(len(groups), dtype=float)

    fig, ax = plt.subplots(figsize=(max(10.0, 3.5 * len(groups)), 6.0))
    for idx, (field, label) in enumerate(fields):
        values = [float(group["summary"][field]) for group in groups]
        cis = [group["summary"][f"{field}_ci95_wilson"] for group in groups]
        lowers = [max(0.0, values[i] - float(cis[i][0])) for i in range(len(values))]
        uppers = [max(0.0, float(cis[i][1]) - values[i]) for i in range(len(values))]
        err = np.asarray([lowers, uppers], dtype=float)
        bars = ax.bar(
            x + ((idx - 1.5) * width),
            values,
            width=width,
            yerr=err,
            capsize=5.0,
            label=label,
            color=colors[idx],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x, [group["label"] for group in groups])
    ax.set_ylabel("Flip Rate To Correct")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("MedQA Causal Patching: L27/H20 Null vs MedQA-Localized Head(s)")
    ax.legend(loc="upper right")
    for idx, group in enumerate(groups):
        ymax = max(float(group["summary"][field]) for field, _ in fields)
        ax.text(
            x[idx],
            min(0.98, ymax + 0.12),
            group["p_text"],
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.8},
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _top_level_decision(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidate_summaries,
        key=lambda row: (
            bool(row.get("causal_claim_supported", False)),
            float(row.get("main_flip_rate", 0.0)),
            float(row.get("effect_main_minus_control_A", 0.0)),
        ),
        reverse=True,
    )
    best = ranked[0]
    if bool(best.get("causal_claim_supported", False)):
        text = (
            f"Causal claim supported at {best['cell_name']} "
            f"(flip rate {float(best['main_flip_rate']):.3f}, all controls beaten at McNemar p<0.01)."
        )
        status = "supported"
    elif bool(best.get("partial_support", False)):
        text = (
            f"Partial support at {best['cell_name']}: main patching exceeds controls, "
            "but not at the task's substantive and significance thresholds."
        )
        status = "partial_support"
    else:
        text = (
            "No candidate head produced a substantive, control-beating flip rate. "
            "On this evidence, MedQA commitment is not localizable to a single head."
        )
        status = "distributed_or_unresolved"
    return {
        "status": str(status),
        "selected_candidate": {
            "cell_name": str(best["cell_name"]),
            "candidate_layer": int(best["candidate_layer"]),
            "candidate_head": int(best["candidate_head"]),
            "main_flip_rate": float(best["main_flip_rate"]),
        },
        "text": text,
    }


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    pair_payload = read_json((PROJECT_ROOT / args.pair_path).resolve())
    pair_rows = pair_payload.get("pairs", [])
    if not isinstance(pair_rows, list) or not pair_rows:
        raise RuntimeError(f"No patching pairs found in {(PROJECT_ROOT / args.pair_path).resolve()}")

    decision_payload = read_json((PROJECT_ROOT / args.localization_decision_path).resolve())
    candidate_heads = _candidate_heads(decision_payload)

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
        target_layer=max(int(row["layer"]) for row in candidate_heads),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )

    baseline_summary_path = (PROJECT_ROOT / args.baseline_null_summary_path).resolve()
    baseline_pair_index_path = (PROJECT_ROOT / args.baseline_null_pair_index_path).resolve()
    baseline_summary = read_json(baseline_summary_path) if baseline_summary_path.exists() else None
    baseline_pair_rows = _load_existing_null_rows(baseline_pair_index_path) if baseline_pair_index_path.exists() else {}

    candidate_summaries: list[dict[str, Any]] = []
    for candidate in candidate_heads:
        summary, pair_results = _run_candidate(
            candidate=candidate,
            pair_rows=pair_rows,
            inputs=inputs,
            model=model,
            device=device,
            out_dir=out_dir,
            seed=int(args.seed),
            sampled_control_head_offset=int(args.sampled_control_head_offset),
            sampled_control_layer_offset=int(args.sampled_control_layer_offset),
            substantive_flip_rate=float(args.substantive_flip_rate),
            significance_threshold=float(args.significance_threshold),
        )
        if baseline_pair_rows:
            current_pair_rows = {str(row["pair_id"]): row for row in pair_results}
            shared = sorted(set(current_pair_rows.keys()) & set(baseline_pair_rows.keys()))
            summary["comparison_vs_existing_l27_h20"] = {
                "n_shared_pairs": int(len(shared)),
                "mcnemar_main_flip_vs_l27_h20": mcnemar_from_paired(
                    [bool(baseline_pair_rows[pair_id].get("flip_indicator_main", False)) for pair_id in shared],
                    [bool(current_pair_rows[pair_id].get("flip_indicator_main", False)) for pair_id in shared],
                ),
                "main_flip_rate_diff_vs_l27_h20": float(
                    float(summary["main_flip_rate"]) - float(baseline_summary.get("main_flip_rate", 0.0))
                ) if baseline_summary is not None else 0.0,
            }
        candidate_summaries.append(summary)

    decision = _top_level_decision(candidate_summaries)

    _plot_comparison(
        baseline_summary=baseline_summary,
        candidate_summaries=candidate_summaries,
        out_path=figures_dir / "patching_comparison_bars.png",
    )

    metadata = build_metadata(
        args=args,
        input_files=[
            PROJECT_ROOT / args.conflict_path,
            PROJECT_ROOT / args.matched_subset_path,
            PROJECT_ROOT / args.label_audit_path,
            PROJECT_ROOT / args.h4_results_path,
            PROJECT_ROOT / args.localization_decision_path,
            PROJECT_ROOT / args.pair_path,
            PROJECT_ROOT / args.baseline_null_summary_path,
            PROJECT_ROOT / args.baseline_null_pair_index_path,
        ],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    payload = {
        "candidate_heads": candidate_heads,
        "candidate_summaries": candidate_summaries,
        "existing_l27_h20_null": baseline_summary,
        "decision": decision,
        "metadata": metadata,
    }
    write_json(out_dir / "summary.json", payload)
    write_json(
        out_dir / "comparison.json",
        {
            "existing_l27_h20_null": baseline_summary,
            "candidate_summaries": candidate_summaries,
            "decision": decision,
            "metadata": metadata,
        },
    )
    (out_dir / "decision.txt").write_text(str(decision["text"]) + "\n", encoding="utf-8")

    print({"decision": decision, "n_candidates": len(candidate_summaries)})
    print(f"summary={out_dir / 'summary.json'}")
    print(f"comparison={out_dir / 'comparison.json'}")
    print(f"figure={figures_dir / 'patching_comparison_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
