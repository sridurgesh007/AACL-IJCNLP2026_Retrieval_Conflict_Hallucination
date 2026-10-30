#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from medrag.causal_localization import build_metadata, build_patching_pairs, load_medqa_causal_inputs, predict_letter_from_logits
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json, write_summary_json
from medrag.utils.stats import mcnemar_from_paired, wilson_ci

RATE_FIELDS = [
    "main_flip_rate",
    "control_A_flip_rate",
    "control_A_all_heads_pooled_flip_rate",
    "control_B_flip_rate",
    "control_C_flip_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedQA causal patching for L27/H20.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument("--pair_path", default="results/medqa_final_v3/causal/patching_pairs.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/causal/patching")
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/causal/figures")
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--sampled_control_head_offset", type=int, default=0)
    parser.add_argument("--sampled_control_layer_offset", type=int, default=0)
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


def _plot_summary(summary: dict[str, Any], out_path: Path) -> None:
    labels = ["Main", "Ctrl A", "Ctrl B", "Ctrl C"]
    fields = [
        "main_flip_rate",
        "control_A_flip_rate",
        "control_B_flip_rate",
        "control_C_flip_rate",
    ]
    values = [float(summary[field]) for field in fields]
    cis = [summary[f"{field}_ci95_wilson"] for field in fields]
    lowers = [max(0.0, values[i] - float(cis[i][0])) for i in range(len(values))]
    uppers = [max(0.0, float(cis[i][1]) - values[i]) for i in range(len(values))]
    err = np.asarray([lowers, uppers], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#1f77b4", "#8c8c8c", "#bfa24a", "#5aa469"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=err, capsize=6.0, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Flip Rate To Correct")
    ax.set_ylim(0.0, min(1.0, max(values + [0.35]) + 0.15))
    ax.set_title("MedQA Causal Patching: L27/H20 vs Controls")
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    comparisons = [
        ("mcnemar_main_vs_control_A", 0, 1),
        ("mcnemar_main_vs_control_B", 0, 2),
        ("mcnemar_main_vs_control_C", 0, 3),
    ]
    y_base = max(values) + 0.06
    for offset, (field, left, right) in enumerate(comparisons):
        y = y_base + (offset * 0.05)
        ax.plot([left, left, right, right], [y, y + 0.015, y + 0.015, y], color="black", lw=1.0)
        p_value = float(summary[field]["p_value"])
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

    pair_payload = (PROJECT_ROOT / args.pair_path).resolve()
    if not pair_payload.exists():
        raise RuntimeError(f"Missing patch pair file: {pair_payload}. Run scripts/39_preflight_medqa_causal.py first.")
    pair_rows = json.loads(pair_payload.read_text(encoding="utf-8")).get("pairs", [])
    if not pair_rows:
        raise RuntimeError(f"No patching pairs found in {pair_payload}.")

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
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )

    num_heads = int(model.config.num_attention_heads)
    num_layers = int(model.config.num_hidden_layers)
    control_heads = [int(h) for h in range(num_heads) if int(h) != int(args.target_head)]
    control_layers = [int(l) for l in range(num_layers) if int(l) != int(args.target_layer)]

    pair_results: list[dict[str, Any]] = []
    main_flags: list[bool] = []
    control_a_flags: list[bool] = []
    control_b_flags: list[bool] = []
    control_c_flags: list[bool] = []
    control_a_all_head_flags: list[bool] = []

    for pair_index, pair in enumerate(pair_rows):
        source = inputs.prepared_by_id[str(pair["source_id"])]
        dest = inputs.prepared_by_id[str(pair["dest_id"])]
        dest_row = inputs.h4_by_id[str(pair["dest_id"])]
        correct_letter = str(pair["shared_letter"])

        sampled_head = control_heads[(pair_index + int(args.sampled_control_head_offset)) % len(control_heads)]
        sampled_layer = control_layers[(pair_index + int(args.sampled_control_layer_offset)) % len(control_layers)]
        rng = random.Random(int(args.seed) + int(pair_index) + 17)
        random_position = rng.randrange(0, max(1, int(dest.prompt_ids.shape[0]) - 1))

        source_main = cache_head_output(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=-1,
        )
        patched_main_logits = run_with_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=-1,
            patched_value=source_main,
        )
        patched_main_answer, main_scores = predict_letter_from_logits(patched_main_logits, inputs.letter_token_sets)
        flip_main = bool(patched_main_answer == correct_letter)

        control_a_answers: dict[str, str | None] = {}
        control_a_flags_by_head: dict[str, bool] = {}
        for head in control_heads:
            source_head = cache_head_output(
                model=model,
                prompt_ids=source.prompt_ids.to(device),
                layer=int(args.target_layer),
                head=int(head),
                position=-1,
            )
            patched_logits = run_with_patch(
                model=model,
                prompt_ids=dest.prompt_ids.to(device),
                layer=int(args.target_layer),
                head=int(head),
                position=-1,
                patched_value=source_head,
            )
            answer, _ = predict_letter_from_logits(patched_logits, inputs.letter_token_sets)
            control_a_answers[str(head)] = answer
            control_a_flags_by_head[str(head)] = bool(answer == correct_letter)
            control_a_all_head_flags.append(bool(answer == correct_letter))

        sampled_a_answer = control_a_answers[str(sampled_head)]
        sampled_a_flip = bool(control_a_flags_by_head[str(sampled_head)])

        patched_b_logits = run_with_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=int(random_position),
            patched_value=source_main,
        )
        patched_b_answer, _ = predict_letter_from_logits(patched_b_logits, inputs.letter_token_sets)
        sampled_b_flip = bool(patched_b_answer == correct_letter)

        source_c = cache_head_output(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            layer=int(sampled_layer),
            head=int(args.target_head),
            position=-1,
        )
        patched_c_logits = run_with_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            layer=int(sampled_layer),
            head=int(args.target_head),
            position=-1,
            patched_value=source_c,
        )
        patched_c_answer, _ = predict_letter_from_logits(patched_c_logits, inputs.letter_token_sets)
        sampled_c_flip = bool(patched_c_answer == correct_letter)

        row = {
            "pair_id": str(pair["pair_id"]),
            "source_id": str(pair["source_id"]),
            "dest_id": str(pair["dest_id"]),
            "correct_letter": str(correct_letter),
            "source_answer": source.base_letter,
            "baseline_dest_answer": dest_row.get("base_letter"),
            "source_margin": float(pair["source_margin"]),
            "dest_margin": float(pair["dest_margin"]),
            "patched_answer_h20": patched_main_answer,
            "patched_answer_h20_letter_scores": main_scores,
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
        write_json(out_dir / f"{pair['pair_id']}.json", row)

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
    summary = {
        "n_pairs": int(n_pairs),
        "target_layer": int(args.target_layer),
        "target_head": int(args.target_head),
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
        "decision_rule": {
            "main_flip_rate_gt_0p30": bool(main_rate > 0.30),
            "main_vs_A_p_lt_0p01": bool(float(mcnemar_a["p_value"]) < 0.01),
            "main_vs_B_p_lt_0p01": bool(float(mcnemar_b["p_value"]) < 0.01),
            "main_vs_C_p_lt_0p01": bool(float(mcnemar_c["p_value"]) < 0.01),
        },
        "metadata": build_metadata(
            args=args,
            input_files=[
                PROJECT_ROOT / args.conflict_path,
                PROJECT_ROOT / args.matched_subset_path,
                PROJECT_ROOT / args.label_audit_path,
                PROJECT_ROOT / args.h4_results_path,
                PROJECT_ROOT / args.pair_path,
            ],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    supported = (
        bool(summary["decision_rule"]["main_flip_rate_gt_0p30"])
        and bool(summary["decision_rule"]["main_vs_A_p_lt_0p01"])
        and bool(summary["decision_rule"]["main_vs_B_p_lt_0p01"])
        and bool(summary["decision_rule"]["main_vs_C_p_lt_0p01"])
    )
    summary["causal_claim_supported"] = bool(supported)

    write_summary_json(out_dir / "summary.json", summary, rate_fields=RATE_FIELDS)
    _plot_summary(summary, figures_dir / "causal_patching_bars.png")
    if supported:
        finding = "L27/H20 is the causal commitment site under retrieval conflict."
    else:
        finding = (
            "L27/H20 is not the causal commitment site — patching this head does not produce flips above chance. "
            "The intervention works at this head but the decision is made elsewhere, which is a distinct claim."
        )
    (out_dir / "finding.txt").write_text(finding + "\n", encoding="utf-8")
    write_json(out_dir / "pair_index.json", {"pairs": pair_results})

    print(f"summary={out_dir / 'summary.json'}")
    print(f"figure={figures_dir / 'causal_patching_bars.png'}")
    print(f"finding={out_dir / 'finding.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
