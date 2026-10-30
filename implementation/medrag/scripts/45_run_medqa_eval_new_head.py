#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from medrag.localization_v2 import pairwise_correctness_comparison, read_jsonl_by_key
from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import (
    build_letter_token_sets,
    evaluate_intervention_cell,
    load_conflicts,
    prepare_examples_for_variant,
    set_global_seed,
)
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import read_json, write_json, write_jsonl
from medrag.causal_localization import build_metadata, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MedQA intervention effectiveness at MedQA-localized head(s).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--localization_decision_path",
        default="results/medqa_final_v3/localization_v2/decision.json",
    )
    parser.add_argument(
        "--h4_summary_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/summary.json",
    )
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument(
        "--out_dir",
        default="results/medqa_final_v3/localization_v2/intervention_at_new_head",
    )
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/localization_v2/figures")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--bootstrap_space", default="E_letter", choices=["E_letter", "E_content"])
    parser.add_argument("--tau_margin", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


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


def _standard_error(rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = min(max(float(rate), 0.0), 1.0)
    return float(math.sqrt(max(p * (1.0 - p), 0.0) / float(n)))


def _difference_se(rate_a: float, n_a: int, rate_b: float, n_b: int) -> float:
    return float(math.sqrt(_standard_error(rate_a, n_a) ** 2 + _standard_error(rate_b, n_b) ** 2))


def _plot_intervention_comparison(
    *,
    h4_summary: dict[str, Any],
    candidate_payload: dict[str, Any],
    out_path: Path,
) -> None:
    labels = ["L27/H20 H4", f"{candidate_payload['cell_name']} headline", f"{candidate_payload['cell_name']} balanced"]
    values = [
        float(h4_summary["restoration_rate"]),
        float(candidate_payload["headline"]["summary"]["restoration_rate"]),
        float(candidate_payload["balanced"]["summary"]["restoration_rate"]),
    ]
    cis = [
        h4_summary["restoration_rate_ci95_wilson"],
        candidate_payload["headline"]["summary"]["restoration_rate_ci95_wilson"],
        candidate_payload["balanced"]["summary"]["restoration_rate_ci95_wilson"],
    ]
    lowers = [max(0.0, values[i] - float(cis[i][0])) for i in range(len(values))]
    uppers = [max(0.0, float(cis[i][1]) - values[i]) for i in range(len(values))]
    err = np.asarray([lowers, uppers], dtype=float)
    colors = ["#777777", "#1f77b4", "#5aa469"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=err, capsize=6.0, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Restoration Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("MedQA Intervention Comparison: L27/H20 vs Localized Head")
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
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

    decision_payload = read_json((PROJECT_ROOT / args.localization_decision_path).resolve())
    candidate_heads = _candidate_heads(decision_payload)
    h4_summary = read_json((PROJECT_ROOT / args.h4_summary_path).resolve())
    h4_rows = read_jsonl((PROJECT_ROOT / args.h4_results_path).resolve())
    h4_rows_by_id = read_jsonl_by_key((PROJECT_ROOT / args.h4_results_path).resolve(), "item_id")

    matched_payload = read_json((PROJECT_ROOT / args.matched_subset_path).resolve())
    label_audit = load_label_audit((PROJECT_ROOT / args.label_audit_path).resolve())
    matched_item_ids = {str(x) for x in matched_payload.get("matched_item_ids", [])}
    clean_item_ids = set(clean_subset_item_ids(matched_payload, label_audit))

    conflicts = load_conflicts((PROJECT_ROOT / args.conflict_path).resolve())
    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = build_letter_token_sets(tokenizer)
    cache = ActivationCache((PROJECT_ROOT / args.cache_dir).resolve())

    conditions = [
        ("headline", 0.0, 2.5),
        ("balanced", 1.5, 1.5),
    ]

    prepared_by_layer: dict[int, list[Any]] = {}
    candidate_payloads: list[dict[str, Any]] = []
    for candidate in candidate_heads:
        layer = int(candidate["layer"])
        head = int(candidate["head"])
        cell_name = str(candidate["cell_name"])
        if layer not in prepared_by_layer:
            prepared_by_layer[layer] = prepare_examples_for_variant(
                conflicts=conflicts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                cache=cache,
                model_id=str(args.model_id),
                prompt_variant="vanilla",
                p_set_mode=str(args.p_set_mode),
                target_layer=int(layer),
                max_length=int(args.max_length),
                max_new_tokens=int(args.max_new_tokens),
                letter_token_sets=letter_token_sets,
            )
        prepared_records = prepared_by_layer[layer]

        cell_dir = out_dir / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        candidate_entry: dict[str, Any] = {
            "cell_name": str(cell_name),
            "candidate_layer": int(layer),
            "candidate_head": int(head),
        }
        for condition_id, lambda_pos, lambda_neg in conditions:
            summary, rows = evaluate_intervention_cell(
                prepared_records=prepared_records,
                model=model,
                tokenizer=tokenizer,
                device=device,
                letter_token_sets=letter_token_sets,
                bootstrap_space=str(args.bootstrap_space),
                intervention_space="E_letter",
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                tau_margin=float(args.tau_margin),
                alpha=float(args.alpha),
                target_layer=int(layer),
                target_head=int(head),
                matched_item_ids=matched_item_ids,
                clean_item_ids=clean_item_ids,
                seed=int(args.seed),
                max_new_tokens=int(args.max_new_tokens),
            )
            summary.update(
                {
                    "condition_id": str(condition_id),
                    "candidate_layer": int(layer),
                    "candidate_head": int(head),
                    "cell_name": str(cell_name),
                    "bootstrap_space": str(args.bootstrap_space),
                    "intervention_space": "E_letter",
                    "tau_margin": float(args.tau_margin),
                    "alpha": float(args.alpha),
                    "lambda_pos": float(lambda_pos),
                    "lambda_neg": float(lambda_neg),
                    "p_set_mode": str(args.p_set_mode),
                }
            )
            condition_dir = cell_dir / condition_id
            condition_dir.mkdir(parents=True, exist_ok=True)
            write_json(condition_dir / "summary.json", summary)
            write_jsonl(condition_dir / "results.jsonl", rows)

            rows_by_id = {str(row["item_id"]): row for row in rows if str(row.get("item_id", "")).strip()}
            correctness_comparison = pairwise_correctness_comparison(
                left_rows=h4_rows_by_id,
                right_rows=rows_by_id,
                left_key="ft_correct",
                right_key="ft_correct",
            )
            candidate_rate = float(summary["restoration_rate"])
            h4_rate = float(h4_summary["restoration_rate"])
            n_base_wrong = int(summary["n_base_wrong"])
            h4_base_wrong = int(h4_summary["n_base_wrong"])
            diff_se = _difference_se(candidate_rate, n_base_wrong, h4_rate, h4_base_wrong)
            comparison = {
                "condition_id": str(condition_id),
                "candidate_restoration_rate": float(candidate_rate),
                "candidate_restoration_rate_ci95_wilson": summary["restoration_rate_ci95_wilson"],
                "candidate_harm_rate": float(summary["harm_rate"]),
                "candidate_harm_rate_ci95_wilson": summary["harm_rate_ci95_wilson"],
                "candidate_ft_accuracy": float(summary["ft_accuracy"]),
                "candidate_ft_accuracy_ci95_wilson": summary["ft_accuracy_ci95_wilson"],
                "h4_restoration_rate": float(h4_rate),
                "h4_restoration_rate_ci95_wilson": h4_summary["restoration_rate_ci95_wilson"],
                "h4_harm_rate": float(h4_summary["harm_rate"]),
                "h4_harm_rate_ci95_wilson": h4_summary["harm_rate_ci95_wilson"],
                "h4_ft_accuracy": float(h4_summary["ft_accuracy"]),
                "h4_ft_accuracy_ci95_wilson": h4_summary["ft_accuracy_ci95_wilson"],
                "restoration_rate_diff_vs_h4": float(candidate_rate - h4_rate),
                "restoration_rate_diff_se": float(diff_se),
                "within_one_difference_se_of_h4": bool(abs(candidate_rate - h4_rate) <= diff_se),
                "per_item_correctness_vs_h4": correctness_comparison,
            }
            write_json(condition_dir / "comparison_vs_h4.json", comparison)
            candidate_entry[str(condition_id)] = {
                "summary": summary,
                "comparison_vs_h4": comparison,
            }

        headline_summary = candidate_entry["headline"]["summary"]
        balanced_summary = candidate_entry["balanced"]["summary"]
        best_config = "headline"
        best_summary = headline_summary
        if float(balanced_summary["restoration_rate"]) > float(headline_summary["restoration_rate"]):
            best_config = "balanced"
            best_summary = balanced_summary
        candidate_entry["best_config"] = str(best_config)
        candidate_entry["best_summary"] = best_summary
        candidate_payloads.append(candidate_entry)

    selected_candidate = max(
        candidate_payloads,
        key=lambda row: (
            float(row["headline"]["summary"]["restoration_rate"]),
            float(row["headline"]["summary"]["ft_accuracy"]),
        ),
    )
    selected_headline = selected_candidate["headline"]["summary"]
    headline_comparison = selected_candidate["headline"]["comparison_vs_h4"]
    if bool(headline_comparison["within_one_difference_se_of_h4"]) and float(selected_headline["restoration_rate"]) >= float(h4_summary["restoration_rate"]):
        framing_option = "A"
        framing_reason = (
            f"{selected_candidate['cell_name']} matches or exceeds the L27/H20 headline restoration "
            "within one standard error of the restoration-rate difference."
        )
    elif float(selected_headline["restoration_rate"]) <= 0.05:
        framing_option = "B"
        framing_reason = (
            f"{selected_candidate['cell_name']} localizes the candidate mechanism, but intervention efficacy "
            "at the new head is effectively absent."
        )
    else:
        framing_option = "C"
        framing_reason = (
            f"{selected_candidate['cell_name']} is the better mechanistic candidate, but L27/H20 remains the stronger "
            "intervention site on the matched subset."
        )

    _plot_intervention_comparison(
        h4_summary=h4_summary,
        candidate_payload=selected_candidate,
        out_path=figures_dir / "intervention_comparison_bars.png",
    )

    metadata = build_metadata(
        args=args,
        input_files=[
            PROJECT_ROOT / args.conflict_path,
            PROJECT_ROOT / args.matched_subset_path,
            PROJECT_ROOT / args.label_audit_path,
            PROJECT_ROOT / args.localization_decision_path,
            PROJECT_ROOT / args.h4_summary_path,
            PROJECT_ROOT / args.h4_results_path,
        ],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    summary_payload = {
        "candidate_heads": candidate_payloads,
        "reference_h4": h4_summary,
        "selected_candidate": {
            "cell_name": str(selected_candidate["cell_name"]),
            "candidate_layer": int(selected_candidate["candidate_layer"]),
            "candidate_head": int(selected_candidate["candidate_head"]),
            "best_config": str(selected_candidate["best_config"]),
            "headline_summary": selected_candidate["headline"]["summary"],
            "balanced_summary": selected_candidate["balanced"]["summary"],
        },
        "framing_decision": {
            "option": str(framing_option),
            "reason": str(framing_reason),
        },
        "metadata": metadata,
    }
    write_json(out_dir / "summary.json", summary_payload)
    write_json(
        out_dir / "comparison.json",
        {
            "reference_h4": h4_summary,
            "candidate_heads": candidate_payloads,
            "selected_candidate": summary_payload["selected_candidate"],
            "framing_decision": summary_payload["framing_decision"],
            "metadata": metadata,
        },
    )
    (out_dir / "decision.txt").write_text(f"Option {framing_option}: {framing_reason}\n", encoding="utf-8")

    print(
        {
            "selected_candidate": summary_payload["selected_candidate"]["cell_name"],
            "framing_option": framing_option,
        }
    )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"comparison={out_dir / 'comparison.json'}")
    print(f"figure={figures_dir / 'intervention_comparison_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
