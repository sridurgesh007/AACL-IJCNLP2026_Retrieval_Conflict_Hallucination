#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import build_letter_token_sets, evaluate_intervention_cell, load_conflicts, prepare_examples_for_variant, set_global_seed, summarize_condition_accuracy
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import build_run_metadata, read_json, write_json, write_jsonl, write_summary_json
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedQA v3 Experiment 3: prompting kill-test.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/killtest")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--best_cell_path", default="results/medqa_final_v3/decomposition/best_cell.json")
    parser.add_argument("--headline_path", default="results/medqa_final_v3/pr_sweep/headline.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts_killtest")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _rows_from_prepared(prepared_records: list[Any], matched_ids: set[str], clean_ids: set[str]) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    for rec in prepared_records:
        total_seconds += float(rec.prefill_wall_clock_seconds + rec.generation_wall_clock_seconds)
        rows.append(
            {
                "item_id": rec.item_id,
                "question": rec.question,
                "correct": rec.correct,
                "parametric": rec.parametric,
                "condition_letter": rec.base_letter,
                "condition_correct": bool(rec.base_correct),
                "in_matched_subset": bool(rec.item_id in matched_ids),
                "in_clean_subset": bool(rec.item_id in clean_ids),
                "skipped": False,
                "prompt_variant": rec.prompt_variant,
            }
        )
    return rows, float(total_seconds)


def _condition_accuracy_payload(rows: list[dict[str, Any]], wall_clock_seconds: float, prompt_variant: str, apply_intervention: bool) -> dict[str, Any]:
    summary = summarize_condition_accuracy(rows=rows, correct_key="condition_correct", wall_clock_seconds=wall_clock_seconds)
    summary.update({"prompt_variant": str(prompt_variant), "apply_intervention": bool(apply_intervention)})
    return summary


def _paired_condition_vectors(rows: list[dict[str, Any]], key: str, subset_key: str = "in_clean_subset") -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in rows:
        if not bool(row.get(subset_key, False)) or bool(row.get("skipped", False)):
            continue
        if str(row["item_id"]) in out:
            raise ValueError(f"Duplicate paired item_id: {row['item_id']}")
        out[str(row["item_id"])] = bool(row.get(key, False))
    return out


def _restoration_against_baseline(cond_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], cond_key: str) -> tuple[float, list[float], int, int]:
    base_map = _paired_condition_vectors(baseline_rows, "condition_correct")
    cond_map = _paired_condition_vectors(cond_rows, cond_key)
    eligible = [item_id for item_id, base_ok in base_map.items() if not base_ok and item_id in cond_map]
    k = int(sum(1 for item_id in eligible if bool(cond_map[item_id])))
    n = int(len(eligible))
    ci = list(wilson_ci(k, n)) if n > 0 else [0.0, 0.0]
    return float(k / max(n, 1)), ci, k, n


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    conflicts = load_conflicts((PROJECT_ROOT / args.conflict_path).resolve())
    matched_payload = read_json((PROJECT_ROOT / args.matched_subset_path).resolve())
    matched_ids = set(str(x) for x in matched_payload.get("matched_item_ids", []))
    label_audit = load_label_audit((PROJECT_ROOT / args.label_audit_path).resolve())
    clean_ids = set(clean_subset_item_ids(matched_payload, label_audit))
    best_cell = read_json((PROJECT_ROOT / args.best_cell_path).resolve())
    headline = read_json((PROJECT_ROOT / args.headline_path).resolve())

    lambda_pos = float(best_cell["lambda_pos"])
    lambda_neg = float(best_cell["lambda_neg"])
    tau_margin = float(headline["tau_margin"])
    bootstrap_space = str(headline["bootstrap_space"])

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = build_letter_token_sets(tokenizer)
    cache = ActivationCache((PROJECT_ROOT / args.cache_dir).resolve())

    vanilla = prepare_examples_for_variant(
        conflicts=conflicts, model=model, tokenizer=tokenizer, device=device, cache=cache,
        model_id=str(args.model_id), prompt_variant="vanilla", p_set_mode=str(args.p_set_mode),
        target_layer=int(args.target_layer), max_length=int(args.max_length), max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    trust = prepare_examples_for_variant(
        conflicts=conflicts, model=model, tokenizer=tokenizer, device=device, cache=cache,
        model_id=str(args.model_id), prompt_variant="trust_instruction", p_set_mode=str(args.p_set_mode),
        target_layer=int(args.target_layer), max_length=int(args.max_length), max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    quote = prepare_examples_for_variant(
        conflicts=conflicts, model=model, tokenizer=tokenizer, device=device, cache=cache,
        model_id=str(args.model_id), prompt_variant="authoritative_quote", p_set_mode=str(args.p_set_mode),
        target_layer=int(args.target_layer), max_length=int(args.max_length), max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )

    c1_rows, c1_time = _rows_from_prepared(vanilla, matched_ids, clean_ids)
    c2_rows, c2_time = _rows_from_prepared(trust, matched_ids, clean_ids)
    c3_rows, c3_time = _rows_from_prepared(quote, matched_ids, clean_ids)
    c1_summary = _condition_accuracy_payload(c1_rows, c1_time, "vanilla", False)
    c2_summary = _condition_accuracy_payload(c2_rows, c2_time, "trust_instruction", False)
    c3_summary = _condition_accuracy_payload(c3_rows, c3_time, "authoritative_quote", False)

    c4_summary, c4_rows_raw = evaluate_intervention_cell(
        prepared_records=vanilla,
        model=model,
        tokenizer=tokenizer,
        device=device,
        letter_token_sets=letter_token_sets,
        bootstrap_space=bootstrap_space,
        intervention_space="E_letter",
        lambda_pos=float(lambda_pos),
        lambda_neg=float(lambda_neg),
        tau_margin=float(tau_margin),
        alpha=float(args.alpha),
        target_layer=int(args.target_layer),
        target_head=int(args.target_head),
        matched_item_ids=matched_ids,
        clean_item_ids=clean_ids,
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
    )
    c5_summary, c5_rows_raw = evaluate_intervention_cell(
        prepared_records=trust,
        model=model,
        tokenizer=tokenizer,
        device=device,
        letter_token_sets=letter_token_sets,
        bootstrap_space=bootstrap_space,
        intervention_space="E_letter",
        lambda_pos=float(lambda_pos),
        lambda_neg=float(lambda_neg),
        tau_margin=float(tau_margin),
        alpha=float(args.alpha),
        target_layer=int(args.target_layer),
        target_head=int(args.target_head),
        matched_item_ids=matched_ids,
        clean_item_ids=clean_ids,
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
    )
    c4_summary.update({"prompt_variant": "vanilla", "apply_intervention": True, "condition_accuracy": c4_summary["ft_accuracy"], "condition_accuracy_ci95_wilson": c4_summary["ft_accuracy_ci95_wilson"]})
    c5_summary.update({"prompt_variant": "trust_instruction", "apply_intervention": True, "condition_accuracy": c5_summary["ft_accuracy"], "condition_accuracy_ci95_wilson": c5_summary["ft_accuracy_ci95_wilson"]})
    c4_rows = [{**row, "condition_correct": bool(row.get("ft_correct", False))} for row in c4_rows_raw]
    c5_rows = [{**row, "condition_correct": bool(row.get("ft_correct", False))} for row in c5_rows_raw]

    conditions = {
        "C1": (c1_summary, c1_rows),
        "C2": (c2_summary, c2_rows),
        "C3": (c3_summary, c3_rows),
        "C4": (c4_summary, c4_rows),
        "C5": (c5_summary, c5_rows),
    }

    for cond_id, (summary, rows) in conditions.items():
        cond_dir = out_dir / cond_id
        cond_dir.mkdir(parents=True, exist_ok=True)
        summary["condition_id"] = cond_id
        summary["metadata"] = build_run_metadata(
            args=args,
            input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
            seed=int(args.seed),
            wall_clock_seconds=float(summary["wall_clock_seconds"]),
            project_root=PROJECT_ROOT,
        )
        rate_fields = ["accuracy", "accuracy_clean_subset"] if not bool(summary.get("apply_intervention", False)) else [
            "baseline_accuracy", "ft_accuracy", "ft_accuracy_clean_subset", "restoration_rate", "harm_rate",
            "restoration_rate_clean_subset", "harm_rate_clean_subset", "gate_fire_rate_base_wrong", "gate_fire_rate_base_correct",
            "restoration_rate_param_wrong", "restoration_rate_other_wrong",
        ]
        write_summary_json(cond_dir / "summary.json", summary, rate_fields=rate_fields)
        write_jsonl(cond_dir / "results.jsonl", rows)

    c2_restoration, c2_restoration_ci, _, _ = _restoration_against_baseline(c2_rows, c1_rows, "condition_correct")
    c3_restoration, c3_restoration_ci, _, _ = _restoration_against_baseline(c3_rows, c1_rows, "condition_correct")

    vectors = {key: _paired_condition_vectors(rows, "condition_correct") for key, (_, rows) in conditions.items()}

    def _pairwise(a: str, b: str) -> dict[str, Any]:
        if not vectors[a] or set(vectors[a]) != set(vectors[b]):
            raise ValueError(f"Nonempty identical clean-subset IDs are required for {a} vs {b}")
        shared = sorted(vectors[a])
        result = mcnemar_from_paired([vectors[a][x] for x in shared], [vectors[b][x] for x in shared])
        return {**result, "population": "in_clean_subset", "n_paired": len(shared)}

    comparison = {
        "accuracy_C1": c1_summary["accuracy"],
        "accuracy_C1_ci95_wilson": c1_summary["accuracy_ci95_wilson"],
        "accuracy_C1_clean_subset": c1_summary["accuracy_clean_subset"],
        "accuracy_C1_clean_subset_ci95_wilson": c1_summary["accuracy_clean_subset_ci95_wilson"],
        "accuracy_C2": c2_summary["accuracy"],
        "accuracy_C2_ci95_wilson": c2_summary["accuracy_ci95_wilson"],
        "accuracy_C2_clean_subset": c2_summary["accuracy_clean_subset"],
        "accuracy_C2_clean_subset_ci95_wilson": c2_summary["accuracy_clean_subset_ci95_wilson"],
        "accuracy_C3": c3_summary["accuracy"],
        "accuracy_C3_ci95_wilson": c3_summary["accuracy_ci95_wilson"],
        "accuracy_C3_clean_subset": c3_summary["accuracy_clean_subset"],
        "accuracy_C3_clean_subset_ci95_wilson": c3_summary["accuracy_clean_subset_ci95_wilson"],
        "accuracy_C4": c4_summary["ft_accuracy"],
        "accuracy_C4_ci95_wilson": c4_summary["ft_accuracy_ci95_wilson"],
        "accuracy_C4_clean_subset": c4_summary["ft_accuracy_clean_subset"],
        "accuracy_C4_clean_subset_ci95_wilson": c4_summary["ft_accuracy_clean_subset_ci95_wilson"],
        "accuracy_C5": c5_summary["ft_accuracy"],
        "accuracy_C5_ci95_wilson": c5_summary["ft_accuracy_ci95_wilson"],
        "accuracy_C5_clean_subset": c5_summary["ft_accuracy_clean_subset"],
        "accuracy_C5_clean_subset_ci95_wilson": c5_summary["ft_accuracy_clean_subset_ci95_wilson"],
        "restoration_rate_C2": c2_restoration,
        "restoration_rate_C2_ci95_wilson": c2_restoration_ci,
        "restoration_rate_C3": c3_restoration,
        "restoration_rate_C3_ci95_wilson": c3_restoration_ci,
        "mcnemar_C4_vs_C2": _pairwise("C2", "C4"),
        "mcnemar_C4_vs_C3": _pairwise("C3", "C4"),
        "mcnemar_C5_vs_C4": _pairwise("C4", "C5"),
        "mcnemar_C5_vs_C2": _pairwise("C2", "C5"),
        "latency_ratio_C4_to_C2": float(c4_summary["wall_clock_seconds_per_item"] / max(float(c2_summary["wall_clock_seconds_per_item"]), 1e-8)),
        "latency_ratio_C5_to_C2": float(c5_summary["wall_clock_seconds_per_item"] / max(float(c2_summary["wall_clock_seconds_per_item"]), 1e-8)),
        "headline_bootstrap_space": bootstrap_space,
        "headline_tau_margin": float(tau_margin),
        "lambda_pos": float(lambda_pos),
        "lambda_neg": float(lambda_neg),
        "metadata": build_run_metadata(
            args=args,
            input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "comparison.json", comparison)
    (out_dir / "finding.txt").write_text("Mechanistic intervention is compared directly against prompt-only strengthening on the matched subset.\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"comparison={out_dir / 'comparison.json'}")
    print(f"finding={out_dir / 'finding.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
