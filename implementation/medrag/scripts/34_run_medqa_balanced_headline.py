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
from medrag.medqa_v3 import (
    build_letter_token_sets,
    evaluate_intervention_cell,
    load_conflicts,
    prepare_examples_for_variant,
    set_global_seed,
)
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import git_commit_hash, read_json, sha256_file, write_json, write_jsonl, write_summary_json
from medrag.utils.stats import mcnemar_from_paired, paired_bootstrap_ci

RATE_FIELDS = [
    "baseline_accuracy",
    "ft_accuracy",
    "ft_accuracy_clean_subset",
    "restoration_rate",
    "harm_rate",
    "restoration_rate_clean_subset",
    "harm_rate_clean_subset",
    "gate_fire_rate_base_wrong",
    "gate_fire_rate_base_correct",
    "restoration_rate_param_wrong",
    "restoration_rate_other_wrong",
]

HEADLINE_CONDITIONS: tuple[tuple[str, float, float], ...] = (
    ("H1", 1.5, 1.5),
    ("H2", 2.0, 2.0),
    ("H3", 2.5, 2.5),
    ("H4", 0.0, 2.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify balanced headline candidates for MedQA v3 consolidation.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/consolidation/headline_verify")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--headline_path", default="results/medqa_final_v3/pr_sweep/headline.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bootstrap_space", default="", help="Defaults to the v3 headline bootstrap space.")
    parser.add_argument("--tau_margin", type=float, default=float("nan"), help="Defaults to the v3 headline tau.")
    parser.add_argument("--h4_tolerance", type=float, default=1e-9)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    return parser.parse_args()


def _build_metadata(*, args: argparse.Namespace, wall_clock_seconds: float, input_files: list[Path]) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "git_commit_hash": git_commit_hash(PROJECT_ROOT),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path.resolve()) for path in input_files],
        "input_file_sha256": {str(path.resolve()): sha256_file(path) for path in input_files},
    }


def _flipped_item_ids(rows: list[dict[str, Any]], *, clean_only: bool = False) -> list[str]:
    return sorted(
        {
            str(row.get("item_id"))
            for row in rows
            if not bool(row.get("skipped", False))
            and bool(row.get("in_clean_subset", False) if clean_only else row.get("in_matched_subset", False))
            and str(row.get("ft_letter")) != str(row.get("base_letter"))
        }
    )


def _matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("in_matched_subset", False)) and not bool(row.get("skipped", False))]


def _base_wrong_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in _matched_rows(rows) if not bool(row.get("base_correct", False))]


def _condition_row(
    *,
    condition_id: str,
    lambda_pos: float,
    lambda_neg: float,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    matched_rows = _matched_rows(rows)
    base_wrong_rows = _base_wrong_rows(rows)
    baseline_correct = [1.0 if bool(row.get("base_correct", False)) else 0.0 for row in matched_rows]
    ft_correct = [1.0 if bool(row.get("ft_correct", False)) else 0.0 for row in matched_rows]
    ft_restored = [1.0 if bool(row.get("ft_restored", False)) else 0.0 for row in base_wrong_rows]
    zeros = [0.0] * len(base_wrong_rows)
    ft_acc_delta_ci = paired_bootstrap_ci(baseline_correct, ft_correct, n_boot=int(n_bootstrap), seed=int(seed))
    restoration_ci_boot = paired_bootstrap_ci(zeros, ft_restored, n_boot=int(n_bootstrap), seed=int(seed) + 1)
    return {
        "condition_id": str(condition_id),
        "lambda_pos": float(lambda_pos),
        "lambda_neg": float(lambda_neg),
        "total_lambda_magnitude": float(abs(lambda_pos) + abs(lambda_neg)),
        "baseline_accuracy": float(summary["baseline_accuracy"]),
        "baseline_accuracy_ci95_wilson": summary["baseline_accuracy_ci95_wilson"],
        "ft_accuracy": float(summary["ft_accuracy"]),
        "ft_accuracy_ci95_wilson": summary["ft_accuracy_ci95_wilson"],
        "ft_accuracy_delta_vs_baseline_ci95_paired_bootstrap": [float(ft_acc_delta_ci[0]), float(ft_acc_delta_ci[1])],
        "ft_accuracy_clean_subset": float(summary["ft_accuracy_clean_subset"]),
        "ft_accuracy_clean_subset_ci95_wilson": summary["ft_accuracy_clean_subset_ci95_wilson"],
        "restoration_rate": float(summary["restoration_rate"]),
        "restoration_rate_ci95_wilson": summary["restoration_rate_ci95_wilson"],
        "restoration_rate_ci95_paired_bootstrap": [float(restoration_ci_boot[0]), float(restoration_ci_boot[1])],
        "restoration_rate_clean_subset": float(summary["restoration_rate_clean_subset"]),
        "restoration_rate_clean_subset_ci95_wilson": summary["restoration_rate_clean_subset_ci95_wilson"],
        "harm_rate": float(summary["harm_rate"]),
        "harm_rate_ci95_wilson": summary["harm_rate_ci95_wilson"],
        "harm_rate_clean_subset": float(summary["harm_rate_clean_subset"]),
        "harm_rate_clean_subset_ci95_wilson": summary["harm_rate_clean_subset_ci95_wilson"],
        "gate_fire_rate_base_wrong": float(summary["gate_fire_rate_base_wrong"]),
        "gate_fire_rate_base_wrong_ci95_wilson": summary["gate_fire_rate_base_wrong_ci95_wilson"],
        "gate_fire_rate_base_correct": float(summary["gate_fire_rate_base_correct"]),
        "gate_fire_rate_base_correct_ci95_wilson": summary["gate_fire_rate_base_correct_ci95_wilson"],
        "restoration_improvement": float(summary["restoration_improvement"]),
        "youden_j": float(summary["restoration_rate"] - summary["gate_fire_rate_base_correct"]),
        "latency_per_item": float(summary["wall_clock_seconds_per_item"]),
        "mcnemar_ft_vs_baseline": summary["mcnemar_ft_vs_baseline"],
        "n_examples_matched": int(summary["n_examples_matched"]),
        "n_examples_clean": int(summary["n_examples_clean"]),
        "n_base_wrong": int(summary["n_base_wrong"]),
        "n_restored": int(summary["n_restored"]),
        "n_harmed": int(summary["n_harmed"]),
        "flipped_item_ids": _flipped_item_ids(rows, clean_only=False),
        "flipped_item_ids_clean": _flipped_item_ids(rows, clean_only=True),
        "out_dir": str(out_dir),
    }


def _pairwise_vs_h4(
    *,
    candidate_id: str,
    candidate_rows: list[dict[str, Any]],
    h4_rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    h4_by_id = {str(row["item_id"]): row for row in _matched_rows(h4_rows)}
    candidate_by_id = {str(row["item_id"]): row for row in _matched_rows(candidate_rows)}
    item_ids = sorted(set(h4_by_id.keys()) & set(candidate_by_id.keys()))
    baseline_correct = [bool(h4_by_id[item_id]["ft_correct"]) for item_id in item_ids]
    current_correct = [bool(candidate_by_id[item_id]["ft_correct"]) for item_id in item_ids]
    accuracy_ci = paired_bootstrap_ci(
        [1.0 if x else 0.0 for x in baseline_correct],
        [1.0 if x else 0.0 for x in current_correct],
        n_boot=int(n_bootstrap),
        seed=int(seed),
    )

    base_wrong_ids = sorted(
        item_id for item_id in item_ids if not bool(h4_by_id[item_id].get("base_correct", False))
    )
    h4_restored = [1.0 if bool(h4_by_id[item_id]["ft_restored"]) else 0.0 for item_id in base_wrong_ids]
    candidate_restored = [1.0 if bool(candidate_by_id[item_id]["ft_restored"]) else 0.0 for item_id in base_wrong_ids]
    restoration_ci = paired_bootstrap_ci(h4_restored, candidate_restored, n_boot=int(n_bootstrap), seed=int(seed) + 1)

    return {
        "candidate_id": str(candidate_id),
        "mcnemar_correctness_vs_H4": mcnemar_from_paired(baseline_correct, current_correct),
        "ft_accuracy_diff_vs_H4_ci95_paired_bootstrap": [float(accuracy_ci[0]), float(accuracy_ci[1])],
        "restoration_rate_diff_vs_H4_ci95_paired_bootstrap": [float(restoration_ci[0]), float(restoration_ci[1])],
    }


def _check_h4_reproduction(current: dict[str, Any], reference: dict[str, Any], tolerance: float) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    scalar_fields = [
        "baseline_accuracy",
        "ft_accuracy",
        "ft_accuracy_clean_subset",
        "restoration_rate",
        "restoration_rate_clean_subset",
        "harm_rate",
        "harm_rate_clean_subset",
        "gate_fire_rate_base_wrong",
        "gate_fire_rate_base_correct",
    ]
    integer_fields = ["n_restored", "n_harmed", "n_examples_matched", "n_examples_clean", "n_base_wrong"]
    for field in scalar_fields:
        got = float(current[field])
        ref = float(reference[field])
        checks.append(
            {
                "name": str(field),
                "got": float(got),
                "ref": float(ref),
                "abs_diff": float(abs(got - ref)),
                "passed": bool(abs(got - ref) <= tolerance),
            }
        )
    for field in integer_fields:
        got = int(current[field])
        ref = int(reference[field])
        checks.append(
            {
                "name": str(field),
                "got": int(got),
                "ref": int(ref),
                "abs_diff": int(abs(got - ref)),
                "passed": bool(got == ref),
            }
        )
    current_mc = current["mcnemar_ft_vs_baseline"]
    reference_mc = reference["mcnemar_ft_vs_baseline"]
    for field in ("b", "c", "n_discordant"):
        got = int(current_mc[field])
        ref = int(reference_mc[field])
        checks.append(
            {
                "name": f"mcnemar_{field}",
                "got": int(got),
                "ref": int(ref),
                "abs_diff": int(abs(got - ref)),
                "passed": bool(got == ref),
            }
        )
    got_p = float(current_mc["p_value"])
    ref_p = float(reference_mc["p_value"])
    checks.append(
        {
            "name": "mcnemar_p_value",
            "got": float(got_p),
            "ref": float(ref_p),
            "abs_diff": float(abs(got_p - ref_p)),
            "passed": bool(abs(got_p - ref_p) <= tolerance),
        }
    )
    failed = [check for check in checks if not bool(check["passed"])]
    return {
        "tolerance": float(tolerance),
        "all_passed": bool(not failed),
        "checks": checks,
        "failed_checks": failed,
    }


def _recommend_headline(rows: list[dict[str, Any]], pairwise_vs_h4: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["youden_j"]),
            -float(row["total_lambda_magnitude"]),
        ),
        reverse=True,
    )
    best = dict(ordered[0])
    h4 = next(row for row in rows if str(row["condition_id"]) == "H4")
    if str(best["condition_id"]) == "H4":
        reason = "H4 remains the highest-Youden operating point among H1-H4."
    else:
        pairwise = pairwise_vs_h4[str(best["condition_id"])]
        p_value = float(pairwise["mcnemar_correctness_vs_H4"]["p_value"])
        reason = (
            f"{best['condition_id']} exceeds H4 on Youden's J "
            f"({float(best['youden_j']):.4f} vs {float(h4['youden_j']):.4f}); "
            f"McNemar p vs H4 = {p_value:.4g}."
        )
    return {
        "condition_id": str(best["condition_id"]),
        "lambda_pos": float(best["lambda_pos"]),
        "lambda_neg": float(best["lambda_neg"]),
        "total_lambda_magnitude": float(best["total_lambda_magnitude"]),
        "ft_accuracy": float(best["ft_accuracy"]),
        "restoration_rate": float(best["restoration_rate"]),
        "harm_rate": float(best["harm_rate"]),
        "youden_j": float(best["youden_j"]),
        "reason": reason,
    }


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = (PROJECT_ROOT / args.conflict_path).resolve()
    matched_subset_path = (PROJECT_ROOT / args.matched_subset_path).resolve()
    label_audit_path = (PROJECT_ROOT / args.label_audit_path).resolve()
    headline_path = (PROJECT_ROOT / args.headline_path).resolve()

    conflicts = load_conflicts(conflict_path)
    matched_payload = read_json(matched_subset_path)
    label_audit = load_label_audit(label_audit_path)
    clean_ids = set(clean_subset_item_ids(matched_payload, label_audit))
    matched_ids = {str(x) for x in matched_payload.get("matched_item_ids", [])}
    headline_reference = read_json(headline_path)

    bootstrap_space = str(args.bootstrap_space).strip() or str(headline_reference["bootstrap_space"])
    if bootstrap_space != "E_letter":
        raise ValueError(f"Expected E_letter bootstrap space for headline verification, got {bootstrap_space}")
    tau_margin = float(args.tau_margin)
    if tau_margin != tau_margin:  # NaN check
        tau_margin = float(headline_reference["tau_margin"])

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = build_letter_token_sets(tokenizer)
    cache = ActivationCache((PROJECT_ROOT / args.cache_dir).resolve())
    prepared = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=cache,
        model_id=str(args.model_id),
        prompt_variant="vanilla",
        p_set_mode=str(args.p_set_mode),
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )

    condition_rows: list[dict[str, Any]] = []
    results_by_condition: dict[str, list[dict[str, Any]]] = {}
    summaries_by_condition: dict[str, dict[str, Any]] = {}

    for index, (condition_id, lambda_pos, lambda_neg) in enumerate(HEADLINE_CONDITIONS):
        cell_started = time.perf_counter()
        raw_summary, rows = evaluate_intervention_cell(
            prepared_records=prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=letter_token_sets,
            bootstrap_space=str(bootstrap_space),
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
        cell_dir = out_dir / str(condition_id)
        cell_dir.mkdir(parents=True, exist_ok=True)
        enriched = dict(raw_summary)
        enriched.update(
            {
                "condition_id": str(condition_id),
                "bootstrap_space": str(bootstrap_space),
                "intervention_space": "E_letter",
                "lambda_pos": float(lambda_pos),
                "lambda_neg": float(lambda_neg),
                "tau_margin": float(tau_margin),
                "alpha": float(args.alpha),
                "target_layer": int(args.target_layer),
                "target_head": int(args.target_head),
                "p_set_mode": str(args.p_set_mode),
                "flipped_item_ids": _flipped_item_ids(rows, clean_only=False),
                "flipped_item_ids_clean": _flipped_item_ids(rows, clean_only=True),
                "metadata": _build_metadata(
                    args=args,
                    wall_clock_seconds=float(time.perf_counter() - cell_started),
                    input_files=[conflict_path, matched_subset_path, label_audit_path, headline_path],
                ),
            }
        )
        write_summary_json(cell_dir / "summary.json", enriched, rate_fields=RATE_FIELDS)
        write_jsonl(cell_dir / "results.jsonl", rows)

        summaries_by_condition[str(condition_id)] = enriched
        results_by_condition[str(condition_id)] = rows
        condition_rows.append(
            _condition_row(
                condition_id=str(condition_id),
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                summary=enriched,
                rows=rows,
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed) + (index * 10),
                out_dir=cell_dir,
            )
        )
        print(json.dumps({"condition_id": condition_id, "summary": str(cell_dir / "summary.json")}, indent=2))

    h4_reproduction = _check_h4_reproduction(
        summaries_by_condition["H4"],
        headline_reference,
        tolerance=float(args.h4_tolerance),
    )
    if not bool(h4_reproduction["all_passed"]):
        write_json(
            out_dir / "comparison.json",
            {
                "bootstrap_space": str(bootstrap_space),
                "tau_margin": float(tau_margin),
                "conditions": condition_rows,
                "h4_reproduction": h4_reproduction,
                "metadata": _build_metadata(
                    args=args,
                    wall_clock_seconds=float(time.perf_counter() - started),
                    input_files=[conflict_path, matched_subset_path, label_audit_path, headline_path],
                ),
            },
        )
        raise RuntimeError(f"H4 failed to reproduce v3 headline. See {out_dir / 'comparison.json'}")

    pairwise_vs_h4: dict[str, Any] = {}
    for condition_id in ("H1", "H2", "H3"):
        pairwise_vs_h4[condition_id] = _pairwise_vs_h4(
            candidate_id=condition_id,
            candidate_rows=results_by_condition[condition_id],
            h4_rows=results_by_condition["H4"],
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 500 + int(condition_id[1:]),
        )

    recommended_headline = _recommend_headline(condition_rows, pairwise_vs_h4)
    comparison = {
        "bootstrap_space": str(bootstrap_space),
        "intervention_space": "E_letter",
        "tau_margin": float(tau_margin),
        "alpha": float(args.alpha),
        "conditions": condition_rows,
        "pairwise_vs_H4": pairwise_vs_h4,
        "h4_reproduction": h4_reproduction,
        "recommended_headline": recommended_headline,
        "metadata": _build_metadata(
            args=args,
            wall_clock_seconds=float(time.perf_counter() - started),
            input_files=[conflict_path, matched_subset_path, label_audit_path, headline_path],
        ),
    }
    write_json(out_dir / "comparison.json", comparison)

    print(f"comparison={out_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
