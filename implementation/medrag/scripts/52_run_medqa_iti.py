#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.baselines.iti import ITIController, load_probe_bundle, save_probe_bundle, select_heads, train_probes
from medrag.causal_localization import build_metadata
from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import (
    build_letter_token_sets,
    decode_letter,
    generate_ids,
    load_conflicts,
    prepare_examples_for_variant,
    set_global_seed,
)
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import read_json, write_json, write_jsonl, write_summary_json
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ITI baseline on MedQA using the v3 matched subset protocol.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/baselines/iti")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--killtest_root", default="results/medqa_final_v3/killtest")
    parser.add_argument("--cache_dir", default="cache/medqa_acts_killtest")
    parser.add_argument("--truthfulqa_probe_bundle", default="results/medqa_final_v3/baselines/iti/truthfulqa/truthfulqa_probes.pt")
    parser.add_argument("--run_option_a", type=int, default=1)
    parser.add_argument("--run_option_b", type=int, default=1)
    parser.add_argument("--target_layers", default="", help="Comma-separated layer indices. Empty means all layers.")
    parser.add_argument("--k_values", default="10,20,48")
    parser.add_argument("--alpha_values", default="5,10,15,20")
    parser.add_argument("--position_modes", default="every_position,last_position_only")
    parser.add_argument("--truthfulqa_k", type=int, default=48)
    parser.add_argument("--truthfulqa_alpha", type=float, default=15.0)
    parser.add_argument("--prompt_variant", default="vanilla")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    value = str(value or "").strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value or "").split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _rate(rows: list[dict[str, Any]], key: str) -> tuple[float, list[float], int, int]:
    n = int(len(rows))
    k = int(sum(1 for row in rows if bool(row.get(key, False))))
    ci = list(wilson_ci(k, n))
    return float(k / max(n, 1)), [float(ci[0]), float(ci[1])], k, n


def summarize_iti_results(results: list[dict[str, Any]], wall_clock_seconds: float) -> dict[str, Any]:
    matched_rows = [row for row in results if bool(row.get("in_matched_subset", False)) and not bool(row.get("skipped", False))]
    if not matched_rows:
        raise RuntimeError("ITI evaluation produced zero matched-subset rows.")
    clean_rows = [row for row in matched_rows if bool(row.get("in_clean_subset", False))]
    base_wrong = [row for row in matched_rows if not bool(row.get("base_correct", False))]
    base_correct = [row for row in matched_rows if bool(row.get("base_correct", False))]
    clean_base_wrong = [row for row in clean_rows if not bool(row.get("base_correct", False))]
    clean_base_correct = [row for row in clean_rows if bool(row.get("base_correct", False))]

    baseline_accuracy, baseline_ci, _, _ = _rate(matched_rows, "base_correct")
    ft_accuracy, ft_ci, _, _ = _rate(matched_rows, "iti_correct")
    restoration_rate, restoration_ci, n_restored, _ = _rate(base_wrong, "iti_restored")
    harm_rate, harm_ci, n_harmed, _ = _rate(base_correct, "iti_harmed")
    clean_restoration_rate, clean_restoration_ci, n_restored_clean, _ = _rate(clean_base_wrong, "iti_restored")
    clean_harm_rate, clean_harm_ci, n_harmed_clean, _ = _rate(clean_base_correct, "iti_harmed")
    clean_accuracy, clean_accuracy_ci, _, _ = _rate(clean_rows, "iti_correct")
    summary = {
        "n_examples_raw": int(len(results)),
        "n_examples_matched": int(len(matched_rows)),
        "n_examples_clean": int(len(clean_rows)),
        "n_base_wrong": int(len(base_wrong)),
        "n_base_correct": int(len(base_correct)),
        "baseline_accuracy": float(baseline_accuracy),
        "baseline_accuracy_ci95_wilson": baseline_ci,
        "ft_accuracy": float(ft_accuracy),
        "ft_accuracy_ci95_wilson": ft_ci,
        "ft_accuracy_clean_subset": float(clean_accuracy),
        "ft_accuracy_clean_subset_ci95_wilson": clean_accuracy_ci,
        "restoration_rate": float(restoration_rate),
        "restoration_rate_ci95_wilson": restoration_ci,
        "harm_rate": float(harm_rate),
        "harm_rate_ci95_wilson": harm_ci,
        "restoration_rate_clean_subset": float(clean_restoration_rate),
        "restoration_rate_clean_subset_ci95_wilson": clean_restoration_ci,
        "harm_rate_clean_subset": float(clean_harm_rate),
        "harm_rate_clean_subset_ci95_wilson": clean_harm_ci,
        "restoration_improvement": float(restoration_rate - harm_rate),
        "n_restored": int(n_restored),
        "n_harmed": int(n_harmed),
        "n_restored_clean": int(n_restored_clean),
        "n_harmed_clean": int(n_harmed_clean),
        "mcnemar_ft_vs_baseline": mcnemar_from_paired(
            [bool(row.get("base_correct", False)) for row in matched_rows],
            [bool(row.get("iti_correct", False)) for row in matched_rows],
        ),
        "wall_clock_seconds": float(wall_clock_seconds),
        "wall_clock_seconds_per_item": float(wall_clock_seconds / max(len(matched_rows), 1)),
    }
    return summary


def evaluate_iti_cell(
    *,
    prepared_records: list[Any],
    model,
    tokenizer,
    device: torch.device,
    letter_token_sets: dict[str, set[int]],
    selected_heads,
    alpha: float,
    position_mode: str,
    matched_item_ids: set[str],
    clean_item_ids: set[str],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ITIController(
        model=model,
        selected_heads=selected_heads,
        alpha=float(alpha),
        position_mode=str(position_mode),
    ):
        for rec in prepared_records:
            iti_ids = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens))
            iti_letter = decode_letter(iti_ids, tokenizer, letter_token_sets)
            iti_correct = bool(iti_letter == rec.correct)
            rows.append(
                {
                    "item_id": str(rec.item_id),
                    "question": str(rec.question),
                    "correct": str(rec.correct),
                    "parametric": str(rec.parametric),
                    "base_letter": rec.base_letter,
                    "iti_letter": iti_letter,
                    "base_correct": bool(rec.base_correct),
                    "iti_correct": bool(iti_correct),
                    "iti_restored": bool((not rec.base_correct) and iti_correct),
                    "iti_harmed": bool(rec.base_correct and (not iti_correct)),
                    "in_matched_subset": bool(rec.item_id in matched_item_ids),
                    "in_clean_subset": bool(rec.item_id in clean_item_ids),
                    "skipped": False,
                }
            )
    summary = summarize_iti_results(rows, float(time.perf_counter() - t0))
    return summary, rows


def _probe_summary_payload(probe_metadata: dict[str, Any], probe_results: dict[tuple[int, int], Any], *, top_n: int = 20) -> dict[str, Any]:
    ranked = sorted(
        probe_results.values(),
        key=lambda probe: (float(probe.accuracy), abs(float(probe.separation))),
        reverse=True,
    )
    return {
        **probe_metadata,
        "n_probes": int(len(probe_results)),
        "top_heads": [
            {
                "layer": int(probe.layer),
                "head": int(probe.head),
                "accuracy": float(probe.accuracy),
                "separation": float(probe.separation),
                "threshold": float(probe.threshold),
            }
            for probe in ranked[: min(int(top_n), len(ranked))]
        ],
    }


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No ITI cells were evaluated.")
    return max(
        rows,
        key=lambda row: (
            float(row["restoration_rate_clean_subset"]),
            float(row["restoration_rate"]),
            -float(row["harm_rate_clean_subset"]),
            -float(row["harm_rate"]),
            float(row["ft_accuracy_clean_subset"]),
            float(row["ft_accuracy"]),
            -float(row["alpha"]),
            -int(row["selected_k"]),
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _existing_correctness_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        current_correct = bool(row.get("condition_correct", row.get("ft_correct", False)))
        normalized.append(
            {
                "item_id": str(row["item_id"]),
                "current_correct": bool(current_correct),
                "base_correct": bool(row.get("base_correct", False)),
                "in_matched_subset": bool(row.get("in_matched_subset", False)),
                "in_clean_subset": bool(row.get("in_clean_subset", False)),
            }
        )
    return normalized


def _restoration_against_baseline(
    condition_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    clean_only: bool = False,
) -> tuple[float, list[float]]:
    base_map = {str(row["item_id"]): row for row in baseline_rows if bool(row.get("in_matched_subset", False))}
    cond_map = {str(row["item_id"]): row for row in condition_rows if bool(row.get("in_matched_subset", False))}
    eligible = [
        item_id
        for item_id, row in base_map.items()
        if (not bool(row.get("current_correct", False)))
        and item_id in cond_map
        and ((not clean_only) or bool(row.get("in_clean_subset", False)))
    ]
    restored = int(sum(1 for item_id in eligible if bool(cond_map[item_id].get("current_correct", False))))
    ci = list(wilson_ci(restored, len(eligible)))
    return float(restored / max(len(eligible), 1)), [float(ci[0]), float(ci[1])]


def _harm_against_baseline(
    condition_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    clean_only: bool = False,
) -> tuple[float, list[float]]:
    base_map = {str(row["item_id"]): row for row in baseline_rows if bool(row.get("in_matched_subset", False))}
    cond_map = {str(row["item_id"]): row for row in condition_rows if bool(row.get("in_matched_subset", False))}
    eligible = [
        item_id
        for item_id, row in base_map.items()
        if bool(row.get("current_correct", False))
        and item_id in cond_map
        and ((not clean_only) or bool(row.get("in_clean_subset", False)))
    ]
    harmed = int(sum(1 for item_id in eligible if not bool(cond_map[item_id].get("current_correct", False))))
    ci = list(wilson_ci(harmed, len(eligible)))
    return float(harmed / max(len(eligible), 1)), [float(ci[0]), float(ci[1])]


def _pairwise_mcnemar(reference_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref = {str(row["item_id"]): bool(row["current_correct"]) for row in reference_rows if bool(row.get("in_matched_subset", False))}
    cur = {str(row["item_id"]): bool(row["current_correct"]) for row in current_rows if bool(row.get("in_matched_subset", False))}
    item_ids = sorted(set(ref.keys()) & set(cur.keys()))
    return mcnemar_from_paired([ref[item_id] for item_id in item_ids], [cur[item_id] for item_id in item_ids])


def build_comparison(
    *,
    killtest_root: Path,
    best_option_a: dict[str, Any] | None,
    best_option_b: dict[str, Any] | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    c1_rows = _existing_correctness_rows(killtest_root / "C1" / "results.jsonl")
    c2_rows = _existing_correctness_rows(killtest_root / "C2" / "results.jsonl")
    c3_rows = _existing_correctness_rows(killtest_root / "C3" / "results.jsonl")
    c4_rows = _existing_correctness_rows(killtest_root / "C4" / "results.jsonl")
    c5_rows = _existing_correctness_rows(killtest_root / "C5" / "results.jsonl")
    c1_summary = read_json(killtest_root / "C1" / "summary.json")
    c2_summary = read_json(killtest_root / "C2" / "summary.json")
    c3_summary = read_json(killtest_root / "C3" / "summary.json")
    c4_summary = read_json(killtest_root / "C4" / "summary.json")
    c5_summary = read_json(killtest_root / "C5" / "summary.json")

    c2_restore, c2_restore_ci = _restoration_against_baseline(c2_rows, c1_rows, clean_only=False)
    c2_restore_clean, c2_restore_clean_ci = _restoration_against_baseline(c2_rows, c1_rows, clean_only=True)
    c2_harm, c2_harm_ci = _harm_against_baseline(c2_rows, c1_rows, clean_only=False)
    c3_restore, c3_restore_ci = _restoration_against_baseline(c3_rows, c1_rows, clean_only=False)
    c3_restore_clean, c3_restore_clean_ci = _restoration_against_baseline(c3_rows, c1_rows, clean_only=True)
    c3_harm, c3_harm_ci = _harm_against_baseline(c3_rows, c1_rows, clean_only=False)
    rows.extend(
        [
            {
                "condition_id": "C1",
                "label": "C1: Vanilla",
                "source": "v3",
                "restoration_rate_clean_subset": 0.0,
                "restoration_rate_clean_subset_ci95_wilson": [0.0, 0.0],
                "restoration_rate": 0.0,
                "restoration_rate_ci95_wilson": [0.0, 0.0],
                "harm_rate": 0.0,
                "harm_rate_ci95_wilson": [0.0, 0.0],
                "accuracy": float(c1_summary["accuracy"]),
                "accuracy_ci95_wilson": c1_summary["accuracy_ci95_wilson"],
            },
            {
                "condition_id": "C2",
                "label": "C2: Trust instruction",
                "source": "v3",
                "restoration_rate_clean_subset": float(c2_restore_clean),
                "restoration_rate_clean_subset_ci95_wilson": c2_restore_clean_ci,
                "restoration_rate": float(c2_restore),
                "restoration_rate_ci95_wilson": c2_restore_ci,
                "harm_rate": float(c2_harm),
                "harm_rate_ci95_wilson": c2_harm_ci,
                "accuracy": float(c2_summary["accuracy"]),
                "accuracy_ci95_wilson": c2_summary["accuracy_ci95_wilson"],
            },
            {
                "condition_id": "C3",
                "label": "C3: Authoritative quote",
                "source": "v3",
                "restoration_rate_clean_subset": float(c3_restore_clean),
                "restoration_rate_clean_subset_ci95_wilson": c3_restore_clean_ci,
                "restoration_rate": float(c3_restore),
                "restoration_rate_ci95_wilson": c3_restore_ci,
                "harm_rate": float(c3_harm),
                "harm_rate_ci95_wilson": c3_harm_ci,
                "accuracy": float(c3_summary["accuracy"]),
                "accuracy_ci95_wilson": c3_summary["accuracy_ci95_wilson"],
            },
            {
                "condition_id": "C4",
                "label": "C4: Our mechanism",
                "source": "v3",
                "restoration_rate_clean_subset": float(c4_summary["restoration_rate_clean_subset"]),
                "restoration_rate_clean_subset_ci95_wilson": c4_summary["restoration_rate_clean_subset_ci95_wilson"],
                "restoration_rate": float(c4_summary["restoration_rate"]),
                "restoration_rate_ci95_wilson": c4_summary["restoration_rate_ci95_wilson"],
                "harm_rate": float(c4_summary["harm_rate"]),
                "harm_rate_ci95_wilson": c4_summary["harm_rate_ci95_wilson"],
                "accuracy": float(c4_summary["ft_accuracy"]),
                "accuracy_ci95_wilson": c4_summary["ft_accuracy_ci95_wilson"],
            },
            {
                "condition_id": "C5",
                "label": "C5: Mechanism + trust",
                "source": "v3",
                "restoration_rate_clean_subset": float(c5_summary["restoration_rate_clean_subset"]),
                "restoration_rate_clean_subset_ci95_wilson": c5_summary["restoration_rate_clean_subset_ci95_wilson"],
                "restoration_rate": float(c5_summary["restoration_rate"]),
                "restoration_rate_ci95_wilson": c5_summary["restoration_rate_ci95_wilson"],
                "harm_rate": float(c5_summary["harm_rate"]),
                "harm_rate_ci95_wilson": c5_summary["harm_rate_ci95_wilson"],
                "accuracy": float(c5_summary["ft_accuracy"]),
                "accuracy_ci95_wilson": c5_summary["ft_accuracy_ci95_wilson"],
            },
        ]
    )

    reference_rows = {"C1": c1_rows, "C2": c2_rows, "C4": c4_rows, "C5": c5_rows}
    for condition_id, payload in (("C6", best_option_a), ("C7", best_option_b)):
        if payload is None:
            continue
        payload_rows = _read_jsonl(Path(str(payload["results_path"])))
        current_rows = [
            {
                "item_id": str(row["item_id"]),
                "current_correct": bool(row["iti_correct"]),
                "in_matched_subset": bool(row["in_matched_subset"]),
            }
            for row in payload_rows
        ]
        rows.append(
            {
                "condition_id": condition_id,
                "label": (
                    "C6: ITI (domain-matched)"
                    if condition_id == "C6"
                    else "C7: ITI (TruthfulQA probes)"
                ),
                "source": "iti",
                "position_mode": str(payload["position_mode"]),
                "probe_training_source": str(payload["probe_training_source"]),
                "selected_k": int(payload["selected_k"]),
                "selected_alpha": float(payload["alpha"]),
                "restoration_rate_clean_subset": float(payload["restoration_rate_clean_subset"]),
                "restoration_rate_clean_subset_ci95_wilson": payload["restoration_rate_clean_subset_ci95_wilson"],
                "restoration_rate": float(payload["restoration_rate"]),
                "restoration_rate_ci95_wilson": payload["restoration_rate_ci95_wilson"],
                "harm_rate": float(payload["harm_rate"]),
                "harm_rate_ci95_wilson": payload["harm_rate_ci95_wilson"],
                "accuracy": float(payload["ft_accuracy"]),
                "accuracy_ci95_wilson": payload["ft_accuracy_ci95_wilson"],
                "mcnemar_vs_C1": _pairwise_mcnemar(reference_rows["C1"], current_rows),
                "mcnemar_vs_C2": _pairwise_mcnemar(reference_rows["C2"], current_rows),
                "mcnemar_vs_C4": _pairwise_mcnemar(reference_rows["C4"], current_rows),
                "mcnemar_vs_C5": _pairwise_mcnemar(reference_rows["C5"], current_rows),
            }
        )
    return {"table_rows": rows}


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    target_layers = parse_csv_ints(args.target_layers)
    if not target_layers:
        target_layers = list(range(int(getattr(model.config, "num_hidden_layers"))))
    position_modes = parse_csv_strings(args.position_modes)
    k_values = parse_csv_ints(args.k_values)
    alpha_values = parse_csv_floats(args.alpha_values)

    conflicts = load_conflicts((PROJECT_ROOT / args.conflict_path).resolve())
    matched_payload = read_json((PROJECT_ROOT / args.matched_subset_path).resolve())
    label_audit = load_label_audit((PROJECT_ROOT / args.label_audit_path).resolve())
    matched_ids = {str(x) for x in matched_payload.get("matched_item_ids", [])}
    clean_ids = set(clean_subset_item_ids(matched_payload, label_audit))
    cache = ActivationCache((PROJECT_ROOT / args.cache_dir).resolve())
    prepared_records = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=cache,
        model_id=str(args.model_id),
        prompt_variant=str(args.prompt_variant),
        p_set_mode=str(args.p_set_mode),
        target_layer=int(target_layers[-1]),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=build_letter_token_sets(tokenizer),
    )
    letter_token_sets = build_letter_token_sets(tokenizer)
    medqa_probe_dataset = [
        {
            "item_id": str(rec.item_id),
            "prompt_ids": rec.prompt_ids.detach().clone().cpu(),
            "label": int(bool(rec.base_correct)),
        }
        for rec in prepared_records
        if rec.base_letter in {"A", "B", "C", "D"}
    ]
    if not medqa_probe_dataset:
        raise RuntimeError("No MedQA probe-training examples were available after prompt preparation.")

    aggregate: dict[str, Any] = {
        "probe_method": "mass_mean_direction",
        "target_layers": [int(x) for x in target_layers],
        "position_modes": position_modes,
        "medqa_option_a_cells": [],
        "truthfulqa_option_b_cells": [],
    }
    best_option_a: dict[str, Any] | None = None
    best_option_b: dict[str, Any] | None = None

    if bool(int(args.run_option_a)):
        option_a_dir = out_dir / "domain_matched"
        option_a_dir.mkdir(parents=True, exist_ok=True)
        probe_results = train_probes(
            model=model,
            tokenizer=tokenizer,
            dataset=medqa_probe_dataset,
            target_layers=target_layers,
            device=device,
        )
        probe_metadata = {
            "probe_method": "mass_mean_direction",
            "probe_training_source": "medqa_all500",
            "n_probe_training_items": int(len(medqa_probe_dataset)),
            "n_probe_training_matched_overlap": int(sum(1 for rec in medqa_probe_dataset if str(rec["item_id"]) in matched_ids)),
            "n_probe_training_clean_overlap": int(sum(1 for rec in medqa_probe_dataset if str(rec["item_id"]) in clean_ids)),
            "target_layers": [int(x) for x in target_layers],
        }
        save_probe_bundle(option_a_dir / "medqa_probes.pt", probe_results=probe_results, metadata=probe_metadata)
        write_json(option_a_dir / "probe_summary.json", _probe_summary_payload(probe_metadata, probe_results))

        for position_mode in position_modes:
            for k in k_values:
                selected_heads = select_heads(probe_results, int(k))
                for alpha in alpha_values:
                    summary, rows = evaluate_iti_cell(
                        prepared_records=prepared_records,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        letter_token_sets=letter_token_sets,
                        selected_heads=selected_heads,
                        alpha=float(alpha),
                        position_mode=str(position_mode),
                        matched_item_ids=matched_ids,
                        clean_item_ids=clean_ids,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    summary.update(
                        {
                            "probe_training_source": "medqa_all500",
                            "position_mode": str(position_mode),
                            "selected_k": int(k),
                            "alpha": float(alpha),
                            "n_items_evaluated": int(len(matched_ids)),
                            "n_items_in_probe_training": int(len(medqa_probe_dataset)),
                            "n_items_in_probe_training_and_matched_subset": int(
                                sum(1 for rec in medqa_probe_dataset if str(rec["item_id"]) in matched_ids)
                            ),
                            "selected_heads_top20": [
                                {
                                    "layer": int(head.layer),
                                    "head": int(head.head),
                                    "accuracy": float(head.accuracy),
                                    "separation": float(head.separation),
                                }
                                for head in selected_heads[: min(20, len(selected_heads))]
                            ],
                            "metadata": build_metadata(
                                args=args,
                                input_files=[
                                    PROJECT_ROOT / args.conflict_path,
                                    PROJECT_ROOT / args.matched_subset_path,
                                    PROJECT_ROOT / args.label_audit_path,
                                ],
                                seed=int(args.seed),
                                wall_clock_seconds=float(summary["wall_clock_seconds"]),
                                project_root=PROJECT_ROOT,
                            ),
                        }
                    )
                    cell_dir = option_a_dir / str(position_mode) / f"k_{int(k)}" / f"alpha_{slug_float(alpha)}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    write_summary_json(
                        cell_dir / "summary.json",
                        summary,
                        rate_fields=[
                            "baseline_accuracy",
                            "ft_accuracy",
                            "ft_accuracy_clean_subset",
                            "restoration_rate",
                            "harm_rate",
                            "restoration_rate_clean_subset",
                            "harm_rate_clean_subset",
                        ],
                    )
                    write_jsonl(cell_dir / "results.jsonl", rows)
                    aggregate["medqa_option_a_cells"].append({**summary, "results_path": str((cell_dir / "results.jsonl").resolve())})
        best_option_a = _select_best(aggregate["medqa_option_a_cells"])
        write_json(option_a_dir / "best.json", best_option_a)

    if bool(int(args.run_option_b)):
        option_b_dir = out_dir / "truthfulqa_transfer"
        option_b_dir.mkdir(parents=True, exist_ok=True)
        truthfulqa_meta, truthfulqa_probes = load_probe_bundle(PROJECT_ROOT / args.truthfulqa_probe_bundle)
        write_json(option_b_dir / "probe_summary.json", truthfulqa_meta)
        for position_mode in position_modes:
            selected_heads = select_heads(truthfulqa_probes, int(args.truthfulqa_k))
            alpha = float(args.truthfulqa_alpha)
            summary, rows = evaluate_iti_cell(
                prepared_records=prepared_records,
                model=model,
                tokenizer=tokenizer,
                device=device,
                letter_token_sets=letter_token_sets,
                selected_heads=selected_heads,
                alpha=float(alpha),
                position_mode=str(position_mode),
                matched_item_ids=matched_ids,
                clean_item_ids=clean_ids,
                max_new_tokens=int(args.max_new_tokens),
            )
            summary.update(
                {
                    "probe_training_source": "truthfulqa_mc1",
                    "position_mode": str(position_mode),
                    "selected_k": int(args.truthfulqa_k),
                    "alpha": float(alpha),
                    "n_items_evaluated": int(len(matched_ids)),
                    "n_items_in_probe_training": int(truthfulqa_meta.get("n_train", 0)),
                    "n_items_in_probe_training_and_matched_subset": 0,
                    "selected_heads_top20": [
                        {
                            "layer": int(head.layer),
                            "head": int(head.head),
                            "accuracy": float(head.accuracy),
                            "separation": float(head.separation),
                        }
                        for head in selected_heads[: min(20, len(selected_heads))]
                    ],
                    "metadata": build_metadata(
                        args=args,
                        input_files=[
                            PROJECT_ROOT / args.conflict_path,
                            PROJECT_ROOT / args.matched_subset_path,
                            PROJECT_ROOT / args.label_audit_path,
                            PROJECT_ROOT / args.truthfulqa_probe_bundle,
                        ],
                        seed=int(args.seed),
                        wall_clock_seconds=float(summary["wall_clock_seconds"]),
                        project_root=PROJECT_ROOT,
                    ),
                }
            )
            cell_dir = option_b_dir / str(position_mode) / f"k_{int(args.truthfulqa_k)}" / f"alpha_{slug_float(alpha)}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            write_summary_json(
                cell_dir / "summary.json",
                summary,
                rate_fields=[
                    "baseline_accuracy",
                    "ft_accuracy",
                    "ft_accuracy_clean_subset",
                    "restoration_rate",
                    "harm_rate",
                    "restoration_rate_clean_subset",
                    "harm_rate_clean_subset",
                ],
            )
            write_jsonl(cell_dir / "results.jsonl", rows)
            aggregate["truthfulqa_option_b_cells"].append({**summary, "results_path": str((cell_dir / "results.jsonl").resolve())})
        best_option_b = _select_best(aggregate["truthfulqa_option_b_cells"])
        write_json(option_b_dir / "best.json", best_option_b)

    comparison: dict[str, Any] = {}
    killtest_root = (PROJECT_ROOT / args.killtest_root).resolve()
    if killtest_root.exists():
        comparison = build_comparison(
            killtest_root=killtest_root,
            best_option_a=best_option_a,
            best_option_b=best_option_b,
        )
        comparison["metadata"] = build_metadata(
            args=args,
            input_files=[
                PROJECT_ROOT / args.conflict_path,
                PROJECT_ROOT / args.matched_subset_path,
                PROJECT_ROOT / args.label_audit_path,
            ],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=PROJECT_ROOT,
        )
        write_json(out_dir / "comparison.json", comparison)

    aggregate.update(
        {
            "best_option_a": best_option_a,
            "best_option_b": best_option_b,
            "comparison_path": (str((out_dir / "comparison.json").resolve()) if comparison else ""),
            "metadata": build_metadata(
                args=args,
                input_files=[
                    PROJECT_ROOT / args.conflict_path,
                    PROJECT_ROOT / args.matched_subset_path,
                    PROJECT_ROOT / args.label_audit_path,
                ],
                seed=int(args.seed),
                wall_clock_seconds=float(time.perf_counter() - t0),
                project_root=PROJECT_ROOT,
            ),
        }
    )
    write_json(out_dir / "aggregate.json", aggregate)
    print(json.dumps({"best_option_a": best_option_a, "best_option_b": best_option_b}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
