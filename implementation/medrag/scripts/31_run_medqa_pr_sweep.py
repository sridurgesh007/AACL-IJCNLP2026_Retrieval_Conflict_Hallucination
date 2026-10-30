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
from medrag.medqa_v3 import build_letter_token_sets, evaluate_intervention_cell, load_conflicts, prepare_examples_for_variant, set_global_seed
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import build_run_metadata, read_json, write_json, write_jsonl, write_summary_json

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedQA v3 Experiment 2: PR sweep over tau and bootstrap space.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/pr_sweep")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--best_cell_path", default="results/medqa_final_v3/decomposition/best_cell.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--tau_values", default="-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0")
    parser.add_argument("--bootstrap_spaces", default="E_content,E_letter")
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


def _parse_grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if str(x).strip()]


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
    lambda_pos = float(best_cell["lambda_pos"])
    lambda_neg = float(best_cell["lambda_neg"])

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

    tau_grid = _parse_grid(str(args.tau_values))
    bootstrap_spaces = [x.strip() for x in str(args.bootstrap_spaces).split(",") if x.strip()]
    aggregate_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None

    for bootstrap_space in bootstrap_spaces:
        for tau_value in tau_grid:
            summary, rows = evaluate_intervention_cell(
                prepared_records=prepared,
                model=model,
                tokenizer=tokenizer,
                device=device,
                letter_token_sets=letter_token_sets,
                bootstrap_space=str(bootstrap_space),
                intervention_space="E_letter",
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                tau_margin=float(tau_value),
                alpha=float(args.alpha),
                target_layer=int(args.target_layer),
                target_head=int(args.target_head),
                matched_item_ids=matched_ids,
                clean_item_ids=clean_ids,
                seed=int(args.seed),
                max_new_tokens=int(args.max_new_tokens),
            )
            cell_name = f"boot_{bootstrap_space}_tau_{str(tau_value).replace('-', 'm').replace('.', 'p')}"
            cell_dir = out_dir / cell_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            summary.update(
                {
                    "experiment": "pr_sweep",
                    "bootstrap_space": str(bootstrap_space),
                    "intervention_space": "E_letter",
                    "lambda_pos": float(lambda_pos),
                    "lambda_neg": float(lambda_neg),
                    "tau_margin": float(tau_value),
                    "youden_j": float(summary["restoration_rate"] - summary["gate_fire_rate_base_correct"]),
                }
            )
            summary["metadata"] = build_run_metadata(
                args=args,
                input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
                seed=int(args.seed),
                wall_clock_seconds=float(summary["wall_clock_seconds"]),
                project_root=PROJECT_ROOT,
            )
            write_summary_json(cell_dir / "summary.json", summary, rate_fields=RATE_FIELDS)
            write_jsonl(cell_dir / "results.jsonl", rows)
            aggregate_row = {
                "bootstrap_space": str(bootstrap_space),
                "tau_margin": float(tau_value),
                "lambda_pos": float(lambda_pos),
                "lambda_neg": float(lambda_neg),
                "restoration_rate": float(summary["restoration_rate"]),
                "restoration_rate_ci95_wilson": summary["restoration_rate_ci95_wilson"],
                "gate_fire_rate_base_correct": float(summary["gate_fire_rate_base_correct"]),
                "gate_fire_rate_base_correct_ci95_wilson": summary["gate_fire_rate_base_correct_ci95_wilson"],
                "harm_rate": float(summary["harm_rate"]),
                "harm_rate_ci95_wilson": summary["harm_rate_ci95_wilson"],
                "youden_j": float(summary["youden_j"]),
                "out_dir": str(cell_dir),
            }
            aggregate_rows.append(aggregate_row)
            if best_summary is None or float(summary["youden_j"]) > float(best_summary["youden_j"]):
                best_summary = dict(summary)
            print(json.dumps(summary, indent=2))
            print(f"summary={cell_dir / 'summary.json'}")

    if best_summary is None:
        raise RuntimeError("No PR sweep cells completed.")

    pr_curve = [
        {
            "tau": float(row["tau_margin"]),
            "bootstrap_space": str(row["bootstrap_space"]),
            "gate_fire_correct": float(row["gate_fire_rate_base_correct"]),
            "gate_fire_correct_ci": row["gate_fire_rate_base_correct_ci95_wilson"],
            "restoration_rate": float(row["restoration_rate"]),
            "restoration_rate_ci": row["restoration_rate_ci95_wilson"],
        }
        for row in aggregate_rows
    ]
    headline = dict(best_summary)
    aggregate = {
        "experiment": "pr_sweep",
        "lambda_pos": float(lambda_pos),
        "lambda_neg": float(lambda_neg),
        "rows": aggregate_rows,
        "pr_curve": pr_curve,
        "headline_operating_point": headline,
        "metadata": build_run_metadata(
            args=args,
            input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "aggregate.json", aggregate)
    write_summary_json(out_dir / "headline.json", headline, rate_fields=RATE_FIELDS)
    print(f"aggregate={out_dir / 'aggregate.json'}")
    print(f"headline={out_dir / 'headline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
