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
    parser = argparse.ArgumentParser(description="MedQA v3 Experiment 1: mechanism decomposition.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/decomposition")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--lambda_pos_values", default="0.0,2.5,5.0,7.5,10.0")
    parser.add_argument("--lambda_neg_values", default="0.0,2.5,5.0,7.5,10.0")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--tau_margin", type=float, default=0.0)
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
    vals: list[float] = []
    seen: set[float] = set()
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value in seen:
            continue
        seen.add(value)
        vals.append(float(value))
    if not vals:
        raise ValueError("Empty grid")
    return vals


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


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

    lp_grid = _parse_grid(str(args.lambda_pos_values))
    ln_grid = _parse_grid(str(args.lambda_neg_values))
    aggregate_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None

    for lp in lp_grid:
        for ln in ln_grid:
            summary, rows = evaluate_intervention_cell(
                prepared_records=prepared,
                model=model,
                tokenizer=tokenizer,
                device=device,
                letter_token_sets=letter_token_sets,
                bootstrap_space="E_letter",
                intervention_space="E_letter",
                lambda_pos=float(lp),
                lambda_neg=float(ln),
                tau_margin=float(args.tau_margin),
                alpha=float(args.alpha),
                target_layer=int(args.target_layer),
                target_head=int(args.target_head),
                matched_item_ids=matched_ids,
                clean_item_ids=clean_ids,
                seed=int(args.seed),
                max_new_tokens=int(args.max_new_tokens),
            )
            cell_dir = out_dir / f"lp{_slug(lp)}_ln{_slug(ln)}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            summary.update(
                {
                    "experiment": "decomposition",
                    "bootstrap_space": "E_letter",
                    "intervention_space": "E_letter",
                    "lambda_pos": float(lp),
                    "lambda_neg": float(ln),
                    "p_set_mode": str(args.p_set_mode),
                    "target_layer": int(args.target_layer),
                    "target_head": int(args.target_head),
                    "alpha": float(args.alpha),
                    "tau_margin": float(args.tau_margin),
                    "n_items_clean": int(len(clean_ids)),
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
                "lambda_pos": float(lp),
                "lambda_neg": float(ln),
                "restoration_rate": float(summary["restoration_rate"]),
                "restoration_rate_ci95_wilson": summary["restoration_rate_ci95_wilson"],
                "harm_rate": float(summary["harm_rate"]),
                "harm_rate_ci95_wilson": summary["harm_rate_ci95_wilson"],
                "gate_fire_rate_base_correct": float(summary["gate_fire_rate_base_correct"]),
                "gate_fire_rate_base_correct_ci95_wilson": summary["gate_fire_rate_base_correct_ci95_wilson"],
                "restoration_improvement": float(summary["restoration_improvement"]),
                "youden_j": float(summary["youden_j"]),
                "out_dir": str(cell_dir),
            }
            aggregate_rows.append(aggregate_row)
            if best_summary is None or float(summary["youden_j"]) > float(best_summary["youden_j"]):
                best_summary = dict(summary)
            print(json.dumps(summary, indent=2))
            print(f"summary={cell_dir / 'summary.json'}")

    if best_summary is None:
        raise RuntimeError("No decomposition cells completed.")

    pure_promo = max((row for row in aggregate_rows if abs(float(row["lambda_neg"])) < 1e-9), key=lambda x: float(x["restoration_rate"]))
    pure_suppress = max((row for row in aggregate_rows if abs(float(row["lambda_pos"])) < 1e-9), key=lambda x: float(x["restoration_rate"]))
    best_dual = max(aggregate_rows, key=lambda x: float(x["youden_j"]))
    if float(best_dual["restoration_rate"]) > max(float(pure_promo["restoration_rate"]), float(pure_suppress["restoration_rate"])):
        finding = "Dual promotion+suppression outperforms the best pure one-sided condition at matched-subset evaluation."
    elif float(pure_promo["restoration_rate"]) >= float(pure_suppress["restoration_rate"]):
        finding = "Suppression does not add beyond pure promotion at the same operating scale on the matched subset."
    else:
        finding = "Pure suppression contributes more than pure promotion under this letter-space decomposition sweep."

    aggregate = {
        "experiment": "decomposition",
        "rows": aggregate_rows,
        "best_decomposition_cell": best_summary,
        "metadata": build_run_metadata(
            args=args,
            input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "aggregate.json", aggregate)
    write_summary_json(out_dir / "best_cell.json", best_summary, rate_fields=RATE_FIELDS)
    (out_dir / "finding.txt").write_text(finding + "\n", encoding="utf-8")
    print(f"aggregate={out_dir / 'aggregate.json'}")
    print(f"best_cell={out_dir / 'best_cell.json'}")
    print(f"finding={out_dir / 'finding.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
