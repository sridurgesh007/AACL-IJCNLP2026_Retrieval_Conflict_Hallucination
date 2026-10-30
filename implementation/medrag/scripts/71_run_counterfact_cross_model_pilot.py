#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import (
    build_metadata,
    choose_best_lambda,
    evaluate_intervention,
    load_counterfact_prepared,
    load_target_site,
    parse_float_grid,
    prepare_baseline_rows,
    select_tau,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CounterFact cross-model pilot for entity token-logit intervention.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument(
        "--evidence_mode",
        default="cache",
        choices=["cache", "question_answer", "relation_or_question_answer"],
        help="How to construct CounterFact evidence context. Use relation_or_question_answer for v2 runs.",
    )
    parser.add_argument("--localization_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_items", type=int, default=500)
    parser.add_argument("--pilot_examples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--lambda_neg_grid", default="2.5,5.0,7.5,10.0")
    parser.add_argument("--lambda_pos_grid", default="0.0,2.5")
    parser.add_argument("--n_bias_steps", type=int, default=1)
    parser.add_argument("--min_margin_auc", type=float, default=0.70)
    parser.add_argument("--min_base_hallucinated", type=int, default=5)
    parser.add_argument("--min_restored", type=int, default=1)
    parser.add_argument("--override_target_layer", type=int, default=None)
    parser.add_argument("--override_target_head", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    pilot_dir = out_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    records, cache_path = load_counterfact_prepared(
        project_root=PROJECT_ROOT,
        cache_path=args.cache_path,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        max_items=int(args.max_items),
        evidence_mode=str(args.evidence_mode),
    )
    records = records[: min(int(args.pilot_examples), len(records))]
    baseline_rows = prepare_baseline_rows(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
    )
    tau_summary, tau_rows = select_tau(baseline_rows)
    if tau_summary["selected_tau"] is None:
        raise RuntimeError("Could not select tau; pilot lacks both faithful and hallucinated base labels.")
    selected_tau = float(tau_summary["selected_tau"])
    site = load_target_site(
        localization_path=(PROJECT_ROOT / args.localization_path).resolve(),
        override_target_layer=args.override_target_layer,
        override_target_head=args.override_target_head,
    )

    lambda_rows: list[dict] = []
    sweep_cache: dict[tuple[float, float], tuple[list[dict], dict]] = {}
    for lambda_pos in parse_float_grid(args.lambda_pos_grid):
        for lambda_neg in parse_float_grid(args.lambda_neg_grid):
            rows, summary = evaluate_intervention(
                baseline_rows=baseline_rows,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tau=float(selected_tau),
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                n_bias_steps=int(args.n_bias_steps),
                max_new_tokens=int(args.max_new_tokens),
            )
            summary.update(
                {
                    "lambda_pos": float(lambda_pos),
                    "lambda_neg": float(lambda_neg),
                    "youden_j": float(summary["restoration_rate"] - summary["harm_rate"]),
                }
            )
            lambda_rows.append(summary)
            sweep_cache[(float(lambda_pos), float(lambda_neg))] = (rows, summary)

    best = choose_best_lambda(lambda_rows)
    pilot_go = bool(
        tau_summary["margin_auc"] is not None
        and float(tau_summary["margin_auc"]) >= float(args.min_margin_auc)
        and int(tau_summary["n_hallucinated"]) >= int(args.min_base_hallucinated)
        and int(best["n_restored"]) >= int(args.min_restored)
    )
    selected_key = (float(best["lambda_pos"]), float(best["lambda_neg"]))
    selected_rows, _ = sweep_cache[selected_key]
    payload = {
        "model_id": str(args.model_id),
        "model_name": str(args.model_name or args.model_id),
        "architecture": str(args.architecture),
        "pilot_go": bool(pilot_go),
        "go_thresholds": {
            "min_margin_auc": float(args.min_margin_auc),
            "min_base_hallucinated": int(args.min_base_hallucinated),
            "min_restored": int(args.min_restored),
        },
        "target_site": site,
        "evidence_mode": str(args.evidence_mode),
        "pilot_examples": int(len(baseline_rows)),
        "margin_auc": tau_summary["margin_auc"],
        "selected_tau": float(selected_tau),
        "tau_summary": tau_summary,
        "tau_candidates": tau_rows,
        "lambda_sweep": lambda_rows,
        "best_row": best,
        "selected_lambda_pos": float(best["lambda_pos"]),
        "selected_lambda_neg": float(best["lambda_neg"]),
        "n_bias_steps": int(args.n_bias_steps),
        "metadata": build_metadata(
            args=args,
            input_files=[cache_path, PROJECT_ROOT / args.localization_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=REPO_ROOT,
        ),
    }
    write_json(pilot_dir / "margin_auc.json", payload)
    write_jsonl(pilot_dir / "baseline_rows.jsonl", baseline_rows)
    write_jsonl(pilot_dir / "selected_intervention_rows.jsonl", selected_rows)
    print(json.dumps({k: payload[k] for k in ["pilot_go", "margin_auc", "selected_tau", "selected_lambda_pos", "selected_lambda_neg"]}, indent=2))
    print(f"pilot={pilot_dir / 'margin_auc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
