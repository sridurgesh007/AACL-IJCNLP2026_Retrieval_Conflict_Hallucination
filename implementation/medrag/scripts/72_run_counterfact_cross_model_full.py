#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import (
    build_metadata,
    evaluate_intervention,
    load_counterfact_prepared,
    load_target_site,
    prepare_baseline_rows,
    select_tau,
    write_counterfact_bundle,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full CounterFact cross-model entity token-logit intervention.")
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
    parser.add_argument("--pilot_summary_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_items", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=5)
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
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot = read_json((PROJECT_ROOT / args.pilot_summary_path).resolve())
    if not bool(pilot.get("pilot_go", False)):
        raise RuntimeError(f"Refusing full CounterFact run because pilot_go is false: {PROJECT_ROOT / args.pilot_summary_path}")
    selected_tau = float(pilot["selected_tau"])
    lambda_pos = float(pilot["selected_lambda_pos"])
    lambda_neg = float(pilot["selected_lambda_neg"])
    n_bias_steps = int(pilot.get("n_bias_steps", 1))

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    records, cache_path = load_counterfact_prepared(
        project_root=PROJECT_ROOT,
        cache_path=args.cache_path,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        max_items=int(args.max_items),
        evidence_mode=str(args.evidence_mode),
    )
    baseline_rows = prepare_baseline_rows(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
    )
    full_tau_summary, full_tau_rows = select_tau(baseline_rows)
    c3_rows, c3_summary = evaluate_intervention(
        baseline_rows=baseline_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        tau=float(selected_tau),
        lambda_pos=float(lambda_pos),
        lambda_neg=float(lambda_neg),
        n_bias_steps=int(n_bias_steps),
        max_new_tokens=int(args.max_new_tokens),
    )
    site = load_target_site(
        localization_path=(PROJECT_ROOT / args.localization_path).resolve(),
        override_target_layer=args.override_target_layer,
        override_target_head=args.override_target_head,
    )
    base_faithful = sum(1 for row in baseline_rows if str(row["base_label"]) == "faithful")
    base_hallucinated = sum(1 for row in baseline_rows if str(row["base_label"]) == "hallucinated")
    base_other = len(baseline_rows) - base_faithful - base_hallucinated
    summary = {
        "model_id": str(args.model_id),
        "model_name": str(args.model_name or args.model_id),
        "architecture": str(args.architecture),
        "task": "CounterFact",
        "intervention_impl": "token_logit_bias",
        "intervention_space": "E_content/P_content_first_token",
        "evidence_mode": str(args.evidence_mode),
        "target_site": site,
        "n_items": int(len(baseline_rows)),
        "base_counts": {
            "faithful": int(base_faithful),
            "hallucinated": int(base_hallucinated),
            "other": int(base_other),
        },
        "base_faithful_rate": float(base_faithful / max(len(baseline_rows), 1)),
        "base_hallucinated_rate": float(base_hallucinated / max(len(baseline_rows), 1)),
        "selected_tau": float(selected_tau),
        "selected_lambda_pos": float(lambda_pos),
        "selected_lambda_neg": float(lambda_neg),
        "n_bias_steps": int(n_bias_steps),
        "pilot_summary_path": str((PROJECT_ROOT / args.pilot_summary_path).resolve()),
        "pilot": pilot,
        "margin_auc_full": full_tau_summary["margin_auc"],
        "full_tau_summary": full_tau_summary,
        "full_tau_candidates": full_tau_rows,
        "conditions": {
            "C2": {
                "condition_id": "C2",
                "condition_name": "CounterFact evidence context, no intervention",
                "n_items": int(len(baseline_rows)),
                "faithful_rate": float(base_faithful / max(len(baseline_rows), 1)),
                "hallucinated_rate": float(base_hallucinated / max(len(baseline_rows), 1)),
                "other_rate": float(base_other / max(len(baseline_rows), 1)),
            },
            "C3": {
                "condition_id": "C3",
                "condition_name": "CounterFact evidence context + token-logit mechanism",
                **c3_summary,
                "tau": float(selected_tau),
                "lambda_pos": float(lambda_pos),
                "lambda_neg": float(lambda_neg),
            },
        },
        "restoration_rate": float(c3_summary["restoration_rate"]),
        "harm_rate": float(c3_summary["harm_rate"]),
        "gate_fire_rate": float(c3_summary["gate_fire_rate"]),
        "metadata": build_metadata(
            args=args,
            input_files=[
                cache_path,
                PROJECT_ROOT / args.localization_path,
                PROJECT_ROOT / args.pilot_summary_path,
            ],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=REPO_ROOT,
        ),
    }
    write_counterfact_bundle(out_dir=out_dir, summary=summary, rows=c3_rows)
    write_jsonl(out_dir / "baseline_rows.jsonl", baseline_rows)
    write_json(out_dir / "full_margin_auc.json", full_tau_summary)
    print(json.dumps({"restoration_rate": summary["restoration_rate"], "harm_rate": summary["harm_rate"], "gate_fire_rate": summary["gate_fire_rate"]}, indent=2))
    print(f"summary={out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
