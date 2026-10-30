#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.final_paper import (
    auc_rank,
    build_cross_model_metadata,
    load_medqa_cross_model_inputs,
    load_top_site,
    model_summary_row,
    parse_float_grid,
    run_medqa_pilot_grid,
    write_medqa_cross_model_bundle,
    full_intervention_summary,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-model MedQA intervention on Qwen.")
    parser.add_argument("--model_id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--model_name", default="Qwen2-7B-Instruct")
    parser.add_argument("--architecture", default="gqa_residual_stream")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--headline_path", default="results/medqa_final_v3/pr_sweep/headline.json")
    parser.add_argument("--localization_path", default="results/cross_model/qwen_localization/layer_importance.json")
    parser.add_argument("--out_dir", default="results/cross_model/qwen_intervention")
    parser.add_argument("--cache_dir", default="cache/cross_model/qwen_medqa")
    parser.add_argument("--prompt_variant", default="vanilla")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--pilot_examples", type=int, default=50)
    parser.add_argument("--tau_grid", default="-1.0,-0.5,0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--lambda_neg_grid", default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--pos_ratio_grid", default="0.0,0.5,1.0")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
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
    headline = read_json((PROJECT_ROOT / args.headline_path).resolve())
    bootstrap_space = str(headline.get("bootstrap_space", "E_letter"))

    site = load_top_site((PROJECT_ROOT / args.localization_path).resolve(), site_mode="layer")
    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    shared = load_medqa_cross_model_inputs(
        project_root=PROJECT_ROOT,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=args.conflict_path,
        matched_subset_path=args.matched_subset_path,
        label_audit_path=args.label_audit_path,
        cache_dir=args.cache_dir,
        target_layer=int(site["target_layer"]),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        prompt_variant=str(args.prompt_variant),
        p_set_mode=str(args.p_set_mode),
    )
    tau_grid = parse_float_grid(args.tau_grid, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    lambda_neg_grid = parse_float_grid(args.lambda_neg_grid, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    pos_ratio_grid = parse_float_grid(args.pos_ratio_grid, [0.0, 0.5, 1.0])

    pilot_grid, pilot_best, _ = run_medqa_pilot_grid(
        prepared_records=shared.prepared_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        letter_token_sets=shared.letter_token_sets,
        matched_ids=shared.matched_ids,
        clean_ids=shared.clean_ids,
        bootstrap_space=bootstrap_space,
        target_layer=int(site["target_layer"]),
        target_head=0,
        alpha=float(args.alpha),
        max_new_tokens=int(args.max_new_tokens),
        tau_grid=tau_grid,
        lambda_neg_grid=lambda_neg_grid,
        pos_ratio_grid=pos_ratio_grid,
        pilot_examples=int(args.pilot_examples),
        seed=int(args.seed),
    )

    full_summary, full_rows = full_intervention_summary(
        prepared_records=shared.prepared_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        letter_token_sets=shared.letter_token_sets,
        matched_ids=shared.matched_ids,
        clean_ids=shared.clean_ids,
        bootstrap_space=bootstrap_space,
        target_layer=int(site["target_layer"]),
        target_head=0,
        alpha=float(args.alpha),
        max_new_tokens=int(args.max_new_tokens),
        tau_margin=float(pilot_best["tau_margin"]),
        lambda_pos=float(pilot_best["lambda_pos"]),
        lambda_neg=float(pilot_best["lambda_neg"]),
        seed=int(args.seed),
    )
    auc = auc_rank(
        [float(row.get("prefill_margin", 0.0)) for row in full_rows if not bool(row.get("skipped", False))],
        [1 if bool(row.get("base_correct", False)) else 0 for row in full_rows if not bool(row.get("skipped", False))],
    )
    full_summary.update(
        {
            "model_id": str(args.model_id),
            "model_name": str(args.model_name),
            "architecture": str(args.architecture),
            "bootstrap_space": str(bootstrap_space),
            "intervention_space": "E_letter",
            "site_mode": str(site["site_mode"]),
            "site_description": str(site["site_description"]),
            "target_layer": int(site["target_layer"]),
            "target_head": 0,
            "selected_tau_margin": float(pilot_best["tau_margin"]),
            "selected_lambda_pos": float(pilot_best["lambda_pos"]),
            "selected_lambda_neg": float(pilot_best["lambda_neg"]),
            "margin_auc": None if auc is None else float(auc),
            "architecture_note": "Grouped-query attention deployment uses the late residual-stream layer rather than a per-head intervention site.",
        }
    )
    metadata = build_cross_model_metadata(
        args=args,
        input_files=shared.metadata_inputs + [(PROJECT_ROOT / args.headline_path).resolve(), (PROJECT_ROOT / args.localization_path).resolve()],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    write_medqa_cross_model_bundle(
        out_dir=out_dir,
        summary=full_summary,
        results=full_rows,
        pilot_grid=pilot_grid,
        pilot_best=pilot_best,
        metadata=metadata,
    )
    write_json(
        out_dir / "comparison_row.json",
        model_summary_row(
            full_summary,
            model_name=str(args.model_name),
            architecture=str(args.architecture),
            intervention_site=str(site["site_description"]),
        ),
    )
    print(json.dumps({"selected": pilot_best, "margin_auc": auc}, indent=2))
    print(f"summary={out_dir / 'summary.json'}")
    print(f"results={out_dir / 'results.jsonl'}")
    print(f"pilot_grid={out_dir / 'pilot_grid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
