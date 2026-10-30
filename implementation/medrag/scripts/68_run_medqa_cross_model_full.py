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
from medrag.final_paper import (
    auc_rank,
    build_cross_model_metadata,
    ci_overlap,
    collect_margin_rows,
    compare_condition_summaries,
    evaluate_cross_model_intervention,
    evaluate_head_ov_hook_intervention,
    evaluate_residual_stream_hook_intervention,
    full_intervention_summary,
    load_medqa_cross_model_inputs,
    load_top_site,
    model_summary_row,
    paired_mcnemar_from_rows,
    resolve_intervention_impl,
    summary_from_prepared_records,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl, write_summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full cross-model MedQA evaluation from pilot-selected tau/lambda.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--site_mode", required=True, choices=["head", "layer"])
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--headline_path", default="results/medqa_final_v3/pr_sweep/headline.json")
    parser.add_argument("--llama_summary_path", default="results/medqa_final_v3/consolidation/headline_verify/H4/summary.json")
    parser.add_argument("--localization_path", required=True)
    parser.add_argument("--override_target_layer", type=int, default=None)
    parser.add_argument("--override_target_head", type=int, default=None)
    parser.add_argument("--pilot_summary_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--c1_prompt_variant", default="vanilla")
    parser.add_argument("--c2_prompt_variant", default="trust_instruction")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--intervention_impl",
        default="auto",
        choices=[
            "auto",
            "head_logit_bias",
            "head_ov_hook",
            "token_logit_bias",
            "residual_stream_hook",
            "llama_medqa_running_margin_gate",
            "first_token_logit_bias",
            "true_head_ov_hook",
            "head_causal_hook",
            "running_margin_head_hook",
            "residual_stream_token_bias",
        ],
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _apply_site_override(site: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.override_target_layer is None and args.override_target_head is None:
        return site
    if args.override_target_layer is None:
        raise ValueError("--override_target_layer is required when overriding the intervention site.")
    if str(site["site_mode"]) == "head" and args.override_target_head is None:
        raise ValueError("--override_target_head is required for head-mode intervention overrides.")
    out = dict(site)
    out["target_layer"] = int(args.override_target_layer)
    if args.override_target_head is not None:
        out["target_head"] = int(args.override_target_head)
    out["site_description"] = (
        f"layer={int(out['target_layer'])}, head={int(out['target_head'])}"
        if str(out["site_mode"]) == "head"
        else f"layer={int(out['target_layer'])}, residual-stream"
    )
    payload = dict(out.get("site_payload") or {})
    payload["override_target_layer"] = int(out["target_layer"])
    payload["override_target_head"] = int(out["target_head"])
    out["site_payload"] = payload
    return out


def _condition_rate_fields() -> list[str]:
    return [
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


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    headline = read_json((PROJECT_ROOT / args.headline_path).resolve())
    bootstrap_space = str(headline.get("bootstrap_space", "E_letter"))
    pilot = read_json((PROJECT_ROOT / args.pilot_summary_path).resolve())
    if not bool(pilot.get("pilot_go", False)):
        raise RuntimeError(f"Pilot GO/NO-GO failed at {Path(args.pilot_summary_path)}; refusing full evaluation.")
    selected_tau = float(pilot["selected_tau_margin"])
    selected_lambda_pos = float(pilot.get("selected_lambda_pos", 0.0))
    selected_lambda_neg = float(pilot["selected_lambda_neg"])

    site = _apply_site_override(load_top_site((PROJECT_ROOT / args.localization_path).resolve(), site_mode=str(args.site_mode)), args)
    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)

    c1_shared = load_medqa_cross_model_inputs(
        project_root=PROJECT_ROOT,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=args.conflict_path,
        matched_subset_path=args.matched_subset_path,
        label_audit_path=args.label_audit_path,
        cache_dir=str(Path(args.cache_dir) / str(args.c1_prompt_variant)),
        target_layer=int(site["target_layer"]),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        prompt_variant=str(args.c1_prompt_variant),
        p_set_mode=str(args.p_set_mode),
    )
    c2_shared = load_medqa_cross_model_inputs(
        project_root=PROJECT_ROOT,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=args.conflict_path,
        matched_subset_path=args.matched_subset_path,
        label_audit_path=args.label_audit_path,
        cache_dir=str(Path(args.cache_dir) / str(args.c2_prompt_variant)),
        target_layer=int(site["target_layer"]),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        prompt_variant=str(args.c2_prompt_variant),
        p_set_mode=str(args.p_set_mode),
    )

    c1_base_summary, c1_condition_rows = summary_from_prepared_records(
        c1_shared.prepared_records,
        matched_ids=c1_shared.matched_ids,
        clean_ids=c1_shared.clean_ids,
    )
    c1_summary, c1_rows = compare_condition_summaries(
        c1_condition_rows,
        c1_condition_rows,
        wall_clock_seconds=float(c1_base_summary["wall_clock_seconds"]),
    )
    c1_summary.update(
        {
            "condition_id": "C1",
            "prompt_variant": str(args.c1_prompt_variant),
            "apply_intervention": False,
            "condition_accuracy": float(c1_base_summary["accuracy"]),
            "condition_accuracy_ci95_wilson": list(c1_base_summary["accuracy_ci95_wilson"]),
            "condition_accuracy_clean_subset": float(c1_base_summary["accuracy_clean_subset"]),
            "condition_accuracy_clean_subset_ci95_wilson": list(c1_base_summary["accuracy_clean_subset_ci95_wilson"]),
        }
    )

    c2_base_summary, c2_condition_rows = summary_from_prepared_records(
        c2_shared.prepared_records,
        matched_ids=c2_shared.matched_ids,
        clean_ids=c2_shared.clean_ids,
    )
    c2_summary, c2_rows = compare_condition_summaries(
        c1_condition_rows,
        c2_condition_rows,
        wall_clock_seconds=float(c2_base_summary["wall_clock_seconds"]),
    )
    c2_summary.update(
        {
            "condition_id": "C2",
            "prompt_variant": str(args.c2_prompt_variant),
            "apply_intervention": False,
            "condition_accuracy": float(c2_base_summary["accuracy"]),
            "condition_accuracy_ci95_wilson": list(c2_base_summary["accuracy_ci95_wilson"]),
            "condition_accuracy_clean_subset": float(c2_base_summary["accuracy_clean_subset"]),
            "condition_accuracy_clean_subset_ci95_wilson": list(c2_base_summary["accuracy_clean_subset_ci95_wilson"]),
        }
    )

    intervention_impl = resolve_intervention_impl(site_mode=str(args.site_mode), requested=str(args.intervention_impl))
    if intervention_impl == "head_logit_bias":
        c3_summary, c3_rows = full_intervention_summary(
            prepared_records=c1_shared.prepared_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=c1_shared.letter_token_sets,
            matched_ids=c1_shared.matched_ids,
            clean_ids=c1_shared.clean_ids,
            bootstrap_space=bootstrap_space,
            target_layer=int(site["target_layer"]),
            target_head=int(site["target_head"]),
            alpha=float(args.alpha),
            max_new_tokens=int(args.max_new_tokens),
            tau_margin=float(selected_tau),
            lambda_pos=float(selected_lambda_pos),
            lambda_neg=float(selected_lambda_neg),
            seed=int(args.seed),
        )
        c3_intervention_impl = "head_logit_bias"
    elif intervention_impl == "token_logit_bias":
        c3_summary, c3_rows = evaluate_cross_model_intervention(
            prepared_records=c1_shared.prepared_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=c1_shared.letter_token_sets,
            bootstrap_space=bootstrap_space,
            lambda_pos=float(selected_lambda_pos),
            lambda_neg=float(selected_lambda_neg),
            tau_margin=float(selected_tau),
            matched_item_ids=c1_shared.matched_ids,
            clean_item_ids=c1_shared.clean_ids,
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
            site_mode=str(args.site_mode),
        )
        c3_intervention_impl = "token_logit_bias"
    elif intervention_impl == "head_ov_hook":
        c3_summary, c3_rows = evaluate_head_ov_hook_intervention(
            prepared_records=c1_shared.prepared_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=c1_shared.letter_token_sets,
            bootstrap_space=bootstrap_space,
            lambda_pos=float(selected_lambda_pos),
            lambda_neg=float(selected_lambda_neg),
            tau_margin=float(selected_tau),
            alpha=float(args.alpha),
            target_layer=int(site["target_layer"]),
            target_head=int(site["target_head"]),
            matched_item_ids=c1_shared.matched_ids,
            clean_item_ids=c1_shared.clean_ids,
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
        )
        c3_intervention_impl = "head_ov_hook"
    else:
        c3_summary, c3_rows = evaluate_residual_stream_hook_intervention(
            prepared_records=c1_shared.prepared_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=c1_shared.letter_token_sets,
            bootstrap_space=bootstrap_space,
            lambda_pos=float(selected_lambda_pos),
            lambda_neg=float(selected_lambda_neg),
            tau_margin=float(selected_tau),
            target_layer=int(site["target_layer"]),
            target_head=int(site["target_head"]),
            matched_item_ids=c1_shared.matched_ids,
            clean_item_ids=c1_shared.clean_ids,
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
        )
        c3_intervention_impl = "residual_stream_hook"
    c3_summary.update(
        {
            "condition_id": "C3",
            "prompt_variant": str(args.c1_prompt_variant),
            "apply_intervention": True,
            "intervention_impl": c3_intervention_impl,
            "alpha": float(args.alpha),
            "condition_accuracy": float(c3_summary["ft_accuracy"]),
            "condition_accuracy_ci95_wilson": list(c3_summary["ft_accuracy_ci95_wilson"]),
            "condition_accuracy_clean_subset": float(c3_summary["ft_accuracy_clean_subset"]),
            "condition_accuracy_clean_subset_ci95_wilson": list(c3_summary["ft_accuracy_clean_subset_ci95_wilson"]),
        }
    )

    full_margin_rows = collect_margin_rows(
        prepared_records=c1_shared.prepared_records,
        model=model,
        bootstrap_space=bootstrap_space,
        matched_ids=c1_shared.matched_ids,
        limit=0,
    )
    full_margin_auc = auc_rank(
        [float(row["prefill_margin"]) for row in full_margin_rows],
        [1 if bool(row["base_correct"]) else 0 for row in full_margin_rows],
    )

    llama_summary = read_json((PROJECT_ROOT / args.llama_summary_path).resolve())
    restoration_vs_llama = {
        "llama_restoration_rate": float(llama_summary["restoration_rate"]),
        "current_restoration_rate": float(c3_summary["restoration_rate"]),
        "difference": float(c3_summary["restoration_rate"] - float(llama_summary["restoration_rate"])),
        "ci_overlap": bool(
            ci_overlap(
                list(c3_summary["restoration_rate_ci95_wilson"]),
                list(llama_summary["restoration_rate_ci95_wilson"]),
            )
        ),
    }
    harm_vs_llama = {
        "llama_harm_rate": float(llama_summary["harm_rate"]),
        "current_harm_rate": float(c3_summary["harm_rate"]),
        "difference": float(c3_summary["harm_rate"] - float(llama_summary["harm_rate"])),
        "ci_overlap": bool(
            ci_overlap(
                list(c3_summary["harm_rate_ci95_wilson"]),
                list(llama_summary["harm_rate_ci95_wilson"]),
            )
        ),
    }

    metadata = build_cross_model_metadata(
        args=args,
        input_files=(
            c1_shared.metadata_inputs
            + [
                (PROJECT_ROOT / args.headline_path).resolve(),
                (PROJECT_ROOT / args.localization_path).resolve(),
                (PROJECT_ROOT / args.pilot_summary_path).resolve(),
                (PROJECT_ROOT / args.llama_summary_path).resolve(),
            ]
        ),
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )

    for summary in (c1_summary, c2_summary, c3_summary):
        summary.update(
            {
                "model_id": str(args.model_id),
                "model_name": str(args.model_name),
                "architecture": str(args.architecture),
                "bootstrap_space": str(bootstrap_space),
                "site_mode": str(site["site_mode"]),
                "site_description": str(site["site_description"]),
                "target_layer": int(site["target_layer"]),
                "target_head": int(site["target_head"]),
                "selected_tau_margin": float(selected_tau),
                "selected_lambda_pos": float(selected_lambda_pos),
                "selected_lambda_neg": float(selected_lambda_neg),
                "alpha": float(args.alpha),
                "pilot_margin_auc": None if pilot.get("pilot_margin_auc") is None else float(pilot["pilot_margin_auc"]),
                "metadata": metadata,
            }
        )
    if str(args.site_mode) == "layer":
        c3_summary["architecture_note"] = "Grouped-query attention deployment uses the late residual-stream layer rather than a per-head intervention site."

    condition_payloads = {
        "C1": (c1_summary, c1_rows),
        "C2": (c2_summary, c2_rows),
        "C3": (c3_summary, c3_rows),
    }
    for cond_id, (summary, rows) in condition_payloads.items():
        cond_dir = out_dir / cond_id
        cond_dir.mkdir(parents=True, exist_ok=True)
        write_summary_json(cond_dir / "summary.json", summary, rate_fields=_condition_rate_fields())
        write_jsonl(cond_dir / "results.jsonl", rows)

    aggregate = {
        "model_id": str(args.model_id),
        "model_name": str(args.model_name),
        "architecture": str(args.architecture),
        "bootstrap_space": str(bootstrap_space),
        "site_mode": str(site["site_mode"]),
        "site_description": str(site["site_description"]),
        "target_layer": int(site["target_layer"]),
        "target_head": int(site["target_head"]),
        "selected_tau_margin": float(selected_tau),
        "selected_lambda_pos": float(selected_lambda_pos),
        "selected_lambda_neg": float(selected_lambda_neg),
        "alpha": float(args.alpha),
        "intervention_impl": c3_intervention_impl,
        "pilot_margin_auc": None if pilot.get("pilot_margin_auc") is None else float(pilot["pilot_margin_auc"]),
        "pilot_go": bool(pilot.get("pilot_go", False)),
        "margin_auc": None if full_margin_auc is None else float(full_margin_auc),
        "mcnemar_c3_vs_c1": c3_summary["mcnemar_ft_vs_baseline"],
        "mcnemar_c3_vs_c2": paired_mcnemar_from_rows(c2_rows, c3_rows),
        "restoration_vs_llama": restoration_vs_llama,
        "harm_vs_llama": harm_vs_llama,
        "conditions": {
            "C1": c1_summary,
            "C2": c2_summary,
            "C3": c3_summary,
        },
        "metadata": metadata,
    }
    write_json(out_dir / "summary.json", aggregate)
    write_json(
        out_dir / "comparison_row.json",
        model_summary_row(
            c3_summary,
            model_name=str(args.model_name),
            architecture=str(args.architecture),
            intervention_site=str(site["site_description"]),
        ),
    )

    print(json.dumps(aggregate, indent=2))
    print(f"summary={out_dir / 'summary.json'}")
    print(f"c1={out_dir / 'C1' / 'summary.json'}")
    print(f"c2={out_dir / 'C2' / 'summary.json'}")
    print(f"c3={out_dir / 'C3' / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
