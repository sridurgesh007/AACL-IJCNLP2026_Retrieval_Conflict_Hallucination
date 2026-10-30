#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from medrag.final_paper import (
    auc_rank,
    build_cross_model_metadata,
    collect_margin_rows,
    lambda_sweep_at_tau,
    load_medqa_cross_model_inputs,
    load_top_site,
    matched_subset_ids,
    parse_float_grid,
    resolve_intervention_impl,
    select_tau_by_youden,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot calibration for cross-model MedQA intervention.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--site_mode", required=True, choices=["head", "layer"])
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--headline_path", default="results/medqa_final_v3/pr_sweep/headline.json")
    parser.add_argument("--localization_path", required=True)
    parser.add_argument("--override_target_layer", type=int, default=None)
    parser.add_argument("--override_target_head", type=int, default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--prompt_variant", default="vanilla")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--pilot_examples", type=int, default=100)
    parser.add_argument("--min_margin_auc", type=float, default=0.70)
    parser.add_argument("--lambda_pos_grid", default="0.0")
    parser.add_argument("--lambda_neg_grid", default="2.5,5.0,7.5")
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


def _plot_margin_distribution(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    correct = [float(row["prefill_margin"]) for row in rows if bool(row["base_correct"]) and np.isfinite(float(row["prefill_margin"]))]
    wrong = [float(row["prefill_margin"]) for row in rows if (not bool(row["base_correct"])) and np.isfinite(float(row["prefill_margin"]))]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bins = 20
    if correct:
        ax.hist(correct, bins=bins, alpha=0.6, label="Base-correct", color="#2a6f97")
    if wrong:
        ax.hist(wrong, bins=bins, alpha=0.6, label="Base-wrong", color="#c44536")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prefill margin")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    pilot_dir = out_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    headline = read_json((PROJECT_ROOT / args.headline_path).resolve())
    bootstrap_space = str(headline.get("bootstrap_space", "E_letter"))
    site = _apply_site_override(load_top_site((PROJECT_ROOT / args.localization_path).resolve(), site_mode=str(args.site_mode)), args)

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

    pilot_ids_list = matched_subset_ids(shared.prepared_records, shared.matched_ids, int(args.pilot_examples))
    pilot_id_set = set(pilot_ids_list)
    pilot_clean_ids = set(shared.clean_ids) & pilot_id_set
    margin_rows = collect_margin_rows(
        prepared_records=shared.prepared_records,
        model=model,
        bootstrap_space=bootstrap_space,
        matched_ids=shared.matched_ids,
        limit=int(args.pilot_examples),
    )
    if not margin_rows:
        raise RuntimeError("Pilot margin rows are empty.")

    auc = auc_rank(
        [float(row["prefill_margin"]) for row in margin_rows],
        [1 if bool(row["base_correct"]) else 0 for row in margin_rows],
    )
    tau_payload = select_tau_by_youden(
        [float(row["prefill_margin"]) for row in margin_rows],
        [not bool(row["base_correct"]) for row in margin_rows],
    )
    selected_tau = float(tau_payload["tau"])
    pilot_go = bool(auc is not None and float(auc) >= float(args.min_margin_auc))

    lambda_rows: list[dict[str, Any]] = []
    selected_lambda_pos: float | None = None
    selected_lambda_neg: float | None = None
    selected_intervention_impl = resolve_intervention_impl(
        site_mode=str(site["site_mode"]),
        requested=str(args.intervention_impl),
    )
    if pilot_go:
        lambda_rows, best_lambda, _ = lambda_sweep_at_tau(
            prepared_records=shared.prepared_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=shared.letter_token_sets,
            bootstrap_space=bootstrap_space,
            tau_margin=float(selected_tau),
            lambda_neg_grid=parse_float_grid(args.lambda_neg_grid, [2.5, 5.0, 7.5]),
            lambda_pos_grid=parse_float_grid(args.lambda_pos_grid, [0.0]),
            matched_item_ids=pilot_id_set,
            clean_item_ids=pilot_clean_ids,
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
            site_mode=str(args.site_mode),
            target_layer=int(site["target_layer"]),
            target_head=int(site["target_head"]),
            alpha=float(args.alpha),
            intervention_impl=str(args.intervention_impl),
        )
        selected_lambda_pos = float(best_lambda["lambda_pos"])
        selected_lambda_neg = float(best_lambda["lambda_neg"])
        selected_intervention_impl = str(best_lambda.get("intervention_impl", selected_intervention_impl))

    margin_plot_path = pilot_dir / "margin_distribution.png"
    _plot_margin_distribution(
        margin_rows,
        margin_plot_path,
        title=f"{args.model_name} pilot margin distribution ({args.site_mode})",
    )

    payload = {
        "model_id": str(args.model_id),
        "model_name": str(args.model_name),
        "architecture": str(args.architecture),
        "site_mode": str(site["site_mode"]),
        "site_description": str(site["site_description"]),
        "target_layer": int(site["target_layer"]),
        "target_head": int(site["target_head"]),
        "intervention_impl": str(selected_intervention_impl),
        "alpha": float(args.alpha),
        "bootstrap_space": str(bootstrap_space),
        "pilot_examples": int(len(margin_rows)),
        "pilot_margin_auc": None if auc is None else float(auc),
        "selected_tau_margin": float(selected_tau),
        "selected_lambda_pos": 0.0 if selected_lambda_pos is None else float(selected_lambda_pos),
        "selected_lambda_neg": None if selected_lambda_neg is None else float(selected_lambda_neg),
        "pilot_go": bool(pilot_go),
        "min_margin_auc": float(args.min_margin_auc),
        "n_base_wrong": int(sum(1 for row in margin_rows if not bool(row["base_correct"]))),
        "n_base_correct": int(sum(1 for row in margin_rows if bool(row["base_correct"]))),
        "tau_selection": tau_payload,
        "lambda_sweep": lambda_rows,
        "metadata": build_cross_model_metadata(
            args=args,
            input_files=shared.metadata_inputs + [(PROJECT_ROOT / args.headline_path).resolve(), (PROJECT_ROOT / args.localization_path).resolve()],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(pilot_dir / "margin_auc.json", payload)
    write_jsonl(pilot_dir / "margin_rows.jsonl", margin_rows)

    print(json.dumps(payload, indent=2))
    print(f"pilot_summary={pilot_dir / 'margin_auc.json'}")
    print(f"pilot_rows={pilot_dir / 'margin_rows.jsonl'}")
    print(f"pilot_figure={margin_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
