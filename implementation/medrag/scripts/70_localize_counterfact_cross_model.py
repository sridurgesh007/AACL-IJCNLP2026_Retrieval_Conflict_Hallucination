#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import (
    build_metadata,
    load_counterfact_prepared,
    margin_from_logits,
    prefill_logits,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json, write_jsonl

BASE_PHASE1_PATH = EVIDENCE_AWARE_ROOT / "scripts" / "01_phase1_diagnostics.py"
_BASE_SPEC = importlib.util.spec_from_file_location("phase1_base_counterfact_cross_model", BASE_PHASE1_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load phase1 tracer from {BASE_PHASE1_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _BASE
_BASE_SPEC.loader.exec_module(_BASE)

LayerDecompositionHooks = _BASE.LayerDecompositionHooks
ensure_attention_last = _BASE.ensure_attention_last
logit_lens = _BASE.logit_lens
logit_lens_batch = _BASE.logit_lens_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize CounterFact entity-margin heads for a cross-model run.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument(
        "--evidence_mode",
        default="cache",
        choices=["cache", "question_answer", "relation_or_question_answer"],
        help="How to construct CounterFact evidence context. Use relation_or_question_answer for v2 runs.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--figures_dir", default="")
    parser.add_argument("--max_items", type=int, default=500)
    parser.add_argument("--localization_items", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--reference_layer", type=int, default=27)
    parser.add_argument("--reference_head", type=int, default=20)
    parser.add_argument("--min_depth", type=float, default=0.75)
    parser.add_argument("--max_depth", type=float, default=0.95)
    parser.add_argument("--min_cv", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half = float(1.96 * std / np.sqrt(max(arr.size, 1)))
    return mean, std, float(mean - half), float(mean + half)


def _mean_logits(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    idx = torch.as_tensor([int(x) for x in token_ids], dtype=torch.long, device=logits.device)
    return logits.index_select(dim=-1, index=idx).mean(dim=-1)


def _compute_head_margin_delta(
    *,
    model,
    residual_in: torch.Tensor,
    head_pre_o: torch.Tensor,
    e_ids: list[int],
    p_ids: list[int],
    layer_idx: int,
) -> list[float]:
    layer = model.model.layers[int(layer_idx)]
    o_proj = layer.self_attn.o_proj
    num_heads, head_dim = int(head_pre_o.shape[0]), int(head_pre_o.shape[1])
    hidden_size = int(num_heads * head_dim)
    target_dtype = o_proj.weight.dtype
    target_device = o_proj.weight.device

    head_pre = head_pre_o.to(device=target_device, dtype=target_dtype)
    head_only = torch.zeros((num_heads, hidden_size), device=target_device, dtype=target_dtype)
    for head_idx in range(num_heads):
        start = int(head_idx * head_dim)
        end = int(start + head_dim)
        head_only[head_idx, start:end] = head_pre[head_idx]

    contrib = o_proj(head_only).float()
    residual = residual_in.to(device=target_device, dtype=torch.float32)
    logits_pre = logit_lens(model, residual)
    margin_pre = _mean_logits(logits_pre, e_ids) - _mean_logits(logits_pre, p_ids)
    combined = residual.unsqueeze(0).expand(num_heads, -1) + contrib
    logits_batch = logit_lens_batch(model, combined)
    margins = _mean_logits(logits_batch, e_ids) - _mean_logits(logits_batch, p_ids)
    return [float(x) for x in (margins - margin_pre).detach().float().cpu().tolist()]


def _plot_heatmap(rows: list[dict[str, Any]], *, num_layers: int, num_heads: int, reference_layer: int, reference_head: int, out_path: Path) -> None:
    mat = np.full((num_layers, num_heads), np.nan, dtype=np.float64)
    for row in rows:
        mat[int(row["layer"]), int(row["head"])] = float(row["mean_contribution"])
    finite = mat[np.isfinite(mat)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1e-8
    vmax = max(vmax, 1e-8)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), origin="lower")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("CounterFact Head Importance: Entity-Margin Contribution")
    for row in rows[:5]:
        ax.scatter(int(row["head"]), int(row["layer"]), marker="*", s=170, color="black", edgecolors="white", linewidths=0.8)
    if 0 <= int(reference_layer) < num_layers and 0 <= int(reference_head) < num_heads:
        ax.scatter(int(reference_head), int(reference_layer), marker="s", s=90, facecolors="none", edgecolors="gold", linewidths=2.0)
    fig.colorbar(im, ax=ax, label="Mean margin contribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    figures_dir = (PROJECT_ROOT / (args.figures_dir or str(Path(args.out_dir) / "figures"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    records, cache_path = load_counterfact_prepared(
        project_root=PROJECT_ROOT,
        cache_path=args.cache_path,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        max_items=int(args.max_items),
        evidence_mode=str(args.evidence_mode),
    )
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_heads = int(getattr(model.config, "num_attention_heads"))

    base_rows: list[dict[str, Any]] = []
    for rec in records:
        ex = rec.example
        logits = prefill_logits(model, rec.prompt_ids, device)
        margin = margin_from_logits(logits, {int(ex.evidence_token_id)}, {int(ex.parametric_token_id)})
        top_id = int(torch.argmax(logits).item())
        base_rows.append(
            {
                "item_id": str(ex.id),
                "prefill_margin": float(margin),
                "argmax_token_id": int(top_id),
                "argmax_label": "faithful"
                if top_id == int(ex.evidence_token_id)
                else ("hallucinated" if top_id == int(ex.parametric_token_id) else "other"),
            }
        )

    base_by_id = {str(row["item_id"]): row for row in base_rows}
    hallucinated_ids = [str(row["item_id"]) for row in base_rows if str(row["argmax_label"]) == "hallucinated"]
    selected_records = [rec for rec in records if str(rec.example.id) in set(hallucinated_ids)]
    if not selected_records:
        selected_records = records
    selected_records = selected_records[: int(args.localization_items)]
    if not selected_records:
        raise RuntimeError("No records selected for CounterFact localization.")

    hooks = LayerDecompositionHooks(model)
    hooks.register()
    cell_values: dict[tuple[int, int], list[float]] = {(layer, head): [] for layer in range(num_layers) for head in range(num_heads)}
    per_item_rows: list[dict[str, Any]] = []
    try:
        for rec in selected_records:
            ex = rec.example
            input_ids = rec.prompt_ids.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)
            hooks.clear()
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            if not ensure_attention_last(hooks, outputs, num_layers):
                raise RuntimeError(f"Missing attention outputs while tracing item_id={ex.id}")
            bundle = hooks.bundle(num_layers=num_layers)
            per_layer: dict[str, list[float]] = {}
            for layer_idx in range(num_layers):
                deltas = _compute_head_margin_delta(
                    model=model,
                    residual_in=bundle.residual_in[layer_idx],
                    head_pre_o=bundle.attention_pre_o[layer_idx],
                    e_ids=[int(ex.evidence_token_id)],
                    p_ids=[int(ex.parametric_token_id)],
                    layer_idx=layer_idx,
                )
                per_layer[str(layer_idx)] = [float(x) for x in deltas]
                for head_idx, value in enumerate(deltas):
                    cell_values[(int(layer_idx), int(head_idx))].append(float(value))
            per_item_rows.append(
                {
                    "item_id": str(ex.id),
                    "question": str(ex.question),
                    "evidence": str(ex.evidence),
                    "correct_answer": str(ex.answer_liquid),
                    "parametric_answer": str(ex.answer_solid),
                    "base_argmax_label": str(base_by_id.get(str(ex.id), {}).get("argmax_label", "unknown")),
                    "prefill_margin": float(base_by_id.get(str(ex.id), {}).get("prefill_margin", 0.0)),
                    "per_layer_head_margin_contribution": per_layer,
                }
            )
    finally:
        hooks.close()

    head_rows: list[dict[str, Any]] = []
    for layer in range(num_layers):
        for head in range(num_heads):
            values = cell_values[(int(layer), int(head))]
            mean, std, ci_low, ci_high = _mean_ci95(values)
            cv = float(std / max(abs(mean), 1e-8))
            depth = float(layer / max(num_layers - 1, 1))
            head_rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "mean_contribution": float(mean),
                    "std": float(std),
                    "coefficient_of_variation": float(cv),
                    "depth": float(depth),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "n_items": int(len(values)),
                }
            )
    head_rows.sort(key=lambda row: (float(row["mean_contribution"]), -int(row["layer"]), -int(row["head"])), reverse=True)
    for rank, row in enumerate(head_rows, start=1):
        row["rank"] = int(rank)

    candidates = [
        row
        for row in head_rows
        if float(args.min_depth) <= float(row["depth"]) <= float(args.max_depth)
        and float(row["coefficient_of_variation"]) >= float(args.min_cv)
        and float(row["mean_contribution"]) > 0.0
    ]
    recommended = dict(candidates[0] if candidates else head_rows[0])
    reference = next(
        (
            dict(row)
            for row in head_rows
            if int(row["layer"]) == int(args.reference_layer) and int(row["head"]) == int(args.reference_head)
        ),
        None,
    )
    payload = {
        "model_id": str(args.model_id),
        "n_records_prepared": int(len(records)),
        "n_localization_items": int(len(per_item_rows)),
        "n_base_argmax_hallucinated": int(len(hallucinated_ids)),
        "selection_rule": {
            "selected_from": "argmax_hallucinated_rows_if_available",
            "min_depth": float(args.min_depth),
            "max_depth": float(args.max_depth),
            "min_cv": float(args.min_cv),
        },
        "evidence_mode": str(args.evidence_mode),
        "recommended_head": recommended,
        "top_5_heads": [dict(row) for row in head_rows[:5]],
        "candidate_heads": [dict(row) for row in candidates[:20]],
        "reference_head": reference,
        "per_head_rows": head_rows,
        "metadata": build_metadata(
            args=args,
            input_files=[cache_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=REPO_ROOT,
        ),
    }
    write_json(out_dir / "head_importance.json", payload)
    write_jsonl(out_dir / "per_item_contributions.jsonl", per_item_rows)
    write_jsonl(out_dir / "baseline_argmax_rows.jsonl", base_rows)
    _plot_heatmap(
        head_rows,
        num_layers=num_layers,
        num_heads=num_heads,
        reference_layer=int(args.reference_layer),
        reference_head=int(args.reference_head),
        out_path=figures_dir / "head_importance_heatmap.png",
    )
    print(json.dumps({"recommended_head": recommended, "top_5_heads": payload["top_5_heads"]}, indent=2))
    print(f"head_importance={out_dir / 'head_importance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
