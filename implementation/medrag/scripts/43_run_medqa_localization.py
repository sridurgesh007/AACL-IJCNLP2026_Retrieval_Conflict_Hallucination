#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.localization_v2 import build_metadata, choose_localization_candidates, mean_ci95, read_jsonl
from medrag.medqa_v3 import build_letter_token_sets, load_conflicts, set_global_seed, stable_item_id
from medrag.prompt_variants import build_medqa_prompt
from medrag.utils.io import write_json, write_jsonl

BASE_PHASE1_PATH = EVIDENCE_AWARE_ROOT / "scripts" / "01_phase1_diagnostics.py"
_BASE_SPEC = importlib.util.spec_from_file_location("phase1_base", BASE_PHASE1_PATH)
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
    parser = argparse.ArgumentParser(description="Run MedQA-specific head localization for letter commitment.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument("--out_dir", default="results/medqa_final_v3/localization_v2")
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/localization_v2/figures")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--prompt_variant", default="vanilla")
    parser.add_argument("--reference_layer", type=int, default=27)
    parser.add_argument("--reference_head", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _select_rows(h4_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in h4_rows
        if bool(row.get("in_matched_subset", False))
        and (not bool(row.get("base_correct", False)))
        and bool(row.get("conflict_detected", False))
    ]
    if not selected:
        raise RuntimeError("No matched-subset base-wrong conflict rows found in H4 results.")
    return selected


def _mean_logits(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    if not token_ids:
        raise RuntimeError("Token id set for letter margin is empty.")
    idx = torch.as_tensor(token_ids, dtype=torch.long, device=logits.device)
    return logits.index_select(dim=-1, index=idx).mean(dim=-1)


def _compute_head_margin_delta(model, residual_in: torch.Tensor, head_pre_o: torch.Tensor, e_ids: list[int], p_ids: list[int], layer_idx: int) -> list[float]:
    if head_pre_o.ndim != 2:
        raise ValueError(f"head_pre_o must be [num_heads, head_dim], got {tuple(head_pre_o.shape)}")
    layer = model.model.layers[int(layer_idx)]
    o_proj = layer.self_attn.o_proj
    num_heads, head_dim = int(head_pre_o.shape[0]), int(head_pre_o.shape[1])
    hidden_size = num_heads * head_dim

    target_dtype = o_proj.weight.dtype
    target_device = o_proj.weight.device
    head_pre = head_pre_o.to(device=target_device, dtype=target_dtype)

    head_only = torch.zeros((num_heads, hidden_size), device=target_device, dtype=target_dtype)
    for head_idx in range(num_heads):
        start = head_idx * head_dim
        end = start + head_dim
        head_only[head_idx, start:end] = head_pre[head_idx]

    contrib = o_proj(head_only).float()
    residual = residual_in.to(device=target_device, dtype=torch.float32)
    logits_pre = logit_lens(model, residual)
    margin_pre = _mean_logits(logits_pre, e_ids) - _mean_logits(logits_pre, p_ids)
    combined = residual.unsqueeze(0).expand(num_heads, -1) + contrib
    logits_batch = logit_lens_batch(model, combined)
    margins = _mean_logits(logits_batch, e_ids) - _mean_logits(logits_batch, p_ids)
    deltas = (margins - margin_pre).detach().float().cpu().tolist()
    return [float(x) for x in deltas]


def _plot_heatmap(head_rows: list[dict[str, Any]], *, num_layers: int, num_heads: int, reference_layer: int, reference_head: int, out_path: Path) -> None:
    mat = np.full((num_layers, num_heads), np.nan, dtype=np.float64)
    for row in head_rows:
        mat[int(row["layer"]), int(row["head"])] = float(row["mean_contribution"])
    finite = mat[np.isfinite(mat)]
    vmax = float(np.max(np.abs(finite))) if finite.size > 0 else 1e-8
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", norm=norm, origin="lower")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("MedQA Head Importance: Mean Letter-Margin Contribution")
    ax.set_xticks(np.arange(0, num_heads, max(1, num_heads // 8)))
    ax.set_yticks(np.arange(0, num_layers, max(1, num_layers // 8)))

    top5 = head_rows[:5]
    for row in top5:
        ax.scatter(int(row["head"]), int(row["layer"]), marker="*", s=180, color="black", edgecolors="white", linewidths=0.8)
    ref = next(row for row in head_rows if int(row["layer"]) == int(reference_layer) and int(row["head"]) == int(reference_head))
    ax.scatter(int(reference_head), int(reference_layer), marker="s", s=90, facecolors="none", edgecolors="gold", linewidths=2.0)
    ax.text(
        int(reference_head) + 0.3,
        int(reference_layer) + 0.3,
        f"L{reference_layer}/H{reference_head}\nrank={int(ref['rank'])}",
        fontsize=9,
        color="black",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "gold", "alpha": 0.8},
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean margin contribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    conflicts = load_conflicts((PROJECT_ROOT / args.conflict_path).resolve())
    conflict_by_id = {stable_item_id(ex): ex for ex in conflicts}
    h4_rows = read_jsonl((PROJECT_ROOT / args.h4_results_path).resolve())
    selected_rows = _select_rows(h4_rows)
    selected_ids = [str(row["item_id"]) for row in selected_rows]

    missing_ids = sorted(set(selected_ids) - set(conflict_by_id.keys()))
    if missing_ids:
        raise RuntimeError(f"Selected H4 item ids missing from conflict dataset. First missing id: {missing_ids[0]}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = build_letter_token_sets(tokenizer)
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_heads = int(getattr(model.config, "num_attention_heads"))

    hooks = LayerDecompositionHooks(model)
    hooks.register()
    per_item_rows: list[dict[str, Any]] = []
    cell_values: dict[tuple[int, int], list[float]] = {(layer, head): [] for layer in range(num_layers) for head in range(num_heads)}

    try:
        for item_id in selected_ids:
            ex = conflict_by_id[item_id]
            question = str(ex.get("question", "")).strip()
            options = ex.get("options", {}) or {}
            context = str(ex.get("context", "")).strip()
            correct = str(ex.get("correct_answer", "")).strip().upper()
            if not question or not isinstance(options, dict) or not context:
                raise RuntimeError(f"Invalid MedQA row for item_id={item_id}")

            prompt = build_medqa_prompt(
                question=question,
                options={str(k): str(v) for k, v in options.items()},
                context=context,
                prompt_variant=str(args.prompt_variant),
                tokenizer=tokenizer,
                model_id=str(args.model_id),
            )
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length))
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

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
                raise RuntimeError(f"Missing attention outputs while tracing item_id={item_id}")
            bundle = hooks.bundle(num_layers=num_layers)

            e_ids = sorted(int(x) for x in letter_token_sets.get(correct, set()))
            p_ids = sorted(int(x) for letter, ids in letter_token_sets.items() if str(letter) != str(correct) for x in ids)
            if not e_ids or not p_ids:
                raise RuntimeError(f"Empty letter token sets for item_id={item_id} correct={correct}")

            per_layer = {}
            for layer_idx in range(num_layers):
                deltas = _compute_head_margin_delta(
                    model=model,
                    residual_in=bundle.residual_in[layer_idx],
                    head_pre_o=bundle.attention_pre_o[layer_idx],
                    e_ids=e_ids,
                    p_ids=p_ids,
                    layer_idx=layer_idx,
                )
                per_layer[str(layer_idx)] = [float(x) for x in deltas]
                for head_idx, value in enumerate(deltas):
                    cell_values[(int(layer_idx), int(head_idx))].append(float(value))

            per_item_rows.append(
                {
                    "item_id": str(item_id),
                    "correct_letter": str(correct),
                    "parametric_letter": str(ex.get("parametric_answer", "")).strip().upper(),
                    "baseline_letter": str(next(row for row in selected_rows if str(row["item_id"]) == str(item_id)).get("base_letter", "")),
                    "per_layer_head_margin_contribution": per_layer,
                }
            )
    finally:
        hooks.close()

    head_rows: list[dict[str, Any]] = []
    for layer in range(num_layers):
        for head in range(num_heads):
            values = list(cell_values[(int(layer), int(head))])
            mean, std, ci_low, ci_high = mean_ci95(values)
            head_rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "mean_contribution": float(mean),
                    "std": float(std),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "n_items": int(len(values)),
                }
            )
    head_rows.sort(key=lambda row: (float(row["mean_contribution"]), -int(row["layer"]), -int(row["head"])), reverse=True)
    for rank, row in enumerate(head_rows, start=1):
        row["rank"] = int(rank)

    decision = choose_localization_candidates(
        head_rows=head_rows,
        ref_layer=int(args.reference_layer),
        ref_head=int(args.reference_head),
    )
    top5 = [
        {
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "mean_contribution": float(row["mean_contribution"]),
            "rank": int(row["rank"]),
        }
        for row in decision["top_5_heads"]
    ]
    ref_row = decision["reference_head"]

    metadata = build_metadata(
        args=args,
        input_files=[
            PROJECT_ROOT / args.conflict_path,
            PROJECT_ROOT / args.matched_subset_path,
            PROJECT_ROOT / args.label_audit_path,
            PROJECT_ROOT / args.h4_results_path,
        ],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    payload = {
        "n_items_scored": int(len(per_item_rows)),
        "selected_item_ids": selected_ids,
        "per_head_mean_contribution": [
            [
                int(row["layer"]),
                int(row["head"]),
                float(row["mean_contribution"]),
                float(row["std"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
            ]
            for row in head_rows
        ],
        "top_5_heads": top5,
        "candidate_heads": [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "mean_contribution": float(row["mean_contribution"]),
                "rank": int(row["rank"]),
            }
            for row in decision["candidate_heads"]
        ],
        "decision": {
            "mode": str(decision["mode"]),
            "text": str(decision["decision_text"]),
        },
        "L27_H20_rank": int(ref_row["rank"]),
        "L27_H20_mean_contribution": float(ref_row["mean_contribution"]),
        "metadata": metadata,
    }
    write_json(out_dir / "head_importance.json", payload)
    write_json(out_dir / "decision.json", {"decision": payload["decision"], "candidate_heads": payload["candidate_heads"], "metadata": metadata})
    write_jsonl(out_dir / "per_item_contributions.jsonl", per_item_rows)
    (out_dir / "decision.txt").write_text(str(decision["decision_text"]) + "\n", encoding="utf-8")

    heatmap_path = figures_dir / "head_importance_heatmap.png"
    _plot_heatmap(
        head_rows=head_rows,
        num_layers=num_layers,
        num_heads=num_heads,
        reference_layer=int(args.reference_layer),
        reference_head=int(args.reference_head),
        out_path=heatmap_path,
    )

    print(json.dumps({"n_items_scored": len(per_item_rows), "top_5_heads": top5}, indent=2))
    print(f"head_importance={out_dir / 'head_importance.json'}")
    print(f"decision={out_dir / 'decision.txt'}")
    print(f"per_item={out_dir / 'per_item_contributions.jsonl'}")
    print(f"heatmap={heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
