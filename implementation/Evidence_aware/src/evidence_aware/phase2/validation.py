from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F

from evidence_aware.phase2.common import PreparedExample, argmax_non_whitespace_token_id, classify_answer_token, write_json, write_jsonl
from evidence_aware.phase2.modeling import ValidationHooks, logit_lens, logit_lens_batch, to_tensors


def compute_head_delta_attn_evidence_for_layer(
    model,
    layer_idx: int,
    residual_in: torch.Tensor,
    head_pre_o: torch.Tensor,
    evidence_token_id: int,
    p_evidence_pre: float,
) -> list[float]:
    if head_pre_o.ndim != 2:
        raise ValueError(f"head_pre_o must be [num_heads, head_dim], got {tuple(head_pre_o.shape)}")
    layer = model.model.layers[layer_idx]
    o_proj = layer.self_attn.o_proj
    num_heads, head_dim = int(head_pre_o.shape[0]), int(head_pre_o.shape[1])
    hidden_size = num_heads * head_dim

    target_dtype = o_proj.weight.dtype
    target_device = o_proj.weight.device
    pre_heads = head_pre_o.to(device=target_device, dtype=target_dtype)

    head_only = torch.zeros((num_heads, hidden_size), device=target_device, dtype=target_dtype)
    for head_idx in range(num_heads):
        start = head_idx * head_dim
        end = start + head_dim
        head_only[head_idx, start:end] = pre_heads[head_idx]

    contrib = o_proj(head_only).float()
    residual = residual_in.to(device=target_device, dtype=torch.float32).unsqueeze(0).expand(num_heads, -1)
    combined = residual + contrib
    logits_batch = logit_lens_batch(model, combined)
    probs = F.softmax(logits_batch, dim=-1)[:, evidence_token_id]
    deltas = probs - float(p_evidence_pre)
    return [float(x.item()) for x in deltas]


def aggregate_head_matrix(rows: list[dict[str, Any]], analysis_layers: list[int]) -> np.ndarray:
    if not rows:
        raise RuntimeError("aggregate_head_matrix received empty rows.")
    inferred_heads = None
    for row in rows:
        layer_map = row.get("head_delta_attn_evidence_by_layer", {})
        for layer_idx in analysis_layers:
            key = str(layer_idx)
            if key in layer_map and layer_map[key]:
                inferred_heads = len(layer_map[key])
                break
        if inferred_heads is not None:
            break
    if inferred_heads is None:
        raise RuntimeError("Could not infer number of heads from rows.")

    matrix = np.full((inferred_heads, len(analysis_layers)), np.nan, dtype=np.float64)
    for col, layer_idx in enumerate(analysis_layers):
        key = str(layer_idx)
        vectors = [
            np.asarray(row["head_delta_attn_evidence_by_layer"][key], dtype=np.float64)
            for row in rows
            if key in row.get("head_delta_attn_evidence_by_layer", {})
        ]
        if vectors:
            matrix[:, col] = np.mean(np.stack(vectors, axis=0), axis=0)
    return matrix


def rank_top_heads(matrix: np.ndarray, layers: list[int], topk: int, with_values: bool = False) -> list[Any]:
    flat: list[tuple[float, int, int]] = []
    for head_idx in range(matrix.shape[0]):
        for col, layer_idx in enumerate(layers):
            val = matrix[head_idx, col]
            if np.isnan(val):
                continue
            flat.append((float(val), int(layer_idx), int(head_idx)))
    flat.sort(key=lambda x: x[0], reverse=True)
    selected = flat[: max(0, int(topk))]
    if with_values:
        return [{"layer": l, "head": h, "value": v} for (v, l, h) in selected]
    return [(l, h) for (v, l, h) in selected]


def extract_top_heads_from_phase1(path: Path, topk: int, with_values: bool = False) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing phase1 aggregate file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    phase1_map = payload.get("mean_head_delta_attn_evidence_faithful", {})
    flat: list[tuple[float, int, int]] = []
    for layer_str, values in phase1_map.items():
        for head_idx, value in enumerate(values):
            flat.append((float(value), int(layer_str), int(head_idx)))
    flat.sort(key=lambda x: x[0], reverse=True)
    selected = flat[: max(0, int(topk))]
    if with_values:
        return [{"layer": l, "head": h, "value": v} for (v, l, h) in selected]
    return [(l, h) for (v, l, h) in selected]


def plot_validation_heatmap(matrix: np.ndarray, layers: list[int], out_path: Path, title: str) -> None:
    merged = np.nan_to_num(matrix, nan=0.0)
    vmax = float(np.max(np.abs(merged))) if merged.size > 0 else 1e-8
    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers)
    n_heads = matrix.shape[0]
    step = max(1, n_heads // 8)
    ax.set_yticks(np.arange(0, n_heads, step))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean delta p(evidence) from isolated head contribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def head_value(matrix: np.ndarray, layers: list[int], layer: int, head: int) -> float | None:
    if head < 0 or head >= matrix.shape[0]:
        return None
    if layer not in layers:
        return None
    col = layers.index(layer)
    value = matrix[head, col]
    if np.isnan(value):
        return None
    return float(value)


def run_validation(
    model,
    tokenizer,
    device: torch.device,
    examples: list[PreparedExample],
    phase1_aggregate_path: Path,
    out_dir: Path,
    root_out_dir: Path,
) -> dict[str, Any]:
    hooks = ValidationHooks(model)
    hooks.register()
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    analysis_layers = [l for l in range(22, 28) if l < num_layers]
    rows: list[dict[str, Any]] = []
    dropped = {"missing_hook_capture": 0, "no_non_whitespace_token": 0, "label_other": 0}

    try:
        for ex in examples:
            hooks.clear()
            input_ids, attention_mask = to_tensors(ex, device=device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            final_logits = outputs.logits[0, -1, :].float()
            gen_token = argmax_non_whitespace_token_id(final_logits, tokenizer)
            if gen_token is None:
                dropped["no_non_whitespace_token"] += 1
                continue
            label = classify_answer_token(gen_token, ex.evidence_token_id, ex.parametric_token_id)
            if label == "other":
                dropped["label_other"] += 1
                continue

            head_delta_by_layer: dict[str, list[float]] = {}
            valid = True
            for layer_idx in analysis_layers:
                if layer_idx not in hooks.residual_in or layer_idx not in hooks.attention_pre_o:
                    valid = False
                    break
                logits_pre = logit_lens(model, hooks.residual_in[layer_idx])
                p_evidence_pre = float(F.softmax(logits_pre, dim=-1)[ex.evidence_token_id].item())
                head_delta_by_layer[str(layer_idx)] = compute_head_delta_attn_evidence_for_layer(
                    model=model,
                    layer_idx=layer_idx,
                    residual_in=hooks.residual_in[layer_idx],
                    head_pre_o=hooks.attention_pre_o[layer_idx],
                    evidence_token_id=ex.evidence_token_id,
                    p_evidence_pre=p_evidence_pre,
                )
            if not valid:
                dropped["missing_hook_capture"] += 1
                continue
            rows.append({"id": ex.id, "label": label, "head_delta_attn_evidence_by_layer": head_delta_by_layer})
    finally:
        hooks.close()

    faithful_rows = [r for r in rows if r["label"] == "faithful"]
    hall_rows = [r for r in rows if r["label"] == "hallucinated"]
    if not faithful_rows or not hall_rows:
        raise RuntimeError(
            "Validation produced empty faithful or hallucinated group. "
            f"faithful={len(faithful_rows)}, hallucinated={len(hall_rows)}"
        )

    faithful_matrix = aggregate_head_matrix(faithful_rows, analysis_layers)
    hall_matrix = aggregate_head_matrix(hall_rows, analysis_layers)
    plot_validation_heatmap(faithful_matrix, analysis_layers, out_dir / "validation_heatmap_faithful.png", "Validation Head Delta Heatmap (Faithful)")
    plot_validation_heatmap(hall_matrix, analysis_layers, out_dir / "validation_heatmap_hallucinated.png", "Validation Head Delta Heatmap (Hallucinated)")

    phase1_top3 = extract_top_heads_from_phase1(phase1_aggregate_path, topk=3)
    validation_top3 = rank_top_heads(faithful_matrix, analysis_layers, topk=3)
    overlap = sorted(set(phase1_top3) & set(validation_top3))

    l27h20_f = head_value(faithful_matrix, analysis_layers, layer=27, head=20)
    l27h20_h = head_value(hall_matrix, analysis_layers, layer=27, head=20)
    l24h3_f = head_value(faithful_matrix, analysis_layers, layer=24, head=3)
    l24h4_f = head_value(faithful_matrix, analysis_layers, layer=24, head=4)

    c1 = bool(l27h20_f is not None and l27h20_h is not None and l27h20_f > 0.10 and l27h20_h < 0.03)
    c2 = bool((l24h3_f is not None and l24h3_f > 0.05) or (l24h4_f is not None and l24h4_f > 0.05))
    c3 = len(overlap) > 0
    stable = bool(c1 and c2 and c3)

    top5_phase1 = extract_top_heads_from_phase1(phase1_aggregate_path, topk=5, with_values=True)
    top5_val_f = rank_top_heads(faithful_matrix, analysis_layers, topk=5, with_values=True)
    top5_val_h = rank_top_heads(hall_matrix, analysis_layers, topk=5, with_values=True)

    verdict = [
        f"VALIDATION_VERDICT={'STABLE' if stable else 'UNSTABLE'}",
        "",
        "Pass Criteria:",
        f"1) L27H20 faithful > 0.10 and hallucinated < 0.03 -> {c1}",
        f"2) L24H3 or L24H4 faithful > 0.05 -> {c2}",
        f"3) Top-3 overlap with Phase1 top-3 -> {c3}",
        "",
        f"L27H20 faithful={l27h20_f}",
        f"L27H20 hallucinated={l27h20_h}",
        f"L24H3 faithful={l24h3_f}",
        f"L24H4 faithful={l24h4_f}",
        f"top3_overlap={overlap}",
        "",
        f"counts={json.dumps({'rows': len(rows), 'faithful': len(faithful_rows), 'hallucinated': len(hall_rows), 'dropped': dropped}, indent=2)}",
        "",
        f"phase1_top5={json.dumps(top5_phase1, indent=2)}",
        f"validation_top5_faithful={json.dumps(top5_val_f, indent=2)}",
        f"validation_top5_hallucinated={json.dumps(top5_val_h, indent=2)}",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_text = "\n".join(verdict)
    (out_dir / "validation_verdict.txt").write_text(verdict_text, encoding="utf-8")
    (root_out_dir / "validation_verdict.txt").write_text(verdict_text, encoding="utf-8")

    summary = {
        "stable": stable,
        "criteria": {"c1_l27h20": c1, "c2_l24h3_or_h4": c2, "c3_top3_overlap": c3},
        "values": {
            "l27h20_faithful": l27h20_f,
            "l27h20_hallucinated": l27h20_h,
            "l24h3_faithful": l24h3_f,
            "l24h4_faithful": l24h4_f,
            "top3_overlap": overlap,
            "phase1_top3": phase1_top3,
            "validation_top3": validation_top3,
        },
        "counts": {"rows": len(rows), "faithful": len(faithful_rows), "hallucinated": len(hall_rows), "dropped": dropped},
        "analysis_layers": analysis_layers,
        "top5_validation_faithful": top5_val_f,
        "top5_validation_hallucinated": top5_val_h,
    }
    write_json(out_dir / "validation_summary.json", summary)
    write_jsonl(out_dir / "validation_rows.jsonl", rows)
    return summary
