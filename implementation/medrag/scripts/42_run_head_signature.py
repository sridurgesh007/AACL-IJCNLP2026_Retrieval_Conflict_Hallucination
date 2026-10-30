#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import run_with_zero_ablation, run_without_patch
from medrag.causal_localization import (
    build_metadata,
    canonical_letter_token_ids,
    load_medqa_causal_inputs,
    locate_context_positions,
    middle_prompt_position,
    predict_letter_from_logits,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedQA head-signature diagnostics for L27/H20.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/causal/signature")
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/causal/figures")
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--sampled_control_head_offset", type=int, default=0)
    parser.add_argument("--sampled_control_head_global", type=int, default=-1)
    parser.add_argument("--ov_max_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _rate(flags: list[bool]) -> tuple[float, list[float], int, int]:
    k = int(sum(1 for flag in flags if bool(flag)))
    n = int(len(flags))
    rate = float(k / max(n, 1))
    ci = wilson_ci(k, n) if n > 0 else (0.0, 0.0)
    return float(rate), [float(ci[0]), float(ci[1])], int(k), int(n)


def _resolve_control_head(*, num_heads: int, target_head: int, requested: int, offset: int) -> int:
    control_heads = [int(h) for h in range(int(num_heads)) if int(h) != int(target_head)]
    if not control_heads:
        raise RuntimeError("No control heads are available.")
    if int(requested) >= 0:
        if int(requested) == int(target_head):
            raise RuntimeError("Global control head must differ from the target head.")
        return int(requested)
    return int(control_heads[int(offset) % len(control_heads)])


def _attention_from_answer_position(model, prompt_ids: torch.Tensor, layer: int) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(
            prompt_ids.unsqueeze(0),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    attentions = outputs.attentions
    if attentions is None or len(attentions) <= int(layer):
        raise RuntimeError(f"Model did not return attentions for layer {layer}.")
    layer_attn = attentions[int(layer)]
    if layer_attn.ndim != 4:
        raise RuntimeError(f"Unexpected attention tensor shape: {tuple(layer_attn.shape)}")
    return layer_attn[0, :, -1, :].detach().float().cpu()


def _capture_vproj_input(model, prompt_ids: torch.Tensor, layer: int, position: int = -1) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, args):
        del module
        if not args:
            return args
        hidden = args[0]
        if hidden is None or hidden.ndim != 3:
            return args
        pos = int(position)
        if pos < 0:
            pos = int(hidden.shape[1]) + int(pos)
        if not (0 <= int(pos) < int(hidden.shape[1])):
            raise IndexError(f"Position {position} is out of bounds for v_proj input with seq len {hidden.shape[1]}")
        captured["value"] = hidden[:, int(pos), :].detach().float().cpu()
        return args

    handle = model.model.layers[int(layer)].self_attn.v_proj.register_forward_pre_hook(_hook)
    try:
        with torch.inference_mode():
            _ = model(prompt_ids.unsqueeze(0), use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "value" not in captured:
        raise RuntimeError(f"Failed to capture v_proj input for layer {layer}.")
    return captured["value"].squeeze(0)


def _compute_ov_matrix(model, layer: int, head: int) -> torch.Tensor:
    attn = model.model.layers[int(layer)].self_attn
    num_heads = int(getattr(model.config, "num_attention_heads"))
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    hidden_size = int(getattr(model.config, "hidden_size"))
    if hidden_size % num_heads != 0:
        raise RuntimeError("hidden_size is not divisible by num_attention_heads.")
    if num_heads % num_kv_heads != 0:
        raise RuntimeError("num_attention_heads is not divisible by num_key_value_heads.")
    head_dim = hidden_size // num_heads
    kv_group_size = num_heads // num_kv_heads
    kv_head = int(head) // int(kv_group_size)

    o_weight = attn.o_proj.weight.detach().float().cpu()[:, int(head) * head_dim : (int(head) + 1) * head_dim]
    v_weight = attn.v_proj.weight.detach().float().cpu()[int(kv_head) * head_dim : (int(kv_head) + 1) * head_dim, :]
    return o_weight @ v_weight


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = float(torch.linalg.norm(a).item() * torch.linalg.norm(b).item())
    if denom <= 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _decode_token(tokenizer, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False)).replace("\n", "\\n")


def _plot_attention_hist(attention_rows: list[dict[str, Any]], out_path: Path) -> None:
    target = [float(row["attention_on_evidence_span_target"]) for row in attention_rows]
    control = [float(row["attention_on_evidence_span_control_mean"]) for row in attention_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(0.0, 1.0, 16)
    ax.hist(target, bins=bins, alpha=0.65, label="L27/H20", color="#1f77b4")
    ax.hist(control, bins=bins, alpha=0.55, label="L27 control mean", color="#8c8c8c")
    ax.set_xlabel("Attention Mass On Evidence Span")
    ax.set_ylabel("Count")
    ax.set_title("Attention On Evidence From Answer Position")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_signature_panel(
    *,
    attention_rows: list[dict[str, Any]],
    ov_summary: dict[str, Any],
    ablation_summary: dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    target = [float(row["attention_on_evidence_span_target"]) for row in attention_rows]
    control = [float(row["attention_on_evidence_span_control_mean"]) for row in attention_rows]
    bins = np.linspace(0.0, 1.0, 14)
    axes[0].hist(target, bins=bins, alpha=0.65, label="L27/H20", color="#1f77b4")
    axes[0].hist(control, bins=bins, alpha=0.55, label="Control mean", color="#8c8c8c")
    axes[0].set_xlabel("Attention On Evidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("A. Evidence Attention")
    axes[0].legend(frameon=False)

    top_tokens = ov_summary["target_head"]["top5_tokens"]
    token_labels = [str(row["token"]) for row in top_tokens]
    token_scores = [float(row["score"]) for row in top_tokens]
    axes[1].bar(np.arange(len(token_labels)), token_scores, color="#d17c0f")
    axes[1].set_xticks(np.arange(len(token_labels)), token_labels, rotation=25, ha="right")
    axes[1].set_ylabel("Unembedding score")
    axes[1].set_title("B. OV Top-5 Target Tokens")

    categories = ["base_correct_non_conflict", "base_correct_conflict", "base_wrong"]
    labels = ["Correct / no conflict", "Correct / conflict", "Base wrong"]
    target_drop = [float(ablation_summary["categories"][cat]["accuracy_drop_vs_no_ablation"]["target_head"]) for cat in categories]
    control_drop = [float(ablation_summary["categories"][cat]["accuracy_drop_vs_no_ablation"]["control_head"]) for cat in categories]
    middle_drop = [float(ablation_summary["categories"][cat]["accuracy_drop_vs_no_ablation"]["middle_position"]) for cat in categories]
    x = np.arange(len(categories))
    width = 0.24
    axes[2].bar(x - width, target_drop, width=width, label="Target head", color="#c44e52")
    axes[2].bar(x, control_drop, width=width, label="Control head", color="#8c8c8c")
    axes[2].bar(x + width, middle_drop, width=width, label="Middle position", color="#55a868")
    axes[2].set_xticks(x, labels, rotation=15, ha="right")
    axes[2].set_ylabel("Accuracy drop")
    axes[2].set_title("C. Ablation Accuracy Drops")
    axes[2].legend(frameon=False)

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

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    inputs = load_medqa_causal_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=PROJECT_ROOT / args.conflict_path,
        matched_subset_path=PROJECT_ROOT / args.matched_subset_path,
        label_audit_path=PROJECT_ROOT / args.label_audit_path,
        h4_results_path=PROJECT_ROOT / args.h4_results_path,
        cache_dir=PROJECT_ROOT / args.cache_dir,
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )

    num_heads = int(model.config.num_attention_heads)
    control_head_global = _resolve_control_head(
        num_heads=num_heads,
        target_head=int(args.target_head),
        requested=int(args.sampled_control_head_global),
        offset=int(args.sampled_control_head_offset),
    )
    control_heads = [int(h) for h in range(num_heads) if int(h) != int(args.target_head)]

    metadata_inputs = [
        PROJECT_ROOT / args.conflict_path,
        PROJECT_ROOT / args.matched_subset_path,
        PROJECT_ROOT / args.label_audit_path,
        PROJECT_ROOT / args.h4_results_path,
    ]

    attention_started = time.perf_counter()
    attention_candidates = [
        item for item in inputs.causal_items if item.in_matched_subset and (not item.base_correct)
    ]
    attention_rows: list[dict[str, Any]] = []
    target_dom_flags: list[bool] = []
    control_dom_flags: list[bool] = []
    skipped_attention = 0
    for idx, item in enumerate(attention_candidates):
        rec = inputs.prepared_by_id[item.item_id]
        prompt_ids = rec.prompt_ids.to(device)
        span_positions, content_positions = locate_context_positions(
            tokenizer=tokenizer,
            prompt=rec.prompt,
            prompt_ids=rec.prompt_ids,
            context=rec.context,
        )
        evidence_positions = content_positions if content_positions else span_positions
        if not evidence_positions:
            skipped_attention += 1
            continue
        attn = _attention_from_answer_position(model, prompt_ids, layer=int(args.target_layer))
        non_evidence_positions = [int(pos) for pos in range(int(attn.shape[1])) if int(pos) not in set(int(x) for x in evidence_positions)]
        sampled_control_head = control_heads[(idx + int(args.sampled_control_head_offset)) % len(control_heads)]
        target_evidence = float(attn[int(args.target_head), evidence_positions].sum().item())
        target_non = float(attn[int(args.target_head), non_evidence_positions].sum().item()) if non_evidence_positions else 0.0
        sampled_control_evidence = float(attn[int(sampled_control_head), evidence_positions].sum().item())
        sampled_control_non = float(attn[int(sampled_control_head), non_evidence_positions].sum().item()) if non_evidence_positions else 0.0
        control_mean_evidence = float(attn[control_heads][:, evidence_positions].sum(dim=1).mean().item())
        control_mean_non = float(attn[control_heads][:, non_evidence_positions].sum(dim=1).mean().item()) if non_evidence_positions else 0.0
        target_dom = bool(target_evidence > target_non)
        control_dom = bool(sampled_control_evidence > sampled_control_non)
        target_dom_flags.append(target_dom)
        control_dom_flags.append(control_dom)
        attention_rows.append(
            {
                "item_id": str(item.item_id),
                "correct_letter": str(item.correct),
                "baseline_letter": item.base_letter,
                "query_position": int(rec.prompt_ids.shape[0] - 1),
                "sampled_control_head": int(sampled_control_head),
                "n_evidence_positions": int(len(evidence_positions)),
                "attention_on_evidence_span_target": float(target_evidence),
                "attention_on_non_evidence_target": float(target_non),
                "attention_on_evidence_span_control_sampled": float(sampled_control_evidence),
                "attention_on_non_evidence_control_sampled": float(sampled_control_non),
                "attention_on_evidence_span_control_mean": float(control_mean_evidence),
                "attention_on_non_evidence_control_mean": float(control_mean_non),
                "evidence_dominant_target": bool(target_dom),
                "evidence_dominant_control_sampled": bool(control_dom),
            }
        )

    target_rate, target_ci, target_k, target_n = _rate(target_dom_flags)
    control_rate, control_ci, control_k, control_n = _rate(control_dom_flags)
    attention_summary = {
        "n_candidates": int(len(attention_candidates)),
        "n_used": int(len(attention_rows)),
        "n_skipped": int(skipped_attention),
        "target_head": int(args.target_head),
        "sampled_control_head_offset": int(args.sampled_control_head_offset),
        "attention_evidence_dominant_rate_target": float(target_rate),
        "attention_evidence_dominant_rate_target_ci95_wilson": target_ci,
        "attention_evidence_dominant_rate_target_successes": int(target_k),
        "attention_evidence_dominant_rate_control_sampled": float(control_rate),
        "attention_evidence_dominant_rate_control_sampled_ci95_wilson": control_ci,
        "attention_evidence_dominant_rate_control_sampled_successes": int(control_k),
        "mcnemar_target_vs_control_sampled": mcnemar_from_paired(target_dom_flags, control_dom_flags),
        "mean_attention_on_evidence_target": float(np.mean([row["attention_on_evidence_span_target"] for row in attention_rows])) if attention_rows else 0.0,
        "mean_attention_on_evidence_control_mean": float(np.mean([row["attention_on_evidence_span_control_mean"] for row in attention_rows])) if attention_rows else 0.0,
        "wall_clock_seconds": float(time.perf_counter() - attention_started),
        "items": attention_rows,
        "metadata": build_metadata(
            args=args,
            input_files=metadata_inputs,
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "attention_on_evidence.json", attention_summary)

    ov_started = time.perf_counter()
    ov_items = attention_candidates[: max(1, int(args.ov_max_examples))]
    letter_ids = canonical_letter_token_ids(tokenizer)
    unembed = model.lm_head.weight.detach().float().cpu()
    target_ov = _compute_ov_matrix(model, layer=int(args.target_layer), head=int(args.target_head))
    control_ov = _compute_ov_matrix(model, layer=int(args.target_layer), head=int(control_head_global))
    target_ys: list[torch.Tensor] = []
    control_ys: list[torch.Tensor] = []
    vproj_rows: list[dict[str, Any]] = []
    for item in ov_items:
        rec = inputs.prepared_by_id[item.item_id]
        x = _capture_vproj_input(model, rec.prompt_ids.to(device), layer=int(args.target_layer), position=-1)
        y_target = target_ov @ x
        y_control = control_ov @ x
        target_ys.append(y_target)
        control_ys.append(y_control)
        vproj_rows.append({"item_id": str(item.item_id)})

    if not target_ys:
        raise RuntimeError("OV diagnostic had zero usable examples.")

    target_stack = torch.stack(target_ys, dim=0)
    control_stack = torch.stack(control_ys, dim=0)
    target_mean_y = target_stack.mean(dim=0)
    control_mean_y = control_stack.mean(dim=0)

    target_letter_alignment = {
        str(letter): float(np.mean([_cosine(vec, unembed[int(tok_id)]) for vec in target_ys]))
        for letter, tok_id in letter_ids.items()
    }
    control_letter_alignment = {
        str(letter): float(np.mean([_cosine(vec, unembed[int(tok_id)]) for vec in control_ys]))
        for letter, tok_id in letter_ids.items()
    }

    target_vocab_scores = torch.mv(unembed, target_mean_y)
    control_vocab_scores = torch.mv(unembed, control_mean_y)
    target_top_ids = torch.topk(target_vocab_scores, k=5).indices.tolist()
    control_top_ids = torch.topk(control_vocab_scores, k=5).indices.tolist()
    ov_summary = {
        "n_examples_used": int(len(target_ys)),
        "sampled_control_head": int(control_head_global),
        "target_head": {
            "letter_alignment": target_letter_alignment,
            "top5_tokens": [
                {
                    "token_id": int(tok_id),
                    "token": _decode_token(tokenizer, int(tok_id)),
                    "score": float(target_vocab_scores[int(tok_id)].item()),
                    "is_letter_token": bool(int(tok_id) in set(letter_ids.values())),
                }
                for tok_id in target_top_ids
            ],
        },
        "control_head": {
            "letter_alignment": control_letter_alignment,
            "top5_tokens": [
                {
                    "token_id": int(tok_id),
                    "token": _decode_token(tokenizer, int(tok_id)),
                    "score": float(control_vocab_scores[int(tok_id)].item()),
                    "is_letter_token": bool(int(tok_id) in set(letter_ids.values())),
                }
                for tok_id in control_top_ids
            ],
        },
        "wall_clock_seconds": float(time.perf_counter() - ov_started),
        "items": vproj_rows,
        "metadata": build_metadata(
            args=args,
            input_files=metadata_inputs,
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "ov_direction.json", ov_summary)

    ablation_started = time.perf_counter()
    ablation_rows: list[dict[str, Any]] = []
    category_rows: dict[str, list[dict[str, Any]]] = {
        "base_correct_non_conflict": [],
        "base_correct_conflict": [],
        "base_wrong": [],
    }
    for idx, item in enumerate(inputs.causal_items):
        rec = inputs.prepared_by_id[item.item_id]
        prompt_ids = rec.prompt_ids.to(device)
        sampled_control_head = control_heads[(idx + int(args.sampled_control_head_offset)) % len(control_heads)]
        middle_pos = middle_prompt_position(rec.prompt_ids)

        baseline_logits = run_without_patch(model, prompt_ids, use_cache=False)
        target_logits = run_with_zero_ablation(
            model=model,
            prompt_ids=prompt_ids,
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=-1,
        )
        control_logits = run_with_zero_ablation(
            model=model,
            prompt_ids=prompt_ids,
            layer=int(args.target_layer),
            head=int(sampled_control_head),
            position=-1,
        )
        middle_logits = run_with_zero_ablation(
            model=model,
            prompt_ids=prompt_ids,
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=int(middle_pos),
        )

        baseline_letter, _ = predict_letter_from_logits(baseline_logits, inputs.letter_token_sets)
        target_letter, _ = predict_letter_from_logits(target_logits, inputs.letter_token_sets)
        control_letter, _ = predict_letter_from_logits(control_logits, inputs.letter_token_sets)
        middle_letter, _ = predict_letter_from_logits(middle_logits, inputs.letter_token_sets)

        baseline_correct = bool(baseline_letter == item.correct)
        target_correct = bool(target_letter == item.correct)
        control_correct = bool(control_letter == item.correct)
        middle_correct = bool(middle_letter == item.correct)

        if item.base_correct and (not item.conflict_detected):
            category = "base_correct_non_conflict"
        elif item.base_correct and item.conflict_detected:
            category = "base_correct_conflict"
        else:
            category = "base_wrong"

        row = {
            "item_id": str(item.item_id),
            "category": str(category),
            "correct_letter": str(item.correct),
            "baseline_letter": baseline_letter,
            "target_ablation_letter": target_letter,
            "control_head_ablation_letter": control_letter,
            "middle_position_ablation_letter": middle_letter,
            "baseline_correct": bool(baseline_correct),
            "baseline_matches_h4": bool(baseline_correct == bool(item.base_correct)),
            "target_ablation_correct": bool(target_correct),
            "control_head_ablation_correct": bool(control_correct),
            "middle_position_ablation_correct": bool(middle_correct),
            "sampled_control_head": int(sampled_control_head),
            "middle_position": int(middle_pos),
            "conflict_detected": bool(item.conflict_detected),
        }
        ablation_rows.append(row)
        category_rows[str(category)].append(row)

    def _accuracy(rows: list[dict[str, Any]], key: str) -> tuple[float, list[float], int, int]:
        return _rate([bool(row.get(key, False)) for row in rows])

    categories_summary: dict[str, Any] = {}
    for category, rows in category_rows.items():
        baseline_rate, baseline_ci, baseline_k, n_rows = _accuracy(rows, "baseline_correct")
        target_rate_c, target_ci_c, target_k_c, _ = _accuracy(rows, "target_ablation_correct")
        control_rate_c, control_ci_c, control_k_c, _ = _accuracy(rows, "control_head_ablation_correct")
        middle_rate_c, middle_ci_c, middle_k_c, _ = _accuracy(rows, "middle_position_ablation_correct")
        categories_summary[str(category)] = {
            "n_examples": int(n_rows),
            "accuracy": {
                "no_ablation": float(baseline_rate),
                "target_head": float(target_rate_c),
                "control_head": float(control_rate_c),
                "middle_position": float(middle_rate_c),
            },
            "accuracy_ci95_wilson": {
                "no_ablation": baseline_ci,
                "target_head": target_ci_c,
                "control_head": control_ci_c,
                "middle_position": middle_ci_c,
            },
            "n_correct": {
                "no_ablation": int(baseline_k),
                "target_head": int(target_k_c),
                "control_head": int(control_k_c),
                "middle_position": int(middle_k_c),
            },
            "accuracy_drop_vs_no_ablation": {
                "target_head": float(baseline_rate - target_rate_c),
                "control_head": float(baseline_rate - control_rate_c),
                "middle_position": float(baseline_rate - middle_rate_c),
            },
            "mcnemar": {
                "target_vs_no_ablation": mcnemar_from_paired(
                    [bool(row["baseline_correct"]) for row in rows],
                    [bool(row["target_ablation_correct"]) for row in rows],
                ),
                "target_vs_control_head": mcnemar_from_paired(
                    [bool(row["target_ablation_correct"]) for row in rows],
                    [bool(row["control_head_ablation_correct"]) for row in rows],
                ),
                "target_vs_middle_position": mcnemar_from_paired(
                    [bool(row["target_ablation_correct"]) for row in rows],
                    [bool(row["middle_position_ablation_correct"]) for row in rows],
                ),
            },
        }

    ablation_summary = {
        "target_head": int(args.target_head),
        "sampled_control_head_offset": int(args.sampled_control_head_offset),
        "n_examples": int(len(ablation_rows)),
        "categories": categories_summary,
        "wall_clock_seconds": float(time.perf_counter() - ablation_started),
        "items": ablation_rows,
        "metadata": build_metadata(
            args=args,
            input_files=metadata_inputs,
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "ablation.json", ablation_summary)

    attention_hist_path = figures_dir / "attention_on_evidence.png"
    signature_panel_path = figures_dir / "head_signature_panel.png"
    _plot_attention_hist(attention_rows, attention_hist_path)
    _plot_signature_panel(
        attention_rows=attention_rows,
        ov_summary=ov_summary,
        ablation_summary=ablation_summary,
        out_path=signature_panel_path,
    )

    print(f"attention={out_dir / 'attention_on_evidence.json'}")
    print(f"ov={out_dir / 'ov_direction.json'}")
    print(f"ablation={out_dir / 'ablation.json'}")
    print(f"attention_plot={attention_hist_path}")
    print(f"signature_plot={signature_panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
