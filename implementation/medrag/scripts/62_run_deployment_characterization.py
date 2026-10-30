#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.decomposed_gate import DecomposedMarginGate
from medrag.final_paper import build_cross_model_metadata, load_medqa_cross_model_inputs, load_top_site, write_markdown_section
from medrag.medqa_v3 import (
    FirstStepTokenBiasProcessor,
    build_letter_token_sets,
    decode_letter,
    generate_ids,
    set_global_seed,
)
from medrag.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deployment characterization for MedQA cross-model intervention.")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--headline_path", default="results/medqa_final_v3/pr_sweep/headline.json")
    parser.add_argument("--llama_summary_path", default="results/medqa_final_v3/consolidation/headline_verify/H4/summary.json")
    parser.add_argument("--llama_site_path", default="results/medqa_final_v3/localization_v2/head_importance.json")
    parser.add_argument("--mistral_summary_path", default="results/cross_model/mistral_intervention/summary.json")
    parser.add_argument("--mistral_site_path", default="results/cross_model/mistral_localization/head_importance.json")
    parser.add_argument("--qwen_summary_path", default="results/cross_model/qwen_intervention/summary.json")
    parser.add_argument("--qwen_site_path", default="results/cross_model/qwen_localization/layer_importance.json")
    parser.add_argument("--out_dir", default="results/deployment")
    parser.add_argument("--cache_root", default="cache/deployment")
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--warmup_items", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prompt_variant", default="vanilla")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    model_name: str
    architecture: str
    summary_path: Path
    site_path: Path
    site_mode: str
    bootstrap_space: str
    tau_margin: float
    lambda_pos: float
    lambda_neg: float
    target_layer: int
    target_head: int


class BatchedFirstStepTokenBiasProcessor(LogitsProcessor):
    def __init__(self, configs: list[dict[str, Any]]) -> None:
        self._configs = configs
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self._step == 0:
            for row_idx, config in enumerate(self._configs):
                if row_idx >= int(scores.shape[0]):
                    break
                neg_bias = float(config.get("neg_bias", 0.0))
                pos_bias = float(config.get("pos_bias", 0.0))
                if neg_bias > 0.0:
                    for token_id in config.get("p_ids", []):
                        if 0 <= int(token_id) < int(scores.shape[-1]):
                            scores[row_idx, int(token_id)] -= neg_bias
                if pos_bias > 0.0:
                    for token_id in config.get("e_ids", []):
                        if 0 <= int(token_id) < int(scores.shape[-1]):
                            scores[row_idx, int(token_id)] += pos_bias
        self._step += 1
        return scores


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p95": 0.0}
    ordered = sorted(float(x) for x in values)
    return {
        "mean": float(statistics.fmean(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "p95": float(np.quantile(np.asarray(ordered, dtype=np.float64), 0.95)),
    }


def _prefill_capture(model, prompt_ids: torch.Tensor, device: torch.device, target_layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, args):
        del module
        hidden = args[0]
        if hidden is None or hidden.ndim != 3:
            return args
        captured["hidden_pre_norm"] = hidden[:, -1:, :].detach().float().cpu()
        return args

    handle = model.model.layers[int(target_layer)].post_attention_layernorm.register_forward_pre_hook(_hook)
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=prompt_ids.unsqueeze(0).to(device),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu()
    hidden_pre_norm = captured.get("hidden_pre_norm")
    if hidden_pre_norm is None:
        raise RuntimeError("Failed to capture hidden_pre_norm during prefill.")
    return logits, hidden_pre_norm


def _load_specs(args: argparse.Namespace) -> list[ModelSpec]:
    headline = read_json((PROJECT_ROOT / args.headline_path).resolve())
    llama_summary = read_json((PROJECT_ROOT / args.llama_summary_path).resolve())
    mistral_summary = read_json((PROJECT_ROOT / args.mistral_summary_path).resolve())
    qwen_summary = read_json((PROJECT_ROOT / args.qwen_summary_path).resolve())
    llama_site = load_top_site((PROJECT_ROOT / args.llama_site_path).resolve(), site_mode="head")
    mistral_site = load_top_site((PROJECT_ROOT / args.mistral_site_path).resolve(), site_mode="head")
    qwen_site = load_top_site((PROJECT_ROOT / args.qwen_site_path).resolve(), site_mode="layer")
    return [
        ModelSpec(
            key="llama",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            model_name="Llama-3.1-8B",
            architecture="standard_mha",
            summary_path=(PROJECT_ROOT / args.llama_summary_path).resolve(),
            site_path=(PROJECT_ROOT / args.llama_site_path).resolve(),
            site_mode="head",
            bootstrap_space=str(headline["bootstrap_space"]),
            tau_margin=float(headline["tau_margin"]),
            lambda_pos=float(llama_summary["lambda_pos"]),
            lambda_neg=float(llama_summary["lambda_neg"]),
            target_layer=int(llama_site["target_layer"]),
            target_head=int(llama_site["target_head"]),
        ),
        ModelSpec(
            key="mistral",
            model_id=str(mistral_summary["model_id"]),
            model_name=str(mistral_summary["model_name"]),
            architecture=str(mistral_summary["architecture"]),
            summary_path=(PROJECT_ROOT / args.mistral_summary_path).resolve(),
            site_path=(PROJECT_ROOT / args.mistral_site_path).resolve(),
            site_mode="head",
            bootstrap_space=str(mistral_summary["bootstrap_space"]),
            tau_margin=float(mistral_summary["selected_tau_margin"]),
            lambda_pos=float(mistral_summary["selected_lambda_pos"]),
            lambda_neg=float(mistral_summary["selected_lambda_neg"]),
            target_layer=int(mistral_summary["target_layer"]),
            target_head=int(mistral_summary["target_head"]),
        ),
        ModelSpec(
            key="qwen",
            model_id=str(qwen_summary["model_id"]),
            model_name=str(qwen_summary["model_name"]),
            architecture=str(qwen_summary["architecture"]),
            summary_path=(PROJECT_ROOT / args.qwen_summary_path).resolve(),
            site_path=(PROJECT_ROOT / args.qwen_site_path).resolve(),
            site_mode="layer",
            bootstrap_space=str(qwen_summary["bootstrap_space"]),
            tau_margin=float(qwen_summary["selected_tau_margin"]),
            lambda_pos=float(qwen_summary["selected_lambda_pos"]),
            lambda_neg=float(qwen_summary["selected_lambda_neg"]),
            target_layer=int(qwen_summary["target_layer"]),
            target_head=int(qwen_summary["target_head"]),
        ),
    ]


def _choose_bootstrap_sets(rec: Any, bootstrap_space: str) -> tuple[set[int], set[int], int | None, int | None]:
    if str(bootstrap_space) == "E_content":
        return (
            set(rec.ep_sets["E_content"]),
            set(rec.ep_sets["P_content"]),
            rec.representatives["content_ev_id"],
            rec.representatives["content_par_id"],
        )
    if str(bootstrap_space) == "E_letter":
        return (
            set(rec.ep_sets["E_letter"]),
            set(rec.ep_sets["P_letter"]),
            rec.representatives["letter_ev_id"],
            rec.representatives["letter_par_id"],
        )
    raise ValueError(f"Unsupported bootstrap_space: {bootstrap_space}")


def _valid_records(prepared_records: list[Any], bootstrap_space: str) -> list[Any]:
    out: list[Any] = []
    for rec in prepared_records:
        e_boot, p_boot, ev_id, par_id = _choose_bootstrap_sets(rec, bootstrap_space)
        if e_boot and p_boot and ev_id is not None and par_id is not None and set(rec.ep_sets["E_letter"]) and set(rec.ep_sets["P_letter"]):
            out.append(rec)
    return out


def _baseline_time(model, rec: Any, device: torch.device, max_new_tokens: int) -> tuple[float, float]:
    _reset_peak(device)
    _sync(device)
    started = time.perf_counter()
    _ = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens))
    _sync(device)
    elapsed = float(time.perf_counter() - started)
    return elapsed, _max_memory_mb(device)


def _gate_decision(
    *,
    model,
    tokenizer,
    device: torch.device,
    rec: Any,
    spec: ModelSpec,
    alpha: float,
) -> tuple[dict[str, Any], float]:
    logits, hidden_pre_norm = _prefill_capture(model, rec.prompt_ids, device, int(spec.target_layer))
    e_boot, p_boot, ev_id, par_id = _choose_bootstrap_sets(rec, spec.bootstrap_space)
    gate = DecomposedMarginGate(
        model=model,
        tokenizer=tokenizer,
        evidence_set=e_boot,
        parametric_set=p_boot,
        target_layer=int(spec.target_layer),
        target_head=int(spec.target_head),
        tau_margin=float(spec.tau_margin),
        alpha=float(alpha),
        lambda_max=max(float(spec.lambda_pos), float(spec.lambda_neg), 1.0),
    )
    gate.setup_example(int(ev_id), int(par_id), device=device)
    decision = gate.decide(
        hidden_pre_norm=hidden_pre_norm,
        lambda_pos=float(spec.lambda_pos),
        lambda_neg=float(spec.lambda_neg),
    )
    return {
        "decision": decision,
        "e_ids": sorted(int(x) for x in rec.ep_sets["E_letter"]),
        "p_ids": sorted(int(x) for x in rec.ep_sets["P_letter"]),
        "logits": logits,
    }, float(decision.prefill_margin)


def _gate_only_time(model, tokenizer, rec: Any, spec: ModelSpec, alpha: float, device: torch.device) -> tuple[float, float]:
    _reset_peak(device)
    _sync(device)
    started = time.perf_counter()
    _gate_decision(model=model, tokenizer=tokenizer, device=device, rec=rec, spec=spec, alpha=alpha)
    _sync(device)
    elapsed = float(time.perf_counter() - started)
    return elapsed, _max_memory_mb(device)


def _intervention_time(model, tokenizer, rec: Any, spec: ModelSpec, alpha: float, device: torch.device, max_new_tokens: int) -> tuple[float, float]:
    _reset_peak(device)
    _sync(device)
    started = time.perf_counter()
    gate_payload, _ = _gate_decision(model=model, tokenizer=tokenizer, device=device, rec=rec, spec=spec, alpha=alpha)
    decision = gate_payload["decision"]
    processors: list[LogitsProcessor] = []
    if float(decision.lambda_pos_applied) > 0.0 or float(decision.lambda_neg_applied) > 0.0:
        processors.append(
            FirstStepTokenBiasProcessor(
                e_token_ids=set(gate_payload["e_ids"]) if float(decision.lambda_pos_applied) > 0.0 else set(),
                p_token_ids=set(gate_payload["p_ids"]) if float(decision.lambda_neg_applied) > 0.0 else set(),
                pos_bias=float(decision.lambda_pos_applied),
                neg_bias=float(decision.lambda_neg_applied),
            )
        )
    _ = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens), logits_processors=processors)
    _sync(device)
    elapsed = float(time.perf_counter() - started)
    return elapsed, _max_memory_mb(device)


def _deployment_for_model(args: argparse.Namespace, spec: ModelSpec) -> tuple[dict[str, Any], dict[str, Any], list[Any], Any, Any, torch.device]:
    model, tokenizer, device = load_model_and_tokenizer(spec.model_id, args.device, args.torch_dtype)
    shared = load_medqa_cross_model_inputs(
        project_root=PROJECT_ROOT,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(spec.model_id),
        conflict_path=args.conflict_path,
        matched_subset_path=args.matched_subset_path,
        label_audit_path=args.label_audit_path,
        cache_dir=str(Path(args.cache_root) / spec.key),
        target_layer=int(spec.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        prompt_variant=str(args.prompt_variant),
        p_set_mode=str(args.p_set_mode),
    )
    candidates = _valid_records(shared.prepared_records, spec.bootstrap_space)
    needed = int(args.warmup_items) + int(args.n_items)
    if len(candidates) < needed:
        raise RuntimeError(f"{spec.model_name}: need {needed} valid records, found {len(candidates)}")
    selected = candidates[:needed]
    for rec in selected[: int(args.warmup_items)]:
        _baseline_time(model, rec, device, int(args.max_new_tokens))
        _gate_only_time(model, tokenizer, rec, spec, float(args.alpha), device)
        _intervention_time(model, tokenizer, rec, spec, float(args.alpha), device, int(args.max_new_tokens))

    baseline_times: list[float] = []
    gate_times: list[float] = []
    full_times: list[float] = []
    baseline_peaks: list[float] = []
    intervention_peaks: list[float] = []
    for rec in selected[int(args.warmup_items) :]:
        elapsed, peak = _baseline_time(model, rec, device, int(args.max_new_tokens))
        baseline_times.append(float(elapsed))
        baseline_peaks.append(float(peak))
        elapsed, peak = _gate_only_time(model, tokenizer, rec, spec, float(args.alpha), device)
        gate_times.append(float(elapsed))
        elapsed, peak = _intervention_time(model, tokenizer, rec, spec, float(args.alpha), device, int(args.max_new_tokens))
        full_times.append(float(elapsed))
        intervention_peaks.append(float(peak))

    latency_row = {
        "model": str(spec.model_name),
        "architecture": str(spec.architecture),
        "intervention_site": f"layer={int(spec.target_layer)}, head={int(spec.target_head)}" if spec.site_mode == "head" else f"layer={int(spec.target_layer)}, residual-stream",
        "n_items": int(args.n_items),
        "warmup_items": int(args.warmup_items),
        "baseline": _summary_stats(baseline_times),
        "gate_only": _summary_stats(gate_times),
        "full_intervention": _summary_stats(full_times),
    }
    latency_row["overhead_gate_only_vs_baseline"] = float(latency_row["gate_only"]["mean"] / max(latency_row["baseline"]["mean"], 1e-8))
    latency_row["overhead_full_vs_baseline"] = float(latency_row["full_intervention"]["mean"] / max(latency_row["baseline"]["mean"], 1e-8))

    memory_row = {
        "model": str(spec.model_name),
        "architecture": str(spec.architecture),
        "baseline_peak_mb": float(max(baseline_peaks) if baseline_peaks else 0.0),
        "intervention_peak_mb": float(max(intervention_peaks) if intervention_peaks else 0.0),
    }
    memory_row["delta_mb"] = float(memory_row["intervention_peak_mb"] - memory_row["baseline_peak_mb"])
    return latency_row, memory_row, selected, model, tokenizer, device


def _pad_batch(prompt_tensors: list[torch.Tensor], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(t.shape[0]) for t in prompt_tensors)
    batch = []
    mask = []
    for tensor in prompt_tensors:
        pad = max_len - int(tensor.shape[0])
        if pad > 0:
            batch.append(torch.cat([torch.full((pad,), int(pad_token_id), dtype=torch.long), tensor], dim=0))
            mask.append(torch.cat([torch.zeros((pad,), dtype=torch.long), torch.ones_like(tensor, dtype=torch.long)], dim=0))
        else:
            batch.append(tensor)
            mask.append(torch.ones_like(tensor, dtype=torch.long))
    return torch.stack(batch, dim=0), torch.stack(mask, dim=0)


def _resolve_pad_token_id(model, tokenizer) -> int:
    for candidate in (
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None
        if candidate is not None:
            return int(candidate)
    return 0


def _batched_generation(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, max_new_tokens: int) -> list[list[int]]:
    pad_token_id = _resolve_pad_token_id(model, tokenizer)
    prompt_batch, attention_mask = _pad_batch([row["prompt_ids"] for row in rows], pad_token_id)
    processor = BatchedFirstStepTokenBiasProcessor(
        [
            {
                "e_ids": row["e_ids"],
                "p_ids": row["p_ids"],
                "pos_bias": row["pos_bias"],
                "neg_bias": row["neg_bias"],
            }
            for row in rows
        ]
    )
    with torch.inference_mode():
        out = model.generate(
            prompt_batch.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=int(pad_token_id),
            logits_processor=LogitsProcessorList([processor]),
        )
    generated_ids: list[list[int]] = []
    for idx, row in enumerate(rows):
        prompt_len = int(row["prompt_ids"].shape[0])
        total_len = int(prompt_batch.shape[1])
        gen = [int(x) for x in out[idx, total_len:].detach().cpu().tolist()]
        generated_ids.append(gen)
    return generated_ids


def _run_batching_check(args: argparse.Namespace, spec: ModelSpec, model, tokenizer, device: torch.device, records: list[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rec in records[: int(args.batch_size)]:
        gate_payload, _ = _gate_decision(model=model, tokenizer=tokenizer, device=device, rec=rec, spec=spec, alpha=float(args.alpha))
        decision = gate_payload["decision"]
        rows.append(
            {
                "item_id": str(rec.item_id),
                "prompt_ids": rec.prompt_ids,
                "e_ids": gate_payload["e_ids"],
                "p_ids": gate_payload["p_ids"],
                "pos_bias": float(decision.lambda_pos_applied),
                "neg_bias": float(decision.lambda_neg_applied),
            }
        )
    letter_token_sets = build_letter_token_sets(tokenizer)
    single_letters = {}
    single_token_ids: dict[str, list[int]] = {}
    for row, rec in zip(rows, records[: int(args.batch_size)]):
        processors: list[LogitsProcessor] = []
        if float(row["pos_bias"]) > 0.0 or float(row["neg_bias"]) > 0.0:
            processors.append(
                FirstStepTokenBiasProcessor(
                    e_token_ids=set(row["e_ids"]) if float(row["pos_bias"]) > 0.0 else set(),
                    p_token_ids=set(row["p_ids"]) if float(row["neg_bias"]) > 0.0 else set(),
                    pos_bias=float(row["pos_bias"]),
                    neg_bias=float(row["neg_bias"]),
                )
            )
        ids = generate_ids(model, rec.prompt_ids.to(device), int(args.max_new_tokens), logits_processors=processors)
        single_token_ids[str(rec.item_id)] = list(ids)
        single_letters[str(rec.item_id)] = decode_letter(ids, tokenizer, letter_token_sets)
    batch_ids = _batched_generation(model, tokenizer, rows, device, int(args.max_new_tokens))
    batch_letters = {str(rec.item_id): decode_letter(ids, tokenizer, letter_token_sets) for rec, ids in zip(records[: int(args.batch_size)], batch_ids)}
    letter_mismatches = [
        {
            "item_id": str(rec.item_id),
            "single_letter": single_letters[str(rec.item_id)],
            "batch_letter": batch_letters[str(rec.item_id)],
            "single_ids": single_token_ids[str(rec.item_id)],
            "batch_ids": batch_ids[idx],
        }
        for idx, rec in enumerate(records[: int(args.batch_size)])
        if single_letters[str(rec.item_id)] != batch_letters[str(rec.item_id)]
    ]
    token_id_mismatches = [
        {
            "item_id": str(rec.item_id),
            "single_letter": single_letters[str(rec.item_id)],
            "batch_letter": batch_letters[str(rec.item_id)],
            "single_ids": single_token_ids[str(rec.item_id)],
            "batch_ids": batch_ids[idx],
        }
        for idx, rec in enumerate(records[: int(args.batch_size)])
        if single_token_ids[str(rec.item_id)] != batch_ids[idx]
    ]
    return {
        "model": str(spec.model_name),
        "batch_size": int(args.batch_size),
        "n_compared": int(min(len(records), int(args.batch_size))),
        "all_match": bool(not letter_mismatches),
        "letter_mismatches": letter_mismatches,
        "token_id_mismatches": token_id_mismatches,
    }


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_specs(args)
    latency_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    batching_payload: dict[str, Any] | None = None
    metadata_inputs = [
        (PROJECT_ROOT / args.conflict_path).resolve(),
        (PROJECT_ROOT / args.matched_subset_path).resolve(),
        (PROJECT_ROOT / args.label_audit_path).resolve(),
        (PROJECT_ROOT / args.headline_path).resolve(),
        (PROJECT_ROOT / args.llama_summary_path).resolve(),
        (PROJECT_ROOT / args.mistral_summary_path).resolve(),
        (PROJECT_ROOT / args.qwen_summary_path).resolve(),
    ]

    for spec in specs:
        latency_row, memory_row, selected_records, model, tokenizer, device = _deployment_for_model(args, spec)
        latency_rows.append(latency_row)
        memory_rows.append(memory_row)
        if spec.key == "llama":
            batching_payload = _run_batching_check(args, spec, model, tokenizer, device, selected_records)

    latency_payload = {
        "rows": latency_rows,
        "metadata": build_cross_model_metadata(
            args=args,
            input_files=metadata_inputs,
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    memory_payload = {
        "rows": memory_rows,
        "metadata": latency_payload["metadata"],
    }
    if batching_payload is None:
        batching_payload = {"all_match": False, "reason": "llama spec missing"}
    batching_payload["metadata"] = latency_payload["metadata"]
    write_json(out_dir / "latency.json", latency_payload)
    write_json(out_dir / "memory.json", memory_payload)
    write_json(out_dir / "batching.json", batching_payload)

    lines = [
        "## Latency",
        "",
        "| Model | Baseline mean (s) | Gate-only mean (s) | Full mean (s) | Gate/Baseline | Full/Baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in latency_rows:
        lines.append(
            f"| {row['model']} | {row['baseline']['mean']:.4f} | {row['gate_only']['mean']:.4f} | {row['full_intervention']['mean']:.4f} | "
            f"{row['overhead_gate_only_vs_baseline']:.3f} | {row['overhead_full_vs_baseline']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Memory",
            "",
            "| Model | Baseline peak (MB) | Intervention peak (MB) | Delta (MB) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in memory_rows:
        lines.append(f"| {row['model']} | {row['baseline_peak_mb']:.1f} | {row['intervention_peak_mb']:.1f} | {row['delta_mb']:.1f} |")
    lines.extend(
        [
            "",
            "## Batching",
            f"- Llama batch-size-{int(args.batch_size)} decoded-letter parity: `{bool(batching_payload['all_match'])}`",
            f"- Compared items: `{int(batching_payload['n_compared'])}`",
            f"- Token-id-only mismatches: `{len(batching_payload.get('token_id_mismatches', []))}`",
            "",
            "## Synthesis",
            "The deployment path measures the current inference stack directly: baseline generation, prefill margin-gate computation, and prefill plus first-step token-bias intervention.",
            "The batching check verifies row-specific first-step biasing yields the same outputs in batched and single-item inference for Llama.",
        ]
    )
    write_markdown_section(out_dir / "SECTION.md", "Deployment Characterization", lines)
    print(json.dumps({"latency": str(out_dir / "latency.json"), "memory": str(out_dir / "memory.json"), "batching": str(out_dir / "batching.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
