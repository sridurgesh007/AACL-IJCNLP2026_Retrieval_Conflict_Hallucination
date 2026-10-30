from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import (
    FirstStepTokenBiasProcessor,
    build_letter_token_sets,
    decode_letter,
    evaluate_head_ov_hook_intervention,
    evaluate_intervention_cell,
    generate_ids,
    load_conflicts,
    prepare_examples_for_variant,
    set_global_seed,
    summarize_condition_accuracy,
    summarize_intervention_results,
)
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import build_run_metadata, git_commit_hash, read_json, sha256_file, write_json, write_jsonl
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


def parse_float_grid(raw: str, fallback: list[float]) -> list[float]:
    text = str(raw or "").strip()
    if not text:
        return [float(x) for x in fallback]
    out: list[float] = []
    seen: set[float] = set()
    for piece in text.split(","):
        token = piece.strip()
        if not token:
            continue
        value = float(token)
        if value in seen:
            continue
        seen.add(value)
        out.append(float(value))
    return out or [float(x) for x in fallback]


def slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = float(sum(float(x) for x in values) / float(n))
    if n == 1:
        return mean, 0.0, mean, mean
    var = float(sum((float(x) - mean) ** 2 for x in values) / float(max(n - 1, 1)))
    std = math.sqrt(max(var, 0.0))
    half = 1.96 * (std / math.sqrt(float(n)))
    return mean, std, float(mean - half), float(mean + half)


@dataclass(frozen=True)
class MedQACrossModelInputs:
    conflicts: list[dict[str, Any]]
    matched_ids: set[str]
    clean_ids: set[str]
    letter_token_sets: dict[str, set[int]]
    prepared_records: list[Any]
    metadata_inputs: list[Path]


def load_medqa_cross_model_inputs(
    *,
    project_root: Path,
    model,
    tokenizer,
    device,
    model_id: str,
    conflict_path: str | Path,
    matched_subset_path: str | Path,
    label_audit_path: str | Path,
    cache_dir: str | Path,
    target_layer: int,
    max_length: int,
    max_new_tokens: int,
    prompt_variant: str = "vanilla",
    p_set_mode: str = "all_wrong",
) -> MedQACrossModelInputs:
    conflict_path = (project_root / conflict_path).resolve()
    matched_subset_path = (project_root / matched_subset_path).resolve()
    label_audit_path = (project_root / label_audit_path).resolve()
    cache_dir = (project_root / cache_dir).resolve()

    conflicts = load_conflicts(conflict_path)
    matched_payload = read_json(matched_subset_path)
    matched_ids = {str(x) for x in matched_payload.get("matched_item_ids", [])}
    label_audit = load_label_audit(label_audit_path)
    clean_ids = set(clean_subset_item_ids(matched_payload, label_audit))
    letter_token_sets = build_letter_token_sets(tokenizer)
    cache = ActivationCache(cache_dir)
    prepared_records = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=cache,
        model_id=str(model_id),
        prompt_variant=str(prompt_variant),
        p_set_mode=str(p_set_mode),
        target_layer=int(target_layer),
        max_length=int(max_length),
        max_new_tokens=int(max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    return MedQACrossModelInputs(
        conflicts=conflicts,
        matched_ids=matched_ids,
        clean_ids=clean_ids,
        letter_token_sets=letter_token_sets,
        prepared_records=prepared_records,
        metadata_inputs=[conflict_path, matched_subset_path, label_audit_path],
    )


def load_top_site(localization_path: str | Path, *, site_mode: str) -> dict[str, Any]:
    payload = read_json(localization_path)
    if str(site_mode) == "head":
        candidates = list(payload.get("candidate_heads") or payload.get("top_5_heads") or [])
        if not candidates:
            raise RuntimeError(f"No candidate_heads/top_5_heads in {localization_path}")
        top = dict(candidates[0])
        return {
            "site_mode": "head",
            "target_layer": int(top["layer"]),
            "target_head": int(top["head"]),
            "site_description": f"layer={int(top['layer'])}, head={int(top['head'])}",
            "site_payload": top,
        }
    if str(site_mode) == "layer":
        top = dict(payload.get("top_layer") or {})
        if not top:
            rows = list(payload.get("layers") or [])
            if not rows:
                raise RuntimeError(f"No top_layer/layers in {localization_path}")
            top = dict(rows[0])
        return {
            "site_mode": "layer",
            "target_layer": int(top["layer"]),
            "target_head": int(top.get("head", 0)),
            "site_description": f"layer={int(top['layer'])}, residual-stream",
            "site_payload": top,
        }
    raise ValueError(f"Unsupported site_mode: {site_mode}")


def matched_subset_ids(prepared_records: list[Any], matched_ids: set[str], limit: int) -> list[str]:
    ordered = [str(rec.item_id) for rec in prepared_records if str(rec.item_id) in matched_ids]
    if int(limit) > 0:
        ordered = ordered[: int(limit)]
    return ordered


def score_intervention_summary(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["restoration_improvement"]),
        -float(summary["harm_rate"]),
        float(summary["ft_accuracy"]),
        -float(summary["gate_fire_rate_base_correct"]),
    )


def run_medqa_pilot_grid(
    *,
    prepared_records: list[Any],
    model,
    tokenizer,
    device,
    letter_token_sets: dict[str, set[int]],
    matched_ids: set[str],
    clean_ids: set[str],
    bootstrap_space: str,
    target_layer: int,
    target_head: int,
    alpha: float,
    max_new_tokens: int,
    tau_grid: list[float],
    lambda_neg_grid: list[float],
    pos_ratio_grid: list[float],
    pilot_examples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    pilot_ids_list = matched_subset_ids(prepared_records, matched_ids, int(pilot_examples))
    pilot_id_set = set(pilot_ids_list)
    pilot_clean_ids = set(clean_ids) & pilot_id_set
    if not pilot_id_set:
        raise RuntimeError("No pilot ids found for matched subset.")

    rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_results: list[dict[str, Any]] | None = None
    best_config: dict[str, Any] | None = None
    for tau_margin in tau_grid:
        for lambda_neg in lambda_neg_grid:
            for pos_ratio in pos_ratio_grid:
                lambda_pos = float(pos_ratio) * float(lambda_neg)
                summary, results = evaluate_intervention_cell(
                    prepared_records=prepared_records,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    letter_token_sets=letter_token_sets,
                    bootstrap_space=str(bootstrap_space),
                    intervention_space="E_letter",
                    lambda_pos=float(lambda_pos),
                    lambda_neg=float(lambda_neg),
                    tau_margin=float(tau_margin),
                    alpha=float(alpha),
                    target_layer=int(target_layer),
                    target_head=int(target_head),
                    matched_item_ids=pilot_id_set,
                    clean_item_ids=pilot_clean_ids,
                    seed=int(seed),
                    max_new_tokens=int(max_new_tokens),
                )
                row = {
                    "tau_margin": float(tau_margin),
                    "lambda_pos": float(lambda_pos),
                    "lambda_neg": float(lambda_neg),
                    "pos_ratio": float(pos_ratio),
                    "baseline_accuracy": float(summary["baseline_accuracy"]),
                    "ft_accuracy": float(summary["ft_accuracy"]),
                    "restoration_rate": float(summary["restoration_rate"]),
                    "harm_rate": float(summary["harm_rate"]),
                    "restoration_improvement": float(summary["restoration_improvement"]),
                    "gate_fire_rate_base_wrong": float(summary["gate_fire_rate_base_wrong"]),
                    "gate_fire_rate_base_correct": float(summary["gate_fire_rate_base_correct"]),
                    "n_base_wrong": int(summary["n_base_wrong"]),
                    "n_restored": int(summary["n_restored"]),
                    "n_harmed": int(summary["n_harmed"]),
                }
                rows.append(row)
                if best_summary is None or score_intervention_summary(summary) > score_intervention_summary(best_summary):
                    best_summary = dict(summary)
                    best_results = list(results)
                    best_config = dict(row)
    if best_summary is None or best_results is None or best_config is None:
        raise RuntimeError("Pilot grid produced no results.")
    best_config["matched_item_ids"] = pilot_ids_list
    best_config["summary"] = best_summary
    return rows, best_config, best_results


def full_intervention_summary(
    *,
    prepared_records: list[Any],
    model,
    tokenizer,
    device,
    letter_token_sets: dict[str, set[int]],
    matched_ids: set[str],
    clean_ids: set[str],
    bootstrap_space: str,
    target_layer: int,
    target_head: int,
    alpha: float,
    max_new_tokens: int,
    tau_margin: float,
    lambda_pos: float,
    lambda_neg: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return evaluate_intervention_cell(
        prepared_records=prepared_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        letter_token_sets=letter_token_sets,
        bootstrap_space=str(bootstrap_space),
        intervention_space="E_letter",
        lambda_pos=float(lambda_pos),
        lambda_neg=float(lambda_neg),
        tau_margin=float(tau_margin),
        alpha=float(alpha),
        target_layer=int(target_layer),
        target_head=int(target_head),
        matched_item_ids=matched_ids,
        clean_item_ids=clean_ids,
        seed=int(seed),
        max_new_tokens=int(max_new_tokens),
    )


def write_medqa_cross_model_bundle(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    pilot_grid: list[dict[str, Any]],
    pilot_best: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = dict(summary)
    summary_payload["pilot_selection"] = pilot_best
    summary_payload["metadata"] = metadata
    write_json(out_dir / "summary.json", summary_payload)
    write_json(out_dir / "pilot_grid.json", {"rows": pilot_grid, "best": pilot_best, "metadata": metadata})
    write_jsonl(out_dir / "results.jsonl", results)


def rows_from_prepared_records(
    prepared_records: list[Any],
    *,
    matched_ids: set[str],
    clean_ids: set[str],
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    for rec in prepared_records:
        total_seconds += float(rec.prefill_wall_clock_seconds + rec.generation_wall_clock_seconds)
        rows.append(
            {
                "item_id": str(rec.item_id),
                "question": str(rec.question),
                "correct": str(rec.correct),
                "parametric": str(rec.parametric),
                "condition_letter": rec.base_letter,
                "condition_correct": bool(rec.base_correct),
                "in_matched_subset": bool(rec.item_id in matched_ids),
                "in_clean_subset": bool(rec.item_id in clean_ids),
                "skipped": False,
                "prompt_variant": str(rec.prompt_variant),
            }
        )
    return rows, float(total_seconds)


def summary_from_prepared_records(
    prepared_records: list[Any],
    *,
    matched_ids: set[str],
    clean_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, wall_clock_seconds = rows_from_prepared_records(prepared_records, matched_ids=matched_ids, clean_ids=clean_ids)
    summary = summarize_condition_accuracy(rows=rows, correct_key="condition_correct", wall_clock_seconds=wall_clock_seconds)
    return summary, rows


def build_transition_rows(
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(row["item_id"]): row for row in baseline_rows}
    rows: list[dict[str, Any]] = []
    for cur in current_rows:
        item_id = str(cur["item_id"])
        base = baseline_by_id.get(item_id)
        if base is None:
            continue
        base_correct = bool(base.get("condition_correct", False))
        cur_correct = bool(cur.get("condition_correct", False))
        rows.append(
            {
                "item_id": item_id,
                "question": str(cur.get("question", "")),
                "correct": str(cur.get("correct", "")),
                "parametric": str(cur.get("parametric", "")),
                "base_letter": base.get("condition_letter"),
                "ft_letter": cur.get("condition_letter"),
                "base_correct": base_correct,
                "ft_correct": cur_correct,
                "ft_restored": bool((not base_correct) and cur_correct),
                  "ft_harmed": bool(base_correct and (not cur_correct)),
                  "base_matches_parametric": bool(str(base.get("condition_letter", "")) == str(cur.get("parametric", ""))),
                  "conflict_detected": False,
                  # Non-intervention C1/C2 transition rows do not carry a meaningful
                  # prefill margin, but downstream summary helpers expect a numeric field.
                  "prefill_margin": 0.0,
                  "gate_lambda": 0.0,
                  "lambda_pos_applied": 0.0,
                  "lambda_neg_applied": 0.0,
                "lambda_clipped": False,
                "n_E_content": 0,
                "n_P_content": 0,
                "n_E_letter": 0,
                "n_P_letter": 0,
                "bootstrap_space": "",
                "intervention_space": "",
                "in_matched_subset": bool(cur.get("in_matched_subset", False)),
                "in_clean_subset": bool(cur.get("in_clean_subset", False)),
                "skipped": False,
            }
        )
    return rows


def compare_condition_summaries(
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    *,
    wall_clock_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transition_rows = build_transition_rows(baseline_rows, current_rows)
    summary = summarize_intervention_results(results=transition_rows, wall_clock_seconds=float(wall_clock_seconds))
    return summary, transition_rows


def _logits_from_hidden(model, hidden_pre_norm: torch.Tensor) -> torch.Tensor:
    hidden = hidden_pre_norm
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim != 3:
        raise ValueError(f"Unexpected hidden_pre_norm shape: {tuple(hidden.shape)}")
    lm_weight = model.lm_head.weight.detach().float()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias_fp32 = lm_bias.detach().float() if lm_bias is not None else None
    target_device = model.lm_head.weight.device
    target_dtype = model.lm_head.weight.dtype
    hidden = hidden.to(device=target_device, dtype=target_dtype)
    with torch.no_grad():
        normed = model.model.norm(hidden)
        logits = F.linear(normed.float(), lm_weight, lm_bias_fp32)[0, -1, :]
    return logits


def margin_from_hidden(
    model,
    hidden_pre_norm: torch.Tensor,
    *,
    evidence_set: set[int],
    parametric_set: set[int],
) -> float:
    logits = _logits_from_hidden(model, hidden_pre_norm)
    vocab_size = int(logits.numel())
    e_ids = [int(tid) for tid in evidence_set if 0 <= int(tid) < vocab_size]
    p_ids = [int(tid) for tid in parametric_set if 0 <= int(tid) < vocab_size]
    if not e_ids or not p_ids:
        return float("nan")
    log_p = torch.log_softmax(logits, dim=-1)
    e_idx = torch.as_tensor(e_ids, device=log_p.device, dtype=torch.long)
    p_idx = torch.as_tensor(p_ids, device=log_p.device, dtype=torch.long)
    log_p_e = torch.logsumexp(log_p.index_select(0, e_idx), dim=0)
    log_p_p = torch.logsumexp(log_p.index_select(0, p_idx), dim=0)
    return float((log_p_e - log_p_p).item())


def collect_margin_rows(
    *,
    prepared_records: list[Any],
    model,
    bootstrap_space: str,
    matched_ids: set[str],
    limit: int = 0,
) -> list[dict[str, Any]]:
    selected_ids = set(matched_subset_ids(prepared_records, matched_ids, int(limit))) if int(limit) > 0 else set()
    rows: list[dict[str, Any]] = []
    for rec in prepared_records:
        if selected_ids and str(rec.item_id) not in selected_ids:
            continue
        if str(bootstrap_space) == "E_content":
            e_boot = set(rec.ep_sets["E_content"])
            p_boot = set(rec.ep_sets["P_content"])
        elif str(bootstrap_space) == "E_letter":
            e_boot = set(rec.ep_sets["E_letter"])
            p_boot = set(rec.ep_sets["P_letter"])
        else:
            raise ValueError(f"Unsupported bootstrap_space: {bootstrap_space}")
        if not e_boot or not p_boot:
            continue
        margin = margin_from_hidden(model, rec.hidden_pre_norm, evidence_set=e_boot, parametric_set=p_boot)
        rows.append(
            {
                "item_id": str(rec.item_id),
                "prompt_variant": str(rec.prompt_variant),
                "base_letter": rec.base_letter,
                "base_correct": bool(rec.base_correct),
                "prefill_margin": float(margin),
            }
        )
    return rows


def evaluate_cross_model_intervention(
    *,
    prepared_records: list[Any],
    model,
    tokenizer,
    device: torch.device,
    letter_token_sets: dict[str, set[int]],
    bootstrap_space: str,
    lambda_pos: float,
    lambda_neg: float,
    tau_margin: float,
    matched_item_ids: set[str],
    clean_item_ids: set[str],
    seed: int,
    max_new_tokens: int = 5,
    site_mode: str = "head",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del site_mode
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for rec in prepared_records:
        if str(bootstrap_space) == "E_content":
            e_boot = set(rec.ep_sets["E_content"])
            p_boot = set(rec.ep_sets["P_content"])
        elif str(bootstrap_space) == "E_letter":
            e_boot = set(rec.ep_sets["E_letter"])
            p_boot = set(rec.ep_sets["P_letter"])
        else:
            raise ValueError(f"Unsupported bootstrap_space: {bootstrap_space}")

        e_int = set(rec.ep_sets["E_letter"])
        p_int = set(rec.ep_sets["P_letter"])
        if not e_boot or not p_boot or not e_int or not p_int:
            results.append(
                {
                    "item_id": str(rec.item_id),
                    "question": str(rec.question),
                    "correct": str(rec.correct),
                    "parametric": str(rec.parametric),
                    "base_letter": rec.base_letter,
                    "ft_letter": rec.base_letter,
                    "base_correct": bool(rec.base_correct),
                    "ft_correct": bool(rec.base_correct),
                    "ft_restored": False,
                    "ft_harmed": False,
                    "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                    "conflict_detected": False,
                    "prefill_margin": float("nan"),
                    "gate_lambda": 0.0,
                    "lambda_pos_applied": 0.0,
                    "lambda_neg_applied": 0.0,
                    "lambda_clipped": False,
                    "n_E_content": int(rec.ep_sets["n_E_content"]),
                    "n_P_content": int(rec.ep_sets["n_P_content"]),
                    "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                    "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                    "bootstrap_space": str(bootstrap_space),
                    "intervention_space": "E_letter",
                    "in_matched_subset": bool(rec.item_id in matched_item_ids),
                    "in_clean_subset": bool(rec.item_id in clean_item_ids),
                    "skipped": True,
                    "skip_reason": "missing_token_sets",
                    "seed": int(seed),
                }
            )
            continue

        margin = margin_from_hidden(model, rec.hidden_pre_norm, evidence_set=e_boot, parametric_set=p_boot)
        conflict_detected = bool(math.isfinite(float(margin)) and float(margin) < float(tau_margin))
        lambda_pos_applied = float(lambda_pos) if conflict_detected else 0.0
        lambda_neg_applied = float(lambda_neg) if conflict_detected else 0.0
        logits_processors: list[Any] = []
        if (lambda_pos_applied > 0.0 and e_int) or (lambda_neg_applied > 0.0 and p_int):
            logits_processors.append(
                FirstStepTokenBiasProcessor(
                    e_token_ids=(set(e_int) if lambda_pos_applied > 0.0 else set()),
                    p_token_ids=(set(p_int) if lambda_neg_applied > 0.0 else set()),
                    pos_bias=float(lambda_pos_applied),
                    neg_bias=float(lambda_neg_applied),
                )
            )
        ft_ids = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens), logits_processors=logits_processors)
        ft_letter = decode_letter(ft_ids, tokenizer, letter_token_sets)
        ft_correct = bool(ft_letter == rec.correct)
        results.append(
            {
                "item_id": str(rec.item_id),
                "question": str(rec.question),
                "correct": str(rec.correct),
                "parametric": str(rec.parametric),
                "base_letter": rec.base_letter,
                "ft_letter": ft_letter,
                "base_correct": bool(rec.base_correct),
                "ft_correct": bool(ft_correct),
                "ft_restored": bool((not rec.base_correct) and ft_correct),
                "ft_harmed": bool(rec.base_correct and (not ft_correct)),
                "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                "conflict_detected": bool(conflict_detected),
                "prefill_margin": float(margin),
                "gate_lambda": float(max(float(tau_margin) - float(margin), 0.0)) if math.isfinite(float(margin)) else 0.0,
                "lambda_pos_applied": float(lambda_pos_applied),
                "lambda_neg_applied": float(lambda_neg_applied),
                "lambda_clipped": False,
                "n_E_content": int(rec.ep_sets["n_E_content"]),
                "n_P_content": int(rec.ep_sets["n_P_content"]),
                "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                "bootstrap_space": str(bootstrap_space),
                "intervention_space": "E_letter",
                "in_matched_subset": bool(rec.item_id in matched_item_ids),
                "in_clean_subset": bool(rec.item_id in clean_item_ids),
                "skipped": False,
                "seed": int(seed),
            }
        )
    elapsed = float(time.perf_counter() - t0)
    summary = summarize_intervention_results(results=results, wall_clock_seconds=elapsed)
    return summary, results


def _valid_vocab_ids(model, token_ids: set[int]) -> list[int]:
    vocab_size = int(model.lm_head.weight.shape[0])
    return sorted({int(tid) for tid in token_ids if 0 <= int(tid) < vocab_size})


def _residual_letter_bias_delta(
    model,
    *,
    e_token_ids: set[int],
    p_token_ids: set[int],
    lambda_pos: float,
    lambda_neg: float,
    device: torch.device,
) -> torch.Tensor:
    e_ids = _valid_vocab_ids(model, set(e_token_ids))
    p_ids = _valid_vocab_ids(model, set(p_token_ids) - set(e_ids))
    if not e_ids and not p_ids:
        raise ValueError("Residual hook requires at least one valid E/P token id.")

    token_ids = e_ids + p_ids
    desired = [float(lambda_pos) for _ in e_ids] + [-float(lambda_neg) for _ in p_ids]
    if not any(abs(float(x)) > 0.0 for x in desired):
        return torch.zeros(int(model.config.hidden_size), dtype=torch.float32, device=device)

    weight_device = model.lm_head.weight.device
    lm_weight = model.lm_head.weight.detach()
    idx = torch.as_tensor(token_ids, dtype=torch.long, device=weight_device)
    w = lm_weight.index_select(0, idx).detach().float().to(device)
    target = torch.as_tensor(desired, dtype=torch.float32, device=device)
    gram = w.matmul(w.transpose(0, 1))
    diag_mean = float(torch.diagonal(gram).abs().mean().detach().cpu().item()) if gram.numel() else 1.0
    ridge = max(diag_mean * 1e-6, 1e-6)
    eye = torch.eye(int(gram.shape[0]), dtype=torch.float32, device=device)
    coeff = torch.linalg.solve(gram + ridge * eye, target)
    return w.transpose(0, 1).matmul(coeff).detach().float()


def _residual_delta_effect_stats(
    model,
    *,
    delta: torch.Tensor,
    e_token_ids: set[int],
    p_token_ids: set[int],
    lambda_pos: float,
    lambda_neg: float,
    device: torch.device,
) -> dict[str, Any]:
    e_ids = _valid_vocab_ids(model, set(e_token_ids))
    p_ids = _valid_vocab_ids(model, set(p_token_ids) - set(e_ids))
    weight_device = model.lm_head.weight.device
    delta_vec = delta.detach().float().to(device)

    def _stats(ids: list[int]) -> dict[str, float | None]:
        if not ids:
            return {"mean": None, "min": None, "max": None}
        idx = torch.as_tensor(ids, dtype=torch.long, device=weight_device)
        w = model.lm_head.weight.detach().index_select(0, idx).float().to(device)
        effects = w.matmul(delta_vec)
        return {
            "mean": float(effects.mean().detach().cpu().item()),
            "min": float(effects.min().detach().cpu().item()),
            "max": float(effects.max().detach().cpu().item()),
        }

    e_stats = _stats(e_ids)
    p_stats = _stats(p_ids)
    e_mean = e_stats["mean"]
    p_mean = p_stats["mean"]
    return {
        "residual_delta_target_lambda_pos": float(lambda_pos),
        "residual_delta_target_lambda_neg": float(lambda_neg),
        "residual_delta_n_e_ids": int(len(e_ids)),
        "residual_delta_n_p_ids": int(len(p_ids)),
        "residual_delta_e_mean_effect": e_mean,
        "residual_delta_e_min_effect": e_stats["min"],
        "residual_delta_e_max_effect": e_stats["max"],
        "residual_delta_p_mean_effect": p_mean,
        "residual_delta_p_min_effect": p_stats["min"],
        "residual_delta_p_max_effect": p_stats["max"],
        "residual_delta_mean_e_minus_p_effect": (
            float(e_mean) - float(p_mean)
            if e_mean is not None and p_mean is not None
            else None
        ),
        "residual_delta_promotes_e": bool(e_mean is not None and float(e_mean) > 0.0) if e_ids else False,
        "residual_delta_suppresses_p": bool(p_mean is not None and float(p_mean) < 0.0) if p_ids else False,
    }


def _generate_ids_with_residual_delta(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_layer: int,
    delta: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], bool]:
    applied = {"value": False}

    def _hook(module, args):
        del module
        if applied["value"] or not args:
            return args
        hidden = args[0]
        if hidden is None or hidden.ndim != 3:
            return args
        hidden_mod = hidden.clone()
        update = delta.to(device=hidden_mod.device, dtype=hidden_mod.dtype).view(1, 1, -1)
        hidden_mod[:, -1:, :] = hidden_mod[:, -1:, :] + update
        applied["value"] = True
        return (hidden_mod,) + tuple(args[1:])

    handle = model.model.layers[int(target_layer)].post_attention_layernorm.register_forward_pre_hook(_hook)
    try:
        output_ids = generate_ids(model, prompt_ids, int(max_new_tokens))
    finally:
        handle.remove()
    return output_ids, bool(applied["value"])


def evaluate_residual_stream_hook_intervention(
    *,
    prepared_records: list[Any],
    model,
    tokenizer,
    device: torch.device,
    letter_token_sets: dict[str, set[int]],
    bootstrap_space: str,
    lambda_pos: float,
    lambda_neg: float,
    tau_margin: float,
    target_layer: int,
    target_head: int,
    matched_item_ids: set[str],
    clean_item_ids: set[str],
    seed: int,
    max_new_tokens: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    delta_cache: dict[tuple[tuple[int, ...], tuple[int, ...], float, float], dict[str, Any]] = {}
    t0 = time.perf_counter()
    for rec in prepared_records:
        if str(bootstrap_space) == "E_content":
            e_boot = set(rec.ep_sets["E_content"])
            p_boot = set(rec.ep_sets["P_content"])
        elif str(bootstrap_space) == "E_letter":
            e_boot = set(rec.ep_sets["E_letter"])
            p_boot = set(rec.ep_sets["P_letter"])
        else:
            raise ValueError(f"Unsupported bootstrap_space: {bootstrap_space}")

        e_int = set(rec.ep_sets["E_letter"])
        p_int = set(rec.ep_sets["P_letter"])
        if not e_boot or not p_boot or not e_int or not p_int:
            results.append(
                {
                    "item_id": str(rec.item_id),
                    "question": str(rec.question),
                    "correct": str(rec.correct),
                    "parametric": str(rec.parametric),
                    "base_letter": rec.base_letter,
                    "ft_letter": rec.base_letter,
                    "base_correct": bool(rec.base_correct),
                    "ft_correct": bool(rec.base_correct),
                    "ft_restored": False,
                    "ft_harmed": False,
                    "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                    "conflict_detected": False,
                    "prefill_margin": float("nan"),
                    "gate_lambda": 0.0,
                    "lambda_pos_applied": 0.0,
                    "lambda_neg_applied": 0.0,
                    "lambda_clipped": False,
                    "n_E_content": int(rec.ep_sets["n_E_content"]),
                    "n_P_content": int(rec.ep_sets["n_P_content"]),
                    "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                    "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                    "bootstrap_space": str(bootstrap_space),
                    "intervention_space": "E_letter",
                    "intervention_impl": "residual_stream_hook",
                    "target_layer": int(target_layer),
                    "target_head": int(target_head),
                    "residual_hook_applied": False,
                    "in_matched_subset": bool(rec.item_id in matched_item_ids),
                    "in_clean_subset": bool(rec.item_id in clean_item_ids),
                    "skipped": True,
                    "skip_reason": "missing_token_sets",
                    "seed": int(seed),
                }
            )
            continue

        margin = margin_from_hidden(model, rec.hidden_pre_norm, evidence_set=e_boot, parametric_set=p_boot)
        conflict_detected = bool(math.isfinite(float(margin)) and float(margin) < float(tau_margin))
        lambda_pos_applied = float(lambda_pos) if conflict_detected else 0.0
        lambda_neg_applied = float(lambda_neg) if conflict_detected else 0.0
        hook_applied = False
        delta_stats: dict[str, Any] = {}
        if lambda_pos_applied > 0.0 or lambda_neg_applied > 0.0:
            delta_key = (tuple(sorted(e_int)), tuple(sorted(p_int)), float(lambda_pos_applied), float(lambda_neg_applied))
            delta_payload = delta_cache.get(delta_key)
            if delta_payload is None:
                delta = _residual_letter_bias_delta(
                    model,
                    e_token_ids=e_int,
                    p_token_ids=p_int,
                    lambda_pos=float(lambda_pos_applied),
                    lambda_neg=float(lambda_neg_applied),
                    device=device,
                )
                delta_stats = _residual_delta_effect_stats(
                    model,
                    delta=delta,
                    e_token_ids=e_int,
                    p_token_ids=p_int,
                    lambda_pos=float(lambda_pos_applied),
                    lambda_neg=float(lambda_neg_applied),
                    device=device,
                )
                delta_payload = {"delta": delta.detach().cpu(), "stats": dict(delta_stats)}
                delta_cache[delta_key] = delta_payload
            else:
                delta_stats = dict(delta_payload["stats"])
            ft_ids, hook_applied = _generate_ids_with_residual_delta(
                model,
                rec.prompt_ids.to(device),
                target_layer=int(target_layer),
                delta=delta_payload["delta"].to(device),
                max_new_tokens=int(max_new_tokens),
            )
        else:
            ft_ids = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens))

        ft_letter = decode_letter(ft_ids, tokenizer, letter_token_sets)
        ft_correct = bool(ft_letter == rec.correct)
        results.append(
            {
                "item_id": str(rec.item_id),
                "question": str(rec.question),
                "correct": str(rec.correct),
                "parametric": str(rec.parametric),
                "base_letter": rec.base_letter,
                "ft_letter": ft_letter,
                "base_correct": bool(rec.base_correct),
                "ft_correct": bool(ft_correct),
                "ft_restored": bool((not rec.base_correct) and ft_correct),
                "ft_harmed": bool(rec.base_correct and (not ft_correct)),
                "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                "conflict_detected": bool(conflict_detected),
                "prefill_margin": float(margin),
                "gate_lambda": float(max(float(tau_margin) - float(margin), 0.0)) if math.isfinite(float(margin)) else 0.0,
                "lambda_pos_applied": float(lambda_pos_applied),
                "lambda_neg_applied": float(lambda_neg_applied),
                "lambda_clipped": False,
                "n_E_content": int(rec.ep_sets["n_E_content"]),
                "n_P_content": int(rec.ep_sets["n_P_content"]),
                "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                "bootstrap_space": str(bootstrap_space),
                "intervention_space": "E_letter",
                "intervention_impl": "residual_stream_hook",
                "target_layer": int(target_layer),
                "target_head": int(target_head),
                "residual_hook_applied": bool(hook_applied),
                **delta_stats,
                "in_matched_subset": bool(rec.item_id in matched_item_ids),
                "in_clean_subset": bool(rec.item_id in clean_item_ids),
                "skipped": False,
                "seed": int(seed),
            }
        )
    elapsed = float(time.perf_counter() - t0)
    summary = summarize_intervention_results(results=results, wall_clock_seconds=elapsed)
    return summary, results


def resolve_intervention_impl(*, site_mode: str, requested: str = "auto") -> str:
    raw = str(requested or "auto").strip().lower()
    if raw == "auto":
        return "head_logit_bias" if str(site_mode) == "head" else "token_logit_bias"
    aliases = {
        "llama_medqa_running_margin_gate": "head_logit_bias",
        "first_token_logit_bias": "head_logit_bias",
        "residual_stream_token_bias": "token_logit_bias",
        "true_head_ov_hook": "head_ov_hook",
        "head_causal_hook": "head_ov_hook",
        "running_margin_head_hook": "head_ov_hook",
    }
    raw = aliases.get(raw, raw)
    allowed = {"head_logit_bias", "token_logit_bias", "residual_stream_hook", "head_ov_hook"}
    if raw not in allowed:
        raise ValueError(f"Unsupported intervention_impl: {requested}. Allowed: {sorted(allowed | {'auto'})}")
    return raw


def lambda_sweep_at_tau(
    *,
    prepared_records: list[Any],
    model,
    tokenizer,
    device: torch.device,
    letter_token_sets: dict[str, set[int]],
    bootstrap_space: str,
    tau_margin: float,
    lambda_neg_grid: list[float],
    matched_item_ids: set[str],
    clean_item_ids: set[str],
    seed: int,
    max_new_tokens: int,
    site_mode: str,
    target_layer: int,
    target_head: int,
    alpha: float = 1.0,
    intervention_impl: str = "auto",
    lambda_pos_grid: list[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_results: list[dict[str, Any]] | None = None
    best_row: dict[str, Any] | None = None
    resolved_impl = resolve_intervention_impl(site_mode=str(site_mode), requested=str(intervention_impl))
    lambda_pos_values = [0.0] if lambda_pos_grid is None else [float(value) for value in lambda_pos_grid]
    if not lambda_pos_values:
        lambda_pos_values = [0.0]
    for lambda_pos in lambda_pos_values:
        for lambda_neg in lambda_neg_grid:
            if resolved_impl == "head_logit_bias":
                summary, results = evaluate_intervention_cell(
                    prepared_records=prepared_records,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    letter_token_sets=letter_token_sets,
                    bootstrap_space=str(bootstrap_space),
                    intervention_space="E_letter",
                    lambda_pos=float(lambda_pos),
                    lambda_neg=float(lambda_neg),
                    tau_margin=float(tau_margin),
                    alpha=float(alpha),
                    target_layer=int(target_layer),
                    target_head=int(target_head),
                    matched_item_ids=matched_item_ids,
                    clean_item_ids=clean_item_ids,
                    seed=int(seed),
                    max_new_tokens=int(max_new_tokens),
                )
                row_impl = "head_logit_bias"
            elif resolved_impl == "token_logit_bias":
                summary, results = evaluate_cross_model_intervention(
                    prepared_records=prepared_records,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    letter_token_sets=letter_token_sets,
                    bootstrap_space=str(bootstrap_space),
                    lambda_pos=float(lambda_pos),
                    lambda_neg=float(lambda_neg),
                    tau_margin=float(tau_margin),
                    matched_item_ids=matched_item_ids,
                    clean_item_ids=clean_item_ids,
                    seed=int(seed),
                    max_new_tokens=int(max_new_tokens),
                    site_mode=str(site_mode),
                )
                row_impl = "token_logit_bias"
            elif resolved_impl == "head_ov_hook":
                summary, results = evaluate_head_ov_hook_intervention(
                    prepared_records=prepared_records,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    letter_token_sets=letter_token_sets,
                    bootstrap_space=str(bootstrap_space),
                    lambda_pos=float(lambda_pos),
                    lambda_neg=float(lambda_neg),
                    tau_margin=float(tau_margin),
                    alpha=float(alpha),
                    target_layer=int(target_layer),
                    target_head=int(target_head),
                    matched_item_ids=matched_item_ids,
                    clean_item_ids=clean_item_ids,
                    seed=int(seed),
                    max_new_tokens=int(max_new_tokens),
                )
                row_impl = "head_ov_hook"
            else:
                summary, results = evaluate_residual_stream_hook_intervention(
                    prepared_records=prepared_records,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    letter_token_sets=letter_token_sets,
                    bootstrap_space=str(bootstrap_space),
                    lambda_pos=float(lambda_pos),
                    lambda_neg=float(lambda_neg),
                    tau_margin=float(tau_margin),
                    target_layer=int(target_layer),
                    target_head=int(target_head),
                    matched_item_ids=matched_item_ids,
                    clean_item_ids=clean_item_ids,
                    seed=int(seed),
                    max_new_tokens=int(max_new_tokens),
                )
                row_impl = "residual_stream_hook"
            hook_rows = [
                row
                for row in results
                if bool(row.get("in_matched_subset", False)) and bool(row.get("residual_hook_applied", False))
            ]

            def _mean_result_field(key: str) -> float | None:
                values = [float(row[key]) for row in hook_rows if row.get(key) is not None]
                return float(sum(values) / len(values)) if values else None

            row = {
                "tau_margin": float(tau_margin),
                "lambda_pos": float(lambda_pos),
                "lambda_neg": float(lambda_neg),
                "alpha": float(alpha),
                "site_mode": str(site_mode),
                "target_layer": int(target_layer),
                "target_head": int(target_head),
                "intervention_impl": row_impl,
                "restoration_rate": float(summary["restoration_rate"]),
                "harm_rate": float(summary["harm_rate"]),
                "ft_accuracy": float(summary["ft_accuracy"]),
                "youden_j": float(summary["restoration_rate"] - summary["harm_rate"]),
                "n_base_wrong": int(summary["n_base_wrong"]),
                "n_restored": int(summary["n_restored"]),
                "n_harmed": int(summary["n_harmed"]),
                "n_residual_hook_applied": int(len(hook_rows)),
                "n_residual_delta_promotes_e": int(sum(bool(row.get("residual_delta_promotes_e", False)) for row in hook_rows)),
                "n_residual_delta_suppresses_p": int(sum(bool(row.get("residual_delta_suppresses_p", False)) for row in hook_rows)),
                "mean_residual_delta_e_mean_effect": _mean_result_field("residual_delta_e_mean_effect"),
                "mean_residual_delta_p_mean_effect": _mean_result_field("residual_delta_p_mean_effect"),
                "mean_residual_delta_mean_e_minus_p_effect": _mean_result_field("residual_delta_mean_e_minus_p_effect"),
            }
            rows.append(row)
            if best_row is None or (
                float(row["youden_j"]),
                float(row["ft_accuracy"]),
                -float(row["harm_rate"]),
                -abs(float(row["lambda_pos"]) - float(row["lambda_neg"])),
                -float(row["lambda_neg"]),
                -float(row["lambda_pos"]),
            ) > (
                float(best_row["youden_j"]),
                float(best_row["ft_accuracy"]),
                -float(best_row["harm_rate"]),
                -abs(float(best_row["lambda_pos"]) - float(best_row["lambda_neg"])),
                -float(best_row["lambda_neg"]),
                -float(best_row["lambda_pos"]),
            ):
                best_row = dict(row)
                best_summary = dict(summary)
                best_results = list(results)
    if best_row is None or best_summary is None or best_results is None:
        raise RuntimeError("Lambda sweep produced no results.")
    best_row["summary"] = best_summary
    return rows, best_row, best_results


def ci_overlap(left: list[float] | tuple[float, float], right: list[float] | tuple[float, float]) -> bool:
    l0, l1 = float(left[0]), float(left[1])
    r0, r1 = float(right[0]), float(right[1])
    return not (l1 < r0 or r1 < l0)


def model_summary_row(summary: dict[str, Any], *, model_name: str, architecture: str, intervention_site: str) -> dict[str, Any]:
    return {
        "model": str(model_name),
        "architecture": str(architecture),
        "intervention_site": str(intervention_site),
        "restoration_rate": float(summary["restoration_rate"]),
        "restoration_rate_ci95_wilson": list(summary["restoration_rate_ci95_wilson"]),
        "harm_rate": float(summary["harm_rate"]),
        "harm_rate_ci95_wilson": list(summary["harm_rate_ci95_wilson"]),
        "ft_accuracy": float(summary["ft_accuracy"]),
        "ft_accuracy_ci95_wilson": list(summary["ft_accuracy_ci95_wilson"]),
        "gate_fire_rate_base_wrong": float(summary["gate_fire_rate_base_wrong"]),
        "gate_fire_rate_base_correct": float(summary["gate_fire_rate_base_correct"]),
        "mcnemar_ft_vs_baseline": summary["mcnemar_ft_vs_baseline"],
    }


def risk_coverage_curve(scores: list[float], correct: list[bool]) -> list[dict[str, float]]:
    if len(scores) != len(correct):
        raise ValueError("scores and correct must have the same length.")
    pairs = sorted(
        [(float(score), bool(ok)) for score, ok in zip(scores, correct) if math.isfinite(float(score))],
        key=lambda item: item[0],
        reverse=True,
    )
    if not pairs:
        return []
    curve: list[dict[str, float]] = []
    kept_correct = 0
    kept_total = 0
    for idx, (_, ok) in enumerate(pairs, start=1):
        kept_total += 1
        if bool(ok):
            kept_correct += 1
        coverage = float(kept_total / len(pairs))
        accuracy = float(kept_correct / kept_total)
        risk = float(1.0 - accuracy)
        curve.append({"coverage": coverage, "risk": risk, "accuracy": accuracy, "threshold_rank": float(idx)})
    return curve


def auc_rank(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if math.isfinite(float(score))]
    n_pos = sum(1 for _, label in pairs if int(label) == 1)
    n_neg = sum(1 for _, label in pairs if int(label) == 0)
    if n_pos <= 0 or n_neg <= 0:
        return None
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][0] == ordered[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(ranks[idx] for idx, (_, label) in enumerate(pairs) if label == 1)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def aurc_from_curve(curve: list[dict[str, float]]) -> float:
    if not curve:
        return 1.0
    xs = [0.0] + [float(row["coverage"]) for row in curve]
    ys = [float(curve[0]["risk"])] + [float(row["risk"]) for row in curve]
    area = 0.0
    for left, right, y_left, y_right in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        area += 0.5 * (float(y_left) + float(y_right)) * float(right - left)
    return float(area)


def select_tau_by_youden(scores: list[float], labels_positive_when_low: list[bool]) -> dict[str, Any]:
    if len(scores) != len(labels_positive_when_low):
        raise ValueError("scores and labels_positive_when_low must have the same length.")
    pairs = [(float(score), bool(label)) for score, label in zip(scores, labels_positive_when_low) if math.isfinite(float(score))]
    if not pairs:
        return {"tau": 0.0, "youden_j": 0.0, "rows": []}
    unique = sorted({score for score, _ in pairs})
    thresholds = [unique[0] - 1e-6] + [0.5 * (unique[i] + unique[i + 1]) for i in range(len(unique) - 1)] + [unique[-1] + 1e-6]
    n_pos = sum(1 for _, label in pairs if label)
    n_neg = sum(1 for _, label in pairs if not label)
    rows: list[dict[str, float]] = []
    for tau in thresholds:
        tp = sum(1 for score, label in pairs if label and score < tau)
        fp = sum(1 for score, label in pairs if (not label) and score < tau)
        tpr = float(tp / max(n_pos, 1))
        fpr = float(fp / max(n_neg, 1))
        rows.append({"tau": float(tau), "tpr": tpr, "fpr": fpr, "youden_j": float(tpr - fpr)})
    best = sorted(rows, key=lambda row: (-float(row["youden_j"]), float(row["fpr"]), float(row["tau"])))[0]
    return {"tau": float(best["tau"]), "youden_j": float(best["youden_j"]), "rows": rows}


def write_markdown_section(path: str | Path, title: str, lines: list[str]) -> None:
    payload = [f"# {title}", ""]
    payload.extend(lines)
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(payload).rstrip() + "\n", encoding="utf-8")


def build_cross_model_metadata(
    *,
    args: Any,
    input_files: list[Path],
    seed: int,
    wall_clock_seconds: float,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "git_commit_hash": git_commit_hash(project_root),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path.resolve()) for path in input_files],
        "input_file_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in input_files
            if path.exists()
        },
    }


def paired_mcnemar_from_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], *, key: str = "ft_correct") -> dict[str, Any]:
    left = {str(row["item_id"]): bool(row.get(key, False)) for row in left_rows if not bool(row.get("skipped", False))}
    right = {str(row["item_id"]): bool(row.get(key, False)) for row in right_rows if not bool(row.get("skipped", False))}
    shared = sorted(set(left.keys()) & set(right.keys()))
    return mcnemar_from_paired([left[item_id] for item_id in shared], [right[item_id] for item_id in shared])


def rate_from_rows(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    n = int(len(rows))
    k = int(sum(1 for row in rows if bool(row.get(key, False))))
    return {
        "n": n,
        "k": k,
        "rate": float(k / max(n, 1)),
        "ci95_wilson": list(wilson_ci(k, n)),
    }
