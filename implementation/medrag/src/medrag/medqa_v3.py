from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from medrag.decomposed_gate import DecomposedMarginGate
from medrag.prompt_variants import build_medqa_prompt
from medrag.running_margin_gate import RunningMarginGate
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.stats import mcnemar_from_paired, wilson_ci

VALID_LETTERS = ("A", "B", "C", "D")
LETTER_REGEX = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


class FirstStepTokenBiasProcessor(LogitsProcessor):
    def __init__(self, *, p_token_ids: set[int], e_token_ids: set[int], neg_bias: float, pos_bias: float) -> None:
        self._p_ids = sorted(int(x) for x in p_token_ids)
        self._e_ids = sorted(int(x) for x in e_token_ids)
        self._neg_bias = float(neg_bias)
        self._pos_bias = float(pos_bias)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self._step == 0:
            if self._neg_bias > 0.0 and self._p_ids:
                p_idx = torch.as_tensor(self._p_ids, dtype=torch.long, device=scores.device)
                scores[:, p_idx] = scores[:, p_idx] - float(self._neg_bias)
            if self._pos_bias > 0.0 and self._e_ids:
                e_idx = torch.as_tensor(self._e_ids, dtype=torch.long, device=scores.device)
                scores[:, e_idx] = scores[:, e_idx] + float(self._pos_bias)
        self._step += 1
        return scores


@dataclass
class PreparedExample:
    item_id: str
    prompt_variant: str
    question: str
    context: str
    options: dict[str, str]
    correct: str
    parametric: str
    prompt: str
    prompt_hash: str
    prompt_ids: torch.Tensor
    cached_logits: torch.Tensor
    hidden_pre_norm: torch.Tensor
    base_ids: list[int]
    base_letter: str | None
    base_correct: bool
    ep_sets: dict[str, Any]
    representatives: dict[str, int | None]
    cache_hit: bool
    cache_path: str
    prefill_wall_clock_seconds: float
    generation_wall_clock_seconds: float


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_item_id(ex: dict[str, Any]) -> str:
    key = "|".join(
        [
            str(ex.get("question", "")).strip(),
            str(ex.get("correct_answer", "")).strip().upper(),
            str(ex.get("parametric_answer", "")).strip().upper(),
            str(ex.get("source", "medqa")).strip(),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_conflicts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def row_key(ex: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(ex.get("question", "")).strip(),
        str(ex.get("correct_answer", "")).strip().upper(),
        str(ex.get("parametric_answer", "")).strip().upper(),
    )


def build_letter_token_sets(tokenizer) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {k: set() for k in VALID_LETTERS}
    for letter in VALID_LETTERS:
        for variant in (letter, f" {letter}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tok in ids:
                out[letter].add(int(tok))
    return out


def choose_representative(logits: torch.Tensor, token_ids: set[int]) -> int | None:
    if not token_ids:
        return None
    vocab_size = int(logits.numel())
    valid = [int(tid) for tid in token_ids if 0 <= int(tid) < vocab_size]
    if not valid:
        return None
    return int(max(valid, key=lambda t: float(logits[int(t)].item())))


def decode_letter(output_ids: list[int], tokenizer, letter_token_sets: dict[str, set[int]]) -> str | None:
    rev: dict[int, str] = {}
    for letter, ids in letter_token_sets.items():
        for tid in ids:
            rev[int(tid)] = str(letter)
    for tid in output_ids:
        if int(tid) in rev:
            return rev[int(tid)]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    match = LETTER_REGEX.search(text)
    if match:
        return str(match.group(1)).upper()
    return None


def _coerce_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_token_id(item)
            if coerced is not None:
                return coerced
        return None
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        return _coerce_token_id(flat)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return _coerce_token_id(value.detach().cpu().reshape(-1).tolist())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_pad_token_id(model) -> int:
    generation_config = getattr(model, "generation_config", None)
    config = getattr(model, "config", None)
    for candidate in (
        getattr(generation_config, "pad_token_id", None),
        getattr(config, "pad_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(config, "eos_token_id", None),
    ):
        token_id = _coerce_token_id(candidate)
        if token_id is not None and token_id >= 0:
            return int(token_id)
    return 0


def generate_ids(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    logits_processors: list[LogitsProcessor] | None = None,
) -> list[int]:
    kwargs: dict[str, Any] = {}
    if logits_processors:
        kwargs["logits_processor"] = LogitsProcessorList(logits_processors)
    attention_mask = torch.ones_like(prompt_ids).unsqueeze(0)
    pad_token_id = _resolve_pad_token_id(model)
    with torch.inference_mode():
        out = model.generate(
            prompt_ids.unsqueeze(0),
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=pad_token_id,
            **kwargs,
        )
    return [int(x) for x in out[0, prompt_ids.shape[0] :].tolist()]


def _clean_token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return str(text).replace("\u2581", " ").replace("\u0120", " ").replace("\u010A", " ").strip()


def _keep_content_token(tokenizer, token_id: int) -> bool:
    all_special_ids = getattr(tokenizer, "all_special_ids", []) or []
    if int(token_id) in {int(x) for x in all_special_ids}:
        return False
    cleaned = _clean_token_text(tokenizer, int(token_id))
    if not cleaned:
        return False
    return any(ch.isalnum() for ch in cleaned)


def _tokenize_text_variants(tokenizer, text: str) -> set[int]:
    ids: set[int] = set()
    text = str(text or "").strip()
    if not text:
        return ids
    for variant in (text, f" {text}"):
        toks = tokenizer.encode(variant, add_special_tokens=False)
        ids |= {int(tok) for tok in toks if _keep_content_token(tokenizer, int(tok))}
    return ids


def build_content_ep_sets(tokenizer, options: dict[str, str], correct: str, letter_token_sets: dict[str, set[int]]) -> dict[str, Any]:
    option_content_token_sets: dict[str, set[int]] = {}
    for letter in VALID_LETTERS:
        option_text = str(options.get(letter, "")).strip()
        option_content_token_sets[letter] = _tokenize_text_variants(tokenizer, option_text)

    correct_letter = str(correct).upper()
    wrong_letters = [letter for letter in VALID_LETTERS if letter != correct_letter]
    e_content_raw = set(option_content_token_sets.get(correct_letter, set()))
    p_content_all_wrong_raw: set[int] = set()
    for letter in wrong_letters:
        p_content_all_wrong_raw |= set(option_content_token_sets.get(letter, set()))

    return {
        "correct_letter": correct_letter,
        "wrong_letters": wrong_letters,
        "option_content_token_sets": option_content_token_sets,
        "E_content_raw": e_content_raw,
        "P_content_all_wrong_raw": p_content_all_wrong_raw,
        "E_letter": set(int(x) for x in letter_token_sets.get(correct_letter, set())),
        "P_letter_all_wrong": set(int(x) for letter in wrong_letters for x in letter_token_sets.get(letter, set())),
    }


def materialize_ep_sets(
    ep_base: dict[str, Any],
    *,
    parametric_answer: str,
    p_set_mode: str,
    letter_token_sets: dict[str, set[int]],
) -> dict[str, Any]:
    correct_letter = str(ep_base["correct_letter"])
    wrong_letters = [str(x) for x in ep_base["wrong_letters"]]
    parametric_letter = str(parametric_answer).upper()
    if str(p_set_mode) == "param_only" and parametric_letter in wrong_letters:
        p_letters = [parametric_letter]
    else:
        p_letters = list(wrong_letters)

    option_content_token_sets = ep_base["option_content_token_sets"]
    e_content_raw = set(ep_base["E_content_raw"])
    p_content_raw: set[int] = set()
    for letter in p_letters:
        p_content_raw |= set(option_content_token_sets.get(letter, set()))
    content_overlap = set(e_content_raw) & set(p_content_raw)
    e_content = set(e_content_raw) - content_overlap
    p_content = set(p_content_raw) - content_overlap

    e_letter_raw = set(ep_base["E_letter"])
    p_letter_raw: set[int] = set()
    for letter in p_letters:
        p_letter_raw |= {int(x) for x in letter_token_sets.get(letter, set())}
    letter_overlap = set(e_letter_raw) & set(p_letter_raw)
    e_letter = set(e_letter_raw) - letter_overlap
    p_letter = set(p_letter_raw) - letter_overlap

    return {
        "p_letters": list(p_letters),
        "E_content": e_content,
        "P_content": p_content,
        "E_letter": e_letter,
        "P_letter": p_letter,
        "n_E_content": int(len(e_content)),
        "n_P_content": int(len(p_content)),
        "n_E_letter": int(len(e_letter)),
        "n_P_letter": int(len(p_letter)),
        "overlap_removed_count": int(len(content_overlap) + len(letter_overlap)),
    }


def prepare_examples_for_variant(
    *,
    conflicts: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    cache: ActivationCache,
    model_id: str,
    prompt_variant: str,
    p_set_mode: str,
    target_layer: int,
    max_length: int,
    max_new_tokens: int,
    letter_token_sets: dict[str, set[int]],
) -> list[PreparedExample]:
    prepared: list[PreparedExample] = []
    for ex in conflicts:
        question = str(ex.get("question", "")).strip()
        options = ex.get("options", {}) or {}
        context = str(ex.get("context", "")).strip()
        correct = str(ex.get("correct_answer", "")).strip().upper()
        param = str(ex.get("parametric_answer", "")).strip().upper()
        if not question or not isinstance(options, dict) or not context:
            continue
        if correct not in VALID_LETTERS or param not in VALID_LETTERS or param == correct:
            continue

        item_id = stable_item_id(ex)
        prompt = build_medqa_prompt(
            question=question,
            options=options,
            context=context,
            prompt_variant=prompt_variant,
            tokenizer=tokenizer,
            model_id=model_id,
        )
        cache_entry = cache.get_or_compute(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_id=model_id,
            item_id=item_id,
            prompt=prompt,
            target_layer=int(target_layer),
            max_length=int(max_length),
        )
        prompt_ids_cpu = cache_entry["prompt_ids"].detach().cpu()
        prompt_ids = prompt_ids_cpu.to(device)
        cached_logits = cache_entry["cached_logits"].detach().cpu()
        hidden_pre_norm = cache_entry["hidden_pre_norm"].detach().cpu()

        t0 = time.perf_counter()
        base_ids = generate_ids(model, prompt_ids, int(max_new_tokens))
        gen_elapsed = float(time.perf_counter() - t0)
        base_letter = decode_letter(base_ids, tokenizer, letter_token_sets)
        base_correct = bool(base_letter == correct)

        ep_base = build_content_ep_sets(tokenizer, options, correct, letter_token_sets)
        ep_sets = materialize_ep_sets(ep_base, parametric_answer=param, p_set_mode=p_set_mode, letter_token_sets=letter_token_sets)
        representatives = {
            "content_ev_id": choose_representative(cached_logits, set(ep_sets["E_content"])),
            "content_par_id": choose_representative(cached_logits, set(ep_sets["P_content"])),
            "letter_ev_id": choose_representative(cached_logits, set(ep_sets["E_letter"])),
            "letter_par_id": choose_representative(cached_logits, set(ep_sets["P_letter"])),
        }
        prepared.append(
            PreparedExample(
                item_id=item_id,
                prompt_variant=str(prompt_variant),
                question=question,
                context=context,
                options={str(k): str(v) for k, v in options.items()},
                correct=correct,
                parametric=param,
                prompt=prompt,
                prompt_hash=str(cache_entry["prompt_hash"]),
                prompt_ids=prompt_ids_cpu,
                cached_logits=cached_logits,
                hidden_pre_norm=hidden_pre_norm,
                base_ids=base_ids,
                base_letter=base_letter,
                base_correct=bool(base_correct),
                ep_sets=ep_sets,
                representatives=representatives,
                cache_hit=bool(cache_entry.get("cache_hit", False)),
                cache_path=str(cache_entry.get("cache_path", "")),
                prefill_wall_clock_seconds=float(cache_entry.get("wall_clock_seconds", 0.0)),
                generation_wall_clock_seconds=float(gen_elapsed),
            )
        )
    return prepared

def diagnostics_for_matched_subset(
    *,
    vanilla_records: list[PreparedExample],
    trust_records: list[PreparedExample],
    authoritative_records: list[PreparedExample],
) -> list[dict[str, Any]]:
    by_variant = {
        "vanilla": {rec.item_id: rec for rec in vanilla_records},
        "trust_instruction": {rec.item_id: rec for rec in trust_records},
        "authoritative_quote": {rec.item_id: rec for rec in authoritative_records},
    }
    item_ids = sorted(set(by_variant["vanilla"].keys()) | set(by_variant["trust_instruction"].keys()) | set(by_variant["authoritative_quote"].keys()))
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        reasons: list[str] = []
        vanilla = by_variant["vanilla"].get(item_id)
        trust = by_variant["trust_instruction"].get(item_id)
        quote = by_variant["authoritative_quote"].get(item_id)
        if vanilla is None:
            reasons.append("missing_vanilla")
        if trust is None:
            reasons.append("missing_trust_instruction")
        if quote is None:
            reasons.append("missing_authoritative_quote")
        for name, rec in (("vanilla", vanilla), ("trust_instruction", trust)):
            if rec is None:
                continue
            if rec.base_letter not in VALID_LETTERS:
                reasons.append(f"{name}_invalid_base_letter")
            if not rec.ep_sets["E_content"] or not rec.ep_sets["P_content"]:
                reasons.append(f"{name}_empty_content_sets")
            if not rec.ep_sets["E_letter"] or not rec.ep_sets["P_letter"]:
                reasons.append(f"{name}_empty_letter_sets")
            if rec.representatives["content_ev_id"] is None or rec.representatives["content_par_id"] is None:
                reasons.append(f"{name}_missing_content_representatives")
            if rec.representatives["letter_ev_id"] is None or rec.representatives["letter_par_id"] is None:
                reasons.append(f"{name}_missing_letter_representatives")
        if quote is not None and quote.base_letter not in VALID_LETTERS:
            reasons.append("authoritative_quote_invalid_base_letter")
        rows.append({"item_id": item_id, "exclude_reasons": sorted(set(reasons))})
    return rows


def evaluate_intervention_cell(
    *,
    prepared_records: list[PreparedExample],
    model,
    tokenizer,
    device: torch.device,
    letter_token_sets: dict[str, set[int]],
    bootstrap_space: str,
    intervention_space: str,
    lambda_pos: float,
    lambda_neg: float,
    tau_margin: float,
    alpha: float,
    target_layer: int,
    target_head: int,
    matched_item_ids: set[str],
    clean_item_ids: set[str],
    seed: int,
    max_new_tokens: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for rec in prepared_records:
        if str(bootstrap_space) == "E_content":
            e_boot = set(rec.ep_sets["E_content"])
            p_boot = set(rec.ep_sets["P_content"])
            ev_id = rec.representatives["content_ev_id"]
            par_id = rec.representatives["content_par_id"]
        elif str(bootstrap_space) == "E_letter":
            e_boot = set(rec.ep_sets["E_letter"])
            p_boot = set(rec.ep_sets["P_letter"])
            ev_id = rec.representatives["letter_ev_id"]
            par_id = rec.representatives["letter_par_id"]
        else:
            raise ValueError(f"Unsupported bootstrap_space: {bootstrap_space}")

        if str(intervention_space) != "E_letter":
            raise ValueError(f"Unsupported intervention_space: {intervention_space}")
        e_int = set(rec.ep_sets["E_letter"])
        p_int = set(rec.ep_sets["P_letter"])

        if not e_boot or not p_boot or not e_int or not p_int or ev_id is None or par_id is None:
            row = {
                "item_id": rec.item_id,
                "question": rec.question,
                "correct": rec.correct,
                "parametric": rec.parametric,
                "base_letter": rec.base_letter,
                "ft_letter": rec.base_letter,
                "base_correct": bool(rec.base_correct),
                "ft_correct": bool(rec.base_correct),
                "ft_restored": False,
                "ft_harmed": False,
                "conflict_detected": False,
                "skipped": True,
                "skip_reason": "missing_sets_or_representatives",
                "in_matched_subset": bool(rec.item_id in matched_item_ids),
                "in_clean_subset": bool(rec.item_id in clean_item_ids),
            }
            results.append(row)
            continue

        gate = DecomposedMarginGate(
            model=model,
            tokenizer=tokenizer,
            evidence_set=e_boot,
            parametric_set=p_boot,
            target_layer=int(target_layer),
            target_head=int(target_head),
            tau_margin=float(tau_margin),
            alpha=float(alpha),
            lambda_max=max(float(lambda_pos), float(lambda_neg), 1.0),
        )
        gate.setup_example(int(ev_id), int(par_id), device=device)
        decision = gate.decide(
            hidden_pre_norm=rec.hidden_pre_norm.to(device),
            lambda_pos=float(lambda_pos),
            lambda_neg=float(lambda_neg),
        )
        logits_processors: list[LogitsProcessor] = []
        if (decision.lambda_pos_applied > 0.0 and e_int) or (decision.lambda_neg_applied > 0.0 and p_int):
            logits_processors.append(
                FirstStepTokenBiasProcessor(
                    e_token_ids=(set(e_int) if decision.lambda_pos_applied > 0.0 else set()),
                    p_token_ids=(set(p_int) if decision.lambda_neg_applied > 0.0 else set()),
                    pos_bias=float(decision.lambda_pos_applied),
                    neg_bias=float(decision.lambda_neg_applied),
                )
            )
        ft_ids = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens), logits_processors=logits_processors)
        ft_letter = decode_letter(ft_ids, tokenizer, letter_token_sets)
        ft_correct = bool(ft_letter == rec.correct)
        ft_restored = bool((not rec.base_correct) and ft_correct)
        ft_harmed = bool(rec.base_correct and (not ft_correct))
        results.append(
            {
                "item_id": rec.item_id,
                "question": rec.question,
                "correct": rec.correct,
                "parametric": rec.parametric,
                "base_letter": rec.base_letter,
                "ft_letter": ft_letter,
                "base_correct": bool(rec.base_correct),
                "ft_correct": bool(ft_correct),
                "ft_restored": bool(ft_restored),
                "ft_harmed": bool(ft_harmed),
                "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                "conflict_detected": bool(decision.conflict_detected),
                "prefill_margin": float(decision.prefill_margin),
                "gate_lambda": float(decision.gate_lambda),
                "lambda_pos_applied": float(decision.lambda_pos_applied),
                "lambda_neg_applied": float(decision.lambda_neg_applied),
                "lambda_clipped": bool(decision.lambda_clipped),
                "n_E_content": int(rec.ep_sets["n_E_content"]),
                "n_P_content": int(rec.ep_sets["n_P_content"]),
                "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                "bootstrap_space": str(bootstrap_space),
                "intervention_space": str(intervention_space),
                "in_matched_subset": bool(rec.item_id in matched_item_ids),
                "in_clean_subset": bool(rec.item_id in clean_item_ids),
                "skipped": False,
                "seed": int(seed),
            }
        )
    elapsed = float(time.perf_counter() - t0)
    summary = summarize_intervention_results(results=results, wall_clock_seconds=elapsed)
    return summary, results


def evaluate_head_ov_hook_intervention(
    *,
    prepared_records: list[PreparedExample],
    model,
    tokenizer,
    device: torch.device,
    letter_token_sets: dict[str, set[int]],
    bootstrap_space: str,
    lambda_pos: float,
    lambda_neg: float,
    tau_margin: float,
    alpha: float,
    target_layer: int,
    target_head: int,
    matched_item_ids: set[str],
    clean_item_ids: set[str],
    seed: int,
    max_new_tokens: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Causal head-slice intervention through the target attention output path.

    Unlike FirstStepTokenBiasProcessor, this modifies only the selected head slice
    before the layer's output projection. The grid values are interpreted as a
    lambda cap; the OV direction itself is derived from the E/P representative
    token unembedding difference projected into the target head subspace.
    """
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    lambda_cap = float(max(float(lambda_pos), float(lambda_neg), 1.0))
    for rec in prepared_records:
        if str(bootstrap_space) == "E_content":
            e_boot = set(rec.ep_sets["E_content"])
            p_boot = set(rec.ep_sets["P_content"])
            ev_id = rec.representatives["content_ev_id"]
            par_id = rec.representatives["content_par_id"]
        elif str(bootstrap_space) == "E_letter":
            e_boot = set(rec.ep_sets["E_letter"])
            p_boot = set(rec.ep_sets["P_letter"])
            ev_id = rec.representatives["letter_ev_id"]
            par_id = rec.representatives["letter_par_id"]
        else:
            raise ValueError(f"Unsupported bootstrap_space: {bootstrap_space}")

        if not e_boot or not p_boot or ev_id is None or par_id is None:
            results.append(
                {
                    "item_id": rec.item_id,
                    "question": rec.question,
                    "correct": rec.correct,
                    "parametric": rec.parametric,
                    "base_letter": rec.base_letter,
                    "ft_letter": rec.base_letter,
                    "base_correct": bool(rec.base_correct),
                    "ft_correct": bool(rec.base_correct),
                    "ft_restored": False,
                    "ft_harmed": False,
                    "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                    "conflict_detected": False,
                    "prefill_margin": 0.0,
                    "gate_lambda": 0.0,
                    "lambda_pos_applied": 0.0,
                    "lambda_neg_applied": 0.0,
                    "lambda_clipped": False,
                    "n_E_content": int(rec.ep_sets["n_E_content"]),
                    "n_P_content": int(rec.ep_sets["n_P_content"]),
                    "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                    "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                    "bootstrap_space": str(bootstrap_space),
                    "intervention_space": "target_head_ov",
                    "intervention_impl": "head_ov_hook",
                    "target_layer": int(target_layer),
                    "target_head": int(target_head),
                    "in_matched_subset": bool(rec.item_id in matched_item_ids),
                    "in_clean_subset": bool(rec.item_id in clean_item_ids),
                    "skipped": True,
                    "skip_reason": "missing_sets_or_representatives",
                    "seed": int(seed),
                }
            )
            continue

        gate = RunningMarginGate(
            model=model,
            tokenizer=tokenizer,
            evidence_set=e_boot,
            parametric_set=p_boot,
            target_layer=int(target_layer),
            target_head=int(target_head),
            tau_margin=float(tau_margin),
            alpha=float(alpha),
            lambda_max=float(lambda_cap),
            first_token_only=True,
        )
        gate.setup_example(int(ev_id), int(par_id), device=device)
        hidden = rec.hidden_pre_norm.to(device)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0).unsqueeze(0)
        elif hidden.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected hidden_pre_norm shape: {tuple(hidden.shape)}")
        prefill_margin = float(gate._compute_margin(hidden))
        gate._current_margin = float(prefill_margin)
        gate._margin_at_evidence_pos = float(prefill_margin)
        gate._lambda, gate._lambda_clipped = gate._compute_prefill_lambda(float(prefill_margin))
        gate._evidence_position = -1
        gate._position_found = False
        conflict_detected = bool(np.isfinite(prefill_margin) and prefill_margin < float(tau_margin))

        gate.attach(device=device)
        try:
            ft_ids = generate_ids(model, rec.prompt_ids.to(device), int(max_new_tokens), logits_processors=None)
        finally:
            gate.detach()

        gate_summary = gate.summary()
        head_lambda = float(gate_summary.get("mean_lambda", 0.0))
        if head_lambda <= 0.0 and conflict_detected:
            head_lambda = float(gate._lambda)
        n_steps_steered = int(gate_summary.get("n_steps_steered", 0))
        ft_letter = decode_letter(ft_ids, tokenizer, letter_token_sets)
        ft_correct = bool(ft_letter == rec.correct)
        ft_restored = bool((not rec.base_correct) and ft_correct)
        ft_harmed = bool(rec.base_correct and (not ft_correct))
        results.append(
            {
                "item_id": rec.item_id,
                "question": rec.question,
                "correct": rec.correct,
                "parametric": rec.parametric,
                "base_letter": rec.base_letter,
                "ft_letter": ft_letter,
                "base_correct": bool(rec.base_correct),
                "ft_correct": bool(ft_correct),
                "ft_restored": bool(ft_restored),
                "ft_harmed": bool(ft_harmed),
                "base_matches_parametric": bool(rec.base_letter == rec.parametric),
                "conflict_detected": bool(conflict_detected and n_steps_steered > 0),
                "prefill_margin": float(prefill_margin),
                "gate_lambda": float(head_lambda),
                "lambda_pos_applied": float(head_lambda),
                "lambda_neg_applied": float(head_lambda),
                "lambda_clipped": bool(float(gate_summary.get("clip_fraction", 0.0)) > 0.0),
                "lambda_cap": float(lambda_cap),
                "lambda_pos_config": float(lambda_pos),
                "lambda_neg_config": float(lambda_neg),
                "n_E_content": int(rec.ep_sets["n_E_content"]),
                "n_P_content": int(rec.ep_sets["n_P_content"]),
                "n_E_letter": int(rec.ep_sets["n_E_letter"]),
                "n_P_letter": int(rec.ep_sets["n_P_letter"]),
                "bootstrap_space": str(bootstrap_space),
                "intervention_space": "target_head_ov",
                "intervention_impl": "head_ov_hook",
                "target_layer": int(target_layer),
                "target_head": int(target_head),
                "head_hook_n_steps_total": int(gate_summary.get("n_steps_total", 0)),
                "head_hook_n_steps_steered": int(n_steps_steered),
                "head_hook_frac_steered": float(gate_summary.get("frac_steered", 0.0)),
                "head_hook_mean_lambda": float(gate_summary.get("mean_lambda", 0.0)),
                "head_hook_prefill_margin": float(gate_summary.get("prefill_margin", prefill_margin)),
                "head_hook_margin_errors": int(gate_summary.get("margin_errors", 0)),
                "in_matched_subset": bool(rec.item_id in matched_item_ids),
                "in_clean_subset": bool(rec.item_id in clean_item_ids),
                "skipped": False,
                "seed": int(seed),
            }
        )
    elapsed = float(time.perf_counter() - t0)
    summary = summarize_intervention_results(results=results, wall_clock_seconds=elapsed)
    return summary, results


def summarize_intervention_results(results: list[dict[str, Any]], wall_clock_seconds: float) -> dict[str, Any]:
    eval_rows = [row for row in results if bool(row.get("in_matched_subset", False)) and not bool(row.get("skipped", False))]
    clean_rows = [row for row in eval_rows if bool(row.get("in_clean_subset", False))]
    if not eval_rows:
        raise RuntimeError("No evaluable rows on matched subset.")

    def _rate(rows: list[dict[str, Any]], key: str) -> tuple[float, tuple[float, float], int]:
        if not rows:
            return 0.0, (0.0, 0.0), 0
        k = int(sum(1 for row in rows if bool(row.get(key, False))))
        n = int(len(rows))
        return float(k / max(n, 1)), wilson_ci(k, n), k

    base_wrong = [row for row in eval_rows if not bool(row["base_correct"])]
    base_correct = [row for row in eval_rows if bool(row["base_correct"])]
    clean_base_wrong = [row for row in clean_rows if not bool(row["base_correct"])]
    clean_base_correct = [row for row in clean_rows if bool(row["base_correct"])]

    baseline_accuracy, baseline_ci, _ = _rate(eval_rows, "base_correct")
    ft_accuracy, ft_ci, _ = _rate(eval_rows, "ft_correct")
    restoration_rate, restoration_ci, n_restored = _rate(base_wrong, "ft_restored")
    harm_rate, harm_ci, n_harmed = _rate(base_correct, "ft_harmed")
    gate_wrong_rate, gate_wrong_ci, _ = _rate(base_wrong, "conflict_detected")
    gate_correct_rate, gate_correct_ci, _ = _rate(base_correct, "conflict_detected")

    clean_restoration_rate, clean_restoration_ci, n_restored_clean = _rate(clean_base_wrong, "ft_restored")
    clean_harm_rate, clean_harm_ci, n_harmed_clean = _rate(clean_base_correct, "ft_harmed")
    clean_accuracy, clean_accuracy_ci, _ = _rate(clean_rows, "ft_correct")

    mcnemar = mcnemar_from_paired(
        baseline_correct=[bool(row["base_correct"]) for row in eval_rows],
        current_correct=[bool(row["ft_correct"]) for row in eval_rows],
    )

    wrong_match_param = [row for row in base_wrong if bool(row.get("base_matches_parametric", False))]
    wrong_other = [row for row in base_wrong if not bool(row.get("base_matches_parametric", False))]
    restoration_param, restoration_param_ci, _ = _rate(wrong_match_param, "ft_restored")
    restoration_other, restoration_other_ci, _ = _rate(wrong_other, "ft_restored")

    summary = {
        "n_examples_raw": int(len(results)),
        "n_examples_matched": int(len(eval_rows)),
        "n_examples_clean": int(len(clean_rows)),
        "n_base_wrong": int(len(base_wrong)),
        "n_base_correct": int(len(base_correct)),
        "baseline_accuracy": float(baseline_accuracy),
        "baseline_accuracy_ci95_wilson": [float(baseline_ci[0]), float(baseline_ci[1])],
        "ft_accuracy": float(ft_accuracy),
        "ft_accuracy_ci95_wilson": [float(ft_ci[0]), float(ft_ci[1])],
        "ft_accuracy_clean_subset": float(clean_accuracy),
        "ft_accuracy_clean_subset_ci95_wilson": [float(clean_accuracy_ci[0]), float(clean_accuracy_ci[1])],
        "restoration_rate": float(restoration_rate),
        "restoration_rate_ci95_wilson": [float(restoration_ci[0]), float(restoration_ci[1])],
        "harm_rate": float(harm_rate),
        "harm_rate_ci95_wilson": [float(harm_ci[0]), float(harm_ci[1])],
        "restoration_rate_clean_subset": float(clean_restoration_rate),
        "restoration_rate_clean_subset_ci95_wilson": [float(clean_restoration_ci[0]), float(clean_restoration_ci[1])],
        "harm_rate_clean_subset": float(clean_harm_rate),
        "harm_rate_clean_subset_ci95_wilson": [float(clean_harm_ci[0]), float(clean_harm_ci[1])],
        "restoration_improvement": float(restoration_rate - harm_rate),
        "gate_fire_rate_base_wrong": float(gate_wrong_rate),
        "gate_fire_rate_base_wrong_ci95_wilson": [float(gate_wrong_ci[0]), float(gate_wrong_ci[1])],
        "gate_fire_rate_base_correct": float(gate_correct_rate),
        "gate_fire_rate_base_correct_ci95_wilson": [float(gate_correct_ci[0]), float(gate_correct_ci[1])],
        "fraction_negative_margin_base_wrong": float(sum(1 for row in base_wrong if float(row.get("prefill_margin", 0.0)) < 0.0) / max(len(base_wrong), 1)),
        "mean_prefill_margin_base_wrong": float(np.mean([float(row.get("prefill_margin", 0.0)) for row in base_wrong])) if base_wrong else 0.0,
        "mean_prefill_margin_base_correct": float(np.mean([float(row.get("prefill_margin", 0.0)) for row in base_correct])) if base_correct else 0.0,
        "n_base_wrong_matches_param": int(len(wrong_match_param)),
        "restoration_rate_param_wrong": float(restoration_param),
        "restoration_rate_param_wrong_ci95_wilson": [float(restoration_param_ci[0]), float(restoration_param_ci[1])],
        "restoration_rate_other_wrong": float(restoration_other),
        "restoration_rate_other_wrong_ci95_wilson": [float(restoration_other_ci[0]), float(restoration_other_ci[1])],
        "mean_lambda_pos_when_conflict": float(np.mean([float(row.get("lambda_pos_applied", 0.0)) for row in eval_rows if bool(row.get("conflict_detected", False))])) if any(bool(row.get("conflict_detected", False)) for row in eval_rows) else 0.0,
        "mean_lambda_neg_when_conflict": float(np.mean([float(row.get("lambda_neg_applied", 0.0)) for row in eval_rows if bool(row.get("conflict_detected", False))])) if any(bool(row.get("conflict_detected", False)) for row in eval_rows) else 0.0,
        "mcnemar_ft_vs_baseline": mcnemar,
        "n_restored": int(n_restored),
        "n_harmed": int(n_harmed),
        "n_restored_clean": int(n_restored_clean),
        "n_harmed_clean": int(n_harmed_clean),
        "wall_clock_seconds": float(wall_clock_seconds),
        "wall_clock_seconds_per_item": float(wall_clock_seconds / max(len(eval_rows), 1)),
    }
    return summary


def summarize_condition_accuracy(
    *,
    rows: list[dict[str, Any]],
    correct_key: str,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    eval_rows = [row for row in rows if bool(row.get("in_matched_subset", False)) and not bool(row.get("skipped", False))]
    clean_rows = [row for row in eval_rows if bool(row.get("in_clean_subset", False))]
    k = int(sum(1 for row in eval_rows if bool(row.get(correct_key, False))))
    n = int(len(eval_rows))
    kc = int(sum(1 for row in clean_rows if bool(row.get(correct_key, False))))
    nc = int(len(clean_rows))
    ci = wilson_ci(k, n)
    ci_clean = wilson_ci(kc, nc)
    return {
        "n_examples_matched": int(n),
        "n_examples_clean": int(nc),
        "accuracy": float(k / max(n, 1)),
        "accuracy_ci95_wilson": [float(ci[0]), float(ci[1])],
        "accuracy_clean_subset": float(kc / max(nc, 1)),
        "accuracy_clean_subset_ci95_wilson": [float(ci_clean[0]), float(ci_clean[1])],
        "wall_clock_seconds": float(wall_clock_seconds),
        "wall_clock_seconds_per_item": float(wall_clock_seconds / max(n, 1)),
    }
