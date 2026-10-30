from __future__ import annotations

import re
from typing import Optional

import torch

DOSAGE_PATTERN = re.compile(
    r"\b\d+\.?\d*\s*(?:mg|mcg|ug|ml|g|kg|units?|iu|mmol|meq|%)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+\.?\d*")


def extract_clinical_entities_regex(text: str) -> set[str]:
    values = set()
    if not text:
        return values
    for match in DOSAGE_PATTERN.findall(str(text).lower()):
        match = str(match).strip()
        if match:
            values.add(match)
    return values


def _select_representative(token_ids: set[int], cached_logits: torch.Tensor) -> Optional[int]:
    if not token_ids:
        return None
    logits = cached_logits.float()
    return int(max((int(t) for t in token_ids), key=lambda t: float(logits[int(t)].item())))


def _extract_numbers(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in NUMBER_PATTERN.findall(str(text)) if str(x).strip()]


def build_ep_from_explicit(
    tokenizer,
    context_answer: str,
    parametric_answer: str,
    cached_logits: torch.Tensor,
    k: int = 50,
) -> tuple[set[int], set[int], Optional[int], Optional[int]]:
    # Dosage tasks are driven primarily by number tokens (e.g., "10" vs "30"),
    # while unit tokens ("mg") are typically non-discriminative.
    context_numbers = _extract_numbers(context_answer)
    parametric_numbers = _extract_numbers(parametric_answer)
    if context_numbers and parametric_numbers:
        e_raw = set(int(x) for x in tokenizer.encode(context_numbers[0], add_special_tokens=False))
        p_raw = set(int(x) for x in tokenizer.encode(parametric_numbers[0], add_special_tokens=False))
    else:
        e_raw = set(int(x) for x in tokenizer.encode(context_answer.strip(), add_special_tokens=False))
        p_raw = set(int(x) for x in tokenizer.encode(parametric_answer.strip(), add_special_tokens=False))

    shared = e_raw & p_raw
    e = e_raw - shared
    p = p_raw - shared
    if not e:
        e = e_raw
    if not p:
        p = p_raw

    # For explicit-answer examples (e.g., ClashEval dosage), do NOT intersect with
    # model top-k logits. We know the target strings explicitly; filtering by top-k
    # can erase the correct E/P token sets and collapse steering.

    ev_id = _select_representative(e, cached_logits)
    par_id = _select_representative(p, cached_logits)
    return e, p, ev_id, par_id


def build_ep_from_text(
    tokenizer,
    guideline_text: str,
    parametric_text: str,
    cached_logits: torch.Tensor,
    k: int = 50,
) -> tuple[set[int], set[int], Optional[int], Optional[int]]:
    guid_dosages = extract_clinical_entities_regex(guideline_text)
    param_dosages = extract_clinical_entities_regex(parametric_text)

    discriminative_guid = guid_dosages - param_dosages
    discriminative_param = param_dosages - guid_dosages

    e_raw: set[int] = set()
    for dosage in discriminative_guid:
        e_raw.update(int(x) for x in tokenizer.encode(dosage, add_special_tokens=False))

    p_raw: set[int] = set()
    for dosage in discriminative_param:
        p_raw.update(int(x) for x in tokenizer.encode(dosage, add_special_tokens=False))

    top_k_ids = set(int(x) for x in cached_logits.float().topk(int(k)).indices.tolist())
    guid_all_ids = set(int(x) for x in tokenizer.encode(guideline_text, add_special_tokens=False))

    if not e_raw:
        e_raw = top_k_ids & guid_all_ids
    if not p_raw:
        p_raw = top_k_ids - guid_all_ids

    if e_raw & p_raw:
        e = e_raw - p_raw
        p = p_raw - e_raw
    else:
        e, p = e_raw, p_raw

    ev_id = _select_representative(e, cached_logits)
    par_id = _select_representative(p, cached_logits)
    return e, p, ev_id, par_id


def build_ep(
    tokenizer,
    example: dict,
    cached_logits: torch.Tensor,
    k: int = 50,
) -> tuple[set[int], set[int], Optional[int], Optional[int]]:
    if example.get("context_answer") and example.get("parametric_answer"):
        return build_ep_from_explicit(
            tokenizer=tokenizer,
            context_answer=str(example["context_answer"]),
            parametric_answer=str(example["parametric_answer"]),
            cached_logits=cached_logits,
            k=int(k),
        )
    return build_ep_from_text(
        tokenizer=tokenizer,
        guideline_text=str(example.get("guideline_text", example.get("context", ""))),
        parametric_text=str(example.get("parametric_answer", "")),
        cached_logits=cached_logits,
        k=int(k),
    )
