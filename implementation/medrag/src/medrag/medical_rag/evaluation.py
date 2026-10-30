from __future__ import annotations

import re
from typing import Optional

import numpy as np

DOSAGE_PATTERN = re.compile(
    r"\b\d+\.?\d*\s*(?:mg|mcg|ug|ml|g|kg|units?|iu|mmol|meq|%)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+\.?\d*")

CONTRAINDICATION_KEYWORDS = {
    "avoid",
    "contraindicated",
    "contraindication",
    "do not",
    "don't",
    "never",
    "caution",
    "warning",
    "not recommended",
    "stop",
    "discontinue",
    "prohibited",
}


def compute_gar(
    generated_text: str,
    example: dict,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    gen_lower = str(generated_text).lower().strip()

    context_answer = str(example.get("context_answer", "")).strip()
    if context_answer:
        context_numbers = set(x.strip() for x in NUMBER_PATTERN.findall(context_answer.lower()) if x.strip())
        generated_numbers = set(x.strip() for x in NUMBER_PATTERN.findall(gen_lower) if x.strip())
        if context_numbers:
            overlap = len(context_numbers & generated_numbers) / max(len(context_numbers), 1)
            return bool(overlap >= 1.0), float(overlap)
        found = context_answer.lower() in gen_lower or context_answer.lower().replace(" ", "") in gen_lower.replace(" ", "")
        return bool(found), 1.0 if found else 0.0

    guideline_text = str(example.get("guideline_text", example.get("context", "")))
    guid_dosages = set(DOSAGE_PATTERN.findall(guideline_text.lower()))
    gen_dosages = set(DOSAGE_PATTERN.findall(gen_lower))

    if not guid_dosages:
        guid_words = set(guideline_text.lower().split())
        gen_words = set(gen_lower.split())
        overlap = len(guid_words & gen_words) / max(len(guid_words), 1)
        return bool(overlap >= float(threshold)), float(overlap)

    overlap = len(gen_dosages & guid_dosages) / max(len(guid_dosages), 1)
    return bool(overlap >= float(threshold)), float(overlap)


def compute_csr(
    generated_text: str,
    example: dict,
) -> Optional[bool]:
    context_lower = str(example.get("context", example.get("guideline_text", ""))).lower()
    gen_lower = str(generated_text).lower()

    has_contra = any(kw in context_lower for kw in CONTRAINDICATION_KEYWORDS)
    if not has_contra:
        return None
    return bool(any(kw in gen_lower for kw in CONTRAINDICATION_KEYWORDS))


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n_examples": 0, "error": "no results"}

    def _csr_rate(key: str) -> Optional[float]:
        applicable = [r for r in results if r.get(key) is not None]
        if not applicable:
            return None
        return float(sum(1 for r in applicable if bool(r[key])) / len(applicable))

    return {
        "n_examples": int(n),
        "baseline_gar": float(sum(1 for r in results if bool(r["baseline_adherent"])) / n),
        "rg_gar": float(sum(1 for r in results if bool(r["rg_adherent"])) / n),
        "rg_harm_rate": float(sum(1 for r in results if bool(r["rg_harmed"])) / n),
        "baseline_csr": _csr_rate("base_csr"),
        "rg_csr": _csr_rate("rg_csr"),
        "rg_mean_steps": float(np.mean([float(r["rg_n_steered"]) for r in results])),
        "rg_mean_frac": float(np.mean([float(r["rg_frac_steered"]) for r in results])),
        "note_ft": "first-token-only removed for multi-token medical answers",
        "n_clasheval": int(sum(1 for r in results if str(r.get("source", "")) == "clasheval")),
        "n_recall": int(sum(1 for r in results if str(r.get("source", "")) == "recall")),
        "n_medrag": int(sum(1 for r in results if str(r.get("source", "")) == "medrag")),
    }
