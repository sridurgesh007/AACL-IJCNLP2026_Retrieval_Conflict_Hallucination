from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

TREATMENT_KEYWORDS = [
    "treatment",
    "dosage",
    "dose",
    "medication",
    "drug",
    "administer",
    "prescribe",
    "therapy",
    "contraindicated",
    "management",
    "avoid",
    "antibiotic",
    "prophylaxis",
]

DOSAGE_PATTERN = re.compile(
    r"\d+\.?\d*\s*(?:mg|mcg|ug|ml|g|kg|units?|iu|mmol|meq|%)",
    re.IGNORECASE,
)

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
    "dangerous",
}

try:
    import spacy as _spacy

    _nlp = _spacy.load("en_core_sci_md")
    _SPACY_OK = True
except Exception:
    _nlp = None
    _SPACY_OK = False


def _extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _extract_guideline_answer(row: dict[str, Any]) -> str:
    for key in ("guideline_answer", "answer", "long_answer", "target", "reference"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _extract_passages(row: dict[str, Any]) -> list[str]:
    for key in ("retrieved_passages", "passages", "contexts", "ctxs"):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, list):
            out = [str(x).strip() for x in val if str(x).strip()]
            if out:
                return out
        if isinstance(val, str) and val.strip():
            return [val.strip()]
    if "context" in row and str(row["context"]).strip():
        return [str(row["context"]).strip()]
    return []


def load_medrag_treatment_questions(n: int = 2000, db_dir: str = "data/medrag_statpearls") -> list[dict[str, Any]]:
    q_path = Path(db_dir) / "test.jsonl"
    if q_path.exists():
        questions: list[dict[str, Any]] = []
        with q_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = _extract_question(row)
                if not q:
                    continue
                if not any(kw in q.lower() for kw in TREATMENT_KEYWORDS):
                    continue
                questions.append(
                    {
                        "question": q,
                        "retrieved_passages": _extract_passages(row),
                        "guideline_answer": _extract_guideline_answer(row),
                        "source": "medrag",
                    }
                )
        if questions:
            print(f"Loaded {len(questions)} treatment questions from {q_path}")
            return questions[:n]

    print("MedRAG file not found or empty; falling back to PubMedQA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "PubMedQA fallback requires `datasets` package. "
            "Install with `pip install datasets` or provide MedRAG test.jsonl."
        ) from exc

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    questions = []
    for ex in ds:
        q = str(ex.get("question", "")).strip()
        if not q:
            continue
        if not any(kw in q.lower() for kw in TREATMENT_KEYWORDS):
            continue
        context = ex.get("context", {}) or {}
        ctx_list = context.get("contexts", []) if isinstance(context, dict) else []
        passages = [str(x).strip() for x in ctx_list if str(x).strip()]
        questions.append(
            {
                "question": q,
                "retrieved_passages": passages,
                "guideline_answer": str(ex.get("long_answer", "")).strip(),
                "source": "pubmedqa",
            }
        )
    print(f"Loaded {len(questions)} treatment questions from PubMedQA")
    return questions[:n]


def build_medical_sets(
    tokenizer,
    guideline_text: str,
    cached_logits: torch.Tensor,
    k: int = 50,
) -> tuple[set[int], set[int]]:
    top_k_ids = set(int(x) for x in cached_logits.topk(int(k)).indices.tolist())
    guideline_ids = set(int(x) for x in tokenizer.encode(guideline_text, add_special_tokens=False))
    evidence_set = top_k_ids & guideline_ids
    parametric_set = top_k_ids - evidence_set
    return evidence_set, parametric_set


def most_representative_token(token_set: set[int], cached_logits: torch.Tensor) -> int | None:
    if not token_set:
        return None
    return max((int(t) for t in token_set), key=lambda t: float(cached_logits[int(t)].item()))


def extract_clinical_entities(text: str) -> set[str]:
    entities: set[str] = set()
    if _SPACY_OK and _nlp is not None:
        try:
            doc = _nlp(text)
            for ent in doc.ents:
                val = ent.text.lower().strip()
                if val:
                    entities.add(val)
        except Exception:
            pass
    for match in DOSAGE_PATTERN.finditer(text):
        val = match.group(0).lower().strip()
        if val:
            entities.add(val)
    return entities


def compute_entity_overlap(entities_a: set[str], entities_b: set[str]) -> float:
    if not entities_b:
        return 1.0
    return float(len(entities_a & entities_b) / max(len(entities_b), 1))


def compute_gar(generated_text: str, guideline_text: str, threshold: float = 0.5) -> tuple[bool, float]:
    gen_ents = extract_clinical_entities(generated_text)
    guid_ents = extract_clinical_entities(guideline_text)
    score = compute_entity_overlap(gen_ents, guid_ents)
    return bool(score >= float(threshold)), float(score)


def contraindication_surfaced(generated_text: str, guideline_text: str) -> bool | None:
    g_low = guideline_text.lower()
    t_low = generated_text.lower()
    has_contra = any(kw in g_low for kw in CONTRAINDICATION_KEYWORDS)
    if not has_contra:
        return None
    return bool(any(kw in t_low for kw in CONTRAINDICATION_KEYWORDS))
