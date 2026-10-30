from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

CONTEXT_ANSWER_FIELDS = [
    "context_answer",
    "substituted_answer",
    "correct_answer",
    "evidence_answer",
]

PARAMETRIC_ANSWER_FIELDS = [
    "parametric_answer",
    "original_answer",
    "model_answer",
    "memorized_answer",
]

CONTEXT_FIELDS = [
    "context",
    "passage",
    "retrieved_passage",
    "evidence",
]

DEFAULT_CLASHEVAL_CANDIDATES = [
    "clasheval/data/medical.jsonl",
    "clasheval/data/medical.json",
    "clasheval/data/clasheval_medical.jsonl",
    "clasheval/data/test.jsonl",
    "clasheval/clasheval.json",
    "clasheval/data/data.json",
]

DEFAULT_RECALL_CANDIDATES = [
    "recall/data/medical.jsonl",
    "recall/data/medical.json",
    "recall/data/test.jsonl",
    "recall/data/data.jsonl",
]


def resolve_existing_path(candidates: Iterable[str | Path], root: Path | None = None) -> Path | None:
    base = root.resolve() if root is not None else Path.cwd().resolve()
    for candidate in candidates:
        path = Path(candidate)
        path = path if path.is_absolute() else (base / path)
        if path.exists():
            return path.resolve()
    return None


def _get_field(ex: dict, candidates: list[str], fallback: str = "") -> str:
    for field in candidates:
        if field in ex and ex[field]:
            return str(ex[field]).strip()
    return fallback


def _load_json_or_jsonl(path: Path) -> list[dict]:
    raw_examples: list[dict] = []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return raw_examples
    if content.startswith("["):
        loaded = json.loads(content)
        if isinstance(loaded, list):
            return [x for x in loaded if isinstance(x, dict)]
        return []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if isinstance(raw, dict):
            raw_examples.append(raw)
    return raw_examples


def load_clasheval(
    data_path: str | Path,
    domain: str | None = "medical",
    min_answer_length: int = 1,
    max_examples: int = 1000,
) -> list[dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"ClashEval file not found: {path}")

    raw_examples = _load_json_or_jsonl(path)
    examples: list[dict] = []
    skipped = 0

    domain_lower = domain.lower() if isinstance(domain, str) else None
    for raw in raw_examples:
        if len(examples) >= int(max_examples):
            break

        ex_domain = str(raw.get("domain", "")).lower().strip()
        if domain_lower is not None and domain_lower not in ex_domain:
            skipped += 1
            continue

        question = str(raw.get("question", "")).strip()
        context = _get_field(raw, CONTEXT_FIELDS)
        context_answer = _get_field(raw, CONTEXT_ANSWER_FIELDS)
        parametric_answer = _get_field(raw, PARAMETRIC_ANSWER_FIELDS)

        if not question:
            skipped += 1
            continue
        if len(context_answer) < int(min_answer_length):
            skipped += 1
            continue
        if len(parametric_answer) < int(min_answer_length):
            skipped += 1
            continue
        if context_answer.lower() == parametric_answer.lower():
            skipped += 1
            continue

        examples.append(
            {
                "question": question,
                "context": context,
                "context_answer": context_answer,
                "parametric_answer": parametric_answer,
                "domain": ex_domain or (domain_lower or "unknown"),
                "source": "clasheval",
                "_raw_fields": list(raw.keys()),
            }
        )

    print(f"ClashEval: loaded {len(examples)} examples, skipped {skipped}")
    return examples
