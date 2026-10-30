from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from evidence_aware.phase2.common import Example, PreparedExample, build_prompt, first_answer_token_id_in_prompt


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_cache_rows(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")
    payload = read_json(cache_path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Cache file must be a list: {cache_path}")
    return payload


def normalize_example_rows(rows: list[dict[str, Any]]) -> list[Example]:
    out: list[Example] = []
    for row in rows:
        needed = ["id", "question", "evidence", "answer_liquid", "answer_solid"]
        if not all(k in row for k in needed):
            continue
        out.append(
            Example(
                id=int(row["id"]),
                question=str(row["question"]),
                evidence=str(row["evidence"]),
                answer_liquid=str(row["answer_liquid"]),
                answer_solid=str(row["answer_solid"]),
            )
        )
    return out


def prepare_examples(rows: list[Example], tokenizer, max_length: int) -> list[PreparedExample]:
    prepared: list[PreparedExample] = []
    for row in rows:
        evidence_token_id = first_answer_token_id_in_prompt(tokenizer, row.answer_liquid)
        parametric_token_id = first_answer_token_id_in_prompt(tokenizer, row.answer_solid)
        if evidence_token_id is None or parametric_token_id is None:
            continue
        if evidence_token_id == parametric_token_id:
            continue
        prompt = build_prompt(row.evidence, row.question)
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
        ids = encoded.get("input_ids", [])
        if not ids:
            continue
        prepared.append(
            PreparedExample(
                id=row.id,
                question=row.question,
                evidence=row.evidence,
                answer_liquid=row.answer_liquid,
                answer_solid=row.answer_solid,
                prompt=prompt,
                input_ids=[int(x) for x in ids],
                evidence_token_id=int(evidence_token_id),
                parametric_token_id=int(parametric_token_id),
            )
        )
    return prepared


def split_by_fixed_indices(rows: list[Example]) -> tuple[list[Example], list[Example], list[Example]]:
    phase1 = [r for r in rows if 0 <= r.id <= 499]
    phase2_val = [r for r in rows if 500 <= r.id <= 699]
    eval_rows = [r for r in rows if r.id >= 700]
    return phase1, phase2_val, eval_rows


def find_evidence_positions(prompt_ids: list[int], evidence_ids: list[int]) -> list[int]:
    if not prompt_ids or not evidence_ids or len(evidence_ids) > len(prompt_ids):
        return []
    for i in range(len(prompt_ids) - len(evidence_ids) + 1):
        if prompt_ids[i : i + len(evidence_ids)] == evidence_ids:
            return list(range(i, i + len(evidence_ids)))
    return []


def evidence_positions_with_fallback(tokenizer, prompt_ids: list[int], evidence_text: str) -> tuple[list[int], bool]:
    evidence_ids = tokenizer.encode(evidence_text, add_special_tokens=False)
    pos = find_evidence_positions(prompt_ids, evidence_ids)
    if pos:
        return pos, False
    fallback_n = max(1, int(round(0.4 * len(prompt_ids))))
    return list(range(fallback_n)), True


def roc_auc_binary(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have same length")
    n = len(labels)
    if n == 0:
        return float("nan")
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    sum_pos = sum(ranks[i] for i in pos)
    n_pos = len(pos)
    n_neg = len(neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
