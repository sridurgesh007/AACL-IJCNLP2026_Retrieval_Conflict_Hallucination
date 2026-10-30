from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from evidence_aware.dataset.counterfact import CounterFactLoader
from evidence_aware.phase2.common import Example, PreparedExample, build_prompt, first_answer_token_id_in_prompt


def normalize_row(row: dict[str, Any]) -> Example | None:
    needed = ["id", "question", "evidence", "answer_liquid", "answer_solid"]
    if not all(k in row for k in needed):
        return None
    return Example(
        id=int(row["id"]),
        question=str(row["question"]),
        evidence=str(row["evidence"]),
        answer_liquid=str(row["answer_liquid"]),
        answer_solid=str(row["answer_solid"]),
        subject=str(row.get("subject", "")),
        relation_id=str(row.get("relation_id", "")),
        prompt_template=str(row.get("prompt_template", "")),
    )


def load_examples(cache_path: Path) -> list[Example]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        loader = CounterFactLoader(cache_path=cache_path)
        payload = loader.load(force_refresh=False)
    if not isinstance(payload, list):
        raise RuntimeError(f"Cache must contain a list: {cache_path}")
    examples: list[Example] = []
    for row in payload:
        parsed = normalize_row(row)
        if parsed is not None:
            examples.append(parsed)
    if not examples:
        raise RuntimeError("No valid examples loaded from cache.")
    return examples


def load_phase1_example_ids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing phase1 traces file: {path}")
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "example_id" in row:
                ids.add(int(row["example_id"]))
    if not ids:
        raise RuntimeError(f"Phase1 trace file has no example_id rows: {path}")
    return ids


def make_disjoint_splits(
    examples: list[Example],
    phase1_ids: set[int],
    validation_size: int,
    eval_size: int | None,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    remaining = [ex for ex in examples if ex.id not in phase1_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    if len(remaining) < validation_size:
        raise RuntimeError(
            f"Not enough examples for validation split: available={len(remaining)}, required={validation_size}"
        )
    validation_rows = remaining[:validation_size]
    eval_pool = remaining[validation_size:]
    if eval_size is None:
        eval_rows = eval_pool
    else:
        if len(eval_pool) < eval_size:
            raise RuntimeError(f"Not enough examples for eval split: available={len(eval_pool)}, required={eval_size}")
        eval_rows = eval_pool[:eval_size]
    return validation_rows, eval_rows


def leak_check_or_raise(phase1_ids: set[int], validation_rows: list[Example], eval_rows: list[Example]) -> None:
    validation_ids = {x.id for x in validation_rows}
    eval_ids = {x.id for x in eval_rows}
    if phase1_ids & validation_ids:
        raise RuntimeError("Data leakage: phase1_ids overlaps validation split.")
    if phase1_ids & eval_ids:
        raise RuntimeError("Data leakage: phase1_ids overlaps eval split.")
    if validation_ids & eval_ids:
        raise RuntimeError("Data leakage: validation split overlaps eval split.")


def prepare_examples(rows: Iterable[Example], tokenizer, max_length: int) -> list[PreparedExample]:
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
                subject=row.subject,
                relation_id=row.relation_id,
                prompt_template=row.prompt_template,
            )
        )
    return prepared
