from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MedQARow:
    question: str
    options: dict[str, str]
    correct_answer: str
    source_dataset: str
    source_split: str


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _first_non_empty(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row:
            val = _normalize_text(row.get(key))
            if val:
                return val
    return ""


def _normalize_options_from_mapping(options: dict[Any, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not options:
        return out

    # Fast path: lettered mapping already.
    letter_keys = ["A", "B", "C", "D"]
    if all(_normalize_text(options.get(k)) for k in letter_keys):
        return {k: _normalize_text(options[k]) for k in letter_keys}

    # Numeric mapping -> A/B/C/D by key order.
    numeric_items: list[tuple[int, str]] = []
    for k, v in options.items():
        ks = _normalize_text(k)
        vs = _normalize_text(v)
        if not vs:
            continue
        if ks.isdigit():
            numeric_items.append((int(ks), vs))
    if len(numeric_items) >= 4:
        numeric_items = sorted(numeric_items, key=lambda x: x[0])[:4]
        return {chr(65 + i): txt for i, (_, txt) in enumerate(numeric_items)}

    # Generic mapping fallback: deterministic key order.
    keys = sorted([str(k) for k in options.keys()])
    vals = [_normalize_text(options[k]) for k in keys if _normalize_text(options[k])]
    vals = vals[:4]
    if len(vals) == 4:
        return {chr(65 + i): vals[i] for i in range(4)}
    return out


def _normalize_options_from_list(items: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not items:
        return out

    # List[str]
    if all(not isinstance(x, dict) for x in items):
        vals = [_normalize_text(x) for x in items if _normalize_text(x)]
        if len(vals) >= 4:
            vals = vals[:4]
            return {chr(65 + i): vals[i] for i in range(4)}
        return out

    # List[dict]
    tmp_lettered: dict[str, str] = {}
    ordered_vals: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _first_non_empty(item, ["key", "label", "option", "id"])
        val = _first_non_empty(item, ["value", "text", "content", "option"])
        if not val:
            continue
        if key and key.upper() in {"A", "B", "C", "D"}:
            tmp_lettered[key.upper()] = val
        else:
            ordered_vals.append(val)
    if all(k in tmp_lettered for k in ("A", "B", "C", "D")):
        return {k: tmp_lettered[k] for k in ("A", "B", "C", "D")}
    if len(ordered_vals) >= 4:
        ordered_vals = ordered_vals[:4]
        return {chr(65 + i): ordered_vals[i] for i in range(4)}
    return out


def normalize_options(row: dict[str, Any]) -> dict[str, str]:
    # Common single-field forms.
    if isinstance(row.get("options"), dict):
        out = _normalize_options_from_mapping(row["options"])
        if out:
            return out
    if isinstance(row.get("options"), list):
        out = _normalize_options_from_list(row["options"])
        if out:
            return out

    # Option columns.
    option_col_groups = [
        ("opa", "opb", "opc", "opd"),
        ("option_a", "option_b", "option_c", "option_d"),
        ("ending0", "ending1", "ending2", "ending3"),
        ("A", "B", "C", "D"),
    ]
    for cols in option_col_groups:
        vals = [_normalize_text(row.get(c, "")) for c in cols]
        if all(vals):
            return {chr(65 + i): vals[i] for i in range(4)}

    return {}


def normalize_correct_answer(raw_answer: Any, options: dict[str, str]) -> str:
    ans = _normalize_text(raw_answer)
    if not ans:
        return ""

    a_up = ans.upper()
    if a_up in {"A", "B", "C", "D"}:
        return a_up

    # Numeric index.
    if ans.isdigit():
        idx = int(ans)
        if 0 <= idx <= 3:
            return chr(65 + idx)
        if 1 <= idx <= 4:
            return chr(64 + idx)

    # answer text -> map back to option letter
    for letter, text in options.items():
        if _normalize_text(text).lower() == ans.lower():
            return letter
    return ""


def normalize_medqa_row(row: dict[str, Any], dataset_name: str, split_name: str) -> MedQARow | None:
    if not isinstance(row, dict):
        return None
    question = _first_non_empty(row, ["question", "query", "prompt", "sent1", "text"])
    if not question:
        return None

    options = normalize_options(row)
    if not options or any(not _normalize_text(v) for v in options.values()):
        return None

    # Candidate answer fields across providers.
    answer_candidates = [
        row.get("answer_idx", None),
        row.get("answer", None),
        row.get("label", None),
        row.get("correct_answer", None),
        row.get("target", None),
        row.get("cop", None),
        row.get("gold", None),
    ]
    correct = ""
    for cand in answer_candidates:
        correct = normalize_correct_answer(cand, options)
        if correct:
            break
    if correct not in {"A", "B", "C", "D"}:
        return None

    return MedQARow(
        question=question,
        options=options,
        correct_answer=correct,
        source_dataset=dataset_name,
        source_split=split_name,
    )


def _choose_split(dataset_obj: Any) -> tuple[Any, str]:
    if hasattr(dataset_obj, "keys"):
        # DatasetDict-like
        for split in ("train", "validation", "test"):
            if split in dataset_obj:
                return dataset_obj[split], split
        # fallback first split
        first = next(iter(dataset_obj.keys()))
        return dataset_obj[first], str(first)
    # Dataset-like
    return dataset_obj, "unknown"


def load_medqa_dataset_rows(limit: int | None = None) -> list[MedQARow]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Missing dependency `datasets`. Install it with: pip install datasets"
        ) from exc

    sources: list[tuple[str, str | None]] = [
        ("bigbio/med_qa", "med_qa_en_source"),
        ("GBaker/MedQA-USMLE-4-options", None),
    ]

    rows: list[MedQARow] = []
    errors: list[str] = []
    for dataset_name, cfg in sources:
        try:
            if cfg is None:
                ds = load_dataset(dataset_name, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, cfg, trust_remote_code=True)
            split_ds, split_name = _choose_split(ds)
            normalized: list[MedQARow] = []
            for raw in split_ds:
                item = normalize_medqa_row(raw, dataset_name=dataset_name, split_name=split_name)
                if item is not None:
                    normalized.append(item)
                    if limit is not None and len(normalized) >= int(limit):
                        break
            if normalized:
                rows = normalized
                print(
                    f"MedQA loader: dataset={dataset_name} split={split_name} "
                    f"normalized_rows={len(rows)}"
                )
                return rows
            errors.append(f"{dataset_name}: loaded but yielded 0 normalized rows")
        except Exception as exc:  # pragma: no cover - network/hub variability
            errors.append(f"{dataset_name}: {exc}")

    raise RuntimeError("Unable to load MedQA dataset. Errors:\n- " + "\n- ".join(errors))
