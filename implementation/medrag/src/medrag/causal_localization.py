from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import build_letter_token_sets, load_conflicts, prepare_examples_for_variant
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import git_commit_hash, read_json, sha256_file


@dataclass(frozen=True)
class CausalItem:
    item_id: str
    prepared: Any
    eval_row: dict[str, Any]
    prompt_length: int
    correct: str
    parametric: str
    base_letter: str | None
    base_correct: bool
    conflict_detected: bool
    prefill_margin: float
    in_matched_subset: bool
    in_clean_subset: bool


@dataclass(frozen=True)
class MedQACausalInputs:
    conflicts: list[dict[str, Any]]
    prepared_records: list[Any]
    prepared_by_id: dict[str, Any]
    h4_rows: list[dict[str, Any]]
    h4_by_id: dict[str, dict[str, Any]]
    matched_payload: dict[str, Any]
    label_audit: dict[str, str]
    matched_ids: set[str]
    clean_ids: set[str]
    letter_token_sets: dict[str, set[int]]
    causal_items: list[CausalItem]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def build_metadata(
    *,
    args: Any,
    input_files: list[str | Path],
    seed: int,
    wall_clock_seconds: float,
    project_root: str | Path,
) -> dict[str, Any]:
    resolved_inputs = [Path(path).expanduser().resolve() for path in input_files]
    return {
        "seed": int(seed),
        "git_commit_hash": git_commit_hash(project_root),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path) for path in resolved_inputs],
        "input_file_sha256": {str(path): sha256_file(path) for path in resolved_inputs if path.exists()},
    }


def slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def predict_letter_from_logits(
    logits: torch.Tensor,
    letter_token_sets: dict[str, set[int]],
) -> tuple[str | None, dict[str, float]]:
    scores: dict[str, float] = {}
    vocab = int(logits.numel())
    for letter, token_ids in sorted(letter_token_sets.items()):
        valid = [int(tid) for tid in token_ids if 0 <= int(tid) < vocab]
        if not valid:
            continue
        score = max(float(logits[int(tid)].item()) for tid in valid)
        scores[str(letter)] = float(score)
    if not scores:
        return None, {}
    best_letter = max(scores.items(), key=lambda x: x[1])[0]
    return str(best_letter), scores


def choose_letter_token_id(tokenizer, letter: str) -> int | None:
    candidates = [f" {letter}", str(letter)]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return None


def canonical_letter_token_ids(tokenizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for letter in ("A", "B", "C", "D"):
        token_id = choose_letter_token_id(tokenizer, letter)
        if token_id is None:
            raise RuntimeError(f"Could not resolve a token id for letter {letter}.")
        out[letter] = int(token_id)
    return out


def token_is_content(tokenizer, token_id: int) -> bool:
    all_special_ids = {int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])}
    if int(token_id) in all_special_ids:
        return False
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    cleaned = (
        str(text)
        .replace("\u2581", " ")
        .replace("\u0120", " ")
        .replace("\u010a", " ")
        .strip()
    )
    if not cleaned:
        return False
    return any(ch.isalnum() for ch in cleaned)


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    last_start = len(sequence) - len(subsequence)
    for start in range(last_start + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return int(start)
    return None


def locate_context_positions(
    *,
    tokenizer,
    prompt: str,
    prompt_ids: torch.Tensor,
    context: str,
) -> tuple[list[int], list[int]]:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be 1D.")
    context = str(context or "")
    if not context or context not in prompt:
        return [], []

    token_list = [int(x) for x in prompt_ids.tolist()]
    try:
        encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        input_ids = [int(x) for x in encoded.get("input_ids", [])]
        if isinstance(offsets, list) and len(offsets) == len(input_ids) == len(token_list):
            start_char = int(prompt.index(context))
            end_char = int(start_char + len(context))
            span_positions = [
                int(idx)
                for idx, (lo, hi) in enumerate(offsets)
                if int(hi) > int(start_char) and int(lo) < int(end_char)
            ]
            content_positions = [
                int(idx)
                for idx in span_positions
                if token_is_content(tokenizer, int(token_list[idx]))
            ]
            return span_positions, content_positions
    except Exception:
        pass

    candidates = [
        tokenizer.encode(context, add_special_tokens=False),
        tokenizer.encode(f"Clinical Context:\n{context}", add_special_tokens=False),
    ]
    for ids in candidates:
        found = find_subsequence(token_list, [int(x) for x in ids])
        if found is None:
            continue
        span_positions = list(range(int(found), int(found + len(ids))))
        content_positions = [
            int(idx)
            for idx in span_positions
            if token_is_content(tokenizer, int(token_list[idx]))
        ]
        return span_positions, content_positions
    return [], []


def middle_prompt_position(prompt_ids: torch.Tensor) -> int:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be 1D.")
    seq_len = int(prompt_ids.shape[0])
    if seq_len <= 1:
        return 0
    pos = max(0, min(seq_len - 2, seq_len // 2))
    return int(pos)


def load_medqa_causal_inputs(
    *,
    model,
    tokenizer,
    device: torch.device,
    model_id: str,
    conflict_path: str | Path,
    matched_subset_path: str | Path,
    label_audit_path: str | Path,
    h4_results_path: str | Path,
    cache_dir: str | Path,
    target_layer: int,
    max_length: int,
    max_new_tokens: int,
) -> MedQACausalInputs:
    conflict_path = Path(conflict_path).expanduser().resolve()
    matched_subset_path = Path(matched_subset_path).expanduser().resolve()
    label_audit_path = Path(label_audit_path).expanduser().resolve()
    h4_results_path = Path(h4_results_path).expanduser().resolve()

    conflicts = load_conflicts(conflict_path)
    matched_payload = read_json(matched_subset_path)
    label_audit = load_label_audit(label_audit_path)
    clean_ids = set(clean_subset_item_ids(matched_payload, label_audit))
    matched_ids = {str(x) for x in matched_payload.get("matched_item_ids", [])}
    h4_rows = read_jsonl(h4_results_path)
    h4_by_id = {str(row.get("item_id")): row for row in h4_rows if str(row.get("item_id", "")).strip()}

    letter_token_sets = build_letter_token_sets(tokenizer)
    cache = ActivationCache(Path(cache_dir).expanduser().resolve())
    prepared_records = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=cache,
        model_id=str(model_id),
        prompt_variant="vanilla",
        p_set_mode="all_wrong",
        target_layer=int(target_layer),
        max_length=int(max_length),
        max_new_tokens=int(max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    prepared_by_id = {str(rec.item_id): rec for rec in prepared_records}

    missing = sorted(set(h4_by_id.keys()) - set(prepared_by_id.keys()))
    if missing:
        raise RuntimeError(
            f"H4 results contain {len(missing)} item ids that could not be prepared. "
            f"First missing id: {missing[0]}"
        )

    causal_items: list[CausalItem] = []
    for item_id, row in sorted(h4_by_id.items()):
        rec = prepared_by_id[item_id]
        causal_items.append(
            CausalItem(
                item_id=str(item_id),
                prepared=rec,
                eval_row=row,
                prompt_length=int(rec.prompt_ids.shape[0]),
                correct=str(rec.correct),
                parametric=str(rec.parametric),
                base_letter=(None if rec.base_letter is None else str(rec.base_letter)),
                base_correct=bool(row.get("base_correct", rec.base_correct)),
                conflict_detected=bool(row.get("conflict_detected", False)),
                prefill_margin=float(row.get("prefill_margin", 0.0)),
                in_matched_subset=bool(row.get("in_matched_subset", False)),
                in_clean_subset=bool(row.get("in_clean_subset", False)),
            )
        )
    return MedQACausalInputs(
        conflicts=conflicts,
        prepared_records=prepared_records,
        prepared_by_id=prepared_by_id,
        h4_rows=h4_rows,
        h4_by_id=h4_by_id,
        matched_payload=matched_payload,
        label_audit=label_audit,
        matched_ids=matched_ids,
        clean_ids=clean_ids,
        letter_token_sets=letter_token_sets,
        causal_items=causal_items,
    )


def build_patching_pairs(
    *,
    causal_items: list[CausalItem],
    n_dest_target: int,
    min_dest_required: int,
) -> list[dict[str, Any]]:
    clean_dest = [
        item
        for item in causal_items
        if item.in_clean_subset and (not item.base_correct) and item.conflict_detected
    ]
    extra_dest_pool = [
        item
        for item in causal_items
        if item.in_matched_subset
        and (not item.in_clean_subset)
        and (not item.base_correct)
        and item.conflict_detected
    ]
    clean_dest = sorted(clean_dest, key=lambda x: (x.correct, x.prompt_length, x.item_id))
    extra_dest_pool = sorted(extra_dest_pool, key=lambda x: (x.correct, abs(x.prefill_margin), x.prompt_length, x.item_id))

    extra_needed = max(0, int(n_dest_target) - len(clean_dest))
    dest_items = clean_dest + extra_dest_pool[:extra_needed]
    if len(dest_items) <= 0:
        raise RuntimeError(
            "No destination items available after applying the base-wrong + conflict-detected filters."
        )

    source_items = [
        item
        for item in causal_items
        if item.in_matched_subset and item.base_correct and (not item.conflict_detected)
    ]
    sources_by_letter: dict[str, list[CausalItem]] = {}
    strata_by_letter: dict[str, dict[str, list[CausalItem]]] = {}
    for letter in ("A", "B", "C", "D"):
        bucket = sorted(
            [item for item in source_items if item.correct == letter],
            key=lambda x: (x.prefill_margin, x.prompt_length, x.item_id),
        )
        if not bucket:
            continue
        margins = sorted(float(item.prefill_margin) for item in bucket)
        median = margins[len(margins) // 2]
        low = [item for item in bucket if float(item.prefill_margin) <= float(median)]
        high = [item for item in bucket if float(item.prefill_margin) > float(median)]
        strata_by_letter[letter] = {
            "low": low if low else list(bucket),
            "high": high if high else list(bucket),
        }
        sources_by_letter[letter] = bucket

    usage_counts: dict[str, int] = {}
    letter_dest_counts: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    for index, dest in enumerate(dest_items):
        letter = str(dest.correct)
        if letter not in sources_by_letter:
            raise RuntimeError(f"No source items available for correct letter {letter}.")
        ordinal = int(letter_dest_counts.get(letter, 0))
        letter_dest_counts[letter] = int(ordinal + 1)
        preferred_stratum = "low" if ordinal % 2 == 0 else "high"
        source_pool = list(strata_by_letter[letter][preferred_stratum]) or list(sources_by_letter[letter])

        def _score(src: CausalItem) -> tuple[int, int, float, str]:
            return (
                int(usage_counts.get(src.item_id, 0)),
                abs(int(src.prompt_length) - int(dest.prompt_length)),
                abs(float(src.prefill_margin) - float(dest.prefill_margin)),
                str(src.item_id),
            )

        source = min(source_pool, key=_score)
        usage_counts[source.item_id] = int(usage_counts.get(source.item_id, 0) + 1)
        pair_id = f"{index:03d}_{letter}_{source.item_id[:8]}_{dest.item_id[:8]}"
        pairs.append(
            {
                "pair_id": str(pair_id),
                "source_id": str(source.item_id),
                "dest_id": str(dest.item_id),
                "shared_letter": str(letter),
                "source_margin": float(source.prefill_margin),
                "dest_margin": float(dest.prefill_margin),
                "source_prompt_length": int(source.prompt_length),
                "dest_prompt_length": int(dest.prompt_length),
                "source_margin_stratum": str(preferred_stratum),
                "dest_subset": ("clean" if dest.in_clean_subset else "matched_extra"),
                "source_gate_fires": bool(source.conflict_detected),
                "dest_gate_fires": bool(dest.conflict_detected),
            }
        )
    return pairs
