#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


OLD_CANDIDATES = [
    "results/medqa_final_v3/localization_v2/joint_patching_disruption/results.jsonl",
    "results/medqa_final_v3/causal/joint_patching_disruption/results.jsonl",
    "results/medqa_final_v3/causal/disruption/results.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy and corrected MedQA disruption per-pair outcomes.")
    parser.add_argument("--old_path", default="", help="Legacy Appendix E per-pair disruption results.jsonl.")
    parser.add_argument(
        "--new_path",
        default="results/rebuttal/matched_head_controls_fixed_disruption_v2/disruption_arm_top5.jsonl",
        help="Corrected top5 disruption per-pair rows.",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _find_old(explicit: str) -> Path:
    if explicit:
        p = _resolve(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Explicit --old_path does not exist: {p}")
        return p
    existing = [_resolve(candidate) for candidate in OLD_CANDIDATES if _resolve(candidate).exists()]
    if not existing:
        msg = "\n".join(str(_resolve(candidate)) for candidate in OLD_CANDIDATES)
        raise FileNotFoundError(f"No legacy disruption artifact found. Checked:\n{msg}")
    return existing[0]


def _category(row: dict[str, Any]) -> str:
    if "outcome" in row:
        value = str(row["outcome"])
        mapping = {
            "flip_to_source": "source",
            "flip_to_dest": "dest",
            "flip_to_third": "third",
            "unchanged": "unchanged",
        }
        return mapping.get(value, value)
    if "outcome_category" in row:
        value = str(row["outcome_category"])
        mapping = {
            "flip_to_source_letter": "source",
            "flip_to_dest_correct_letter": "dest",
            "flip_to_third_letter": "third",
            "unchanged": "unchanged",
        }
        return mapping.get(value, value)
    if bool(row.get("flipped_to_source_letter", False)) or bool(row.get("flip_to_source", False)):
        return "source"
    if bool(row.get("flipped_to_dest_correct_letter", False)) or bool(row.get("flip_to_dest", False)):
        return "dest"
    if bool(row.get("flipped_to_third_letter", False)) or bool(row.get("flip_to_third", False)):
        return "third"
    if bool(row.get("unchanged", False)):
        return "unchanged"
    return "unknown"


def _post(row: dict[str, Any]) -> Any:
    return row.get("patched_answer", row.get("post_patch_letter"))


def _join_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("pair_id", "")),
        str(row.get("source_id", "")),
        str(row.get("dest_id", "")),
    )


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id"),
        "source_id": row.get("source_id"),
        "dest_id": row.get("dest_id"),
        "source_correct_letter": row.get("source_correct_letter"),
        "dest_correct_letter": row.get("dest_correct_letter"),
        "dest_base_letter": row.get("baseline_dest_answer", row.get("dest_base_letter")),
        "post_patch_letter": _post(row),
        "category": _category(row),
        "scores": row.get("scores", row.get("patched_scores")),
    }


def main() -> int:
    args = parse_args()
    old_path = _find_old(str(args.old_path))
    new_path = _resolve(str(args.new_path))
    if not new_path.exists():
        raise FileNotFoundError(f"New disruption rows missing: {new_path}")

    old_rows = _read_jsonl(old_path)
    new_rows = _read_jsonl(new_path)
    old_by_pair = {str(row.get("pair_id")): row for row in old_rows if str(row.get("pair_id", ""))}
    new_by_pair = {str(row.get("pair_id")): row for row in new_rows if str(row.get("pair_id", ""))}
    shared_pair_ids = sorted(set(old_by_pair) & set(new_by_pair))

    join_mode = "pair_id"
    if not shared_pair_ids:
        old_by_pair = {f"{row.get('source_id')}->{row.get('dest_id')}": row for row in old_rows}
        new_by_pair = {f"{row.get('source_id')}->{row.get('dest_id')}": row for row in new_rows}
        shared_pair_ids = sorted(set(old_by_pair) & set(new_by_pair))
        join_mode = "source_id->dest_id"

    old_counts: dict[str, int] = {}
    new_counts: dict[str, int] = {}
    for row in old_rows:
        old_counts[_category(row)] = old_counts.get(_category(row), 0) + 1
    for row in new_rows:
        new_counts[_category(row)] = new_counts.get(_category(row), 0) + 1

    print(f"old_path: {old_path}")
    print(f"new_path: {new_path}")
    print(f"old_n: {len(old_rows)} old_counts: {old_counts}")
    print(f"new_n: {len(new_rows)} new_counts: {new_counts}")
    print(f"join_mode: {join_mode} shared: {len(shared_pair_ids)}")

    mismatches = []
    for key in shared_pair_ids:
        old = old_by_pair[key]
        new = new_by_pair[key]
        if _category(old) != _category(new) or _post(old) != _post(new):
            mismatches.append((key, old, new))

    print(f"mismatches: {len(mismatches)}")
    for key, old, new in mismatches:
        cause = "unknown"
        if str(old.get("source_id")) != str(new.get("source_id")) or str(old.get("dest_id")) != str(new.get("dest_id")):
            cause = "different pair set"
        elif _post(old) != _post(new):
            cause = "tie-break / argmax boundary or model nondeterminism"
        elif _category(old) != _category(new):
            cause = "genuine label/accounting fix"
        print("\nDIFF", key, "cause:", cause)
        print("OLD", json.dumps(_compact(old), indent=2, sort_keys=True))
        print("NEW", json.dumps(_compact(new), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
