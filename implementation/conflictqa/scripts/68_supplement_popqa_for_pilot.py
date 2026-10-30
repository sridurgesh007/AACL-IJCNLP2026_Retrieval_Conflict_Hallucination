#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conflictqa_pipeline.utils import build_metadata, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add C2-faithful supplement rows for PopQA pilot AUC only.")
    parser.add_argument("--conflict_path", default="results/cross_model/mistral_popqa_c2_rebuild/popqa_genuine_conflicts.jsonl")
    parser.add_argument("--resolved_path", default="results/cross_model/mistral_popqa_c2_rebuild/popqa_resolved_by_context.jsonl")
    parser.add_argument("--out_path", default="results/cross_model/mistral_popqa_c2_rebuild/popqa_conflicts_supplemented.jsonl")
    parser.add_argument("--summary_path", default="")
    parser.add_argument("--n_supplement", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _mark_rows(rows: list[dict[str, Any]], *, is_supplement: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["is_supplement"] = bool(is_supplement)
        copied["supplement_source"] = "c2_resolved" if is_supplement else ""
        out.append(copied)
    return out


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    rng = random.Random(int(args.seed))

    conflict_path = _resolve_project_path(args.conflict_path)
    resolved_path = _resolve_project_path(args.resolved_path)
    out_path = _resolve_project_path(args.out_path)
    summary_path = _resolve_project_path(args.summary_path) if str(args.summary_path).strip() else out_path.with_suffix(".summary.json")

    conflicts = read_jsonl(conflict_path)
    resolved = read_jsonl(resolved_path)
    if not conflicts:
        raise RuntimeError(f"No genuine conflict rows found: {conflict_path}")
    if not resolved:
        raise RuntimeError(f"No resolved C2-faithful supplement rows found: {resolved_path}")

    conflict_ids = {str(row.get("item_id", "")) for row in conflicts}
    candidates = [row for row in resolved if str(row.get("item_id", "")) not in conflict_ids]
    if not candidates:
        raise RuntimeError("No non-overlapping resolved rows are available for pilot supplement.")

    rng.shuffle(candidates)
    supplement = candidates[: min(int(args.n_supplement), len(candidates))]
    output_rows = _mark_rows(conflicts, is_supplement=False) + _mark_rows(supplement, is_supplement=True)
    rng.shuffle(output_rows)

    write_jsonl(out_path, output_rows)
    summary = {
        "conflict_path": str(conflict_path),
        "resolved_path": str(resolved_path),
        "out_path": str(out_path),
        "n_conflicts": int(len(conflicts)),
        "n_resolved_loaded": int(len(resolved)),
        "n_resolved_candidates_nonoverlap": int(len(candidates)),
        "n_supplement_requested": int(args.n_supplement),
        "n_supplement_written": int(len(supplement)),
        "n_total_written": int(len(output_rows)),
        "prop_counts_conflict": dict(sorted(Counter(str(row.get("prop", "")) for row in conflicts).items())),
        "prop_counts_supplement": dict(sorted(Counter(str(row.get("prop", "")) for row in supplement).items())),
        "metadata": build_metadata(
            args=args,
            input_files=[conflict_path, resolved_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=PROJECT_ROOT.parent,
        ),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"supplemented={out_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
