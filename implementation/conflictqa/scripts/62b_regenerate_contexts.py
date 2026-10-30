#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conflictqa_pipeline.utils import read_jsonl, write_jsonl  # type: ignore

from importlib.util import module_from_spec, spec_from_file_location


_BUILD_SPEC = spec_from_file_location("popqa_build_contexts", PROJECT_ROOT / "scripts" / "62_build_popqa_conflict.py")
if _BUILD_SPEC is None or _BUILD_SPEC.loader is None:
    raise RuntimeError("Could not load 62_build_popqa_conflict.py")
_BUILD = module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(_BUILD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate PopQA natural contexts from an existing 500-item conflict set.")
    parser.add_argument("--input_path", default="data/popqa_conflict_500.jsonl")
    parser.add_argument("--out_path", default="data/popqa_conflict_500_natural.jsonl")
    parser.add_argument("--sample_count", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = (PROJECT_ROOT / args.input_path).resolve()
    out_path = (PROJECT_ROOT / args.out_path).resolve()

    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    out_rows = []
    samples = []
    for idx, row in enumerate(rows):
        old_context = str(row.get("context", "")).strip()
        correct_answer = str(row["correct_answer"]).strip()
        prop = str(row["prop"]).strip()
        question = str(row["question"]).strip()
        natural_context = _BUILD._build_context(question, correct_answer, prop)
        injection_context = str(row.get("explicit_injection_context") or _BUILD._build_injection_context(correct_answer)).strip()

        updated = dict(row)
        updated["context"] = natural_context
        updated["explicit_injection_context"] = injection_context
        out_rows.append(updated)

        if idx < int(args.sample_count):
            samples.append(
                {
                    "item_id": row["item_id"],
                    "question": question,
                    "prop": prop,
                    "old_context": old_context,
                    "new_context": natural_context,
                }
            )

    write_jsonl(out_path, out_rows)
    print(f"wrote={out_path}")
    print(json.dumps({"n_items": len(out_rows), "samples": samples}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
