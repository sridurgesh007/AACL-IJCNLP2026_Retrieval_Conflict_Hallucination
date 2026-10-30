#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify medical conflict dataset quality")
    parser.add_argument("--dataset_path", default="data/medical_conflict_500.jsonl")
    parser.add_argument("--min_examples", type=int, default=100)
    parser.add_argument("--min_explicit_frac", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.dataset_path).resolve()
    if not path.exists():
        print(f"ERROR: missing dataset {path}")
        return 1

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    print(f"total_examples={n}")
    if n < int(args.min_examples):
        print(f"ERROR: too few examples ({n} < {args.min_examples})")
        return 1

    sources: dict[str, int] = {}
    for ex in rows:
        src = str(ex.get("source", "unknown"))
        sources[src] = int(sources.get(src, 0) + 1)
    print(f"sources={json.dumps(sources, indent=2)}")

    explicit = [
        ex
        for ex in rows
        if str(ex.get("context_answer", "")).strip() and str(ex.get("parametric_answer", "")).strip()
    ]
    n_explicit = len(explicit)
    explicit_frac = n_explicit / max(n, 1)
    print(f"explicit_answer_pairs={n_explicit}/{n} ({explicit_frac:.3f})")
    if explicit_frac < float(args.min_explicit_frac):
        print(f"ERROR: explicit fraction below threshold ({explicit_frac:.3f} < {args.min_explicit_frac:.3f})")
        return 1

    same_answer = [
        ex
        for ex in explicit
        if str(ex["context_answer"]).strip().lower() == str(ex["parametric_answer"]).strip().lower()
    ]
    if same_answer:
        print(f"ERROR: found {len(same_answer)} non-conflicts with identical answers")
        return 1

    print("sample_examples:")
    for idx, ex in enumerate(rows[:5], start=1):
        q = str(ex.get("question", ""))[:120]
        e = str(ex.get("context_answer", "N/A"))[:80]
        p = str(ex.get("parametric_answer", "N/A"))[:80]
        src = str(ex.get("source", "unknown"))
        print(f"[{idx}] source={src}")
        print(f"  q={q}")
        print(f"  e={e}")
        print(f"  p={p}")

    print("verify=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
