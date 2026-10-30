from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.dataset.counterfact import CounterFactLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized CounterFact cache for Evidence_aware.")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument(
        "--hf_candidates",
        default="NeelNanda/counterfact-tracing,counterfact",
        help="Comma-separated HF dataset candidates.",
    )
    parser.add_argument(
        "--fallback_url",
        default="https://rome.baulab.info/data/dsets/counterfact.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = CounterFactLoader(
        limit=args.limit,
        cache_path=args.cache_path,
        hf_candidates=[x.strip() for x in args.hf_candidates.split(",") if x.strip()],
        fallback_url=args.fallback_url,
    )
    rows = loader.load(force_refresh=args.force_refresh)
    if not rows:
        raise RuntimeError("No rows loaded.")
    print(f"cached_rows={len(rows)}")
    print(f"sample_keys={list(rows[0].keys())}")


if __name__ == "__main__":
    main()
