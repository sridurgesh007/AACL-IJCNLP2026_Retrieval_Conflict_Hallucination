from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.ag_erha.track_b import run_track_b
from evidence_aware.ag_erha.utils import load_cache_rows, normalize_example_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AG-ERHA Track B only (margin diagnostics).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase1_traces_path", default="results/phase1_diagnostics/per_example_traces.jsonl")
    parser.add_argument("--out_dir", default="results/track_b")
    parser.add_argument("--focus_layer", type=int, default=27)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = normalize_example_rows(load_cache_rows(Path(args.cache_path)))
    examples_by_id = {x.id: x for x in rows}

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), use_fast=True)
    summary = run_track_b(
        tokenizer=tokenizer,
        examples_by_id=examples_by_id,
        traces_path=Path(args.phase1_traces_path),
        out_dir=Path(args.out_dir),
        focus_layer=int(args.focus_layer),
    )
    print(json.dumps(summary, indent=2))
    print(f"track_b_out={args.out_dir}")


if __name__ == "__main__":
    main()
