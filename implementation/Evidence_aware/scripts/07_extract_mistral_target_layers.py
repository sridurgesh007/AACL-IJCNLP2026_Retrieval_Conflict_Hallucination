from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.mistral_multi_head import parse_mistral_extraction_config, run_mistral_target_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Mistral target layers from Track-B margins and top heads from phase1 LD traces."
    )
    parser.add_argument("--track_b_path", default="results/mistral/track_b/margin_trajectories.jsonl")
    parser.add_argument("--phase1_ld_path", default="results/mistral/phase1/per_example_ld.jsonl")
    parser.add_argument("--out_dir", default="results/mistral/multi_head")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_layer", type=int, default=20)
    parser.add_argument("--fallback_layers", default="24,25,26,27,28,29,30,31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_mistral_extraction_config(args)
    summary = run_mistral_target_extraction(config)
    print(json.dumps(summary, indent=2))
    print(f"target_layers_json={Path(config.out_dir) / 'target_layers.json'}")
    print(f"top_heads_json={Path(config.out_dir) / 'top_heads_per_layer.json'}")


if __name__ == "__main__":
    main()
