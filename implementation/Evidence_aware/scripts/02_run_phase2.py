from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.phase2 import parse_phase2_config, run_phase2_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-2 ERHA pipeline from phase2.md")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase1_traces_path", default="results/phase1_diagnostics/per_example_traces.jsonl")
    parser.add_argument("--phase1_aggregate_path", default="results/phase1_diagnostics/aggregate_statistics.json")
    parser.add_argument("--out_dir", default="results/phase2")

    parser.add_argument("--validation_size", type=int, default=200)
    parser.add_argument("--eval_size", type=int, default=2000, help="Set <=0 for full remaining set.")
    parser.add_argument("--min_eval_examples", type=int, default=2000)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--beta_values", default="1.5,2.0,3.0,5.0,8.0")
    parser.add_argument("--include_beta_one_sanity", type=int, default=1, choices=[0, 1])
    parser.add_argument("--target_heads", default="24:3,4;27:20")

    parser.add_argument("--run_dola", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mature_layer", type=int, default=31)
    parser.add_argument("--premature_layers", default="14,16,18,20")

    parser.add_argument("--run_ablation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--random_control_heads_count", type=int, default=3)
    parser.add_argument("--smoke_examples", type=int, default=10)
    parser.add_argument("--allow_unstable_validation", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_phase2_config(args)
    summary = run_phase2_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"final_summary={config.out_dir / 'final_summary.json'}")
    print(f"validation={config.out_dir / 'validation'}")
    print(f"main_eval={config.out_dir / 'main_eval'}")
    print(f"ablation={config.out_dir / 'ablation'}")
    print(f"plots={config.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
