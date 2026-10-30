from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.ag_erha import parse_ag_erha_config, run_ag_erha_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AG-ERHA Track A + Track B pipeline from AG-erha.md")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase1_traces_path", default="results/phase1_diagnostics/per_example_traces.jsonl")
    parser.add_argument("--phase2_main_eval_dir", default="results/phase2/main_eval")
    parser.add_argument("--ag_out_dir", default="results/ag_erha")
    parser.add_argument("--track_b_out_dir", default="results/track_b")
    parser.add_argument("--margin_out_dir", default="results/margin_gated_erha")

    parser.add_argument("--tau_grid", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--tau_margin_grid", default="-2.0,-1.0,-0.5,0.0,0.5")

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_margin_upgrade", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_ag_erha_config(args)
    summary = run_ag_erha_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"track_a={config.ag_out_dir}")
    print(f"track_b={config.track_b_out_dir}")
    print(f"margin_upgrade={config.margin_out_dir}")


if __name__ == "__main__":
    main()
