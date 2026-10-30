from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.mistral_steering import parse_mistral_steering_config, run_mistral_steering_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mistral OV-circuit steering (pilot + full sweep)")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase2_main_eval_dir", default="results/mistral/phase2/main_eval")
    parser.add_argument("--out_dir", default="results/final_correction/mistral_steering")
    parser.add_argument("--target_heads", default="29:2,24:22,31:22")
    parser.add_argument("--selected_head", default="auto")
    parser.add_argument("--lambda_max_override", default="auto")
    parser.add_argument("--alpha_grid", default="0.5,1.0,1.5")
    parser.add_argument("--tau_grid", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--pilot_n", type=int, default=200)
    parser.add_argument("--pilot_alpha", type=float, default=1.0)
    parser.add_argument("--pilot_tau", type=float, default=0.0)
    parser.add_argument("--pilot_min_improvement", type=float, default=0.02)
    parser.add_argument("--pilot_min_stability", type=float, default=0.92)
    parser.add_argument("--pilot_max_harm", type=float, default=0.01)
    parser.add_argument("--pilot_max_other_to_faithful", type=int, default=0)
    parser.add_argument("--force_full_grid", type=int, default=0, choices=[0, 1])
    parser.add_argument("--run_pilot_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gain_sample_n", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_token_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--bootstrap_first_token_margin", type=int, default=1, choices=[0, 1])
    parser.add_argument("--amplification_ceiling", type=float, default=0.013)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_mistral_steering_config(args)
    summary = run_mistral_steering_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"gain_diagnostic={Path(config.out_dir) / 'gain_diagnostic.json'}")
    print(f"pilot={Path(config.out_dir) / 'pilot_200ex.json'}")
    print(f"sweep_summary={Path(config.out_dir) / 'steering_sweep_summary.json'}")
    print(f"verdict={Path(config.out_dir) / 'steering_verdict.txt'}")
    print(f"gate_stats={Path(config.out_dir) / 'gate_stats'}")
    print(f"plots={Path(config.out_dir) / 'plots'}")


if __name__ == "__main__":
    main()
