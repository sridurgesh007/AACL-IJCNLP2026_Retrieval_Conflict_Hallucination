from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.true_multi_head import parse_true_multi_head_config, run_true_multi_head_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRUE multi-head experiments (Experiment A + Experiment B) from steering.md"
    )
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase2_main_eval_dir", default="results/phase2/main_eval")
    parser.add_argument("--ag_track_a_dir", default="results/ag_erha")
    parser.add_argument("--out_dir", default="results/true_multi_head")

    parser.add_argument("--top_heads_path", default="results/multi_head_llama/top_heads_per_layer.json")
    parser.add_argument("--phase1_ld_path", default="results/phase1_output/per_example_traces.jsonl")
    parser.add_argument("--target_layers", default="22,23,24,25,26,27")
    parser.add_argument("--late_layers", default="24,25,26,27")

    parser.add_argument("--exp_a_beta_grid", default="2.0,3.0")
    parser.add_argument("--exp_a_tau_grid", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--exp_a_sanity_taus", default="-999.0,999.0")
    parser.add_argument("--single_beta", type=float, default=5.0)
    parser.add_argument("--single_layer", type=int, default=27)
    parser.add_argument("--single_head", type=int, default=20)

    parser.add_argument("--exp_b_alpha_grid", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--exp_b_tau_grid", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--exp_b_pilot_n", type=int, default=200)
    parser.add_argument("--exp_b_pilot_alpha", type=float, default=1.0)
    parser.add_argument("--exp_b_pilot_tau", type=float, default=0.0)
    parser.add_argument("--exp_b_pilot_min_improvement", type=float, default=0.02)
    parser.add_argument("--exp_b_force_full_grid", type=int, default=0, choices=[0, 1])
    parser.add_argument("--exp_b_lambda_clip_values", default="10.0,20.0")
    parser.add_argument("--exp_b_l22_gain_threshold", type=float, default=0.1)
    parser.add_argument("--exp_b_unit_test_deficit", type=float, default=3.0)
    parser.add_argument("--exp_b_unit_test_max_perturbation", type=float, default=20.0)

    parser.add_argument("--run_experiment_a", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_experiment_b", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanity_tolerance", type=float, default=1e-3)
    parser.add_argument("--first_token_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--bootstrap_first_token_margin", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_true_multi_head_config(args)
    summary = run_true_multi_head_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"final_summary={config.out_dir / 'final_summary.json'}")
    print(f"experiment_a_sweep_summary={config.out_dir / 'experiment_a_sweep_summary.json'}")
    print(f"experiment_a_verdict={config.out_dir / 'experiment_a_verdict.txt'}")
    print(f"experiment_b_sweep_summary={config.out_dir / 'experiment_b_sweep_summary.json'}")
    print(f"experiment_b_verdict={config.out_dir / 'experiment_b_verdict.txt'}")
    print(f"comparison={config.out_dir / 'combined_comparison_table.json'}")
    print(f"plots={config.out_dir / 'plots'}")
    print(f"gate_stats={config.out_dir / 'gate_stats'}")


if __name__ == "__main__":
    main()
