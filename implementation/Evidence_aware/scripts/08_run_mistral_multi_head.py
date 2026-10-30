from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.mistral_multi_head import parse_mistral_multi_head_config, run_mistral_multi_head_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mistral multi-head MG-ERHA pipeline (calibration + full tau sweep)."
    )
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase2_main_eval_dir", default="results/mistral/phase2/main_eval")
    parser.add_argument("--target_layers_path", default="results/mistral/multi_head/target_layers.json")
    parser.add_argument("--top_heads_path", default="results/mistral/multi_head/top_heads_per_layer.json")
    parser.add_argument("--out_dir", default="results/mistral/multi_head")

    parser.add_argument("--tau_values", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--beta_values", default="2.0,3.0,5.0")
    parser.add_argument("--selected_beta", default="none")
    parser.add_argument("--calibration_n", type=int, default=100)
    parser.add_argument("--calibration_only", type=int, default=0, choices=[0, 1])

    parser.add_argument("--single_head_layer", type=int, default=29)
    parser.add_argument("--single_head_index", type=int, default=2)
    parser.add_argument("--single_head_beta", type=float, default=5.0)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--harm_target", type=float, default=0.005)
    parser.add_argument("--harm_reject", type=float, default=0.01)

    parser.add_argument("--generalization_json_path", default="results/generalization/cross_model_comparison.json")
    parser.add_argument("--generalization_plot_path", default="results/generalization/final_cross_model_comparison.png")
    parser.add_argument("--llama_single_restoration", type=float, default=0.057)
    parser.add_argument("--mistral_single_restoration", type=float, default=0.013)
    parser.add_argument("--mistral_single_stability", type=float, default=0.995)
    parser.add_argument("--mistral_single_harm_rate", type=float, default=0.0049)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_mistral_multi_head_config(args)
    summary = run_mistral_multi_head_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"final_summary={Path(config.out_dir) / 'final_summary.json'}")
    print(f"sweep_summary={Path(config.out_dir) / 'multi_head_sweep_summary.json'}")
    print(f"verdict={Path(config.out_dir) / 'multi_head_verdict.txt'}")
    print(f"gate_stats={Path(config.out_dir) / 'gate_stats'}")
    print(f"plots={Path(config.out_dir) / 'plots'}")


if __name__ == "__main__":
    main()
