from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.multi_head import parse_multi_head_config, run_multi_head_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layer-adaptive multi-head MG-ERHA experiment from multi_head.md"
    )
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase1_ld_path", default="results/phase1_diagnostics/per_example_ld.jsonl")
    parser.add_argument("--phase2_main_eval_dir", default="results/phase2/main_eval")
    parser.add_argument("--ag_track_a_dir", default="results/ag_erha")
    parser.add_argument("--out_dir", default="results/multi_head_mg_erha")

    parser.add_argument("--target_layers", default="20,21,22,23,24,25,26,27")
    parser.add_argument("--late_layers", default="24,25,26,27")
    parser.add_argument("--multi_betas", default="2.0,3.0,5.0")
    parser.add_argument("--tau_values", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--single_beta", type=float, default=5.0)

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
    config = parse_multi_head_config(args)
    summary = run_multi_head_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"final_summary={config.out_dir / 'final_summary.json'}")
    print(f"sweep_summary={config.out_dir / 'multi_head_sweep_summary.json'}")
    print(f"top_heads={config.out_dir / 'top_heads_per_layer.json'}")
    print(f"verdict={config.out_dir / 'multi_head_verdict.txt'}")
    print(f"gate_stats={config.out_dir / 'gate_stats'}")
    print(f"plots={config.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
