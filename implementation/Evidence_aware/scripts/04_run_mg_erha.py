from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.mg_erha import parse_mg_erha_config, run_mg_erha_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MG-ERHA sweep from mgerha.md")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase2_main_eval_dir", default="results/phase2/main_eval")
    parser.add_argument("--track_b_margin_path", default="results/track_b/margin_trajectories.jsonl")
    parser.add_argument("--ag_track_a_dir", default="results/ag_erha")
    parser.add_argument("--out_dir", default="results/mg_erha")

    parser.add_argument("--tau_margin_grid", default="-2.0,-1.0,-0.5,0.0,0.5")
    parser.add_argument("--sanity_taus", default="-999.0,999.0")
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sanity_tolerance", type=float, default=1e-3)
    parser.add_argument("--init_prev_margin", type=float, default=0.0)
    parser.add_argument("--precompute_missing_sets", type=int, default=1, choices=[0, 1])
    parser.add_argument("--strict_track_b_sets", type=int, default=0, choices=[0, 1])
    parser.add_argument("--first_token_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bootstrap_first_token_margin", type=int, default=0, choices=[0, 1])
    parser.add_argument("--target_layer_idx", type=int, default=27)
    parser.add_argument("--target_head_idx", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_mg_erha_config(args)
    summary = run_mg_erha_pipeline(config)
    print(json.dumps(summary, indent=2))
    print(f"final_summary={config.out_dir / 'final_summary.json'}")
    print(f"sweep_summary={config.out_dir / 'mg_erha_sweep_summary.json'}")
    print(f"verdict={config.out_dir / 'mg_erha_verdict.txt'}")
    print(f"plots={config.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
