from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.generalization import parse_generalization_config, run_generalization_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full cross-model generalization pipeline from generalization.md")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model_shortname", default="mistral")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--out_root", default="results")
    parser.add_argument("--llama_results_dir", default="results/llama")

    parser.add_argument("--phase1_start_idx", type=int, default=2700)
    parser.add_argument("--phase1_end_idx", type=int, default=3199)
    parser.add_argument("--phase2_val_start_idx", type=int, default=3200)
    parser.add_argument("--phase2_val_end_idx", type=int, default=3399)
    parser.add_argument("--phase2_eval_start_idx", type=int, default=3400)
    parser.add_argument("--phase2_eval_end_idx", type=int, default=5399)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--label_max_new_tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument("--force_prompt_style", default="auto", choices=["auto", "chat", "raw"])
    parser.add_argument("--baseline_probe_examples", type=int, default=50)

    parser.add_argument("--mg_tau_margin_grid", default="-2.0,-1.0,-0.5,0.0,0.5")
    parser.add_argument("--mg_sanity_taus", default="-999.0,999.0")
    parser.add_argument("--mg_first_token_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mg_bootstrap_first_token_margin", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mg_sanity_tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = parse_generalization_config(args)
    summary = run_generalization_pipeline(config)
    print(json.dumps(summary, indent=2))
    mistral_root = Path(config.out_root) / config.model_shortname
    print(f"final_summary={mistral_root / 'generalization_final_summary.json'}")
    print(f"phase1={mistral_root / 'phase1'}")
    print(f"phase2={mistral_root / 'phase2'}")
    print(f"track_b={mistral_root / 'track_b'}")
    print(f"mg_erha={mistral_root / 'mg_erha'}")
    print(f"generalization={Path(config.out_root) / 'generalization'}")


if __name__ == "__main__":
    main()
