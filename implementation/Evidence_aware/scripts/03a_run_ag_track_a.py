from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.ag_erha.track_a import load_baseline_label_map, run_track_a
from evidence_aware.ag_erha.utils import load_cache_rows, normalize_example_rows, prepare_examples, seed_everything
from evidence_aware.phase2.modeling import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AG-ERHA Track A only (sanity + tau sweep).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase2_main_eval_dir", default="results/phase2/main_eval")
    parser.add_argument("--out_dir", default="results/ag_erha")
    parser.add_argument("--tau_grid", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau_grid = [float(x.strip()) for x in str(args.tau_grid).split(",") if x.strip()]
    if not tau_grid:
        raise ValueError("tau_grid cannot be empty.")

    phase2_main_eval_dir = Path(args.phase2_main_eval_dir)
    baseline_map = load_baseline_label_map(phase2_main_eval_dir)
    eval_ids = set(int(x) for x in baseline_map.keys())

    seed_everything(int(args.seed))
    rows = normalize_example_rows(load_cache_rows(Path(args.cache_path)))
    eval_rows = [r for r in rows if int(r.id) in eval_ids]

    model, tokenizer, device = load_model_and_tokenizer(str(args.model_id), str(args.device), str(args.torch_dtype))
    eval_examples = prepare_examples(eval_rows, tokenizer, max_length=int(args.max_length))

    summary = run_track_a(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_examples,
        tau_grid=tau_grid,
        beta=float(args.beta),
        max_new_tokens=int(args.max_new_tokens),
        phase2_main_eval_dir=phase2_main_eval_dir,
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(summary, indent=2))
    print(f"track_a_out={args.out_dir}")


if __name__ == "__main__":
    main()
