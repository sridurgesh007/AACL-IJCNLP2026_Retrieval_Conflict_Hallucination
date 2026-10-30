#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "scripts" / "63_run_popqa_intervention.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("popqa_intervention_base_mistral", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base PopQA intervention script: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_defaults() -> None:
    defaults = {
        "--model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "--input_path": "data/popqa_conflict_500_natural.jsonl",
        "--results_dir": "results/cross_model/mistral_popqa",
        "--pilot_examples": "100",
        "--max_items": "500",
    }
    argv = sys.argv[1:]
    for flag, value in defaults.items():
        if flag in argv or any(str(arg).startswith(f"{flag}=") for arg in argv):
            continue
        sys.argv.extend([flag, value])


def main() -> int:
    _ensure_defaults()
    module = _load_base()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
