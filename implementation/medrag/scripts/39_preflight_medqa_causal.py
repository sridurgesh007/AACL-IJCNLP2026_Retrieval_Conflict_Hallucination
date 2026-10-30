#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import null_patch_test
from medrag.causal_localization import build_metadata, build_patching_pairs, load_medqa_causal_inputs
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight for MedQA causal localization.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/causal")
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--n_dest_target", type=int, default=50)
    parser.add_argument("--min_dest_required", type=int, default=40)
    parser.add_argument("--null_test_examples", type=int, default=10)
    parser.add_argument("--null_test_atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _discover_patching_utility() -> tuple[bool, str | None, list[str]]:
    utility_path = PROJECT_ROOT / "src" / "medrag" / "activation_patch.py"
    if not utility_path.exists():
        return False, None, ["per-head o_proj-input patching utility missing"]
    text = utility_path.read_text(encoding="utf-8")
    missing: list[str] = []
    for required in ("cache_head_output", "run_with_patch", "null_patch_test"):
        if required not in text:
            missing.append(f"missing function: {required}")
    return True, str(utility_path.resolve()), missing


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    utility_exists, utility_location, missing_capabilities = _discover_patching_utility()
    if not utility_exists:
        raise RuntimeError("Activation patching utility is missing.")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    inputs = load_medqa_causal_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=PROJECT_ROOT / args.conflict_path,
        matched_subset_path=PROJECT_ROOT / args.matched_subset_path,
        label_audit_path=PROJECT_ROOT / args.label_audit_path,
        h4_results_path=PROJECT_ROOT / args.h4_results_path,
        cache_dir=PROJECT_ROOT / args.cache_dir,
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )

    pairs = build_patching_pairs(
        causal_items=inputs.causal_items,
        n_dest_target=int(args.n_dest_target),
        min_dest_required=int(args.min_dest_required),
    )
    write_json(out_dir / "patching_pairs.json", {"pairs": pairs})

    null_tests = []
    for rec in inputs.prepared_records[: max(1, int(args.null_test_examples))]:
        null_tests.append(
            {
                "item_id": str(rec.item_id),
                **null_patch_test(
                    model=model,
                    prompt_ids=rec.prompt_ids.to(device),
                    layer=int(args.target_layer),
                    head=int(args.target_head),
                    position=-1,
                    atol=float(args.null_test_atol),
                ),
            }
        )
    null_passed = all(bool(row.get("passed", False)) for row in null_tests)
    if not null_passed:
        write_json(
            out_dir / "preflight.json",
            {
                "patching_utility_exists": bool(utility_exists),
                "location": utility_location,
                "missing_capabilities": missing_capabilities,
                "null_patch_test": {
                    "all_passed": False,
                    "examples": null_tests,
                },
            },
        )
        raise RuntimeError(f"Null patch test failed. See {out_dir / 'preflight.json'}")

    metadata = build_metadata(
        args=args,
        input_files=[
            PROJECT_ROOT / args.conflict_path,
            PROJECT_ROOT / args.matched_subset_path,
            PROJECT_ROOT / args.label_audit_path,
            PROJECT_ROOT / args.h4_results_path,
        ],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    preflight = {
        "patching_utility_exists": bool(utility_exists),
        "location": utility_location,
        "missing_capabilities": missing_capabilities,
        "pairing": {
            "n_pairs": int(len(pairs)),
            "n_dest_target_requested": int(args.n_dest_target),
            "min_dest_required_requested": int(args.min_dest_required),
            "min_dest_requirement_met": bool(len(pairs) >= int(args.min_dest_required)),
            "n_dest_clean": int(sum(1 for pair in pairs if str(pair["dest_subset"]) == "clean")),
            "n_dest_matched_extra": int(sum(1 for pair in pairs if str(pair["dest_subset"]) == "matched_extra")),
            "warning": (
                None
                if len(pairs) >= int(args.min_dest_required)
                else (
                    f"Only {len(pairs)} valid destination items were available under the current "
                    f"base-wrong + conflict-detected + matched/clean filters; continuing with all "
                    f"available pairs instead of the requested minimum {int(args.min_dest_required)}."
                )
            ),
            "pairs_path": str((out_dir / "patching_pairs.json").resolve()),
        },
        "source_pool": {
            "n_base_correct_gate_off_matched": int(
                sum(1 for item in inputs.causal_items if item.in_matched_subset and item.base_correct and not item.conflict_detected)
            ),
        },
        "dest_pool": {
            "n_base_wrong_gate_on_clean": int(
                sum(1 for item in inputs.causal_items if item.in_clean_subset and (not item.base_correct) and item.conflict_detected)
            ),
            "n_base_wrong_gate_on_matched": int(
                sum(1 for item in inputs.causal_items if item.in_matched_subset and (not item.base_correct) and item.conflict_detected)
            ),
        },
        "null_patch_test": {
            "all_passed": bool(null_passed),
            "examples": null_tests,
        },
        "metadata": metadata,
    }
    write_json(out_dir / "preflight.json", preflight)

    print(f"preflight={out_dir / 'preflight.json'}")
    print(f"pairs={out_dir / 'patching_pairs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
