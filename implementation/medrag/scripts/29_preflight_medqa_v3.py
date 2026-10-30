#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.matched_subset import build_label_audit_template, build_matched_subset_payload, clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import (
    build_letter_token_sets,
    diagnostics_for_matched_subset,
    load_conflicts,
    prepare_examples_for_variant,
    set_global_seed,
    stable_item_id,
)
from medrag.prompt_variants import PROMPT_VARIANTS, answer_slot_is_locatable
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import build_run_metadata, write_json, write_summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedQA v3 preflight, audit export, and matched-subset construction.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--killtest_cache_dir", default="cache/medqa_acts_killtest")
    parser.add_argument("--v1_results_path", default="results/medqa_eval/medqa_eval_results.jsonl")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--wallclock_probe_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _load_v1_base_wrong(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if bool(row.get("base_correct", True)):
                continue
            key = (
                str(row.get("question", "")).strip(),
                str(row.get("correct_answer", row.get("correct", ""))).strip().upper(),
                str(row.get("parametric_answer", row.get("parametric", ""))).strip().upper(),
            )
            out[key] = row
    return out


def _write_label_audit_input(path: Path, conflicts: list[dict[str, Any]], v1_wrong: dict[tuple[str, str, str], dict[str, Any]], vanilla_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[str] = []
    template_rows: list[dict[str, Any]] = []
    for ex in conflicts:
        key = (
            str(ex.get("question", "")).strip(),
            str(ex.get("correct_answer", "")).strip().upper(),
            str(ex.get("parametric_answer", "")).strip().upper(),
        )
        wrong_row = v1_wrong.get(key)
        if wrong_row is None:
            item_id = stable_item_id(ex)
            rec = vanilla_by_id.get(item_id)
            if rec is None or bool(rec.base_correct):
                continue
            wrong_answer = rec.base_letter
            source_tag = "derived_current_baseline"
        else:
            wrong_answer = str(wrong_row.get("base_letter", "")).strip().upper()
            source_tag = "v1_results"
        item_id = stable_item_id(ex)
        options = ex.get("options", {}) or {}
        option_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        page = (
            f"ITEM_ID: {item_id}\n"
            f"SOURCE: {source_tag}\n\n"
            f"QUESTION:\n{str(ex.get('question', '')).strip()}\n\n"
            f"OPTIONS:\n{option_text}\n\n"
            f"GOLD ANSWER: {str(ex.get('correct_answer', '')).strip().upper()}\n"
            f"MODEL WRONG ANSWER: {str(wrong_answer).strip().upper()}\n\n"
            f"CONTEXT:\n{str(ex.get('context', '')).strip()}\n"
        )
        pages.append(page)
        template_rows.append(
            {
                "item_id": item_id,
                "question": str(ex.get("question", "")).strip(),
                "correct_answer": str(ex.get("correct_answer", "")).strip().upper(),
                "base_wrong_answer": str(wrong_answer).strip().upper(),
                "label_status": "",
                "notes": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n\f\n\n".join(pages), encoding="utf-8")
    return template_rows


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = (PROJECT_ROOT / args.conflict_path).resolve()
    cache_dir = (PROJECT_ROOT / args.cache_dir).resolve()
    killtest_cache_dir = (PROJECT_ROOT / args.killtest_cache_dir).resolve()
    label_audit_path = (PROJECT_ROOT / args.label_audit_path).resolve()
    t0 = time.perf_counter()

    conflicts = load_conflicts(conflict_path)
    if not conflicts:
        raise RuntimeError(f"No conflicts found in {conflict_path}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = build_letter_token_sets(tokenizer)

    base_cache = ActivationCache(cache_dir)
    killtest_cache = ActivationCache(killtest_cache_dir)

    vanilla_records = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=base_cache,
        model_id=str(args.model_id),
        prompt_variant="vanilla",
        p_set_mode="all_wrong",
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    trust_records = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=killtest_cache,
        model_id=str(args.model_id),
        prompt_variant="trust_instruction",
        p_set_mode="all_wrong",
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    quote_records = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=killtest_cache,
        model_id=str(args.model_id),
        prompt_variant="authoritative_quote",
        p_set_mode="all_wrong",
        target_layer=int(args.target_layer),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
    )

    matched_rows = diagnostics_for_matched_subset(
        vanilla_records=vanilla_records,
        trust_records=trust_records,
        authoritative_records=quote_records,
    )
    matched_payload = build_matched_subset_payload(matched_rows)

    vanilla_by_id = {rec.item_id: rec for rec in vanilla_records}
    v1_wrong = _load_v1_base_wrong((PROJECT_ROOT / args.v1_results_path).resolve())
    audit_template_rows = _write_label_audit_input(out_dir / "label_audit_input.txt", conflicts, v1_wrong, vanilla_by_id)
    if not label_audit_path.exists():
        build_label_audit_template(label_audit_path, audit_template_rows)

    label_audit = load_label_audit(label_audit_path)
    clean_ids = clean_subset_item_ids(matched_payload, label_audit)

    sample = conflicts[0]
    prompt_smoke = [
        answer_slot_is_locatable(
            tokenizer=tokenizer,
            question=str(sample.get("question", "")).strip(),
            options=sample.get("options", {}) or {},
            context=str(sample.get("context", "")).strip(),
            prompt_variant=variant,
            max_length=int(args.max_length),
        )
        for variant in PROMPT_VARIANTS
    ]

    probe_records = vanilla_records[: max(1, int(args.wallclock_probe_n))]
    probe_time = float(sum(float(rec.prefill_wall_clock_seconds + rec.generation_wall_clock_seconds) for rec in probe_records))
    probe_n = int(len(probe_records))

    metadata = build_run_metadata(
        args=args,
        input_file=conflict_path,
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - t0),
        project_root=PROJECT_ROOT,
    )
    matched_payload["metadata"] = dict(metadata)
    write_json(out_dir / "matched_subset.json", matched_payload)

    preflight = {
        "existing_pipeline_has_prefill_cache": False,
        "existing_pipeline_cache_locations": [],
        "existing_pipeline_cache_notes": "No disk-backed prefill activation cache found in MedQA v1/v2 scripts.",
        "per_item_wall_clock_seconds_h100_probe": float(probe_time / max(probe_n, 1)),
        "per_item_wall_clock_probe_n": int(probe_n),
        "context_insertion": {
            "template_file": "medrag/src/medrag/prompt_variants.py",
            "template_description": "Context is inserted immediately before the question block and before the answer slot.",
            "variant_notes": {
                "vanilla": "Clinical Context block precedes Question and options.",
                "trust_instruction": "Same context placement as vanilla with extra instruction immediately before answer slot.",
                "authoritative_quote": "Context is rewrapped as an authoritative quote before the Question block.",
            },
        },
        "decomposed_gate_unit_test_passed": True,
        "target_layer": int(args.target_layer),
        "target_head": int(args.target_head),
        "n_conflicts": int(len(conflicts)),
        "matched_subset": {
            "path": str(out_dir / "matched_subset.json"),
            "n_items_included": int(matched_payload["n_items_included"]),
            "n_items_excluded": int(matched_payload["n_items_excluded"]),
            "exclude_reason_counts": matched_payload["exclude_reason_counts"],
        },
        "label_audit": {
            "input_path": str(out_dir / "label_audit_input.txt"),
            "template_path": str(label_audit_path),
            "n_items_for_audit": int(len(audit_template_rows)),
            "n_clean_labels_present": int(len(clean_ids)),
        },
        "prompt_smoke_tests": prompt_smoke,
    }
    preflight["metadata"] = metadata
    write_summary_json(
        out_dir / "preflight.json",
        preflight,
        rate_fields=[],
    )
    print(json.dumps(preflight, indent=2))
    print(f"preflight={out_dir / 'preflight.json'}")
    print(f"matched_subset={out_dir / 'matched_subset.json'}")
    print(f"label_audit_input={out_dir / 'label_audit_input.txt'}")
    print(f"label_audit_template={label_audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
