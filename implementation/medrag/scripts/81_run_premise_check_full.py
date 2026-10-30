#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.data import load_examples
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import build_counterfact_evidence_text, build_metadata, project_path
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json


def _load_v2_module():
    path = PROJECT_ROOT / "scripts" / "79_run_counterfact_premise_check_v2.py"
    spec = importlib.util.spec_from_file_location("premise_check_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen v2 premise logic from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_v2_module()


# Baseline-label counts for the frozen L27/H20 tau=0.0 OV file.
# Note: 1077 is the other->other transition count, not baseline_label="other".
EXPECTED_OV_COUNTS = {"other": 1110, "faithful": 611, "hallucinated": 279}
EXPECTED_N_OV = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-set CounterFact belief verification, n=2000.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument(
        "--ov_results_path",
        default=(
            "../Evidence_aware/results/true_multi_head/exp_b_corrected/gate_stats/"
            "gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl"
        ),
    )
    parser.add_argument(
        "--evidence_mode",
        default="relation_or_question_answer",
        choices=["cache", "question_answer", "relation_or_question_answer"],
    )
    parser.add_argument("--out_dir", default="results/rebuttal/belief_verification_full")
    parser.add_argument("--max_items", type=int, default=0, help="0 means all OV ids.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--c1_max_new_tokens", type=int, default=16)
    parser.add_argument("--prompt_diff_items", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _verify_ov(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(path)
    ids = [int(row["example_id"]) for row in rows if "example_id" in row]
    label_counts = Counter(str(row.get("baseline_label")) for row in rows)
    config_counts = Counter(
        (
            row.get("tau_margin"),
            row.get("alpha"),
            row.get("lambda_max"),
        )
        for row in rows
    )
    verification = {
        "ov_results_path": str(path),
        "n_rows": len(rows),
        "n_unique_example_id": len(set(ids)),
        "baseline_label_counts": dict(label_counts),
        "expected_n_rows": EXPECTED_N_OV,
        "expected_baseline_label_counts": EXPECTED_OV_COUNTS,
        "config_counts": {str(k): v for k, v in config_counts.items()},
        "passed": True,
        "failures": [],
    }
    if len(rows) != EXPECTED_N_OV:
        verification["failures"].append(f"Expected {EXPECTED_N_OV} OV rows, got {len(rows)}")
    if len(set(ids)) != EXPECTED_N_OV:
        verification["failures"].append(f"Expected {EXPECTED_N_OV} unique example_id values, got {len(set(ids))}")
    for key, expected in EXPECTED_OV_COUNTS.items():
        actual = int(label_counts.get(key, 0))
        if actual != expected:
            verification["failures"].append(f"baseline_label {key}: expected {expected}, got {actual}")
    verification["passed"] = not verification["failures"]
    if not verification["passed"]:
        raise RuntimeError("OV input verification failed: " + "; ".join(verification["failures"]))
    return rows, verification


def _select_examples(cache_path: Path, ov_rows: list[dict[str, Any]], max_items: int) -> tuple[list[Any], dict[str, Any]]:
    examples = load_examples(cache_path)
    by_id = {int(getattr(ex, "id")): ex for ex in examples}
    ids = [int(row["example_id"]) for row in ov_rows]
    if max_items > 0:
        ids = ids[:max_items]
    missing = [item_id for item_id in ids if item_id not in by_id]
    manifest = {
        "cache_path": str(cache_path),
        "cache_examples": len(examples),
        "requested_ids": len(ids),
        "selected_examples": len(ids) - len(missing),
        "missing_ids": len(missing),
        "missing_id_sample": missing[:20],
        "first_selected_ids": ids[:20],
    }
    if missing:
        raise RuntimeError(f"CounterFact cache missing OV ids: n={len(missing)} sample={missing[:20]}")
    return [by_id[item_id] for item_id in ids], manifest


def _prefill_logits(model, tokenizer, prompt: str, max_length: int, device) -> tuple[torch.Tensor, int, str]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask"))
    logits = out.logits[0, -1, :].detach().float().cpu()
    argmax_id = int(torch.argmax(logits).item())
    argmax_token = tokenizer.decode([argmax_id], skip_special_tokens=False)
    return logits, argmax_id, argmax_token


def _summary(per_item: list[dict[str, Any]], dropped: list[dict[str, Any]], args: argparse.Namespace, started: float, cache_path: Path, ov_path: Path) -> dict[str, Any]:
    n = len(per_item)
    ranks = [int(row["rank_target_true"]) for row in per_item if row.get("rank_target_true") is not None]
    p_true = [float(row["p_target_true"]) for row in per_item]
    margin_values = [float(row["logit_margin_c1"]) for row in per_item if row.get("logit_margin_c1") is not None]
    bucket_counts = {
        "rank_1": sum(1 for row in per_item if row["rank_bucket"] == "rank_1"),
        "rank_2_5": sum(1 for row in per_item if row["rank_bucket"] == "rank_2_5"),
        "rank_6_20": sum(1 for row in per_item if row["rank_bucket"] == "rank_6_20"),
        "rank_21_100": sum(1 for row in per_item if row["rank_bucket"] == "rank_21_100"),
        "rank_over_100": sum(1 for row in per_item if row["rank_bucket"] == "rank_over_100"),
        "missing": sum(1 for row in per_item if row["rank_bucket"] == "missing"),
    }
    return {
        "n": int(n),
        "n_dropped": int(len(dropped)),
        "agreement": {
            "agree_first_token": V2._frac(sum(1 for row in per_item if bool(row["agree_first_token"])), n),
            "agree_token_set": V2._frac(sum(1 for row in per_item if bool(row["agree_token_set"])), n),
            "agree_string": V2._frac(sum(1 for row in per_item if bool(row["agree_string"])), n),
            "agree_substring": V2._frac(sum(1 for row in per_item if bool(row["agree_substring"])), n),
        },
        "belief": {
            "frac_rank_1": V2._frac(bucket_counts["rank_1"], n),
            "frac_rank_2_5": V2._frac(bucket_counts["rank_2_5"], n),
            "frac_rank_6_20": V2._frac(bucket_counts["rank_6_20"], n),
            "frac_rank_21_100": V2._frac(bucket_counts["rank_21_100"], n),
            "frac_rank_over_100": V2._frac(bucket_counts["rank_over_100"], n),
            "n_rank_missing": int(bucket_counts["missing"]),
            "median_p_target_true": float(median(p_true)) if p_true else None,
            "median_rank_target_true": int(median(ranks)) if ranks else None,
            "frac_logit_margin_c1_positive": V2._frac(sum(1 for value in margin_values if value > 0.0), len(margin_values)),
            "rank_bucket_counts": bucket_counts,
        },
        "metadata": build_metadata(
            args=args,
            input_files=[cache_path, ov_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=REPO_ROOT,
        ),
    }


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = out_dir / "prediction.json"
    if not prediction_path.exists():
        write_json(
            prediction_path,
            {
                "predicted_n_hallucinated_no_belief": 0,
                "predicted_fisher_p_lt": 1e-12,
                "frozen_at_utc": _now_iso(),
                "note": "prediction recorded before full-set execution",
            },
        )

    ov_path = project_path(PROJECT_ROOT, str(args.ov_results_path))
    ov_rows, ov_verification = _verify_ov(ov_path)
    cache_path = project_path(PROJECT_ROOT, str(args.cache_path))
    examples, cache_manifest = _select_examples(cache_path, ov_rows, int(args.max_items))
    input_verification = ov_verification | {"cache_manifest": cache_manifest, "passed": True}
    write_json(out_dir / "input_verification.json", input_verification)

    config = {
        "created_at_utc": _now_iso(),
        "purpose": "Full-set CounterFact belief verification using frozen v2 premise logic.",
        "source_logic": "79_run_counterfact_premise_check_v2.py",
        "model_id": str(args.model_id),
        "cache_path": str(cache_path),
        "ov_results_path": str(ov_path),
        "out_dir": str(args.out_dir),
        "max_items": int(args.max_items),
        "checkpoint_every": int(args.checkpoint_every),
        "frozen_before_model_load": True,
        "do_sample": False,
    }
    write_json(out_dir / "config.json", config)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    prefill_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    prompt_diff_rows: list[dict[str, Any]] = []
    ckpt_every = max(1, int(args.checkpoint_every))

    for idx, ex in enumerate(examples, start=1):
        item_id = str(getattr(ex, "id", idx - 1))
        question = str(getattr(ex, "question", "")).strip()
        target_true = str(getattr(ex, "answer_solid", "")).strip()
        target_new = str(getattr(ex, "answer_liquid", "")).strip()
        if not question or not target_true or not target_new:
            dropped.append({"item_id": item_id, "reason": "missing_question_or_answer"})
            continue

        evidence_text = build_counterfact_evidence_text(ex, str(args.evidence_mode))
        c1_prompt = V2._chat_prompt(tokenizer, V2._c1_user_text(question))
        c2_prompt = V2._chat_prompt(tokenizer, V2._c2_user_text(evidence_text, question))
        if len(prompt_diff_rows) < int(args.prompt_diff_items):
            prompt_diff_rows.append({"item_id": item_id, "c1_prompt": c1_prompt, "c2_prompt": c2_prompt})

        logits, argmax_id, argmax_token = _prefill_logits(model, tokenizer, c1_prompt, int(args.max_length), device)
        true_stats = V2._answer_token_stats(tokenizer, logits, target_true)
        new_stats = V2._answer_token_stats(tokenizer, logits, target_new)
        true_selected = true_stats["selected"]
        new_selected = new_stats["selected"]
        logit_margin = None
        if true_selected["logit"] is not None and new_selected["logit"] is not None:
            logit_margin = float(true_selected["logit"]) - float(new_selected["logit"])

        prefill_rows.append(
            {
                "item_id": item_id,
                "query": question,
                "target_true": target_true,
                "target_new": target_new,
                "c1_prompt": c1_prompt,
                "c2_prompt": c2_prompt,
                "p_target_true": float(true_selected["prob"]),
                "p_target_new": float(new_selected["prob"]),
                "rank_target_true": true_selected["rank"],
                "rank_target_new": new_selected["rank"],
                "logit_margin_c1": logit_margin,
                "argmax_token": str(argmax_token),
                "argmax_token_id": int(argmax_id),
                "target_true_token": true_stats,
                "target_new_token": new_stats,
                "rank_bucket": V2._rank_bucket(true_selected["rank"]),
            }
        )
        if idx % ckpt_every == 0:
            _write_jsonl(out_dir / "premise_partial.jsonl", prefill_rows)
            write_json(out_dir / "premise_partial_manifest.json", {"completed_prefill": len(prefill_rows), "updated_at_utc": _now_iso()})

    _write_jsonl(out_dir / "premise_partial.jsonl", prefill_rows)
    (out_dir / "prompt_diff_check.txt").write_text(V2._build_prompt_diff(prompt_diff_rows), encoding="utf-8")

    per_item: list[dict[str, Any]] = []
    for idx, row in enumerate(prefill_rows, start=1):
        c1_generation, c1_ids, _logits, _argmax_id, _argmax_token = V2._generate_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=row["c1_prompt"],
            max_length=int(args.max_length),
            max_new_tokens=int(args.c1_max_new_tokens),
            device=device,
        )
        extracted, extraction_failed = V2._extract_answer_span(c1_generation)
        gold_first_token_text = str(row["target_true_token"]["selected"].get("token_text") or "")
        extracted_first_token_id = V2._first_token_id(tokenizer, extracted, leading_space=True)
        extracted_first_token_text = (
            "" if extracted_first_token_id is None else tokenizer.decode([int(extracted_first_token_id)], skip_special_tokens=False)
        )
        norm_true = V2._normalize_text(row["target_true"])
        norm_generation = V2._normalize_text(c1_generation)
        final = row | {
            "c1_generation": c1_generation,
            "c1_generated_token_ids": c1_ids,
            "extracted_answer": extracted,
            "extraction_failed": bool(extraction_failed),
            "agree_first_token": V2._normalize_text(gold_first_token_text) == V2._normalize_text(extracted_first_token_text),
            "agree_token_set": bool(V2.token_set_from_text(tokenizer, row["target_true"]) & V2.token_set_from_text(tokenizer, extracted)),
            "agree_string": V2._normalize_text(extracted) == V2._normalize_text(row["target_true"]),
            "agree_substring": bool(norm_true and norm_true in norm_generation),
        }
        per_item.append(final)
        if idx % ckpt_every == 0:
            _write_jsonl(out_dir / "per_item.jsonl", per_item)

    _write_jsonl(out_dir / "per_item.jsonl", per_item)
    _write_jsonl(out_dir / "dropped_items.jsonl", dropped)
    write_json(out_dir / "summary.json", _summary(per_item, dropped, args, started, cache_path, ov_path))
    readme = (
        "# Full CounterFact Belief Verification\n\n"
        f"Completed {len(per_item)} rows with frozen v2 premise logic. "
        "Use `82_run_stratification_full.py` for the primary outcome join and tests.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"per_item={out_dir / 'per_item.jsonl'}")
    print(f"summary={out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
