#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.final_paper import (
    build_cross_model_metadata,
    compare_condition_summaries,
    paired_mcnemar_from_rows,
    summary_from_prepared_records,
)
from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import build_letter_token_sets, load_conflicts, prepare_examples_for_variant, set_global_seed
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import read_json, write_json, write_jsonl, write_summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add prompt-only C4/C5 baselines to an existing MedQA cross-model C1/C2/C3 run."
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--base_results_dir", required=True, help="Existing C1/C2/C3 run directory.")
    parser.add_argument("--out_dir", required=True, help="New C1-C5 output directory. Must differ from base_results_dir.")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--target_layer", type=int, default=0, help="Layer used only for cache extraction.")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--c1_prompt_variant", default="vanilla")
    parser.add_argument("--c4_prompt_variant", default="authoritative_quote")
    parser.add_argument(
        "--c5_prompt_variant",
        default="trust_instruction",
        help="Prompt wrapper used after inserting the explicit answer into the context.",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _condition_rate_fields() -> list[str]:
    return [
        "baseline_accuracy",
        "ft_accuracy",
        "ft_accuracy_clean_subset",
        "restoration_rate",
        "harm_rate",
        "restoration_rate_clean_subset",
        "harm_rate_clean_subset",
        "gate_fire_rate_base_wrong",
        "gate_fire_rate_base_correct",
        "restoration_rate_param_wrong",
        "restoration_rate_other_wrong",
    ]


def _copy_existing_condition_dirs(base_results_dir: Path, out_dir: Path) -> None:
    for cond_id in ("C1", "C2", "C3"):
        src = base_results_dir / cond_id
        if not src.exists():
            raise FileNotFoundError(f"Missing existing {cond_id} directory: {src}")
        dst = out_dir / cond_id
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _condition_rows_from_c1_results(c1_results_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(c1_results_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "item_id": str(row["item_id"]),
                "question": str(row.get("question", "")),
                "correct": str(row.get("correct", "")),
                "parametric": str(row.get("parametric", "")),
                "condition_letter": row.get("ft_letter", row.get("base_letter")),
                "condition_correct": bool(row.get("ft_correct", row.get("base_correct", False))),
                "in_matched_subset": bool(row.get("in_matched_subset", False)),
                "in_clean_subset": bool(row.get("in_clean_subset", False)),
                "skipped": bool(row.get("skipped", False)),
                "prompt_variant": str(row.get("prompt_variant", "vanilla")),
            }
        )
    if not out:
        raise RuntimeError(f"No C1 rows found in {c1_results_path}")
    return out


def _explicit_answer_conflicts(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in conflicts:
        ex = dict(row)
        correct_letter = str(ex.get("correct_answer", "")).strip().upper()
        options = ex.get("options", {}) or {}
        option_text = ""
        if isinstance(options, dict):
            option_text = str(options.get(correct_letter, "")).strip()
        if option_text:
            injection = f"According to reference sources, the correct answer is {correct_letter}: {option_text}."
        else:
            injection = f"According to reference sources, the correct answer is {correct_letter}."
        context = str(ex.get("context", "")).strip()
        ex["context"] = f"{injection} {context}".strip()
        ex["c5_explicit_answer_injection"] = True
        out.append(ex)
    return out


def _prepare_condition(
    *,
    conflicts: list[dict[str, Any]],
    model,
    tokenizer,
    device,
    model_id: str,
    prompt_variant: str,
    p_set_mode: str,
    target_layer: int,
    cache_dir: Path,
    max_length: int,
    max_new_tokens: int,
    letter_token_sets: dict[str, set[int]],
    matched_ids: set[str],
    clean_ids: set[str],
    c1_condition_rows: list[dict[str, Any]],
    condition_id: str,
    condition_name: str,
    apply_intervention: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = prepare_examples_for_variant(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache=ActivationCache(cache_dir),
        model_id=model_id,
        prompt_variant=prompt_variant,
        p_set_mode=p_set_mode,
        target_layer=int(target_layer),
        max_length=int(max_length),
        max_new_tokens=int(max_new_tokens),
        letter_token_sets=letter_token_sets,
    )
    condition_base_summary, condition_rows = summary_from_prepared_records(
        prepared,
        matched_ids=matched_ids,
        clean_ids=clean_ids,
    )
    transition_summary, transition_rows = compare_condition_summaries(
        c1_condition_rows,
        condition_rows,
        wall_clock_seconds=float(condition_base_summary["wall_clock_seconds"]),
    )
    transition_summary.update(
        {
            "condition_id": str(condition_id),
            "condition_name": str(condition_name),
            "prompt_variant": str(prompt_variant),
            "apply_intervention": bool(apply_intervention),
            "condition_accuracy": float(condition_base_summary["accuracy"]),
            "condition_accuracy_ci95_wilson": list(condition_base_summary["accuracy_ci95_wilson"]),
            "condition_accuracy_clean_subset": float(condition_base_summary["accuracy_clean_subset"]),
            "condition_accuracy_clean_subset_ci95_wilson": list(
                condition_base_summary["accuracy_clean_subset_ci95_wilson"]
            ),
        }
    )
    return transition_summary, transition_rows, condition_rows


def _common_condition_fields(base_summary: dict[str, Any], args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "model_id": str(args.model_id),
        "model_name": str(args.model_name),
        "architecture": str(args.architecture),
        "metadata": metadata,
    }
    for key in (
        "bootstrap_space",
        "site_mode",
        "site_description",
        "target_layer",
        "target_head",
        "selected_tau_margin",
        "selected_lambda_pos",
        "selected_lambda_neg",
        "alpha",
        "pilot_margin_auc",
    ):
        if key in base_summary:
            fields[key] = base_summary[key]
    return fields


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    base_results_dir = _project_path(args.base_results_dir)
    out_dir = _project_path(args.out_dir)
    cache_dir = _project_path(args.cache_dir)
    if base_results_dir == out_dir:
        raise RuntimeError("--out_dir must differ from --base_results_dir so existing results are not modified.")
    if not (base_results_dir / "summary.json").exists():
        raise FileNotFoundError(f"Missing base summary: {base_results_dir / 'summary.json'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _copy_existing_condition_dirs(base_results_dir, out_dir)

    base_summary = read_json(base_results_dir / "summary.json")
    conflicts_path = _project_path(args.conflict_path)
    matched_subset_path = _project_path(args.matched_subset_path)
    label_audit_path = _project_path(args.label_audit_path)

    conflicts = load_conflicts(conflicts_path)
    matched_payload = read_json(matched_subset_path)
    matched_ids = set(str(x) for x in matched_payload.get("matched_item_ids", []))
    clean_ids = set(clean_subset_item_ids(matched_payload, load_label_audit(label_audit_path)))

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = build_letter_token_sets(tokenizer)
    target_layer = int(args.target_layer)
    if target_layer <= 0:
        target_layer = int(base_summary.get("target_layer", 0))
    if target_layer <= 0:
        raise RuntimeError("Could not infer target layer for cache extraction; pass --target_layer explicitly.")

    # Use the original C1 rows from the base run. Recomputing C1 can change a small
    # number of Qwen outputs and invalidates direct C3-vs-C4/C5 comparisons.
    c1_condition_rows = _condition_rows_from_c1_results(out_dir / "C1" / "results.jsonl")

    c4_summary, c4_rows, _ = _prepare_condition(
        conflicts=conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        prompt_variant=str(args.c4_prompt_variant),
        p_set_mode=str(args.p_set_mode),
        target_layer=target_layer,
        cache_dir=cache_dir / "C4_authoritative_quote",
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
        matched_ids=matched_ids,
        clean_ids=clean_ids,
        c1_condition_rows=c1_condition_rows,
        condition_id="C4",
        condition_name="Authoritative-context prompt baseline",
        apply_intervention=False,
    )

    c5_conflicts = _explicit_answer_conflicts(conflicts)
    write_jsonl(out_dir / "c5_explicit_answer_conflicts.jsonl", c5_conflicts)
    c5_summary, c5_rows, _ = _prepare_condition(
        conflicts=c5_conflicts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        prompt_variant=str(args.c5_prompt_variant),
        p_set_mode=str(args.p_set_mode),
        target_layer=target_layer,
        cache_dir=cache_dir / "C5_explicit_answer",
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        letter_token_sets=letter_token_sets,
        matched_ids=matched_ids,
        clean_ids=clean_ids,
        c1_condition_rows=c1_condition_rows,
        condition_id="C5",
        condition_name="Explicit-answer-in-context prompt baseline",
        apply_intervention=False,
    )
    c5_summary["explicit_answer_in_context"] = True

    metadata = build_cross_model_metadata(
        args=args,
        input_files=[
            conflicts_path,
            matched_subset_path,
            label_audit_path,
            base_results_dir / "summary.json",
        ],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=PROJECT_ROOT,
    )
    common_fields = _common_condition_fields(base_summary, args, metadata)
    c4_summary.update(common_fields)
    c5_summary.update(common_fields)

    for cond_id, summary, rows in (("C4", c4_summary, c4_rows), ("C5", c5_summary, c5_rows)):
        cond_dir = out_dir / cond_id
        cond_dir.mkdir(parents=True, exist_ok=True)
        write_summary_json(cond_dir / "summary.json", summary, rate_fields=_condition_rate_fields())
        write_jsonl(cond_dir / "results.jsonl", rows)

    c2_rows = _read_jsonl(out_dir / "C2" / "results.jsonl")
    c3_rows = _read_jsonl(out_dir / "C3" / "results.jsonl")
    aggregate = dict(base_summary)
    aggregate.update(
        {
            "c45_added": True,
            "c45_note": "C4/C5 are prompt-only baselines added in a separate output directory; existing C1/C2/C3 results were copied from base_results_dir.",
            "c45_baseline_source": "copied_base_C1_results",
            "base_results_dir": str(base_results_dir),
            "conditions": {
                **dict(base_summary.get("conditions", {})),
                "C4": c4_summary,
                "C5": c5_summary,
            },
            "mcnemar_c3_vs_c4": paired_mcnemar_from_rows(c3_rows, c4_rows),
            "mcnemar_c3_vs_c5": paired_mcnemar_from_rows(c3_rows, c5_rows),
            "mcnemar_c4_vs_c5": paired_mcnemar_from_rows(c4_rows, c5_rows),
            "mcnemar_c2_vs_c4": paired_mcnemar_from_rows(c2_rows, c4_rows),
            "mcnemar_c2_vs_c5": paired_mcnemar_from_rows(c2_rows, c5_rows),
            "metadata_c45": metadata,
        }
    )
    write_json(out_dir / "summary.json", aggregate)
    write_json(
        out_dir / "c45_comparison_row.json",
        {
            "model": str(args.model_name),
            "base_results_dir": str(base_results_dir),
            "c3_restoration_rate": float(aggregate["conditions"]["C3"]["restoration_rate"]),
            "c3_harm_rate": float(aggregate["conditions"]["C3"]["harm_rate"]),
            "c4_restoration_rate": float(c4_summary["restoration_rate"]),
            "c4_harm_rate": float(c4_summary["harm_rate"]),
            "c5_restoration_rate": float(c5_summary["restoration_rate"]),
            "c5_harm_rate": float(c5_summary["harm_rate"]),
            "mcnemar_c3_vs_c4": aggregate["mcnemar_c3_vs_c4"],
            "mcnemar_c3_vs_c5": aggregate["mcnemar_c3_vs_c5"],
        },
    )

    print(json.dumps(aggregate, indent=2))
    print(f"summary={out_dir / 'summary.json'}")
    print(f"c4={out_dir / 'C4' / 'summary.json'}")
    print(f"c5={out_dir / 'C5' / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
