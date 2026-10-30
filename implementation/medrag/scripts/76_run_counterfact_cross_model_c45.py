#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

import torch

from evidence_aware.phase2.common import build_prompt
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import (
    CounterFactPrepared,
    build_metadata,
    condition_summary,
    load_counterfact_prepared,
    prepare_baseline_rows,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl
from medrag.utils.stats import mcnemar_from_paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add prompt-only C4/C5 baselines to an existing CounterFact cross-model C2/C3 run."
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--base_results_dir", required=True, help="Existing CounterFact C2/C3 full-run directory.")
    parser.add_argument("--out_dir", required=True, help="New C2-C5 output directory. Must differ from base_results_dir.")
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument(
        "--evidence_mode",
        default="relation_or_question_answer",
        choices=["cache", "question_answer", "relation_or_question_answer"],
    )
    parser.add_argument("--max_items", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=256)
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


def _copy_existing_outputs(base_results_dir: Path, out_dir: Path) -> None:
    for name in ("summary.json", "results.jsonl", "baseline_rows.jsonl", "full_margin_auc.json"):
        src = base_results_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    src_figures = base_results_dir / "figures"
    if src_figures.exists():
        shutil.copytree(src_figures, out_dir / "figures_original", dirs_exist_ok=True)


def _authoritative_evidence(evidence: str) -> str:
    text = str(evidence).strip()
    return f"According to authoritative reference sources, the following evidence is correct: {text}"


def _explicit_answer_evidence(evidence: str, answer: str) -> str:
    text = str(evidence).strip()
    answer_text = str(answer).strip()
    return f"According to reference sources, the answer is {answer_text}. {text}".strip()


def _rewrite_records(
    *,
    records: list[CounterFactPrepared],
    tokenizer,
    max_length: int,
    mode: str,
) -> list[CounterFactPrepared]:
    rewritten: list[CounterFactPrepared] = []
    for rec in records:
        ex = rec.example
        if mode == "authoritative":
            evidence = _authoritative_evidence(str(ex.evidence))
        elif mode == "explicit_answer":
            evidence = _explicit_answer_evidence(str(ex.evidence), str(ex.answer_liquid))
        else:
            raise ValueError(f"Unsupported CounterFact prompt baseline mode: {mode}")
        prompt = build_prompt(evidence, str(ex.question))
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=int(max_length))
        ids = [int(x) for x in encoded.get("input_ids", [])]
        if not ids:
            continue
        new_ex = replace(ex, evidence=evidence, prompt=prompt, input_ids=ids)
        prompt_ids = torch.tensor(ids, dtype=torch.long)
        rewritten.append(CounterFactPrepared(example=new_ex, prompt_ids=prompt_ids, prompt_length=int(prompt_ids.numel())))
    if len(rewritten) != len(records):
        raise RuntimeError(f"Prompt rewrite dropped records: kept {len(rewritten)} of {len(records)}")
    return rewritten


def _transition_rows(
    *,
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    condition_id: str,
    condition_name: str,
) -> list[dict[str, Any]]:
    baseline_by_id = {str(row["item_id"]): row for row in baseline_rows}
    out: list[dict[str, Any]] = []
    for cur in current_rows:
        item_id = str(cur["item_id"])
        base = baseline_by_id.get(item_id)
        if base is None:
            continue
        base_label = str(base["base_label"])
        ft_label = str(cur["base_label"])
        row = dict(base)
        row.update(
            {
                "condition_id": str(condition_id),
                "condition_name": str(condition_name),
                "ft_label": ft_label,
                "ft_first_token_id": cur.get("base_first_token_id"),
                "ft_generated_answer": str(cur.get("base_generated_answer", "")),
                "ft_generated_token_ids": list(cur.get("base_generated_token_ids", [])),
                "ft_faithful": bool(ft_label == "faithful"),
                "ft_hallucinated": bool(ft_label == "hallucinated"),
                "restored": bool(base_label == "hallucinated" and ft_label == "faithful"),
                "harmed": bool(base_label == "faithful" and ft_label != "faithful"),
                "gate_fired": False,
                "tau": None,
                "lambda_pos": 0.0,
                "lambda_neg": 0.0,
                "n_bias_steps": 0,
                "prompt_only_baseline": True,
                "prompt": str(cur.get("prompt", "")),
                "evidence": str(cur.get("evidence", "")),
            }
        )
        out.append(row)
    return out


def _condition_counts(rows: list[dict[str, Any]], *, label_key: str) -> dict[str, Any]:
    n = int(len(rows))
    faithful = sum(1 for row in rows if str(row.get(label_key, "")) == "faithful")
    hallucinated = sum(1 for row in rows if str(row.get(label_key, "")) == "hallucinated")
    other = n - faithful - hallucinated
    return {
        "n_items": n,
        "faithful": int(faithful),
        "hallucinated": int(hallucinated),
        "other": int(other),
        "faithful_rate": float(faithful / max(n, 1)),
        "hallucinated_rate": float(hallucinated / max(n, 1)),
        "other_rate": float(other / max(n, 1)),
    }


def _paired_mcnemar(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], left_key: str, right_key: str) -> dict[str, Any]:
    left = {str(row["item_id"]): str(row.get(left_key, "")) == "faithful" for row in left_rows}
    right = {str(row["item_id"]): str(row.get(right_key, "")) == "faithful" for row in right_rows}
    shared = sorted(set(left) & set(right))
    return mcnemar_from_paired([left[item_id] for item_id in shared], [right[item_id] for item_id in shared])


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    base_results_dir = _project_path(args.base_results_dir)
    out_dir = _project_path(args.out_dir)
    if base_results_dir == out_dir:
        raise RuntimeError("--out_dir must differ from --base_results_dir so existing results are not modified.")
    if not (base_results_dir / "summary.json").exists():
        raise FileNotFoundError(f"Missing base summary: {base_results_dir / 'summary.json'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_existing_outputs(base_results_dir, out_dir)

    base_summary = read_json(base_results_dir / "summary.json")
    c2_rows = _read_jsonl(base_results_dir / "baseline_rows.jsonl")
    c3_rows = _read_jsonl(base_results_dir / "results.jsonl")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    records, cache_path = load_counterfact_prepared(
        project_root=PROJECT_ROOT,
        cache_path=args.cache_path,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        max_items=int(args.max_items),
        evidence_mode=str(args.evidence_mode),
    )

    c4_records = _rewrite_records(records=records, tokenizer=tokenizer, max_length=int(args.max_length), mode="authoritative")
    c4_condition_rows = prepare_baseline_rows(
        records=c4_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
    )
    c4_rows = _transition_rows(
        baseline_rows=c2_rows,
        current_rows=c4_condition_rows,
        condition_id="C4",
        condition_name="CounterFact authoritative evidence prompt, no intervention",
    )
    c4_summary = condition_summary(c4_rows)
    c4_counts = _condition_counts(c4_condition_rows, label_key="base_label")
    c4_summary.update(
        {
            "condition_id": "C4",
            "condition_name": "CounterFact authoritative evidence prompt, no intervention",
            "prompt_only_baseline": True,
            "condition_counts": c4_counts,
            "condition_faithful_rate": float(c4_counts["faithful_rate"]),
            "condition_hallucinated_rate": float(c4_counts["hallucinated_rate"]),
            "condition_other_rate": float(c4_counts["other_rate"]),
        }
    )

    c5_records = _rewrite_records(records=records, tokenizer=tokenizer, max_length=int(args.max_length), mode="explicit_answer")
    c5_condition_rows = prepare_baseline_rows(
        records=c5_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
    )
    c5_rows = _transition_rows(
        baseline_rows=c2_rows,
        current_rows=c5_condition_rows,
        condition_id="C5",
        condition_name="CounterFact explicit-answer prompt, no intervention",
    )
    c5_summary = condition_summary(c5_rows)
    c5_counts = _condition_counts(c5_condition_rows, label_key="base_label")
    c5_summary.update(
        {
            "condition_id": "C5",
            "condition_name": "CounterFact explicit-answer prompt, no intervention",
            "prompt_only_baseline": True,
            "explicit_answer_in_context": True,
            "condition_counts": c5_counts,
            "condition_faithful_rate": float(c5_counts["faithful_rate"]),
            "condition_hallucinated_rate": float(c5_counts["hallucinated_rate"]),
            "condition_other_rate": float(c5_counts["other_rate"]),
        }
    )

    metadata = build_metadata(
        args=args,
        input_files=[cache_path, base_results_dir / "summary.json"],
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - started),
        project_root=REPO_ROOT,
    )
    for summary in (c4_summary, c5_summary):
        summary.update(
            {
                "model_id": str(args.model_id),
                "model_name": str(args.model_name or args.model_id),
                "architecture": str(args.architecture),
                "task": "CounterFact",
                "evidence_mode": str(args.evidence_mode),
                "metadata": metadata,
            }
        )

    c2_condition = dict(base_summary.get("conditions", {}).get("C2", {}))
    c3_condition = dict(base_summary.get("conditions", {}).get("C3", {}))
    aggregate = dict(base_summary)
    aggregate.update(
        {
            "c45_added": True,
            "c45_note": "C4/C5 are prompt-only baselines added in a separate output directory; existing C2/C3 results were copied from base_results_dir.",
            "base_results_dir": str(base_results_dir),
            "conditions": {
                "C2": c2_condition,
                "C3": c3_condition,
                "C4": c4_summary,
                "C5": c5_summary,
            },
            "mcnemar_c3_vs_c4": _paired_mcnemar(c3_rows, c4_rows, "ft_label", "ft_label"),
            "mcnemar_c3_vs_c5": _paired_mcnemar(c3_rows, c5_rows, "ft_label", "ft_label"),
            "mcnemar_c4_vs_c5": _paired_mcnemar(c4_rows, c5_rows, "ft_label", "ft_label"),
            "mcnemar_c2_vs_c4": _paired_mcnemar(c2_rows, c4_rows, "base_label", "ft_label"),
            "mcnemar_c2_vs_c5": _paired_mcnemar(c2_rows, c5_rows, "base_label", "ft_label"),
            "metadata_c45": metadata,
        }
    )

    write_json(out_dir / "summary.json", aggregate)
    write_jsonl(out_dir / "C4_results.jsonl", c4_rows)
    write_jsonl(out_dir / "C5_results.jsonl", c5_rows)
    write_jsonl(out_dir / "C4_condition_rows.jsonl", c4_condition_rows)
    write_jsonl(out_dir / "C5_condition_rows.jsonl", c5_condition_rows)
    write_json(
        out_dir / "c45_comparison_row.json",
        {
            "model": str(args.model_name or args.model_id),
            "base_results_dir": str(base_results_dir),
            "c3_restoration_rate": float(c3_condition.get("restoration_rate", base_summary.get("restoration_rate", 0.0))),
            "c3_harm_rate": float(c3_condition.get("harm_rate", base_summary.get("harm_rate", 0.0))),
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
    print(f"c4={out_dir / 'C4_results.jsonl'}")
    print(f"c5={out_dir / 'C5_results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
