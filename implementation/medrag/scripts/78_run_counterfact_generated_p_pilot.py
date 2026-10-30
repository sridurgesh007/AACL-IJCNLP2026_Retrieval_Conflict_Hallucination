#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.common import classify_answer_token
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import (
    NStepEntityBiasProcessor,
    auc_rank,
    build_metadata,
    evaluate_intervention,
    generate_label,
    load_counterfact_prepared,
    margin_from_logits,
    prefill_logits,
    prepare_baseline_rows,
    select_tau,
)
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json, write_jsonl


ARTICLES = {"the", "a", "an"}
LIST_MARKER_RE = re.compile(r"^\s*\d+[\.\)]\s*$")
PURE_PUNCT_RE = re.compile(rf"^[\s{re.escape(string.punctuation)}]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CounterFact generated-P rebuttal pilot: replace oracle P with no-context model generation."
    )
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument(
        "--evidence_mode",
        default="relation_or_question_answer",
        choices=["cache", "question_answer", "relation_or_question_answer"],
    )
    parser.add_argument("--out_dir", default="results/rebuttal/generated_p_pilot")
    parser.add_argument("--max_items", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--c1_max_new_tokens", type=int, default=8)
    parser.add_argument("--c1_prompt_template", default="Question: {question}\nAnswer: ")
    parser.add_argument("--lambda_pos", type=float, default=0.0)
    parser.add_argument("--lambda_neg", type=float, default=2.5)
    parser.add_argument("--n_bias_steps", type=int, default=1)
    parser.add_argument(
        "--tau",
        default="auto",
        help="Gate threshold. Use 'auto' to select from oracle_P baseline on this frozen pilot.",
    )
    parser.add_argument("--premise_go_threshold", type=float, default=0.50)
    parser.add_argument(
        "--run_main",
        default="auto",
        choices=["auto", "0", "1"],
        help="auto stops before intervention when premise agreement is below premise_go_threshold.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_token_text(text: str) -> str:
    token = str(text)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.strip()
    token = token.strip(string.punctuation)
    return token.strip()


def _norm_for_match(text: str) -> str:
    return re.sub(r"\s+", "", _clean_token_text(text).casefold())


def _is_skip_token(text: str) -> bool:
    raw = str(text)
    cleaned = _clean_token_text(raw)
    if not cleaned:
        return True
    if PURE_PUNCT_RE.match(raw):
        return True
    if LIST_MARKER_RE.match(raw):
        return True
    if cleaned.isdigit():
        return True
    if cleaned.casefold() in ARTICLES:
        return True
    return False


def first_content_token_from_generated_ids(
    generated_ids: list[int],
    tokenizer,
) -> dict[str, Any]:
    """Frozen extraction rule from GENERATED_P_PILOT.md Section 4."""
    skipped: list[dict[str, Any]] = []
    for token_id in generated_ids:
        raw = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        cleaned = _clean_token_text(raw)
        if _is_skip_token(raw):
            skipped.append({"token_id": int(token_id), "raw": raw, "cleaned": cleaned})
            continue
        return {
            "extraction_failed": False,
            "token_id": int(token_id),
            "raw": raw,
            "cleaned": cleaned,
            "skipped": skipped,
        }
    return {
        "extraction_failed": True,
        "token_id": None,
        "raw": "",
        "cleaned": "",
        "skipped": skipped,
    }


def first_answer_token_id(tokenizer, answer_text: str) -> int | None:
    answer = str(answer_text).strip()
    if not answer:
        return None
    variants = [f" {answer}", answer]
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False).get("input_ids", [])
        if ids:
            return int(ids[0])
    return None


def token_set_from_text(tokenizer, text: str) -> set[int]:
    cleaned = _clean_token_text(text)
    forms = [cleaned, cleaned.lower(), cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned]
    out: set[int] = set()
    for form in forms:
        form = str(form).strip()
        if not form:
            continue
        for variant in (f" {form}", form):
            ids = tokenizer(variant, add_special_tokens=False).get("input_ids", [])
            if ids:
                out.add(int(ids[0]))
    return out


def generate_c1(
    *,
    model,
    tokenizer,
    device: torch.device,
    question: str,
    prompt_template: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = str(prompt_template).format(question=str(question))
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = [int(x) for x in out[0, input_ids.shape[1] :].detach().cpu().tolist()]
    extracted = first_content_token_from_generated_ids(new_ids, tokenizer)
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "c1_prompt": prompt,
        "c1_generated_answer": generated_text,
        "c1_generated_token_ids": new_ids,
        "generated_p": str(extracted["cleaned"]),
        "generated_p_token_id": extracted["token_id"],
        "extraction_failed": bool(extracted["extraction_failed"]),
        "first_content_token": extracted,
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_hall = [row for row in rows if str(row.get("base_label")) == "hallucinated"]
    base_faith = [row for row in rows if str(row.get("base_label")) == "faithful"]
    restored = [row for row in base_hall if bool(row.get("restored"))]
    harmed = [row for row in base_faith if bool(row.get("harmed"))]
    gate = [row for row in rows if bool(row.get("gate_fired"))]
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        label = str(row.get("base_label"))
        if label == "faithful":
            scores.append(float(row["prefill_margin"]))
            labels.append(1)
        elif label == "hallucinated":
            scores.append(float(row["prefill_margin"]))
            labels.append(0)
    return {
        "n": int(len(rows)),
        "margin_auc": auc_rank(scores, labels),
        "n_base_faithful": int(len(base_faith)),
        "n_base_hallucinated": int(len(base_hall)),
        "n_restored": int(len(restored)),
        "restoration": float(len(restored) / max(len(base_hall), 1)),
        "n_harmed": int(len(harmed)),
        "harm": float(len(harmed) / max(len(base_faith), 1)),
        "n_gate_fired": int(len(gate)),
        "gate_fire_rate": float(len(gate) / max(len(rows), 1)),
    }


def _evaluate_generated_condition(
    *,
    baseline_rows: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    tau: float,
    lambda_pos: float,
    lambda_neg: float,
    n_bias_steps: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        generated_p_ids = {int(x) for x in row.get("generated_p_token_ids", [])}
        e_ids = {int(row["evidence_token_id"])}
        prompt_ids = torch.tensor([int(x) for x in row["prompt_token_ids"]], dtype=torch.long)
        logits = prefill_logits(model, prompt_ids, device)
        margin = margin_from_logits(logits, e_ids, generated_p_ids)
        p_id_for_label = sorted(generated_p_ids)[0] if generated_p_ids else -1
        base_first = None if row.get("base_first_token_id") is None else int(row["base_first_token_id"])
        if base_first in e_ids:
            base_label = "faithful"
        elif base_first in generated_p_ids:
            base_label = "hallucinated"
        else:
            base_label = "other"
        gate_fired = bool(generated_p_ids and math.isfinite(float(margin)) and float(margin) < float(tau))
        processor = None
        if gate_fired:
            processor = NStepEntityBiasProcessor(
                promote_ids=e_ids,
                suppress_ids=generated_p_ids,
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                n_steps=int(n_bias_steps),
            )
        raw_label, first_id, generated, new_ids = generate_label(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            evidence_token_id=int(row["evidence_token_id"]),
            parametric_token_id=int(p_id_for_label),
            max_new_tokens=int(max_new_tokens),
            processor=processor,
        )
        del raw_label
        if first_id in e_ids:
            label = "faithful"
        elif first_id in generated_p_ids:
            label = "hallucinated"
        else:
            label = "other"
        out = dict(row)
        out.update(
            {
                "condition": "generated_P",
                "p_token_source": "c1_no_context_first_content_token",
                "prefill_margin": float(margin) if math.isfinite(float(margin)) else None,
                "base_label": str(base_label),
                "ft_label": str(label),
                "ft_first_token_id": None if first_id is None else int(first_id),
                "ft_generated_answer": str(generated),
                "ft_generated_token_ids": new_ids,
                "gate_fired": bool(gate_fired),
                "restored": bool(base_label == "hallucinated" and label == "faithful"),
                "harmed": bool(base_label == "faithful" and label != "faithful"),
                "tau": float(tau),
                "lambda_pos": float(lambda_pos),
                "lambda_neg": float(lambda_neg),
                "n_bias_steps": int(n_bias_steps),
            }
        )
        rows.append(out)
    return rows


def _oracle_rows_from_existing(
    rows: list[dict[str, Any]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        rec.update(
            {
                "condition": condition,
                "p_token_source": "dataset_target_true",
                "gold_parametric": str(row.get("parametric_answer", "")),
                "generated_p": str(row.get("generated_p", "")),
                "agreement": bool(row.get("agreement", False)),
                "extraction_failed": bool(row.get("extraction_failed", False)),
            }
        )
        out.append(rec)
    return out


def _write_readme(
    *,
    out_dir: Path,
    premise: dict[str, Any],
    summary: dict[str, Any] | None,
    ran_main: bool,
) -> None:
    if ran_main and summary is not None:
        verdict = (
            "The premise held sufficiently to run the intervention."
            if float(premise["agreement_rate"]) >= 0.8
            else "The premise was partial; intervention was run only as a scoped diagnostic."
        )
    else:
        verdict = "The premise failed the configured gate, so intervention was not run."
    text = (
        "# Generated-P CounterFact Pilot\n\n"
        f"Ran the generated-P premise check on {int(premise['n'])} CounterFact items. "
        f"The no-context generated-P agreement rate with dataset target_true was "
        f"{float(premise['agreement_rate']):.4f} "
        f"({int(premise['n_agree'])}/{int(premise['n'])}); "
        f"extraction failures were {int(premise['n_extraction_failed'])}. "
        f"{verdict}\n\n"
        "Sanity check: generated_P token sets are sourced from C1 no-context greedy generation; "
        "the dataset parametric field is retained only as `gold_parametric` for premise comparison "
        "and oracle_P baseline reporting.\n"
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_config = {
        "created_at_utc": _now_iso(),
        "purpose": "Generated-P non-oracle pilot for CounterFact rebuttal.",
        "model_id": str(args.model_id),
        "cache_path": str(args.cache_path),
        "evidence_mode": str(args.evidence_mode),
        "out_dir": str(args.out_dir),
        "max_items": int(args.max_items),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "c1_max_new_tokens": int(args.c1_max_new_tokens),
        "c1_prompt_template": str(args.c1_prompt_template),
        "lambda_pos": float(args.lambda_pos),
        "lambda_neg": float(args.lambda_neg),
        "n_bias_steps": int(args.n_bias_steps),
        "tau": str(args.tau),
        "premise_go_threshold": float(args.premise_go_threshold),
        "run_main": str(args.run_main),
        "seed": int(args.seed),
        "device": str(args.device),
        "torch_dtype": str(args.torch_dtype),
        "frozen_before_model_run": True,
    }
    write_json(out_dir / "config.json", frozen_config)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    records, cache_path = load_counterfact_prepared(
        project_root=PROJECT_ROOT,
        cache_path=args.cache_path,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        max_items=int(args.max_items),
        evidence_mode=str(args.evidence_mode),
    )

    c1_by_id: dict[str, dict[str, Any]] = {}
    premise_rows: list[dict[str, Any]] = []
    for rec in records:
        ex = rec.example
        c1 = generate_c1(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=str(ex.question),
            prompt_template=str(args.c1_prompt_template),
            max_new_tokens=int(args.c1_max_new_tokens),
        )
        gold_token_id = first_answer_token_id(tokenizer, str(ex.answer_solid))
        gold_token_text = "" if gold_token_id is None else tokenizer.decode([int(gold_token_id)], skip_special_tokens=False)
        generated_norm = _norm_for_match(str(c1["generated_p"]))
        gold_norm = _norm_for_match(gold_token_text)
        agreement = bool(
            not bool(c1["extraction_failed"])
            and generated_norm != ""
            and gold_norm != ""
            and generated_norm == gold_norm
        )
        generated_ids = token_set_from_text(tokenizer, str(c1["generated_p"])) if not c1["extraction_failed"] else set()
        row = {
            "item_id": str(ex.id),
            "query": str(ex.question),
            "evidence": str(ex.evidence),
            "correct_answer": str(ex.answer_liquid),
            "gold_parametric": str(ex.answer_solid),
            "gold_parametric_first_token_id": None if gold_token_id is None else int(gold_token_id),
            "gold_parametric_first_token_text": str(gold_token_text),
            "generated_p": str(c1["generated_p"]),
            "generated_p_token_id": c1["generated_p_token_id"],
            "generated_p_token_ids": sorted(int(x) for x in generated_ids),
            "generated_p_norm": generated_norm,
            "gold_parametric_norm": gold_norm,
            "agreement": bool(agreement),
            "extraction_failed": bool(c1["extraction_failed"]),
            "c1_prompt": str(c1["c1_prompt"]),
            "c1_generated_answer": str(c1["c1_generated_answer"]),
            "c1_generated_token_ids": c1["c1_generated_token_ids"],
            "first_content_token": c1["first_content_token"],
        }
        c1_by_id[str(ex.id)] = row
        premise_rows.append(row)

    n = len(premise_rows)
    n_agree = sum(1 for row in premise_rows if bool(row["agreement"]))
    n_failed = sum(1 for row in premise_rows if bool(row["extraction_failed"]))
    disagree_examples = [
        {
            "query": row["query"],
            "gold_parametric": row["gold_parametric"],
            "generated_p": row["generated_p"],
            "c1_generated_answer": row["c1_generated_answer"],
            "extraction_failed": bool(row["extraction_failed"]),
        }
        for row in premise_rows
        if not bool(row["agreement"])
    ][:20]
    premise = {
        "n": int(n),
        "agreement_rate": float(n_agree / max(n, 1)),
        "n_agree": int(n_agree),
        "n_disagree": int(n - n_agree),
        "n_extraction_failed": int(n_failed),
        "extraction_failure_rate": float(n_failed / max(n, 1)),
        "agreement_definition": "normalized first content token from C1 generation equals normalized first BPE token of dataset target_true",
        "decision_gate": {
            "threshold": float(args.premise_go_threshold),
            "passes": bool((n_agree / max(n, 1)) >= float(args.premise_go_threshold)),
        },
        "disagree_examples": disagree_examples,
    }
    write_json(out_dir / "premise_check.json", premise)
    write_jsonl(out_dir / "premise_rows.jsonl", premise_rows)

    should_run_main = (
        str(args.run_main) == "1"
        or (str(args.run_main) == "auto" and float(premise["agreement_rate"]) >= float(args.premise_go_threshold))
    )
    if not should_run_main:
        _write_readme(out_dir=out_dir, premise=premise, summary=None, ran_main=False)
        print(json.dumps({"premise_check": premise, "ran_main": False}, indent=2))
        return 0

    baseline_rows = prepare_baseline_rows(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
    )
    for row in baseline_rows:
        c1 = c1_by_id[str(row["item_id"])]
        row.update(
            {
                "gold_parametric": c1["gold_parametric"],
                "generated_p": c1["generated_p"],
                "generated_p_token_id": c1["generated_p_token_id"],
                "generated_p_token_ids": c1["generated_p_token_ids"],
                "agreement": bool(c1["agreement"]),
                "extraction_failed": bool(c1["extraction_failed"]),
                "c1_generated_answer": c1["c1_generated_answer"],
                "c1_generated_token_ids": c1["c1_generated_token_ids"],
            }
        )

    tau_summary, _ = select_tau(baseline_rows)
    if str(args.tau).strip().lower() == "auto":
        if tau_summary["selected_tau"] is None:
            raise RuntimeError("Cannot auto-select tau: oracle_P baseline lacks faithful/hallucinated classes.")
        selected_tau = float(tau_summary["selected_tau"])
        tau_source = "oracle_P_auto_selected_on_frozen_pilot"
    else:
        selected_tau = float(args.tau)
        tau_source = "cli_supplied"

    oracle_rows_raw, oracle_condition_summary = evaluate_intervention(
        baseline_rows=baseline_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        tau=float(selected_tau),
        lambda_pos=float(args.lambda_pos),
        lambda_neg=float(args.lambda_neg),
        n_bias_steps=int(args.n_bias_steps),
        max_new_tokens=int(args.max_new_tokens),
    )
    oracle_rows = _oracle_rows_from_existing(oracle_rows_raw, condition="oracle_P")
    generated_rows = _evaluate_generated_condition(
        baseline_rows=baseline_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        tau=float(selected_tau),
        lambda_pos=float(args.lambda_pos),
        lambda_neg=float(args.lambda_neg),
        n_bias_steps=int(args.n_bias_steps),
        max_new_tokens=int(args.max_new_tokens),
    )

    per_item_rows: list[dict[str, Any]] = []
    generated_by_id = {str(row["item_id"]): row for row in generated_rows}
    oracle_by_id = {str(row["item_id"]): row for row in oracle_rows}
    for item_id in sorted(generated_by_id, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        o = oracle_by_id[item_id]
        g = generated_by_id[item_id]
        per_item_rows.append(
            {
                "item_id": item_id,
                "query": str(o["question"]),
                "evidence": str(o["evidence"]),
                "correct_answer": str(o["correct_answer"]),
                "gold_P": str(o["gold_parametric"]),
                "generated_P": str(g["generated_p"]),
                "agreement": bool(g["agreement"]),
                "extraction_failed": bool(g["extraction_failed"]),
                "oracle_P": {
                    "gate_fired": bool(o["gate_fired"]),
                    "base_correct": str(o["base_label"]) == "faithful",
                    "post_correct": str(o["ft_label"]) == "faithful",
                    "base_label": str(o["base_label"]),
                    "post_label": str(o["ft_label"]),
                    "margin": o["prefill_margin"],
                    "restored": bool(o["restored"]),
                    "harmed": bool(o["harmed"]),
                    "base_generated_answer": str(o["base_generated_answer"]),
                    "post_generated_answer": str(o["ft_generated_answer"]),
                },
                "generated_P_condition": {
                    "gate_fired": bool(g["gate_fired"]),
                    "base_correct": str(g["base_label"]) == "faithful",
                    "post_correct": str(g["ft_label"]) == "faithful",
                    "base_label": str(g["base_label"]),
                    "post_label": str(g["ft_label"]),
                    "margin": g["prefill_margin"],
                    "restored": bool(g["restored"]),
                    "harmed": bool(g["harmed"]),
                    "base_generated_answer": str(g["base_generated_answer"]),
                    "post_generated_answer": str(g["ft_generated_answer"]),
                },
            }
        )

    oracle_summary = {
        "margin_auc": tau_summary["margin_auc"],
        "restoration": float(oracle_condition_summary["restoration_rate"]),
        "harm": float(oracle_condition_summary["harm_rate"]),
        "n_gate_fired": int(sum(1 for row in oracle_rows if bool(row["gate_fired"]))),
        "n_base_hallucinated": int(oracle_condition_summary["n_base_hallucinated"]),
        "n_base_faithful": int(oracle_condition_summary["n_base_faithful"]),
    }
    generated_summary = _condition_summary(generated_rows)
    summary = {
        "n": int(n),
        "n_extraction_failed": int(n_failed),
        "agreement_rate": float(premise["agreement_rate"]),
        "n_agree": int(n_agree),
        "n_disagree": int(n - n_agree),
        "tau": float(selected_tau),
        "tau_source": tau_source,
        "lambda_pos": float(args.lambda_pos),
        "lambda_neg": float(args.lambda_neg),
        "n_bias_steps": int(args.n_bias_steps),
        "oracle_P": oracle_summary,
        "generated_P": {
            "margin_auc": generated_summary["margin_auc"],
            "restoration": generated_summary["restoration"],
            "harm": generated_summary["harm"],
            "n_gate_fired": generated_summary["n_gate_fired"],
            "n_base_hallucinated": generated_summary["n_base_hallucinated"],
            "n_base_faithful": generated_summary["n_base_faithful"],
            "n_extraction_failed_counted_as_non_restored": int(n_failed),
        },
        "restoration_delta": float(generated_summary["restoration"] - oracle_summary["restoration"]),
        "gold_dependency_sanity": {
            "generated_p_token_source": "C1 no-context greedy generation",
            "dataset_parametric_used_for_generated_p_tokens": False,
            "dataset_parametric_retained_for_premise_comparison": True,
        },
        "metadata": build_metadata(
            args=args,
            input_files=[cache_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=REPO_ROOT,
        ),
    }

    write_jsonl(out_dir / "oracle_P_rows.jsonl", oracle_rows)
    write_jsonl(out_dir / "generated_P_rows.jsonl", generated_rows)
    write_jsonl(out_dir / "per_item.jsonl", per_item_rows)
    write_json(out_dir / "summary.json", summary)
    _write_readme(out_dir=out_dir, premise=premise, summary=summary, ran_main=True)
    print(json.dumps({"premise": premise, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
