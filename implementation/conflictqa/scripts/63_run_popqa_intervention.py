#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from conflictqa_pipeline.utils import (
    auc_rank,
    build_metadata,
    build_user_chat_prompt,
    classify_answer,
    clean_generated_answer,
    entity_match,
    first_token_set,
    generate_short_answer,
    margin_from_logits,
    mcnemar_from_paired,
    prefill_logits,
    read_jsonl,
    wilson_ci,
    write_json,
    write_jsonl,
)


class NStepBiasProcessor:
    def __init__(
        self,
        *,
        promote_ids: set[int],
        suppress_ids: set[int],
        lambda_pos: float,
        lambda_neg: float,
        n_steps: int = 3,
    ) -> None:
        self.promote_ids = {int(token_id) for token_id in promote_ids}
        self.suppress_ids = {int(token_id) for token_id in suppress_ids}
        self.lambda_pos = float(lambda_pos)
        self.lambda_neg = float(lambda_neg)
        self.n_steps = int(n_steps)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self._step < self.n_steps:
            if self.lambda_pos > 0.0:
                for token_id in self.promote_ids:
                    if 0 <= int(token_id) < int(scores.shape[-1]):
                        scores[..., int(token_id)] += self.lambda_pos
            for token_id in self.suppress_ids:
                if self.lambda_neg > 0.0 and 0 <= int(token_id) < int(scores.shape[-1]):
                    scores[..., int(token_id)] -= self.lambda_neg
        self._step += 1
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PopQA conflict intervention with pilot AUC, tau selection, and lambda sweep.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--input_path", "--conflict_path", dest="input_path", default="data/popqa_conflict_500.jsonl")
    parser.add_argument("--results_dir", default="results/popqa")
    parser.add_argument("--pilot_examples", type=int, default=100)
    parser.add_argument("--max_items", type=int, default=500, help="0 means all input rows.")
    parser.add_argument("--min_margin_auc", type=float, default=0.80)
    parser.add_argument("--min_parametric_pilot", type=int, default=15)
    parser.add_argument("--intervention_mode", default="suppression", choices=["suppression", "balanced"])
    parser.add_argument("--lambda_values", default="2.5,5.0,7.5,10.0")
    parser.add_argument("--lambda_pos_values", default="0.0")
    parser.add_argument("--fixed_tau", type=float, default=None)
    parser.add_argument("--fixed_lambda", type=float, default=None)
    parser.add_argument("--fixed_lambda_pos", type=float, default=None)
    parser.add_argument("--evidence_token_source", default="canonical", choices=["canonical", "aliases"])
    parser.add_argument("--n_suppress_steps", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one lambda value.")
    return values


def _is_supplement(row: dict[str, Any]) -> bool:
    return bool(row.get("is_supplement", False))


def _select_pilot_rows(rows: list[dict[str, Any]], pilot_n: int) -> list[dict[str, Any]]:
    pilot_n = min(int(pilot_n), len(rows))
    if pilot_n <= 0:
        return []

    supplement_rows = [row for row in rows if _is_supplement(row)]
    real_rows = [row for row in rows if not _is_supplement(row)]
    if not supplement_rows or not real_rows:
        return rows[:pilot_n]

    n_supplement = min(len(supplement_rows), max(1, pilot_n // 2))
    n_real = min(len(real_rows), pilot_n - n_supplement)
    if n_real <= 0:
        n_real = min(len(real_rows), 1)
        n_supplement = min(len(supplement_rows), pilot_n - n_real)
    if n_real + n_supplement < pilot_n:
        remaining = pilot_n - n_real - n_supplement
        extra_real = real_rows[n_real : n_real + remaining]
        n_real += len(extra_real)
        remaining = pilot_n - n_real - n_supplement
        if remaining > 0:
            n_supplement += min(len(supplement_rows) - n_supplement, remaining)
    return real_rows[:n_real] + supplement_rows[:n_supplement]


def _matches_any_alias(generated: str, references: Sequence[str]) -> bool:
    return any(entity_match(generated, str(reference)) for reference in references if str(reference).strip())


def _classify_with_aliases(generated: str, correct_aliases: Sequence[str], parametric_answer: str) -> str:
    if _matches_any_alias(generated, correct_aliases):
        return "context_faithful"
    if entity_match(generated, str(parametric_answer)):
        return "parametric"
    return "other"


def _build_c1_prompt(tokenizer, question: str) -> str:
    return build_user_chat_prompt(
        tokenizer,
        f"Question: {str(question).strip()}\nAnswer in 1-3 words only.",
    )


def _build_c2_prompt(tokenizer, question: str, context: str) -> str:
    return build_user_chat_prompt(
        tokenizer,
        f"Context: {str(context).strip()}\nQuestion: {str(question).strip()}\nAnswer in 1-3 words only.",
    )


def _build_c4_prompt(tokenizer, question: str, context: str, correct_answer: str) -> str:
    del correct_answer
    return build_user_chat_prompt(
        tokenizer,
        f"Context: {str(context).strip()}\n"
        f"Question: {str(question).strip()}\n"
        "The context above is from an authoritative source. You MUST answer based on the context, "
        "not your prior knowledge. Answer in 1-3 words only.",
    )


def _build_c5_prompt(tokenizer, question: str, explicit_injection_context: str) -> str:
    return _build_c2_prompt(tokenizer, question, explicit_injection_context)


def _encode_prompt(tokenizer, prompt: str, max_length: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length))
    return encoded["input_ids"][0].detach().cpu().long()


def _whitespace_token_ids(tokenizer) -> set[int]:
    whitespace_strings = [" ", "\n", "\t", "  ", "\n\n", " \n", "\n "]
    ids: set[int] = set()
    for text in whitespace_strings:
        for token_id in tokenizer.encode(text, add_special_tokens=False):
            ids.add(int(token_id))
    return ids


def _normalize_aliases(correct_answer: str, correct_aliases: Any) -> list[str]:
    forms: list[str] = [str(correct_answer)]
    if isinstance(correct_aliases, str):
        try:
            parsed = json.loads(correct_aliases)
        except json.JSONDecodeError:
            parsed = [correct_aliases]
        if isinstance(parsed, list):
            forms.extend(str(item) for item in parsed)
        else:
            forms.append(str(parsed))
    elif isinstance(correct_aliases, Sequence):
        forms.extend(str(item) for item in correct_aliases)

    out: list[str] = []
    seen: set[str] = set()
    for form in forms:
        cleaned = clean_generated_answer(str(form))
        key = cleaned.casefold()
        if cleaned and key not in seen:
            out.append(cleaned)
            seen.add(key)
    return out


def _first_token_sets(
    tokenizer,
    correct_answer: str,
    correct_aliases: Any,
    parametric_answer: str,
    whitespace_ids: set[int],
    evidence_token_source: str,
) -> tuple[set[int], set[int], bool, list[str]]:
    if str(evidence_token_source) == "aliases":
        evidence_forms = _normalize_aliases(correct_answer, correct_aliases)
    else:
        evidence_forms = [clean_generated_answer(str(correct_answer))]

    e_tokens: set[int] = set()
    for form in evidence_forms:
        e_tokens.update(set(first_token_set(tokenizer, str(form))) - set(whitespace_ids))
    p_tokens = set(first_token_set(tokenizer, str(parametric_answer))) - set(whitespace_ids)
    if str(evidence_token_source) == "aliases":
        ambiguous = bool(set(e_tokens) & set(p_tokens))
    else:
        ambiguous = bool(e_tokens == p_tokens)
    return e_tokens, p_tokens, bool(ambiguous), evidence_forms


def _validate_token_sets(e_tokens: set[int], p_tokens: set[int], whitespace_ids: set[int], item_id: Any) -> None:
    if not e_tokens:
        raise RuntimeError(f"Empty evidence token set for item {item_id}")
    if not p_tokens:
        raise RuntimeError(f"Empty parametric token set for item {item_id}")
    if set(e_tokens) & set(whitespace_ids):
        raise RuntimeError(f"Whitespace contamination in evidence token set for item {item_id}")
    if set(p_tokens) & set(whitespace_ids):
        raise RuntimeError(f"Whitespace contamination in parametric token set for item {item_id}")


def _run_condition(
    *,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    condition_name: str,
    prompt_key: str,
    tau: float | None = None,
    lambda_pos: float | None = None,
    lambda_neg: float | None = None,
    intervention_mode: str = "none",
    n_suppress_steps: int = 3,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=condition_name):
        prompt_ids = row[prompt_key]
        processor = None
        gate_fired = False
        if tau is not None and lambda_neg is not None:
            margin = float(row["prefill_margin"])
            if (not bool(row["ambiguous_token_sets"])) and math.isfinite(margin) and margin < float(tau):
                promote_ids = set(row["e_token_set"]) if str(intervention_mode) == "balanced" and float(lambda_pos or 0.0) > 0.0 else set()
                suppress_ids = set(row["p_token_set"]) if float(lambda_neg) > 0.0 else set()
                if promote_ids or suppress_ids:
                    processor = NStepBiasProcessor(
                        promote_ids=promote_ids,
                        suppress_ids=suppress_ids,
                        lambda_pos=float(lambda_pos or 0.0),
                        lambda_neg=float(lambda_neg),
                        n_steps=int(n_suppress_steps),
                    )
                    gate_fired = True
        generated, gen_ids = generate_short_answer(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            max_new_tokens=int(max_new_tokens),
            logits_processor=processor,
        )
        generated = clean_generated_answer(generated)
        classification = _classify_with_aliases(
            generated,
            row.get("correct_aliases", [str(row["correct_answer"])]),
            str(row["parametric_answer"]),
        )
        out_row = dict(row)
        out_row.update(
            {
                "generated_answer": generated,
                "classification": classification,
                "context_faithful": bool(classification == "context_faithful"),
                "parametric": bool(classification == "parametric"),
                "other": bool(classification == "other"),
                "generated_token_count": int(len(gen_ids)),
                "gate_fired": bool(gate_fired),
                "tau": None if tau is None else float(tau),
                "lambda_pos": None if lambda_pos is None else float(lambda_pos),
                "lambda_neg": None if lambda_neg is None else float(lambda_neg),
                "intervention_mode": str(intervention_mode),
                "n_suppress_steps": int(n_suppress_steps) if tau is not None and lambda_neg is not None else 0,
            }
        )
        out_rows.append(out_row)
    return out_rows


def _condition_counts(rows: Sequence[dict[str, Any]]) -> tuple[int, int, int]:
    faithful = sum(1 for row in rows if bool(row["context_faithful"]))
    parametric = sum(1 for row in rows if bool(row["parametric"]))
    other = sum(1 for row in rows if bool(row["other"]))
    return faithful, parametric, other


def _condition_summary(
    *,
    condition_id: str,
    condition_name: str,
    rows: list[dict[str, Any]],
    baseline_c2_rows: list[dict[str, Any]],
    tau: float | None,
    lambda_neg: float | None,
    margin_auc: float | None,
    lambda_pos: float | None = None,
    intervention_mode: str = "none",
) -> dict[str, Any]:
    n_examples = int(len(rows))
    faithful, parametric, other = _condition_counts(rows)
    base_by_id = {str(row["item_id"]): row for row in baseline_c2_rows}
    cur_by_id = {str(row["item_id"]): row for row in rows}
    item_ids = [item_id for item_id in base_by_id.keys() if item_id in cur_by_id]
    baseline_wrong = [item_id for item_id in item_ids if bool(base_by_id[item_id]["parametric"])]
    baseline_right = [item_id for item_id in item_ids if bool(base_by_id[item_id]["context_faithful"])]
    restored = sum(1 for item_id in baseline_wrong if bool(cur_by_id[item_id]["context_faithful"]))
    harmed = sum(1 for item_id in baseline_right if not bool(cur_by_id[item_id]["context_faithful"]))
    gate_fired = sum(1 for row in rows if bool(row.get("gate_fired", False)))
    base_correct = [bool(base_by_id[item_id]["context_faithful"]) for item_id in item_ids]
    cur_correct = [bool(cur_by_id[item_id]["context_faithful"]) for item_id in item_ids]

    return {
        "condition_id": condition_id,
        "condition_name": condition_name,
        "n_examples": int(n_examples),
        "faithfulness_rate": float(faithful / max(n_examples, 1)),
        "faithfulness_rate_ci95_wilson": wilson_ci(faithful, n_examples),
        "parametric_rate": float(parametric / max(n_examples, 1)),
        "parametric_rate_ci95_wilson": wilson_ci(parametric, n_examples),
        "other_rate": float(other / max(n_examples, 1)),
        "other_rate_ci95_wilson": wilson_ci(other, n_examples),
        "restoration_rate": float(restored / max(len(baseline_wrong), 1)),
        "restoration_rate_ci95_wilson": wilson_ci(restored, len(baseline_wrong)),
        "n_restored": int(restored),
        "n_c2_parametric": int(len(baseline_wrong)),
        "harm_rate": float(harmed / max(len(baseline_right), 1)),
        "harm_rate_ci95_wilson": wilson_ci(harmed, len(baseline_right)),
        "n_harmed": int(harmed),
        "n_c2_faithful": int(len(baseline_right)),
        "gate_fire_rate": float(gate_fired / max(n_examples, 1)),
        "gate_fire_rate_ci95_wilson": wilson_ci(gate_fired, n_examples),
        "mcnemar_vs_c2": mcnemar_from_paired(base_correct, cur_correct),
        "margin_auc": None if margin_auc is None else float(margin_auc),
        "tau": None if tau is None else float(tau),
        "lambda_pos": None if lambda_pos is None else float(lambda_pos),
        "lambda_neg": None if lambda_neg is None else float(lambda_neg),
        "intervention_mode": str(intervention_mode),
    }


def _pilot_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    faithful_rows = [row for row in rows if str(row["c2_classification"]) == "context_faithful" and math.isfinite(float(row["prefill_margin"]))]
    parametric_rows = [row for row in rows if str(row["c2_classification"]) == "parametric" and math.isfinite(float(row["prefill_margin"]))]
    faithful_margins = [float(row["prefill_margin"]) for row in faithful_rows]
    parametric_margins = [float(row["prefill_margin"]) for row in parametric_rows]
    auc = auc_rank(faithful_margins + parametric_margins, [1] * len(faithful_margins) + [0] * len(parametric_margins))

    thresholds: list[float] = []
    unique_scores = sorted(set(faithful_margins + parametric_margins))
    if unique_scores:
        thresholds.append(unique_scores[0] - 1e-6)
        for left, right in zip(unique_scores[:-1], unique_scores[1:]):
            thresholds.append((float(left) + float(right)) / 2.0)
        thresholds.append(unique_scores[-1] + 1e-6)

    tau_rows: list[dict[str, float]] = []
    if faithful_margins and parametric_margins:
        for tau in thresholds:
            tpr = float(sum(1 for margin in parametric_margins if margin < tau) / len(parametric_margins))
            fpr = float(sum(1 for margin in faithful_margins if margin < tau) / len(faithful_margins))
            tau_rows.append(
                {
                    "tau": float(tau),
                    "tpr_parametric": tpr,
                    "fpr_faithful": fpr,
                    "youden_j": float(tpr - fpr),
                }
            )

    selected = None
    if tau_rows:
        selected = sorted(
            tau_rows,
            key=lambda row: (-float(row["youden_j"]), -float(row["tpr_parametric"]), float(row["fpr_faithful"]), float(row["tau"])),
        )[0]

    mean_answer_length_tokens = float(sum(int(row["c2_generated_token_count"]) for row in rows) / max(len(rows), 1))
    summary = {
        "pilot_examples": int(len(rows)),
        "n_context_faithful": int(len(faithful_rows)),
        "n_parametric": int(len(parametric_rows)),
        "margin_auc": None if auc is None else float(auc),
        "mean_margin_faithful": float(sum(faithful_margins) / max(len(faithful_margins), 1)) if faithful_margins else None,
        "mean_margin_parametric": float(sum(parametric_margins) / max(len(parametric_margins), 1)) if parametric_margins else None,
        "mean_answer_length_tokens": float(mean_answer_length_tokens),
        "selected_tau": None if selected is None else float(selected["tau"]),
        "selected_tau_metrics": selected,
        "tau_candidates": tau_rows,
        "faithful_margins": faithful_margins,
        "parametric_margins": parametric_margins,
    }
    return summary, tau_rows


def _margin_auc_for_c2(rows: list[dict[str, Any]]) -> float | None:
    labeled = [
        row
        for row in rows
        if str(row["c2_classification"]) in {"context_faithful", "parametric"} and math.isfinite(float(row["prefill_margin"]))
    ]
    if not labeled:
        return None
    scores = [float(row["prefill_margin"]) for row in labeled]
    labels = [1 if str(row["c2_classification"]) == "context_faithful" else 0 for row in labeled]
    return auc_rank(scores, labels)


def _quantile_edges(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0
    q1 = ordered[min(len(ordered) - 1, max(0, len(ordered) // 3))]
    q2 = ordered[min(len(ordered) - 1, max(0, (2 * len(ordered)) // 3))]
    return float(q1), float(q2)


def _stratify_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    matched_rows = [row for row in rows if bool(row["c2_parametric"])]
    values = [float(row[key]) for row in matched_rows]
    q1, q2 = _quantile_edges(values)
    bins = [("low", lambda x: x <= q1), ("mid", lambda x: q1 < x <= q2), ("high", lambda x: x > q2)]
    out_rows = []
    for label, predicate in bins:
        bucket = [row for row in matched_rows if predicate(float(row[key]))]
        restored = sum(1 for row in bucket if bool(row["c3_context_faithful"]))
        out_rows.append(
            {
                "label": label,
                "n": int(len(bucket)),
                "restoration_rate": float(restored / max(len(bucket), 1)),
                "restoration_rate_ci95_wilson": wilson_ci(restored, len(bucket)),
            }
        )
    return {"boundaries": [float(q1), float(q2)], "rows": out_rows}


def _stratify_prop(rows: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    matched_rows = [row for row in rows if bool(row["c2_parametric"])]
    by_prop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched_rows:
        by_prop[str(row["prop"])].append(row)
    ranked = sorted(by_prop.items(), key=lambda item: (-len(item[1]), item[0]))[: int(top_k)]
    out = []
    for prop, bucket in ranked:
        restored = sum(1 for row in bucket if bool(row["c3_context_faithful"]))
        out.append(
            {
                "prop": prop,
                "n": int(len(bucket)),
                "restoration_rate": float(restored / max(len(bucket), 1)),
                "restoration_rate_ci95_wilson": wilson_ci(restored, len(bucket)),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    lambda_values = _parse_float_list(args.lambda_values)
    lambda_pos_values = _parse_float_list(args.lambda_pos_values)
    if str(args.intervention_mode) == "suppression":
        lambda_pos_values = [0.0]
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    t0 = time.perf_counter()

    input_path = (PROJECT_ROOT / args.input_path).resolve()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    rows = sorted(rows, key=lambda row: str(row.get("item_id", "")))
    if int(args.max_items) > 0:
        rows = rows[: int(args.max_items)]
    if not rows:
        raise RuntimeError(f"No PopQA conflict rows found: {input_path}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    whitespace_ids = _whitespace_token_ids(tokenizer)

    base_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="PopQA baseline preparation"):
        question = str(row["question"])
        correct_answer = clean_generated_answer(str(row["correct_answer"]))
        parametric_answer = clean_generated_answer(str(row["parametric_answer"]))
        context = str(row["context"])
        explicit_injection_context = str(row.get("explicit_injection_context") or f"According to reference sources, the answer is {correct_answer}.")
        item_id = str(row["item_id"])

        c1_prompt_ids = _encode_prompt(tokenizer, _build_c1_prompt(tokenizer, question), int(args.max_length))
        c2_prompt_ids = _encode_prompt(tokenizer, _build_c2_prompt(tokenizer, question, context), int(args.max_length))
        c4_prompt_ids = _encode_prompt(tokenizer, _build_c4_prompt(tokenizer, question, context, correct_answer), int(args.max_length))
        c5_prompt_ids = _encode_prompt(tokenizer, _build_c5_prompt(tokenizer, question, explicit_injection_context), int(args.max_length))
        correct_aliases = _normalize_aliases(correct_answer, row.get("correct_aliases", [correct_answer]))
        e_tokens, p_tokens, ambiguous_token_sets, evidence_forms = _first_token_sets(
            tokenizer,
            correct_answer,
            correct_aliases,
            parametric_answer,
            whitespace_ids,
            str(args.evidence_token_source),
        )
        _validate_token_sets(e_tokens, p_tokens, whitespace_ids, item_id)
        c2_logits = prefill_logits(model=model, prompt_ids=c2_prompt_ids, device=device)
        prefill_margin = margin_from_logits(c2_logits, e_tokens, p_tokens)
        base_rows.append(
            {
                "item_id": item_id,
                "question": question,
                "correct_answer": correct_answer,
                "correct_aliases": correct_aliases,
                "parametric_answer": parametric_answer,
                "context": context,
                "explicit_injection_context": explicit_injection_context,
                "is_supplement": _is_supplement(row),
                "supplement_source": str(row.get("supplement_source", "")),
                "prop": str(row["prop"]),
                "s_pop": float(row["s_pop"]),
                "llm_answer_prob": float(row["llm_answer_prob"]),
                "e_token_set": sorted(int(tok) for tok in e_tokens),
                "p_token_set": sorted(int(tok) for tok in p_tokens),
                "e_token_set_size": int(len(e_tokens)),
                "p_token_set_size": int(len(p_tokens)),
                "evidence_token_source": str(args.evidence_token_source),
                "evidence_token_forms": evidence_forms,
                "ambiguous_token_sets": bool(ambiguous_token_sets),
                "prefill_margin": float(prefill_margin),
                "_c1_prompt_ids": c1_prompt_ids,
                "_c2_prompt_ids": c2_prompt_ids,
                "_c4_prompt_ids": c4_prompt_ids,
                "_c5_prompt_ids": c5_prompt_ids,
            }
        )

    c1_rows = _run_condition(
        rows=base_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        condition_name="C1 no context",
        prompt_key="_c1_prompt_ids",
    )
    c2_rows = _run_condition(
        rows=base_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        condition_name="C2 short context",
        prompt_key="_c2_prompt_ids",
    )

    prepared_rows: list[dict[str, Any]] = []
    for base_row, c1_row, c2_row in zip(base_rows, c1_rows, c2_rows):
        row = dict(base_row)
        row.update(
            {
                "c1_generated_answer": str(c1_row["generated_answer"]),
                "c1_classification": str(c1_row["classification"]),
                "c1_context_faithful": bool(c1_row["context_faithful"]),
                "c1_parametric": bool(c1_row["parametric"]),
                "c1_other": bool(c1_row["other"]),
                "c2_generated_answer": str(c2_row["generated_answer"]),
                "c2_classification": str(c2_row["classification"]),
                "c2_context_faithful": bool(c2_row["context_faithful"]),
                "c2_parametric": bool(c2_row["parametric"]),
                "c2_other": bool(c2_row["other"]),
                "c2_generated_token_count": int(c2_row["generated_token_count"]),
            }
        )
        prepared_rows.append(row)

    analysis_prepared_rows = [row for row in prepared_rows if not _is_supplement(row)]
    if not analysis_prepared_rows:
        raise RuntimeError("No non-supplement PopQA conflict rows available for intervention analysis.")

    analysis_ids = {str(row["item_id"]) for row in analysis_prepared_rows}
    c1_rows_analysis = [row for row in c1_rows if str(row["item_id"]) in analysis_ids]

    pilot_n = min(int(args.pilot_examples), len(prepared_rows))
    pilot_rows = _select_pilot_rows(prepared_rows, pilot_n)
    pilot_summary, _ = _pilot_summary(pilot_rows)
    pilot_summary.update(
        {
            "mean_e_token_set_size": float(sum(int(row["e_token_set_size"]) for row in pilot_rows) / max(len(pilot_rows), 1)),
            "mean_p_token_set_size": float(sum(int(row["p_token_set_size"]) for row in pilot_rows) / max(len(pilot_rows), 1)),
            "n_pilot_supplement": int(sum(1 for row in pilot_rows if _is_supplement(row))),
            "n_pilot_real_conflict": int(sum(1 for row in pilot_rows if not _is_supplement(row))),
            "n_input_supplement": int(sum(1 for row in prepared_rows if _is_supplement(row))),
            "n_input_real_conflict": int(len(analysis_prepared_rows)),
            "whitespace_token_ids": sorted(int(token_id) for token_id in whitespace_ids),
            "go_threshold_margin_auc": float(args.min_margin_auc),
            "go_threshold_min_parametric_items": int(args.min_parametric_pilot),
            "go_threshold_mean_answer_length_tokens": 5.0,
            "intervention_mode": str(args.intervention_mode),
            "evidence_token_source": str(args.evidence_token_source),
            "lambda_neg_values": [float(value) for value in lambda_values],
            "lambda_pos_values": [float(value) for value in lambda_pos_values],
        }
    )
    pilot_go = bool(
        pilot_summary["margin_auc"] is not None
        and float(pilot_summary["margin_auc"]) >= float(args.min_margin_auc)
        and float(pilot_summary["mean_answer_length_tokens"]) <= 5.0
        and int(pilot_summary["n_parametric"]) >= int(args.min_parametric_pilot)
    )
    pilot_summary["decision"] = "GO" if pilot_go else "NO-GO"
    write_json(results_dir / "pilot_margin_auc.json", pilot_summary)
    if not pilot_go:
        raise RuntimeError(
            "Pilot GO/NO-GO failed: "
            f"margin_auc={pilot_summary['margin_auc']} "
            f"mean_answer_length_tokens={pilot_summary['mean_answer_length_tokens']} "
            f"n_parametric={pilot_summary['n_parametric']}"
        )

    full_margin_auc = _margin_auc_for_c2(prepared_rows)
    selected_tau = float(args.fixed_tau) if args.fixed_tau is not None else float(pilot_summary["selected_tau"])
    lambda_rows: list[dict[str, Any]] = []
    sweep_cache: dict[tuple[float, float], tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    baseline_c2_rows_all = [
        {
            **row,
            "generated_answer": row["c2_generated_answer"],
            "classification": row["c2_classification"],
            "context_faithful": row["c2_context_faithful"],
            "parametric": row["c2_parametric"],
            "other": row["c2_other"],
            "generated_token_count": row["c2_generated_token_count"],
            "gate_fired": False,
        }
        for row in prepared_rows
    ]
    baseline_c2_rows = [row for row in baseline_c2_rows_all if str(row["item_id"]) in analysis_ids]
    baseline_c2_summary = _condition_summary(
        condition_id="C2",
        condition_name="Short context, no intervention",
        rows=baseline_c2_rows,
        baseline_c2_rows=baseline_c2_rows,
        tau=None,
        lambda_neg=None,
        margin_auc=None if full_margin_auc is None else float(full_margin_auc),
    )

    sweep_values = list(lambda_values)
    if args.fixed_lambda is not None:
        sweep_values = [float(args.fixed_lambda)]
    sweep_pos_values = list(lambda_pos_values)
    if args.fixed_lambda_pos is not None:
        sweep_pos_values = [float(args.fixed_lambda_pos)]

    for lambda_pos in sweep_pos_values:
        for lambda_neg in sweep_values:
            c3_rows = _run_condition(
                rows=analysis_prepared_rows,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                condition_name=f"C3 pos={lambda_pos:g} neg={lambda_neg:g}",
                prompt_key="_c2_prompt_ids",
                tau=float(selected_tau),
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                intervention_mode=str(args.intervention_mode),
                n_suppress_steps=int(args.n_suppress_steps),
            )
            c3_summary = _condition_summary(
                condition_id="C3",
                condition_name=f"Short context + {args.intervention_mode} mechanism pos={lambda_pos:g} neg={lambda_neg:g}",
                rows=c3_rows,
                baseline_c2_rows=baseline_c2_rows,
                tau=float(selected_tau),
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                intervention_mode=str(args.intervention_mode),
                margin_auc=None if full_margin_auc is None else float(full_margin_auc),
            )
            c3_summary["youden_j"] = float(c3_summary["restoration_rate"] - c3_summary["harm_rate"])
            sweep_cache[(float(lambda_pos), float(lambda_neg))] = (c3_rows, c3_summary)
            lambda_rows.append(
                {
                    "intervention_mode": str(args.intervention_mode),
                    "lambda_pos": float(lambda_pos),
                    "lambda_neg": float(lambda_neg),
                    "restoration_rate": float(c3_summary["restoration_rate"]),
                    "restoration_rate_ci95_wilson": c3_summary["restoration_rate_ci95_wilson"],
                    "harm_rate": float(c3_summary["harm_rate"]),
                    "harm_rate_ci95_wilson": c3_summary["harm_rate_ci95_wilson"],
                    "faithfulness_rate": float(c3_summary["faithfulness_rate"]),
                    "gate_fire_rate": float(c3_summary["gate_fire_rate"]),
                    "youden_j": float(c3_summary["youden_j"]),
                }
            )

    def _rank_lambda_row(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            -float(row["youden_j"]),
            -float(row["restoration_rate"]),
            float(row["harm_rate"]),
            float(row["lambda_pos"]) + float(row["lambda_neg"]),
            float(row["lambda_pos"]),
            float(row["lambda_neg"]),
        )

    suppression_rows = [row for row in lambda_rows if float(row["lambda_pos"]) == 0.0]
    positive_promotion_rows = [row for row in lambda_rows if float(row["lambda_pos"]) > 0.0]
    best_suppression_row = sorted(suppression_rows, key=_rank_lambda_row)[0] if suppression_rows else None
    best_balanced_row = sorted(positive_promotion_rows, key=_rank_lambda_row)[0] if positive_promotion_rows else None
    best_overall_row = sorted(lambda_rows, key=_rank_lambda_row)[0]

    if args.fixed_lambda is not None and args.fixed_lambda_pos is not None:
        best_lambda_row = lambda_rows[0]
        selected_lambda_pos = float(args.fixed_lambda_pos)
        selected_lambda_neg = float(args.fixed_lambda)
    elif str(args.intervention_mode) == "balanced" and best_balanced_row is not None:
        best_lambda_row = best_balanced_row
        selected_lambda_pos = float(best_lambda_row["lambda_pos"])
        selected_lambda_neg = float(best_lambda_row["lambda_neg"])
    else:
        best_lambda_row = best_overall_row
        selected_lambda_pos = float(best_lambda_row["lambda_pos"])
        selected_lambda_neg = float(best_lambda_row["lambda_neg"])
    lambda_sweep = {
        "intervention_mode": str(args.intervention_mode),
        "evidence_token_source": str(args.evidence_token_source),
        "selected_tau": float(selected_tau),
        "selected_lambda_pos": float(selected_lambda_pos),
        "selected_lambda_neg": float(selected_lambda_neg),
        "rows": lambda_rows,
        "best_row": best_lambda_row,
        "best_overall_row": best_overall_row,
        "best_suppression_row": best_suppression_row,
        "best_balanced_row": best_balanced_row,
    }
    write_json(results_dir / "lambda_sweep.json", lambda_sweep)

    c3_rows, c3_summary = sweep_cache[(float(selected_lambda_pos), float(selected_lambda_neg))]
    c4_rows = _run_condition(
        rows=analysis_prepared_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        condition_name="C4 authoritative prompt",
        prompt_key="_c4_prompt_ids",
    )
    c4_summary = _condition_summary(
        condition_id="C4",
        condition_name="Authoritative context, no intervention",
        rows=c4_rows,
        baseline_c2_rows=baseline_c2_rows,
        tau=None,
        lambda_neg=None,
        margin_auc=None if full_margin_auc is None else float(full_margin_auc),
    )
    c5_rows = _run_condition(
        rows=analysis_prepared_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        condition_name="C5 explicit answer injection",
        prompt_key="_c5_prompt_ids",
    )
    c5_summary = _condition_summary(
        condition_id="C5",
        condition_name="Explicit answer injection, no intervention",
        rows=c5_rows,
        baseline_c2_rows=baseline_c2_rows,
        tau=None,
        lambda_neg=None,
        margin_auc=None if full_margin_auc is None else float(full_margin_auc),
    )

    c1_summary = _condition_summary(
        condition_id="C1",
        condition_name="No context, no intervention",
        rows=c1_rows_analysis,
        baseline_c2_rows=baseline_c2_rows,
        tau=None,
        lambda_neg=None,
        margin_auc=None if full_margin_auc is None else float(full_margin_auc),
    )
    c3_summary["condition_name"] = "Short context + selected mechanism"

    final_rows: list[dict[str, Any]] = []
    c1_by_id = {str(row["item_id"]): row for row in c1_rows}
    c2_by_id = {str(row["item_id"]): row for row in baseline_c2_rows}
    c3_by_id = {str(row["item_id"]): row for row in c3_rows}
    c4_by_id = {str(row["item_id"]): row for row in c4_rows}
    c5_by_id = {str(row["item_id"]): row for row in c5_rows}
    for row in analysis_prepared_rows:
        item_id = str(row["item_id"])
        c1_row = c1_by_id[item_id]
        c2_row = c2_by_id[item_id]
        c3_row = c3_by_id[item_id]
        c4_row = c4_by_id[item_id]
        c5_row = c5_by_id[item_id]
        final_rows.append(
            {
                "item_id": item_id,
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "correct_aliases": row.get("correct_aliases", [row["correct_answer"]]),
                "parametric_answer": row["parametric_answer"],
                "context": row["context"],
                "explicit_injection_context": row["explicit_injection_context"],
                "is_supplement": bool(row.get("is_supplement", False)),
                "supplement_source": str(row.get("supplement_source", "")),
                "prop": row["prop"],
                "s_pop": row["s_pop"],
                "llm_answer_prob": row["llm_answer_prob"],
                "prefill_margin": row["prefill_margin"],
                "e_token_set": row["e_token_set"],
                "p_token_set": row["p_token_set"],
                "e_token_set_size": row["e_token_set_size"],
                "p_token_set_size": row["p_token_set_size"],
                "evidence_token_source": row["evidence_token_source"],
                "evidence_token_forms": row["evidence_token_forms"],
                "ambiguous_token_sets": row["ambiguous_token_sets"],
                "intervention_mode": str(args.intervention_mode),
                "selected_lambda_pos": float(selected_lambda_pos),
                "selected_lambda_neg": float(selected_lambda_neg),
                "c1_generated_answer": c1_row["generated_answer"],
                "c1_classification": c1_row["classification"],
                "c2_generated_answer": c2_row["generated_answer"],
                "c2_classification": c2_row["classification"],
                "c3_generated_answer": c3_row["generated_answer"],
                "c3_classification": c3_row["classification"],
                "c3_gate_fired": bool(c3_row["gate_fired"]),
                "c4_generated_answer": c4_row["generated_answer"],
                "c4_classification": c4_row["classification"],
                "c5_generated_answer": c5_row["generated_answer"],
                "c5_classification": c5_row["classification"],
                "c3_context_faithful": bool(c3_row["context_faithful"]),
                "c5_context_faithful": bool(c5_row["context_faithful"]),
                "c2_parametric": bool(c2_row["parametric"]),
                "c2_context_faithful": bool(c2_row["context_faithful"]),
            }
        )

    stratification = {
        "s_pop": _stratify_numeric(final_rows, "s_pop"),
        "llm_answer_prob": _stratify_numeric(final_rows, "llm_answer_prob"),
        "prop": _stratify_prop(final_rows, top_k=8),
    }

    summary = {
        "n_items": int(len(analysis_prepared_rows)),
        "n_input_items": int(len(prepared_rows)),
        "n_input_supplement": int(sum(1 for row in prepared_rows if _is_supplement(row))),
        "n_analysis_items": int(len(analysis_prepared_rows)),
        "pilot": pilot_summary,
        "intervention_mode": str(args.intervention_mode),
        "evidence_token_source": str(args.evidence_token_source),
        "margin_auc_full": None if full_margin_auc is None else float(full_margin_auc),
        "selected_tau": float(selected_tau),
        "selected_lambda_pos": float(selected_lambda_pos),
        "selected_lambda_neg": float(selected_lambda_neg),
        "c1_faithfulness": float(c1_summary["faithfulness_rate"]),
        "c2_faithfulness": float(baseline_c2_summary["faithfulness_rate"]),
        "c3_faithfulness": float(c3_summary["faithfulness_rate"]),
        "c4_faithfulness": float(c4_summary["faithfulness_rate"]),
        "c5_faithfulness": float(c5_summary["faithfulness_rate"]),
        "restoration_rate": float(c3_summary["restoration_rate"]),
        "harm_rate": float(c3_summary["harm_rate"]),
        "gate_fire_rate": float(c3_summary["gate_fire_rate"]),
        "best_suppression_row": best_suppression_row,
        "best_balanced_row": best_balanced_row,
        "conditions": {
            "C1": c1_summary,
            "C2": baseline_c2_summary,
            "C3": c3_summary,
            "C4": c4_summary,
            "C5": c5_summary,
        },
        "lambda_sweep": lambda_sweep,
        "mcnemar_c2_vs_c3": c3_summary["mcnemar_vs_c2"],
        "stratification": stratification,
        "metadata": build_metadata(
            args=args,
            input_files=[input_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=REPO_ROOT,
        ),
    }
    write_jsonl(results_dir / "results.jsonl", final_rows)
    write_json(results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"pilot={results_dir / 'pilot_margin_auc.json'}")
    print(f"lambda_sweep={results_dir / 'lambda_sweep.json'}")
    print(f"summary={results_dir / 'summary.json'}")
    print(f"results={results_dir / 'results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
