#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
CONFLICTQA_ROOT = REPO_ROOT / "conflictqa"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))
sys.path.insert(0, str(CONFLICTQA_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from conflictqa_pipeline.utils import (
    auc_rank,
    build_metadata,
    margin_from_logits,
    mcnemar_from_paired,
    wilson_ci,
)

NUMBER_PATTERN = re.compile(r"\d+\.?\d*")


class NStepSuppressProcessor:
    """Suppress parametric tokens for the first n generation steps."""

    def __init__(self, suppress_ids: set[int], lambda_neg: float, n_steps: int = 3):
        self.suppress_ids = {int(token_id) for token_id in suppress_ids}
        self.lambda_neg = float(lambda_neg)
        self.n_steps = int(n_steps)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self._step < self.n_steps:
            for token_id in self.suppress_ids:
                if 0 <= int(token_id) < int(scores.shape[-1]):
                    scores[..., int(token_id)] -= self.lambda_neg
        self._step += 1
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClashEval v3 intervention with whitespace-filtered token sets.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medical_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/clasheval_intervention_v3")
    parser.add_argument("--pilot_examples", type=int, default=100)
    parser.add_argument("--n_examples", type=int, default=500, help="0 means all examples.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--classification_window_tokens", type=int, default=10)
    parser.add_argument("--min_margin_auc", type=float, default=0.70)
    parser.add_argument("--lambda_values", default="2.5,5.0,7.5,10.0,15.0")
    parser.add_argument("--n_suppress_steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one lambda value.")
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(clean) / max(len(clean), 1)) if clean else 0.0


def _load_conflicts(path: Path) -> list[dict[str, Any]]:
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


def _context_text(ex: dict[str, Any]) -> str:
    if ex.get("context"):
        return str(ex["context"]).strip()
    if ex.get("guideline_text"):
        return str(ex["guideline_text"]).strip()
    if ex.get("retrieved_passages"):
        passages = [str(x).strip() for x in (ex.get("retrieved_passages") or []) if str(x).strip()]
        if passages:
            return "\n\n".join(passages)
    return ""


def _first_number(text: str) -> str:
    matches = NUMBER_PATTERN.findall(str(text or ""))
    return str(matches[0]).strip() if matches else ""


def _truncate_context_to_dosage_sentence(context_text: str, context_answer: str) -> tuple[str, bool]:
    context = str(context_text or "").strip()
    dosage = _first_number(context_answer)
    if not context or not dosage:
        return context, False
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", context) if part.strip()]
    for part in parts:
        if dosage in part:
            return part, True
    hit = context.find(dosage)
    if hit < 0:
        return context, False
    left = max(context.rfind(".", 0, hit), context.rfind("!", 0, hit), context.rfind("?", 0, hit), context.rfind("\n", 0, hit))
    right_candidates = [pos for pos in (context.find(".", hit), context.find("!", hit), context.find("?", hit), context.find("\n", hit)) if pos >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(context)
    return context[left + 1 : right].strip(), True


def _build_full_context_prompt(question: str, context_text: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Clinical Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer with ONLY the dosage number and unit. Example: 10 mg"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_fixed_prompt(question: str, context_text: str, context_answer: str) -> tuple[str, str, bool]:
    prompt_context, found = _truncate_context_to_dosage_sentence(context_text, context_answer)
    prompt_context = prompt_context or str(context_text or "").strip()
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Question: {question}\n"
        f"Clinical Context Sentence:\n{prompt_context}\n\n"
        "Complete the answer using ONLY the dosage number and unit.\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Dosage:"
    ), prompt_context, found


def _resolve_pad_token_id(model, tokenizer) -> int:
    candidates = [
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None
        if candidate is not None:
            return int(candidate)
    return 0


def _encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length)).to(device)
    return encoded["input_ids"][0], encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(device)


def _generate_text(
    *,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    logits_processor=None,
) -> tuple[str, list[int]]:
    kwargs: dict[str, Any] = {}
    if logits_processor is not None:
        from transformers import LogitsProcessorList

        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
    with torch.inference_mode():
        out = model.generate(
            input_ids=prompt_ids.unsqueeze(0),
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=_resolve_pad_token_id(model, tokenizer),
            **kwargs,
        )
    new_ids = [int(x) for x in out[0, prompt_ids.shape[0] :].detach().cpu().tolist()]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), new_ids


def _decode_window(tokenizer, ids: list[int], n_tokens: int) -> str:
    return tokenizer.decode(ids[: max(0, int(n_tokens))], skip_special_tokens=True).strip()


def _contains_dosage(text: str, answer_text: str) -> bool:
    dosage = _first_number(answer_text)
    if not dosage:
        return False
    return re.search(rf"(?<![\d.]){re.escape(dosage)}(?![\d.])", str(text or "")) is not None


def _classify_window(window_text: str, context_answer: str, parametric_answer: str) -> str:
    if _contains_dosage(window_text, context_answer):
        return "faithful"
    if _contains_dosage(window_text, parametric_answer):
        return "hallucinated"
    return "other"


def _whitespace_token_ids(tokenizer) -> set[int]:
    whitespace_strings = [" ", "\n", "\t", "  ", "\n\n", " \n", "\n "]
    ids: set[int] = set()
    for text in whitespace_strings:
        for token_id in tokenizer.encode(text, add_special_tokens=False):
            ids.add(int(token_id))
    return ids


def _dosage_token_sets(
    tokenizer,
    context_dosage: str,
    parametric_dosage: str,
    whitespace_ids: set[int],
) -> tuple[set[int], set[int], int, bool]:
    def _variants(dosage: str) -> set[int]:
        tokens: set[int] = set()
        dosage = str(dosage)
        for variant in (dosage, f" {dosage}", f"\n{dosage}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            tokens.update(int(token_id) for token_id in ids[:2])
        tokens -= whitespace_ids
        return tokens

    e_tokens = _variants(context_dosage)
    p_tokens = _variants(parametric_dosage)
    overlap = e_tokens & p_tokens
    identical = e_tokens == p_tokens
    if identical:
        return set(), set(), int(len(overlap)), True
    return e_tokens, p_tokens, int(len(overlap)), False


def _validate_no_whitespace_tokens(token_ids: set[int], whitespace_ids: set[int], label: str, item_id: Any) -> None:
    overlap = sorted(int(token_id) for token_id in (set(token_ids) & set(whitespace_ids)))
    if overlap:
        raise RuntimeError(f"Whitespace token contamination in {label} for item {item_id}: {overlap}")


def _pilot_margin_summary(pilot_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    labeled_rows = [
        row
        for row in pilot_rows
        if str(row["c2_classification"]) in {"faithful", "hallucinated"} and math.isfinite(float(row["prefill_margin"]))
    ]
    faithful_margins = [float(row["prefill_margin"]) for row in labeled_rows if str(row["c2_classification"]) == "faithful"]
    hallucinated_margins = [float(row["prefill_margin"]) for row in labeled_rows if str(row["c2_classification"]) == "hallucinated"]
    labels = [1 for _ in faithful_margins] + [0 for _ in hallucinated_margins]
    scores = faithful_margins + hallucinated_margins
    auc = auc_rank(scores, labels)

    thresholds: list[float] = []
    unique_scores = sorted(set(scores))
    if unique_scores:
        thresholds.append(unique_scores[0] - 1e-6)
        for left, right in zip(unique_scores[:-1], unique_scores[1:]):
            thresholds.append((float(left) + float(right)) / 2.0)
        thresholds.append(unique_scores[-1] + 1e-6)

    tau_rows: list[dict[str, float]] = []
    if faithful_margins and hallucinated_margins:
        for tau in thresholds:
            tpr = float(sum(1 for margin in hallucinated_margins if margin < tau) / len(hallucinated_margins))
            fpr = float(sum(1 for margin in faithful_margins if margin < tau) / len(faithful_margins))
            tau_rows.append(
                {
                    "tau": float(tau),
                    "tpr_hallucinated": tpr,
                    "fpr_faithful": fpr,
                    "youden_j": float(tpr - fpr),
                }
            )

    best_tau = float("nan")
    best_row: dict[str, float] | None = None
    if tau_rows:
        best_row = sorted(
            tau_rows,
            key=lambda row: (-float(row["youden_j"]), -float(row["tpr_hallucinated"]), float(row["fpr_faithful"]), float(row["tau"])),
        )[0]
        best_tau = float(best_row["tau"])

    summary = {
        "pilot_examples": int(len(pilot_rows)),
        "n_margin_faithful": int(len(faithful_margins)),
        "n_margin_hallucinated": int(len(hallucinated_margins)),
        "margin_auc": None if auc is None else float(auc),
        "mean_margin_faithful": _mean(faithful_margins),
        "mean_margin_hallucinated": _mean(hallucinated_margins),
        "selected_tau": None if best_row is None else float(best_tau),
        "selected_tau_metrics": best_row,
        "faithful_margins": faithful_margins,
        "hallucinated_margins": hallucinated_margins,
        "tau_candidates": tau_rows,
    }
    return summary, tau_rows


def _condition_counts(rows: list[dict[str, Any]], key: str) -> tuple[int, int, int]:
    faithful = sum(1 for row in rows if str(row[key]) == "faithful")
    hallucinated = sum(1 for row in rows if str(row[key]) == "hallucinated")
    other = sum(1 for row in rows if str(row[key]) == "other")
    return faithful, hallucinated, other


def _summarize_full_condition(rows: list[dict[str, Any]], *, tau: float, lambda_neg: float, n_suppress_steps: int, margin_auc_full: float | None) -> dict[str, Any]:
    c1_faithful, c1_hallucinated, c1_other = _condition_counts(rows, "c1_classification")
    c2_faithful, c2_hallucinated, c2_other = _condition_counts(rows, "c2_classification")
    c3_faithful, c3_hallucinated, c3_other = _condition_counts(rows, "c3_classification")
    n_examples = int(len(rows))
    c2_hall_rows = [row for row in rows if str(row["c2_classification"]) == "hallucinated"]
    c2_faith_rows = [row for row in rows if str(row["c2_classification"]) == "faithful"]
    n_restored = sum(1 for row in c2_hall_rows if str(row["c3_classification"]) == "faithful")
    n_harmed = sum(1 for row in c2_faith_rows if str(row["c3_classification"]) != "faithful")
    gate_fired_count = sum(1 for row in rows if bool(row["gate_fired"]))
    ambiguous_count = sum(1 for row in rows if bool(row["ambiguous_token_sets"]))

    c2_correct = [str(row["c2_classification"]) == "faithful" for row in rows]
    c3_correct = [str(row["c3_classification"]) == "faithful" for row in rows]

    restoration = float(n_restored / max(len(c2_hall_rows), 1))
    harm = float(n_harmed / max(len(c2_faith_rows), 1))
    return {
        "n_examples": n_examples,
        "c1_faithfulness": float(c1_faithful / max(n_examples, 1)),
        "c1_faithfulness_ci95_wilson": wilson_ci(c1_faithful, n_examples),
        "c2_faithfulness": float(c2_faithful / max(n_examples, 1)),
        "c2_faithfulness_ci95_wilson": wilson_ci(c2_faithful, n_examples),
        "c3_faithfulness": float(c3_faithful / max(n_examples, 1)),
        "c3_faithfulness_ci95_wilson": wilson_ci(c3_faithful, n_examples),
        "c1_hallucinated": float(c1_hallucinated / max(n_examples, 1)),
        "c2_hallucinated": float(c2_hallucinated / max(n_examples, 1)),
        "c3_hallucinated": float(c3_hallucinated / max(n_examples, 1)),
        "c1_other": float(c1_other / max(n_examples, 1)),
        "c2_other": float(c2_other / max(n_examples, 1)),
        "c3_other": float(c3_other / max(n_examples, 1)),
        "restoration_on_c2_hallucinated": restoration,
        "restoration_on_c2_hallucinated_ci95_wilson": wilson_ci(n_restored, len(c2_hall_rows)),
        "n_restored": int(n_restored),
        "n_c2_hallucinated": int(len(c2_hall_rows)),
        "harm_rate": harm,
        "harm_rate_ci95_wilson": wilson_ci(n_harmed, len(c2_faith_rows)),
        "n_harmed": int(n_harmed),
        "n_c2_faithful": int(len(c2_faith_rows)),
        "gate_fire_rate": float(gate_fired_count / max(n_examples, 1)),
        "gate_fire_rate_ci95_wilson": wilson_ci(gate_fired_count, n_examples),
        "margin_auc_full": None if margin_auc_full is None else float(margin_auc_full),
        "mcnemar_c2_vs_c3": mcnemar_from_paired(c2_correct, c3_correct),
        "mean_e_token_set_size": _mean([float(row["e_token_set_size"]) for row in rows]),
        "mean_p_token_set_size": _mean([float(row["p_token_set_size"]) for row in rows]),
        "n_ambiguous_token_sets": int(ambiguous_count),
        "tau": float(tau),
        "lambda_neg": float(lambda_neg),
        "n_suppress_steps": int(n_suppress_steps),
        "mechanism_mode": "n_step_token_suppress_whitespace_filtered",
        "youden_j": float(restoration - harm),
    }


def _run_c3(
    *,
    baseline_rows: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    classification_window_tokens: int,
    tau: float,
    lambda_neg: float,
    n_suppress_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for base_row in tqdm(baseline_rows, desc=f"C3 sweep lambda={lambda_neg:g}"):
        c2_ids, c2_mask = _encode_prompt(tokenizer, str(base_row["_c2_prompt"]), int(max_length), device)
        gate_fired = bool(
            not bool(base_row["ambiguous_token_sets"])
            and math.isfinite(float(base_row["prefill_margin"]))
            and float(base_row["prefill_margin"]) < float(tau)
        )
        processor = (
            NStepSuppressProcessor(
                suppress_ids=set(int(token_id) for token_id in base_row["p_token_set"]),
                lambda_neg=float(lambda_neg),
                n_steps=int(n_suppress_steps),
            )
            if gate_fired
            else None
        )
        c3_text, c3_new_ids = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=c2_ids,
            attention_mask=c2_mask,
            max_new_tokens=int(max_new_tokens),
            logits_processor=processor,
        )
        c3_window = _decode_window(tokenizer, c3_new_ids, int(classification_window_tokens))
        c3_classification = _classify_window(c3_window, str(base_row["context_answer"]), str(base_row["parametric_answer"]))
        row = {
            key: value
            for key, value in base_row.items()
            if not str(key).startswith("_")
        }
        row.update(
            {
                "generated_c3": c3_text,
                "c3_window_text": c3_window,
                "c3_classification": c3_classification,
                "gate_fired": bool(gate_fired),
                "n_suppress_steps": int(n_suppress_steps),
                "lambda_neg": float(lambda_neg),
                "tau": float(tau),
            }
        )
        out_rows.append(row)

    labeled_rows = [
        row
        for row in baseline_rows
        if str(row["c2_classification"]) in {"faithful", "hallucinated"} and math.isfinite(float(row["prefill_margin"]))
    ]
    margin_auc_full = auc_rank(
        [float(row["prefill_margin"]) for row in labeled_rows],
        [1 if str(row["c2_classification"]) == "faithful" else 0 for row in labeled_rows],
    )
    summary = _summarize_full_condition(
        out_rows,
        tau=float(tau),
        lambda_neg=float(lambda_neg),
        n_suppress_steps=int(n_suppress_steps),
        margin_auc_full=margin_auc_full,
    )
    return out_rows, summary


def main() -> int:
    args = parse_args()
    lambda_values = _parse_float_list(args.lambda_values)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    t0 = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = (PROJECT_ROOT / args.conflict_path).resolve()

    rows = _load_conflicts(conflict_path)
    if int(args.n_examples) > 0:
        rows = rows[: int(args.n_examples)]
    if not rows:
        raise RuntimeError(f"No rows found in {conflict_path}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    whitespace_ids = _whitespace_token_ids(tokenizer)

    baseline_rows: list[dict[str, Any]] = []
    skip_stats = {"missing_question_or_context": 0, "missing_numeric_answers": 0}

    for idx, ex in enumerate(tqdm(rows, desc="ClashEval v3 baseline")):
        question = str(ex.get("question", "")).strip()
        context = _context_text(ex)
        context_answer = str(ex.get("context_answer", "")).strip()
        parametric_answer = str(ex.get("parametric_answer", "")).strip()
        if not question or not context:
            skip_stats["missing_question_or_context"] += 1
            continue

        context_dosage = _first_number(context_answer)
        parametric_dosage = _first_number(parametric_answer)
        if not context_dosage or not parametric_dosage:
            skip_stats["missing_numeric_answers"] += 1
            continue

        c1_prompt = _build_full_context_prompt(question, context)
        c2_prompt, prompt_context, context_sentence_found = _build_fixed_prompt(question, context, context_answer)
        c1_ids, c1_mask = _encode_prompt(tokenizer, c1_prompt, int(args.max_length), device)
        c2_ids, c2_mask = _encode_prompt(tokenizer, c2_prompt, int(args.max_length), device)

        c1_text, c1_new_ids = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=c1_ids,
            attention_mask=c1_mask,
            max_new_tokens=int(args.max_new_tokens),
        )
        c2_text, c2_new_ids = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=c2_ids,
            attention_mask=c2_mask,
            max_new_tokens=int(args.max_new_tokens),
        )
        c1_window = _decode_window(tokenizer, c1_new_ids, int(args.classification_window_tokens))
        c2_window = _decode_window(tokenizer, c2_new_ids, int(args.classification_window_tokens))
        c1_classification = _classify_window(c1_window, context_answer, parametric_answer)
        c2_classification = _classify_window(c2_window, context_answer, parametric_answer)

        with torch.inference_mode():
            c2_logits = model(
                input_ids=c2_ids.unsqueeze(0),
                attention_mask=c2_mask,
                use_cache=False,
                return_dict=True,
            ).logits[0, -1, :].detach().float().cpu()

        item_id = ex.get("id", ex.get("item_id", idx))
        e_tokens, p_tokens, overlap_count, ambiguous_token_sets = _dosage_token_sets(
            tokenizer,
            context_dosage,
            parametric_dosage,
            whitespace_ids,
        )
        _validate_no_whitespace_tokens(e_tokens, whitespace_ids, "e_token_set", item_id)
        _validate_no_whitespace_tokens(p_tokens, whitespace_ids, "p_token_set", item_id)
        prefill_margin = margin_from_logits(c2_logits, e_tokens, p_tokens) if not ambiguous_token_sets else float("nan")

        baseline_rows.append(
            {
                "item_id": item_id,
                "source": str(ex.get("source", "")),
                "question": question,
                "context_answer": context_answer,
                "parametric_answer": parametric_answer,
                "context_dosage": context_dosage,
                "parametric_dosage": parametric_dosage,
                "prompt_context": prompt_context,
                "context_sentence_found": bool(context_sentence_found),
                "ambiguous_token_sets": bool(ambiguous_token_sets),
                "e_token_set": sorted(int(tok) for tok in e_tokens),
                "p_token_set": sorted(int(tok) for tok in p_tokens),
                "e_token_set_size": int(len(e_tokens)),
                "p_token_set_size": int(len(p_tokens)),
                "token_overlap_count": int(overlap_count),
                "c1_generated": c1_text,
                "c2_generated": c2_text,
                "c1_window_text": c1_window,
                "c2_window_text": c2_window,
                "c1_classification": c1_classification,
                "c2_classification": c2_classification,
                "prefill_margin": float(prefill_margin) if math.isfinite(float(prefill_margin)) else float("nan"),
                "_c2_prompt": c2_prompt,
            }
        )

    if not baseline_rows:
        raise RuntimeError(f"No usable rows. skip_stats={skip_stats}")

    pilot_n = min(int(args.pilot_examples), len(baseline_rows))
    pilot_rows = baseline_rows[:pilot_n]
    pilot_summary, tau_rows = _pilot_margin_summary(pilot_rows)
    pilot_summary.update(
        {
            "mean_e_token_set_size": _mean([float(row["e_token_set_size"]) for row in pilot_rows]),
            "mean_p_token_set_size": _mean([float(row["p_token_set_size"]) for row in pilot_rows]),
            "whitespace_token_ids": sorted(int(token_id) for token_id in whitespace_ids),
            "min_margin_auc_threshold": float(args.min_margin_auc),
            "proceed_to_sweep": bool(
                pilot_summary["margin_auc"] is not None and float(pilot_summary["margin_auc"]) >= float(args.min_margin_auc)
            ),
        }
    )
    _write_json(out_dir / "pilot_margin_auc.json", pilot_summary)

    if pilot_summary["margin_auc"] is None or float(pilot_summary["margin_auc"]) < float(args.min_margin_auc):
        raise RuntimeError(
            f"Pilot margin_auc below threshold: {pilot_summary['margin_auc']} < {float(args.min_margin_auc):.3f}. "
            "Stopping before lambda sweep."
        )

    selected_tau = float(pilot_summary["selected_tau"])
    lambda_rows: list[dict[str, Any]] = []
    sweep_cache: dict[float, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for lambda_neg in lambda_values:
        c3_rows, c3_summary = _run_c3(
            baseline_rows=baseline_rows,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            classification_window_tokens=int(args.classification_window_tokens),
            tau=float(selected_tau),
            lambda_neg=float(lambda_neg),
            n_suppress_steps=int(args.n_suppress_steps),
        )
        sweep_cache[float(lambda_neg)] = (c3_rows, c3_summary)
        lambda_rows.append(
            {
                "lambda_neg": float(lambda_neg),
                "restoration_on_c2_hallucinated": float(c3_summary["restoration_on_c2_hallucinated"]),
                "restoration_on_c2_hallucinated_ci95_wilson": c3_summary["restoration_on_c2_hallucinated_ci95_wilson"],
                "harm_rate": float(c3_summary["harm_rate"]),
                "harm_rate_ci95_wilson": c3_summary["harm_rate_ci95_wilson"],
                "c3_faithfulness": float(c3_summary["c3_faithfulness"]),
                "gate_fire_rate": float(c3_summary["gate_fire_rate"]),
                "youden_j": float(c3_summary["youden_j"]),
            }
        )

    best_lambda_row = sorted(
        lambda_rows,
        key=lambda row: (-float(row["youden_j"]), -float(row["restoration_on_c2_hallucinated"]), float(row["harm_rate"]), float(row["lambda_neg"])),
    )[0]
    selected_lambda = float(best_lambda_row["lambda_neg"])
    lambda_sweep = {
        "selected_tau": float(selected_tau),
        "lambda_values": [float(value) for value in lambda_values],
        "rows": lambda_rows,
        "best_row": best_lambda_row,
    }
    _write_json(out_dir / "lambda_sweep.json", lambda_sweep)

    final_rows, final_summary = sweep_cache[selected_lambda]
    final_summary.update(
        {
            "pilot_examples": int(pilot_n),
            "pilot_margin_auc": float(pilot_summary["margin_auc"]),
            "selected_tau": float(selected_tau),
            "selected_lambda_neg": float(selected_lambda),
            "whitespace_token_ids": sorted(int(token_id) for token_id in whitespace_ids),
            "skip_stats": skip_stats,
            "metadata": build_metadata(
                args=args,
                input_files=[conflict_path],
                seed=int(args.seed),
                wall_clock_seconds=float(time.perf_counter() - t0),
                project_root=REPO_ROOT,
            ),
        }
    )
    _write_jsonl(out_dir / "results.jsonl", final_rows)
    _write_json(out_dir / "summary.json", final_summary)

    print(json.dumps({"pilot_margin_auc": pilot_summary, "lambda_sweep": lambda_sweep, "summary": final_summary}, indent=2))
    print(f"pilot={out_dir / 'pilot_margin_auc.json'}")
    print(f"lambda_sweep={out_dir / 'lambda_sweep.json'}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"results={out_dir / 'results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
