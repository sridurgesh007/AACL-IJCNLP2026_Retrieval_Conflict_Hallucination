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
        if self._step < self.n_steps:
            for token_id in self.suppress_ids:
                if 0 <= int(token_id) < int(scores.shape[-1]):
                    scores[..., int(token_id)] -= self.lambda_neg
        self._step += 1
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed ClashEval kill-test with first-token suppression.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medical_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/clasheval_intervention_fixed")
    parser.add_argument("--n_examples", type=int, default=500, help="0 means all examples.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--classification_window_tokens", type=int, default=10)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lambda_neg", type=float, default=2.5)
    parser.add_argument("--n_suppress_steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


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
    return float(sum(values) / max(len(values), 1)) if values else 0.0


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


def _first_number_value(text: str) -> float | None:
    value = _first_number(text)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


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
    for cand in candidates:
        if isinstance(cand, (list, tuple)):
            cand = cand[0] if cand else None
        if cand is not None:
            return int(cand)
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


def _dosage_token_sets(tokenizer, context_dosage: str, parametric_dosage: str) -> tuple[set[int], set[int], int]:
    def _variants(dosage: str) -> set[int]:
        tokens: set[int] = set()
        dosage = str(dosage)
        for prefix in ("", " ", "\n"):
            ids = tokenizer.encode(prefix + dosage, add_special_tokens=False)
            tokens.update(int(token_id) for token_id in ids[:2])
            ids2 = tokenizer.encode(dosage, add_special_tokens=False)
            tokens.update(int(token_id) for token_id in ids2[:2])
        return tokens

    e_tokens = _variants(context_dosage)
    p_tokens = _variants(parametric_dosage)
    overlap = e_tokens & p_tokens
    if e_tokens == p_tokens:
        return set(), set(), int(len(overlap))
    return e_tokens, p_tokens, int(len(overlap))


def _condition_counts(rows: list[dict[str, Any]], key: str) -> tuple[int, int, int]:
    faithful = sum(1 for row in rows if str(row[key]) == "faithful")
    hallucinated = sum(1 for row in rows if str(row[key]) == "hallucinated")
    other = sum(1 for row in rows if str(row[key]) == "other")
    return faithful, hallucinated, other


def main() -> int:
    args = parse_args()
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

    per_items: list[dict[str, Any]] = []
    fired_hallucinated_diagnostics: list[dict[str, Any]] = []
    skip_stats = {"missing_question_or_context": 0, "missing_numeric_answers": 0}

    for idx, ex in enumerate(tqdm(rows, desc="ClashEval fixed intervention")):
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

        e_tokens, p_tokens, overlap_count = _dosage_token_sets(tokenizer, context_dosage, parametric_dosage)
        ambiguous_token_sets = not e_tokens or not p_tokens
        prefill_margin = margin_from_logits(c2_logits, e_tokens, p_tokens) if not ambiguous_token_sets else float("nan")
        gate_fired = bool(
            not ambiguous_token_sets
            and math.isfinite(float(prefill_margin))
            and float(prefill_margin) < float(args.tau)
        )
        processor = (
            NStepSuppressProcessor(
                suppress_ids=set(p_tokens),
                lambda_neg=float(args.lambda_neg),
                n_steps=int(args.n_suppress_steps),
            )
            if gate_fired
            else None
        )

        c3_text, c3_new_ids = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=c2_ids,
            attention_mask=c2_mask,
            max_new_tokens=int(args.max_new_tokens),
            logits_processor=processor,
        )
        c3_window = _decode_window(tokenizer, c3_new_ids, int(args.classification_window_tokens))
        c3_classification = _classify_window(c3_window, context_answer, parametric_answer)
        c3_first3_token_ids = [int(token_id) for token_id in c3_new_ids[:3]]
        c3_first3_tokens_text = [
            tokenizer.decode([int(token_id)], skip_special_tokens=True) for token_id in c3_first3_token_ids
        ]
        c3_first3_hits_p_tokens = [bool(int(token_id) in p_tokens) for token_id in c3_first3_token_ids]

        item_id = ex.get("id", ex.get("item_id", idx))
        row = {
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
            "token_overlap_count": int(overlap_count),
            "c1_generated": c1_text,
            "c2_generated": c2_text,
            "generated_c3": c3_text,
            "c1_window_text": c1_window,
            "c2_window_text": c2_window,
            "c3_window_text": c3_window,
            "c1_classification": c1_classification,
            "c2_classification": c2_classification,
            "c3_classification": c3_classification,
            "prefill_margin": float(prefill_margin) if math.isfinite(float(prefill_margin)) else float("nan"),
            "gate_fired": bool(gate_fired),
            "n_steps_suppressed": int(args.n_suppress_steps),
            "c3_first3_token_ids": c3_first3_token_ids,
            "c3_first3_tokens_text": c3_first3_tokens_text,
            "c3_first3_hits_p_tokens": c3_first3_hits_p_tokens,
            "tau": float(args.tau),
            "lambda_neg": float(args.lambda_neg),
        }
        per_items.append(row)

        if bool(gate_fired) and str(c3_classification) == "hallucinated" and len(fired_hallucinated_diagnostics) < 10:
            fired_hallucinated_diagnostics.append(
                {
                    "item_id": item_id,
                    "question": question,
                    "context_dosage": context_dosage,
                    "parametric_dosage": parametric_dosage,
                    "c3_first3_token_ids": c3_first3_token_ids,
                    "c3_first3_tokens_text": c3_first3_tokens_text,
                    "c3_first3_hits_p_tokens": c3_first3_hits_p_tokens,
                    "p_token_set": row["p_token_set"],
                }
            )

    if not per_items:
        raise RuntimeError(f"No usable rows. skip_stats={skip_stats}")

    c1_faithful, c1_hallucinated, c1_other = _condition_counts(per_items, "c1_classification")
    c2_faithful, c2_hallucinated, c2_other = _condition_counts(per_items, "c2_classification")
    c3_faithful, c3_hallucinated, c3_other = _condition_counts(per_items, "c3_classification")
    n_examples = int(len(per_items))

    c2_hall_rows = [row for row in per_items if str(row["c2_classification"]) == "hallucinated"]
    c2_faith_rows = [row for row in per_items if str(row["c2_classification"]) == "faithful"]
    n_restored = sum(1 for row in c2_hall_rows if str(row["c3_classification"]) == "faithful")
    n_harmed = sum(1 for row in c2_faith_rows if str(row["c3_classification"]) != "faithful")
    gate_fired_count = sum(1 for row in per_items if bool(row["gate_fired"]))
    ambiguous_count = sum(1 for row in per_items if bool(row["ambiguous_token_sets"]))

    c2_correct = [str(row["c2_classification"]) == "faithful" for row in per_items]
    c3_correct = [str(row["c3_classification"]) == "faithful" for row in per_items]

    summary = {
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
        "restoration_on_c2_hallucinated": float(n_restored / max(len(c2_hall_rows), 1)),
        "restoration_on_c2_hallucinated_ci95_wilson": wilson_ci(n_restored, len(c2_hall_rows)),
        "n_restored": int(n_restored),
        "n_c2_hallucinated": int(len(c2_hall_rows)),
        "harm_rate": float(n_harmed / max(len(c2_faith_rows), 1)),
        "harm_rate_ci95_wilson": wilson_ci(n_harmed, len(c2_faith_rows)),
        "n_harmed": int(n_harmed),
        "n_c2_faithful": int(len(c2_faith_rows)),
        "gate_fire_rate": float(gate_fired_count / max(n_examples, 1)),
        "gate_fire_rate_ci95_wilson": wilson_ci(gate_fired_count, n_examples),
        "mcnemar_c2_vs_c3": mcnemar_from_paired(c2_correct, c3_correct),
        "mean_e_token_set_size": _mean([float(len(row["e_token_set"])) for row in per_items]),
        "mean_p_token_set_size": _mean([float(len(row["p_token_set"])) for row in per_items]),
        "n_ambiguous_token_sets": int(ambiguous_count),
        "tau": float(args.tau),
        "lambda_neg": float(args.lambda_neg),
        "n_suppress_steps": int(args.n_suppress_steps),
        "mechanism_mode": "n_step_token_suppress",
        "skip_stats": skip_stats,
        "metadata": build_metadata(
            args=args,
            input_files=[conflict_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=REPO_ROOT,
        ),
    }

    _write_jsonl(out_dir / "results.jsonl", per_items)
    _write_json(out_dir / "summary.json", summary)
    if fired_hallucinated_diagnostics:
        print("first3_token_diagnostics_for_gate_fired_hallucinated_items=")
        print(json.dumps(fired_hallucinated_diagnostics, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"summary={out_dir / 'summary.json'}")
    print(f"results={out_dir / 'results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
