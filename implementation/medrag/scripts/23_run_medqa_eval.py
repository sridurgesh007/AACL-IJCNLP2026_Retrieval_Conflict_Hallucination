#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.ag_erha.utils import write_json, write_jsonl
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.running_margin_gate import RunningMarginGate

LETTER_REGEX = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


class FirstStepTokenBiasProcessor(LogitsProcessor):
    """Applies fixed logit bias on first generated step only."""

    def __init__(
        self,
        *,
        p_token_ids: set[int],
        e_token_ids: set[int],
        neg_bias: float,
        pos_bias: float,
    ) -> None:
        self._p_ids = sorted(int(x) for x in p_token_ids)
        self._e_ids = sorted(int(x) for x in e_token_ids)
        self._neg_bias = float(neg_bias)
        self._pos_bias = float(pos_bias)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self._step == 0:
            if self._neg_bias > 0.0 and self._p_ids:
                p_idx = torch.as_tensor(self._p_ids, dtype=torch.long, device=scores.device)
                scores[:, p_idx] = scores[:, p_idx] - float(self._neg_bias)
            if self._pos_bias > 0.0 and self._e_ids:
                e_idx = torch.as_tensor(self._e_ids, dtype=torch.long, device=scores.device)
                scores[:, e_idx] = scores[:, e_idx] + float(self._pos_bias)
        self._step += 1
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedQA multiple-choice evaluation (Path A).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_eval")
    parser.add_argument("--tau_margin", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=100.0)
    parser.add_argument(
        "--lambda_values",
        default="",
        help="Optional comma-separated lambda sweep values, e.g. '50,100,200'.",
    )
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--p_set_mode",
        choices=["all_wrong", "param_only", "both"],
        default="all_wrong",
        help="Competitor set mode: all wrong letters, parametric-only letter, or both (ablation).",
    )
    parser.add_argument(
        "--bootstrap_mode",
        choices=["last_prompt", "decision_position", "evidence_search", "both"],
        default="last_prompt",
        help="Bootstrap margin position selection mode.",
    )
    parser.add_argument(
        "--evidence_aligned_mode",
        choices=["off", "letter_only", "extracted", "both"],
        default="off",
        help="Evidence-aligned promotion mode: none, letter-only, extracted, or both (ablation).",
    )
    parser.add_argument(
        "--evidence_top_n",
        type=int,
        default=8,
        help="Maximum number of extracted evidence-aligned tokens to promote.",
    )
    parser.add_argument(
        "--evidence_expand_topn",
        type=int,
        default=0,
        help="Optional vocab-expansion count via anchor similarity (0 disables expansion).",
    )
    parser.add_argument(
        "--evidence_gamma_ratio",
        type=float,
        default=0.5,
        help="Positive promotion scale gamma = evidence_gamma_ratio * lambda_prefill.",
    )
    parser.add_argument(
        "--evidence_neg_bias_ratio",
        type=float,
        default=0.0,
        help="Optional explicit negative token bias scale on P-set, ratio * lambda_prefill.",
    )
    parser.add_argument(
        "--eval_subset",
        choices=["all", "baseline_wrong"],
        default="all",
        help="Evaluate all rows or only baseline-wrong rows.",
    )
    parser.add_argument(
        "--subset_from_results",
        default="",
        help="Optional reference medqa_eval_results.jsonl path; when set, only baseline-wrong keys from this file are evaluated.",
    )
    parser.add_argument(
        "--debug_wrong_examples",
        type=int,
        default=0,
        help="Print debug lines for first N baseline-wrong examples per lambda.",
    )
    parser.add_argument(
        "--write_wrong_diagnostics",
        type=int,
        default=1,
        choices=[0, 1],
        help="Write per-lambda wrong-case diagnostic jsonl.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _build_grounded_mc_prompt(question: str, options: dict[str, str], context: str) -> str:
    options_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Clinical Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D)."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _build_letter_token_sets(tokenizer) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {k: set() for k in ("A", "B", "C", "D")}
    for letter in ("A", "B", "C", "D"):
        for variant in (letter, f" {letter}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tok in ids:
                out[letter].add(int(tok))
    return out


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    max_start = len(haystack) - len(needle)
    if max_start < 0:
        return None
    for start in range(max_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return int(start)
    return None


def _extract_option_token_spans(
    *,
    tokenizer,
    prompt_ids: list[int],
    options: dict[str, str],
) -> dict[str, tuple[int, int] | None]:
    spans: dict[str, tuple[int, int] | None] = {}
    for letter in ("A", "B", "C", "D"):
        option_text = str(options.get(letter, "")).strip()
        if not option_text:
            spans[letter] = None
            continue
        line = f"{letter}) {option_text}"
        line_variants = [line, f"\n{line}", f" {line}"]
        line_pos: int | None = None
        line_len = 0
        for candidate in line_variants:
            cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
            pos = _find_subsequence(prompt_ids, [int(x) for x in cand_ids])
            if pos is not None:
                line_pos = int(pos)
                line_len = int(len(cand_ids))
                break
        if line_pos is None or line_len <= 0:
            spans[letter] = None
            continue

        line_slice = prompt_ids[line_pos : line_pos + line_len]
        text_pos_rel: int | None = None
        text_len = 0
        text_variants = [option_text, f" {option_text}"]
        for text_candidate in text_variants:
            text_ids = tokenizer.encode(text_candidate, add_special_tokens=False)
            rel = _find_subsequence(line_slice, [int(x) for x in text_ids])
            if rel is not None:
                text_pos_rel = int(rel)
                text_len = int(len(text_ids))
                break
        if text_pos_rel is None or text_len <= 0:
            spans[letter] = (int(line_pos), int(line_pos + line_len))
        else:
            start = int(line_pos + text_pos_rel)
            end = int(start + text_len)
            spans[letter] = (start, end)
    return spans


def _mean_pool_span(hidden_seq: torch.Tensor, span: tuple[int, int] | None) -> torch.Tensor | None:
    if span is None:
        return None
    start, end = int(span[0]), int(span[1])
    if start < 0 or end <= start or end > int(hidden_seq.shape[0]):
        return None
    return hidden_seq[start:end, :].float().mean(dim=0)


def _cosine_similarity(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    a_f = a.float()
    b_f = b.float().to(device=a_f.device)
    a_n = torch.linalg.norm(a_f).item()
    b_n = torch.linalg.norm(b_f).item()
    if a_n <= 1e-12 or b_n <= 1e-12:
        return float("nan")
    return float(torch.dot(a_f, b_f).item() / max(a_n * b_n, 1e-12))


def _build_evidence_aligned_set(
    *,
    tokenizer,
    prompt_ids: torch.Tensor,
    options: dict[str, str],
    correct_letter: str,
    p_set: set[int],
    letter_token_sets: dict[str, set[int]],
    hidden_seq: torch.Tensor,
    decision_hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    mode: str,
    top_n: int,
    expand_topn: int,
) -> tuple[set[int], dict[str, Any]]:
    mode_norm = str(mode or "off")
    prompt_list = [int(x) for x in prompt_ids.tolist()]
    option_spans = _extract_option_token_spans(tokenizer=tokenizer, prompt_ids=prompt_list, options=options)
    correct_span = option_spans.get(str(correct_letter))

    correct_letter_ids = set(int(x) for x in letter_token_sets.get(str(correct_letter), set()))
    correct_option_text_ids: set[int] = set()
    if correct_span is not None:
        start, end = int(correct_span[0]), int(correct_span[1])
        if 0 <= start < end <= len(prompt_list):
            correct_option_text_ids = set(int(x) for x in prompt_list[start:end])

    correct_anchor = _mean_pool_span(hidden_seq, correct_span)
    wrong_anchors = [
        _mean_pool_span(hidden_seq, option_spans.get(letter))
        for letter in ("A", "B", "C", "D")
        if str(letter) != str(correct_letter)
    ]
    wrong_anchors = [x for x in wrong_anchors if x is not None]

    if mode_norm == "off":
        selected = set()
    elif mode_norm == "letter_only":
        selected = set(int(x) for x in correct_letter_ids if int(x) not in p_set)
    else:
        candidate_pool: set[int] = set(correct_letter_ids) | set(correct_option_text_ids)
        if int(expand_topn) > 0 and correct_anchor is not None:
            with torch.no_grad():
                anchor = correct_anchor.float().to(device=lm_weight.device)
                anchor_norm = torch.linalg.norm(anchor).item()
                if anchor_norm > 1e-12:
                    weights = lm_weight.float()
                    sims = torch.matmul(weights, anchor) / max(anchor_norm, 1e-12)
                    k = min(int(expand_topn), int(sims.numel()))
                    if k > 0:
                        top_ids = torch.topk(sims, k=k).indices.tolist()
                        candidate_pool |= set(int(x) for x in top_ids)

        scored: list[tuple[int, float, float, float, float]] = []
        for tid in sorted(int(x) for x in candidate_pool):
            if int(tid) in p_set:
                continue
            if int(tid) < 0 or int(tid) >= int(lm_weight.shape[0]):
                continue
            token_vec = lm_weight[int(tid)].float()
            sim_correct = _cosine_similarity(token_vec, correct_anchor)
            sim_decision = _cosine_similarity(token_vec, decision_hidden)
            if wrong_anchors:
                wrong_sims = [_cosine_similarity(token_vec, w) for w in wrong_anchors]
                wrong_sims = [float(x) for x in wrong_sims if np.isfinite(float(x))]
                max_wrong = float(max(wrong_sims)) if wrong_sims else 0.0
            else:
                max_wrong = 0.0
            sc = 0.0
            if np.isfinite(sim_correct):
                sc += float(sim_correct)
            if np.isfinite(sim_decision):
                sc += 0.5 * float(sim_decision)
            sc -= float(max_wrong)
            scored.append((int(tid), float(sc), float(sim_correct), float(sim_decision), float(max_wrong)))

        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        selected = set(int(x[0]) for x in scored_sorted[: max(1, int(top_n))])
        if not selected:
            selected = set(int(x) for x in correct_letter_ids if int(x) not in p_set)

    selected = set(int(x) for x in selected if int(x) not in p_set)

    option_spans_dump = {
        letter: (None if span is None else [int(span[0]), int(span[1])])
        for letter, span in option_spans.items()
    }
    diagnostics: dict[str, Any] = {
        "mode": mode_norm,
        "option_spans": option_spans_dump,
        "correct_letter_token_ids": sorted(int(x) for x in correct_letter_ids),
        "correct_option_text_token_ids": sorted(int(x) for x in correct_option_text_ids),
        "selected_evidence_token_ids": sorted(int(x) for x in selected),
        "overlap_with_parametric": int(len(set(selected) & set(p_set))),
    }
    return selected, diagnostics


def _choose_representative(logits: torch.Tensor, token_ids: set[int]) -> int | None:
    if not token_ids:
        return None
    vocab_size = int(logits.numel())
    valid = [tid for tid in token_ids if 0 <= int(tid) < vocab_size]
    if not valid:
        return None
    return int(max(valid, key=lambda t: float(logits[int(t)].item())))


def _decode_letter(
    output_ids: list[int],
    tokenizer,
    letter_token_sets: dict[str, set[int]],
) -> str | None:
    rev: dict[int, str] = {}
    for letter, ids in letter_token_sets.items():
        for tid in ids:
            rev[int(tid)] = letter
    for tid in output_ids:
        if int(tid) in rev:
            return rev[int(tid)]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    m = LETTER_REGEX.search(text)
    if m:
        return m.group(1).upper()
    return None


def _generate_ids(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    logits_processors: list[LogitsProcessor] | None = None,
) -> list[int]:
    kwargs: dict[str, Any] = {}
    if logits_processors:
        kwargs["logits_processor"] = LogitsProcessorList(logits_processors)
    with torch.inference_mode():
        out = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            **kwargs,
        )
    return [int(x) for x in out[0, prompt_ids.shape[0] :].tolist()]


def _answer_scores(logits: torch.Tensor, letter_token_sets: dict[str, set[int]]) -> dict[str, float]:
    vocab_size = int(logits.numel())
    scores: dict[str, float] = {}
    for letter in ("A", "B", "C", "D"):
        ids = [int(t) for t in letter_token_sets.get(letter, set()) if 0 <= int(t) < vocab_size]
        if not ids:
            scores[letter] = float("-inf")
        else:
            scores[letter] = float(max(float(logits[i].item()) for i in ids))
    return scores


def _decision_margin_from_scores(logits: torch.Tensor, e_set: set[int], p_set: set[int]) -> float:
    vocab_size = int(logits.numel())
    e_ids = [int(t) for t in e_set if 0 <= int(t) < vocab_size]
    p_ids = [int(t) for t in p_set if 0 <= int(t) < vocab_size]
    if not e_ids or not p_ids:
        return float("nan")
    log_p = torch.log_softmax(logits.float(), dim=-1)
    e_idx = torch.as_tensor(e_ids, dtype=torch.long, device=log_p.device)
    p_idx = torch.as_tensor(p_ids, dtype=torch.long, device=log_p.device)
    log_p_e = torch.logsumexp(log_p.index_select(0, e_idx), dim=0)
    log_p_p = torch.logsumexp(log_p.index_select(0, p_idx), dim=0)
    return float((log_p_e - log_p_p).item())


def _parse_lambda_grid(lambda_values: str, lambda_max: float) -> list[float]:
    if not str(lambda_values or "").strip():
        return [float(lambda_max)]
    out: list[float] = []
    for raw in str(lambda_values).split(","):
        token = raw.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        out.append(float(lambda_max))
    # Keep deterministic order while removing duplicates.
    deduped: list[float] = []
    seen: set[float] = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(float(x))
    return deduped


def _resolve_p_set_modes(mode: str) -> list[str]:
    if str(mode) == "both":
        return ["param_only", "all_wrong"]
    return [str(mode)]


def _resolve_bootstrap_modes(mode: str) -> list[str]:
    if str(mode) == "both":
        return ["last_prompt", "decision_position"]
    return [str(mode)]


def _resolve_evidence_modes(mode: str) -> list[str]:
    if str(mode) == "both":
        return ["off", "letter_only", "extracted"]
    return [str(mode)]


def _row_key(ex: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(ex.get("question", "")).strip(),
        str(ex.get("correct_answer", "")).strip().upper(),
        str(ex.get("parametric_answer", "")).strip().upper(),
    )


def _load_wrong_case_keyset(path: str) -> set[tuple[str, str, str]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"subset_from_results not found: {p}")
    out: set[tuple[str, str, str]] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if bool(row.get("base_correct", True)):
                continue
            out.add(
                (
                    str(row.get("question", "")).strip(),
                    str(row.get("correct_answer", "")).strip().upper(),
                    str(row.get("parametric_answer", "")).strip().upper(),
                )
            )
    return out


def _configure_gate_bootstrap_from_margin(gate: RunningMarginGate, margin: float) -> None:
    gate._current_margin = float(margin)
    gate._margin_at_evidence_pos = float(margin)
    gate._evidence_position = -1
    gate._position_found = False
    lam, clipped = gate._compute_prefill_lambda(float(margin))
    gate._lambda = float(lam)
    gate._lambda_clipped = bool(clipped)


def _slug_lambda(value: float) -> str:
    text = f"{float(value):g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def _run_single_lambda(
    *,
    model,
    tokenizer,
    device: torch.device,
    conflicts: list[dict[str, Any]],
    letter_token_sets: dict[str, set[int]],
    args: argparse.Namespace,
    lambda_value: float,
    p_set_mode: str,
    bootstrap_mode: str,
    evidence_mode: str,
    subset_keyset: set[tuple[str, str, str]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    n_skipped = 0
    n_filtered_by_ref_subset = 0
    n_filtered_by_eval_subset = 0
    wrong_diagnostics: list[dict[str, Any]] = []
    debug_remaining = max(0, int(args.debug_wrong_examples))

    for ex in tqdm(
        conflicts,
        desc=f"Path A MedQA eval (lambda={lambda_value:g}, ev={evidence_mode})",
    ):
        question = str(ex.get("question", "")).strip()
        options = ex.get("options", {}) or {}
        context = str(ex.get("context", "")).strip()
        correct = str(ex.get("correct_answer", "")).strip().upper()
        param = str(ex.get("parametric_answer", "")).strip().upper()
        if subset_keyset is not None:
            if _row_key(ex) not in subset_keyset:
                n_filtered_by_ref_subset += 1
                continue
        if not question or not isinstance(options, dict) or not context:
            n_skipped += 1
            continue
        if correct not in {"A", "B", "C", "D"} or param not in {"A", "B", "C", "D"}:
            n_skipped += 1
            continue

        prompt = _build_grounded_mc_prompt(question, options, context)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length)).to(device)
        prompt_ids = enc["input_ids"][0]

        with torch.inference_mode():
            model_out = model(
                prompt_ids.unsqueeze(0),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            cached_logits = model_out.logits[0, -1, :].float()
            hidden_seq = model_out.hidden_states[-1][0].float()
            decision_hidden = hidden_seq[-1, :].float()

        e_set = set(int(x) for x in letter_token_sets[correct])
        if str(p_set_mode) == "param_only":
            p_letters = {str(param)}
        else:
            p_letters = {"A", "B", "C", "D"} - {str(correct)}
        p_set: set[int] = set()
        for letter in p_letters:
            p_set |= set(int(x) for x in letter_token_sets[str(letter)])
        ev_id = _choose_representative(cached_logits, e_set)
        par_id = _choose_representative(cached_logits, p_set)
        if ev_id is None or par_id is None:
            n_skipped += 1
            continue

        evidence_ids, evidence_diag = _build_evidence_aligned_set(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            options=options,
            correct_letter=correct,
            p_set=p_set,
            letter_token_sets=letter_token_sets,
            hidden_seq=hidden_seq,
            decision_hidden=decision_hidden,
            lm_weight=model.lm_head.weight,
            mode=str(evidence_mode),
            top_n=int(args.evidence_top_n),
            expand_topn=int(args.evidence_expand_topn),
        )

        letter_scores = _answer_scores(cached_logits, letter_token_sets)
        decision_margin = _decision_margin_from_scores(cached_logits, e_set, p_set)

        base_ids = _generate_ids(model, prompt_ids, int(args.max_new_tokens))
        base_letter = _decode_letter(base_ids, tokenizer, letter_token_sets)
        base_correct = bool(base_letter == correct)
        if str(args.eval_subset) == "baseline_wrong" and bool(base_correct):
            n_filtered_by_eval_subset += 1
            continue

        gate = RunningMarginGate(
            model=model,
            tokenizer=tokenizer,
            evidence_set=e_set,
            parametric_set=p_set,
            target_layer=int(args.target_layer),
            target_head=int(args.target_head),
            tau_margin=float(args.tau_margin),
            alpha=float(args.alpha),
            lambda_max=float(lambda_value),
            first_token_only=True,
        )
        gate.setup_example(int(ev_id), int(par_id), device=device)
        if str(bootstrap_mode) == "evidence_search":
            gate.bootstrap_prefill(
                prompt_ids=prompt_ids,
                device=device,
                context_answer=str(ex.get("context_answer", "")),
                parametric_answer=str(ex.get("parametric_answer", "")),
            )
        elif str(bootstrap_mode) == "decision_position":
            _configure_gate_bootstrap_from_margin(gate, decision_margin)
        else:
            gate.bootstrap_prefill(
                prompt_ids=prompt_ids,
                device=device,
                context_answer=None,
                parametric_answer=None,
            )

        logits_processors: list[LogitsProcessor] = []
        evidence_pos_bias = 0.0
        evidence_neg_bias = 0.0
        if str(evidence_mode) != "off" and len(evidence_ids) > 0 and float(gate._lambda) > 0.0:
            evidence_pos_bias = float(max(0.0, float(args.evidence_gamma_ratio)) * float(gate._lambda))
            evidence_neg_bias = float(max(0.0, float(args.evidence_neg_bias_ratio)) * float(gate._lambda))
            if evidence_pos_bias > 0.0 or evidence_neg_bias > 0.0:
                logits_processors.append(
                    FirstStepTokenBiasProcessor(
                        p_token_ids=(set(p_set) if evidence_neg_bias > 0.0 else set()),
                        e_token_ids=set(evidence_ids),
                        neg_bias=float(evidence_neg_bias),
                        pos_bias=float(evidence_pos_bias),
                    )
                )

        gate.attach(device=device)
        try:
            ft_ids = _generate_ids(
                model,
                prompt_ids,
                int(args.max_new_tokens),
                logits_processors=logits_processors,
            )
        finally:
            gate.detach()
        ft_letter = _decode_letter(ft_ids, tokenizer, letter_token_sets)
        gate_summary = gate.summary()

        ft_correct = bool(ft_letter == correct)
        ft_restored = bool((not base_correct) and ft_correct)
        ft_harmed = bool(base_correct and (not ft_correct))
        first_step = gate.step_log[0] if gate.step_log else {}
        if not base_correct:
            wrong_row = {
                "question": question,
                "correct_answer": correct,
                "parametric_answer": param,
                "base_letter": base_letter,
                "ft_letter": ft_letter,
                "ft_correct": bool(ft_correct),
                "prefill_margin": float(gate_summary.get("prefill_margin", 0.0)),
                "prefill_lambda": float(gate_summary.get("prefill_lambda", 0.0)),
                "n_steps_steered": int(gate_summary.get("n_steps_steered", 0)),
                "n_steps_total": int(gate_summary.get("n_steps_total", 0)),
                "first_step_reason": str(first_step.get("reason", "")),
                "first_step_margin": float(first_step.get("margin", 0.0)) if first_step else 0.0,
                "first_step_lambda": float(first_step.get("lambda", 0.0)) if first_step else 0.0,
                "bootstrap_mode": str(bootstrap_mode),
                "evidence_aligned_mode": str(evidence_mode),
                "lambda_max_used": float(lambda_value),
                "p_set_mode": str(p_set_mode),
                "answer_scores": letter_scores,
                "decision_margin_from_scores": float(decision_margin) if np.isfinite(decision_margin) else float("nan"),
                "base_matches_parametric": bool(str(base_letter) == str(param)),
                "evidence_token_ids": [int(x) for x in sorted(evidence_ids)],
                "parametric_token_ids": [int(x) for x in sorted(p_set)],
                "evidence_overlap_with_parametric": int(evidence_diag.get("overlap_with_parametric", 0)),
                "evidence_pos_bias": float(evidence_pos_bias),
                "evidence_neg_bias": float(evidence_neg_bias),
                "evidence_diagnostics": evidence_diag,
            }
            wrong_diagnostics.append(wrong_row)
            if debug_remaining > 0:
                debug_remaining -= 1
                print(
                    "WRONG_DEBUG "
                    f"prefill_margin={wrong_row['prefill_margin']:.4f} "
                    f"prefill_lambda={wrong_row['prefill_lambda']:.4f} "
                    f"first_step_reason={wrong_row['first_step_reason']} "
                    f"first_step_margin={wrong_row['first_step_margin']:.4f} "
                    f"first_step_lambda={wrong_row['first_step_lambda']:.4f} "
                    f"correct={correct} param={param} base={base_letter} ft={ft_letter}"
                )

        results.append(
            {
                "question": question,
                "source": str(ex.get("source", "medqa")),
                "correct_answer": correct,
                "parametric_answer": param,
                "base_letter": base_letter,
                "ft_letter": ft_letter,
                "base_correct": base_correct,
                "ft_correct": ft_correct,
                "ft_restored": ft_restored,
                "ft_harmed": ft_harmed,
                "ft_n_steered": int(gate_summary["n_steps_steered"]),
                "ft_n_total": int(gate_summary["n_steps_total"]),
                "ft_frac_steered": float(gate_summary["frac_steered"]),
                "ft_lambda": float(gate_summary["mean_lambda"]),
                "prefill_margin": float(gate_summary.get("prefill_margin", 0.0)),
                "prefill_lambda": float(gate_summary.get("prefill_lambda", 0.0)),
                "lambda_max_used": float(lambda_value),
                "p_set_mode": str(p_set_mode),
                "bootstrap_mode": str(bootstrap_mode),
                "evidence_aligned_mode": str(evidence_mode),
                "p_letters": sorted(list(p_letters)),
                "answer_scores": letter_scores,
                "decision_margin_from_scores": float(decision_margin) if np.isfinite(decision_margin) else float("nan"),
                "base_matches_parametric": bool(str(base_letter) == str(param)),
                "evidence_token_ids": [int(x) for x in sorted(evidence_ids)],
                "evidence_overlap_with_parametric": int(evidence_diag.get("overlap_with_parametric", 0)),
                "evidence_pos_bias": float(evidence_pos_bias),
                "evidence_neg_bias": float(evidence_neg_bias),
                "evidence_selected_count": int(len(evidence_ids)),
            }
        )

    if not results:
        raise RuntimeError("No MedQA evaluation rows produced.")

    n = len(results)
    n_base_wrong = int(sum(1 for r in results if not bool(r["base_correct"])))
    n_base_correct = int(sum(1 for r in results if bool(r["base_correct"])))
    n_conflict = int(sum(1 for r in results if str(r["parametric_answer"]) != str(r["correct_answer"])))

    metrics = {
        "task": "medqa_multiple_choice",
        "answer_format": "single_letter_token",
        "n_examples": int(n),
        "n_skipped": int(n_skipped),
        "n_conflicts_by_construction": int(n_conflict),
        "parametric_accuracy_no_context": float(1.0 - (n_conflict / max(n, 1))),
        "baseline_accuracy": float(sum(1 for r in results if bool(r["base_correct"])) / max(n, 1)),
        "ft_accuracy": float(sum(1 for r in results if bool(r["ft_correct"])) / max(n, 1)),
        "ft_restoration_rate": float(sum(1 for r in results if bool(r["ft_restored"])) / max(n_base_wrong, 1)),
        "ft_harm_rate": float(sum(1 for r in results if bool(r["ft_harmed"])) / max(n_base_correct, 1)),
        "ft_restoration_improvement": float(
            (sum(1 for r in results if bool(r["ft_restored"])) / max(n_base_wrong, 1))
            - (sum(1 for r in results if bool(r["ft_harmed"])) / max(n_base_correct, 1))
        ),
        "tau_margin": float(args.tau_margin),
        "alpha": float(args.alpha),
        "lambda_max": float(lambda_value),
        "target_layer": int(args.target_layer),
        "target_head": int(args.target_head),
        "bootstrap_mode": str(bootstrap_mode),
        "evidence_aligned_mode": str(evidence_mode),
        "p_set_mode": str(p_set_mode),
        "eval_subset": str(args.eval_subset),
        "n_filtered_by_ref_subset": int(n_filtered_by_ref_subset),
        "n_filtered_by_eval_subset": int(n_filtered_by_eval_subset),
        "n_wrong_diagnostics": int(len(wrong_diagnostics)),
    }

    wrong_rows = [r for r in results if not bool(r["base_correct"])]
    wrong_rows_with_margin = [
        r for r in wrong_rows if np.isfinite(float(r.get("prefill_margin", float("nan"))))
    ]
    wrong_rows_with_decision_margin = [
        r for r in wrong_rows if np.isfinite(float(r.get("decision_margin_from_scores", float("nan"))))
    ]
    n_gate_fired_wrong = int(sum(1 for r in wrong_rows if int(r.get("ft_n_steered", 0)) > 0))
    n_prefill_negative_wrong = int(sum(1 for r in wrong_rows_with_margin if float(r.get("prefill_margin", 0.0)) < 0.0))
    n_decision_negative_wrong = int(
        sum(1 for r in wrong_rows_with_decision_margin if float(r.get("decision_margin_from_scores", 0.0)) < 0.0)
    )
    metrics["mean_prefill_margin_base_wrong"] = (
        float(np.mean([float(r.get("prefill_margin", 0.0)) for r in wrong_rows])) if wrong_rows else 0.0
    )
    metrics["mean_decision_margin_base_wrong"] = (
        float(np.mean([float(r.get("decision_margin_from_scores", 0.0)) for r in wrong_rows_with_decision_margin]))
        if wrong_rows_with_decision_margin
        else float("nan")
    )
    metrics["fraction_negative_prefill_margin_base_wrong"] = float(
        n_prefill_negative_wrong / max(len(wrong_rows_with_margin), 1)
    )
    metrics["fraction_negative_decision_margin_base_wrong"] = float(
        n_decision_negative_wrong / max(len(wrong_rows_with_decision_margin), 1)
    )
    metrics["n_gate_fired_base_wrong"] = int(n_gate_fired_wrong)
    metrics["gate_fire_rate_base_wrong"] = float(n_gate_fired_wrong / max(n_base_wrong, 1))
    metrics["n_positive_margin_but_wrong"] = int(
        sum(1 for r in wrong_rows_with_margin if float(r.get("prefill_margin", 0.0)) > 0.0)
    )
    evidence_counts = [int(r.get("evidence_selected_count", 0)) for r in results]
    evidence_active_rows = [r for r in results if float(r.get("evidence_pos_bias", 0.0)) > 0.0]
    metrics["mean_evidence_tokens_selected"] = float(np.mean(evidence_counts)) if evidence_counts else 0.0
    metrics["n_rows_with_evidence_promotion_active"] = int(len(evidence_active_rows))
    metrics["evidence_promotion_active_rate"] = float(len(evidence_active_rows) / max(len(results), 1))
    metrics["mean_evidence_pos_bias_when_active"] = (
        float(np.mean([float(r.get("evidence_pos_bias", 0.0)) for r in evidence_active_rows]))
        if evidence_active_rows
        else 0.0
    )
    metrics["mean_evidence_neg_bias_when_active"] = (
        float(np.mean([float(r.get("evidence_neg_bias", 0.0)) for r in evidence_active_rows]))
        if evidence_active_rows
        else 0.0
    )
    n_wrong_match_param = int(sum(1 for r in wrong_rows if bool(r.get("base_matches_parametric", False))))
    n_wrong_not_param = int(sum(1 for r in wrong_rows if not bool(r.get("base_matches_parametric", False))))
    n_restored_match_param = int(
        sum(1 for r in wrong_rows if bool(r.get("base_matches_parametric", False)) and bool(r.get("ft_restored", False)))
    )
    n_restored_not_param = int(
        sum(1 for r in wrong_rows if (not bool(r.get("base_matches_parametric", False))) and bool(r.get("ft_restored", False)))
    )
    metrics["n_base_wrong"] = int(n_base_wrong)
    metrics["n_base_wrong_matches_parametric"] = int(n_wrong_match_param)
    metrics["n_base_wrong_not_parametric"] = int(n_wrong_not_param)
    metrics["n_restored_base_wrong_matches_parametric"] = int(n_restored_match_param)
    metrics["n_restored_base_wrong_not_parametric"] = int(n_restored_not_param)
    metrics["restoration_rate_base_wrong_matches_parametric"] = float(
        n_restored_match_param / max(n_wrong_match_param, 1)
    )
    metrics["restoration_rate_base_wrong_not_parametric"] = float(
        n_restored_not_param / max(n_wrong_not_param, 1)
    )
    return metrics, results, n_skipped, wrong_diagnostics


def main() -> int:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = _parse_lambda_grid(args.lambda_values, float(args.lambda_max))
    p_set_modes = _resolve_p_set_modes(str(args.p_set_mode))
    bootstrap_modes = _resolve_bootstrap_modes(str(args.bootstrap_mode))
    evidence_modes = _resolve_evidence_modes(str(args.evidence_aligned_mode))

    conflicts = _load_jsonl((PROJECT_ROOT / args.conflict_path).resolve())
    if not conflicts:
        raise RuntimeError(f"No rows in conflict file: {args.conflict_path}")
    print(f"Loaded MedQA conflicts: {len(conflicts)}")
    print(f"Lambda grid: {lambda_grid}")
    print(f"P-set modes: {p_set_modes}")
    print(f"Bootstrap modes: {bootstrap_modes}")
    print(f"Evidence-aligned modes: {evidence_modes}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = _build_letter_token_sets(tokenizer)
    subset_keyset: set[tuple[str, str, str]] | None = None
    if str(args.subset_from_results).strip():
        subset_keyset = _load_wrong_case_keyset(str(args.subset_from_results))
        print(f"Loaded reference wrong-case subset size: {len(subset_keyset)}")

    sweep_rows: list[dict[str, Any]] = []
    for ev_mode in evidence_modes:
        for p_mode in p_set_modes:
            for b_mode in bootstrap_modes:
                for lambda_value in lambda_grid:
                    metrics, results, _, wrong_diagnostics = _run_single_lambda(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        conflicts=conflicts,
                        letter_token_sets=letter_token_sets,
                        args=args,
                        lambda_value=float(lambda_value),
                        p_set_mode=str(p_mode),
                        bootstrap_mode=str(b_mode),
                        evidence_mode=str(ev_mode),
                        subset_keyset=subset_keyset,
                    )

                    if len(p_set_modes) == 1 and len(bootstrap_modes) == 1 and len(evidence_modes) == 1:
                        mode_out_dir = out_dir
                    else:
                        mode_out_dir = (
                            out_dir
                            / f"evidence_{ev_mode}"
                            / f"p_set_{p_mode}"
                            / f"bootstrap_{b_mode}"
                        )
                        mode_out_dir.mkdir(parents=True, exist_ok=True)
                    if len(lambda_grid) == 1:
                        lambda_out_dir = mode_out_dir
                    else:
                        lambda_out_dir = mode_out_dir / f"lambda_{_slug_lambda(float(lambda_value))}"
                        lambda_out_dir.mkdir(parents=True, exist_ok=True)

                    write_json(lambda_out_dir / "medqa_eval_metrics.json", metrics)
                    write_jsonl(lambda_out_dir / "medqa_eval_results.jsonl", results)
                    if int(args.write_wrong_diagnostics) == 1:
                        write_jsonl(lambda_out_dir / "medqa_wrong_case_diagnostics.jsonl", wrong_diagnostics)

                    print(json.dumps(metrics, indent=2))
                    print(f"metrics={lambda_out_dir / 'medqa_eval_metrics.json'}")
                    print(f"rows={lambda_out_dir / 'medqa_eval_results.jsonl'}")
                    if int(args.write_wrong_diagnostics) == 1:
                        print(f"wrong_diagnostics={lambda_out_dir / 'medqa_wrong_case_diagnostics.jsonl'}")

                    sweep_rows.append(
                        {
                            "evidence_aligned_mode": str(ev_mode),
                            "p_set_mode": str(p_mode),
                            "bootstrap_mode": str(b_mode),
                            "lambda_max": float(lambda_value),
                            "baseline_accuracy": float(metrics["baseline_accuracy"]),
                            "ft_accuracy": float(metrics["ft_accuracy"]),
                            "ft_restoration_rate": float(metrics["ft_restoration_rate"]),
                            "ft_harm_rate": float(metrics["ft_harm_rate"]),
                            "ft_restoration_improvement": float(metrics["ft_restoration_improvement"]),
                            "mean_prefill_margin_base_wrong": float(metrics["mean_prefill_margin_base_wrong"]),
                            "mean_decision_margin_base_wrong": (
                                float(metrics["mean_decision_margin_base_wrong"])
                                if np.isfinite(float(metrics["mean_decision_margin_base_wrong"]))
                                else float("nan")
                            ),
                            "fraction_negative_prefill_margin_base_wrong": float(
                                metrics["fraction_negative_prefill_margin_base_wrong"]
                            ),
                            "fraction_negative_decision_margin_base_wrong": float(
                                metrics["fraction_negative_decision_margin_base_wrong"]
                            ),
                            "n_gate_fired_base_wrong": int(metrics["n_gate_fired_base_wrong"]),
                            "gate_fire_rate_base_wrong": float(metrics["gate_fire_rate_base_wrong"]),
                            "n_positive_margin_but_wrong": int(metrics["n_positive_margin_but_wrong"]),
                            "n_base_wrong": int(metrics["n_base_wrong"]),
                            "n_base_wrong_matches_parametric": int(metrics["n_base_wrong_matches_parametric"]),
                            "n_base_wrong_not_parametric": int(metrics["n_base_wrong_not_parametric"]),
                            "n_filtered_by_ref_subset": int(metrics["n_filtered_by_ref_subset"]),
                            "n_filtered_by_eval_subset": int(metrics["n_filtered_by_eval_subset"]),
                            "mean_evidence_tokens_selected": float(metrics["mean_evidence_tokens_selected"]),
                            "evidence_promotion_active_rate": float(metrics["evidence_promotion_active_rate"]),
                            "out_dir": str(lambda_out_dir),
                        }
                    )

    if len(lambda_grid) > 1 or len(p_set_modes) > 1 or len(bootstrap_modes) > 1 or len(evidence_modes) > 1:
        sweep_rows_sorted = sorted(
            sweep_rows,
            key=lambda x: (float(x["ft_restoration_improvement"]), -float(x["ft_harm_rate"])),
            reverse=True,
        )
        sweep_payload = {
            "lambda_values": [float(x) for x in lambda_grid],
            "p_set_modes": [str(x) for x in p_set_modes],
            "bootstrap_modes": [str(x) for x in bootstrap_modes],
            "evidence_aligned_modes": [str(x) for x in evidence_modes],
            "eval_subset": str(args.eval_subset),
            "subset_from_results": str(args.subset_from_results),
            "rows": sweep_rows,
            "best_by_restoration_improvement": sweep_rows_sorted[0] if sweep_rows_sorted else None,
        }
        write_json(out_dir / "medqa_eval_ablation_summary.json", sweep_payload)
        print("Ablation summary:")
        print(json.dumps(sweep_payload, indent=2))
        print(f"ablation_summary={out_dir / 'medqa_eval_ablation_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
