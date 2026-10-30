#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.ag_erha.utils import write_json, write_jsonl
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.running_margin_gate import RunningMarginGate


BASE_EVAL_PATH = Path(__file__).with_name("23_run_medqa_eval.py")
_BASE_SPEC = importlib.util.spec_from_file_location("medqa_eval_base", BASE_EVAL_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load base MedQA evaluator from {BASE_EVAL_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

FirstStepTokenBiasProcessor = _BASE.FirstStepTokenBiasProcessor
_build_grounded_mc_prompt = _BASE._build_grounded_mc_prompt
_load_jsonl = _BASE._load_jsonl
_build_letter_token_sets = _BASE._build_letter_token_sets
_choose_representative = _BASE._choose_representative
_decode_letter = _BASE._decode_letter
_generate_ids = _BASE._generate_ids
_answer_scores = _BASE._answer_scores
_decision_margin_from_scores = _BASE._decision_margin_from_scores
_row_key = _BASE._row_key
_load_wrong_case_keyset = _BASE._load_wrong_case_keyset
_configure_gate_bootstrap_from_margin = _BASE._configure_gate_bootstrap_from_margin


VALID_LETTERS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust MedQA multiple-choice evaluation with content/label E-P routing.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_eval_robust")
    parser.add_argument("--n_examples", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--tau_margin", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha_values", default="", help="Optional comma-separated alpha sweep values.")
    parser.add_argument("--lambda_max", type=float, default=100.0)
    parser.add_argument("--lambda_values", default="", help="Optional comma-separated lambda sweep values.")
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--medqa_ep_mode", choices=["letter_only", "content_only", "combined", "both"], default="both")
    parser.add_argument("--p_set_mode", choices=["all_wrong", "param_only", "both"], default="both")
    parser.add_argument("--bootstrap_mode", choices=["last_prompt", "decision_position", "both"], default="both")
    parser.add_argument("--use_ov_steering", choices=["0", "1", "both"], default="1")
    parser.add_argument("--first_token_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--evidence_gamma_ratio", type=float, default=0.5)
    parser.add_argument("--evidence_neg_bias_ratio", type=float, default=0.0)
    parser.add_argument("--eval_subset", choices=["all", "baseline_wrong"], default="all")
    parser.add_argument("--subset_from_results", default="")
    parser.add_argument("--debug_wrong_examples", type=int, default=0)
    parser.add_argument("--write_wrong_diagnostics", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _parse_float_grid(raw: str, fallback: float) -> list[float]:
    if not str(raw or "").strip():
        return [float(fallback)]
    out: list[float] = []
    seen: set[float] = set()
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        if value in seen:
            continue
        seen.add(value)
        out.append(float(value))
    return out or [float(fallback)]


def _resolve_modes(raw: str, both_values: list[str]) -> list[str]:
    if str(raw) == "both":
        return list(both_values)
    return [str(raw)]


def _resolve_ov_modes(raw: str) -> list[int]:
    if str(raw) == "both":
        return [0, 1]
    return [int(raw)]


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean([float(x) for x in values]))


def _clean_token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return str(text).replace("?", " ").replace("G", " ").strip()


def _keep_content_token(tokenizer, token_id: int) -> bool:
    all_special_ids = getattr(tokenizer, "all_special_ids", []) or []
    if int(token_id) in {int(x) for x in all_special_ids}:
        return False
    cleaned = _clean_token_text(tokenizer, int(token_id))
    if not cleaned:
        return False
    return any(ch.isalnum() for ch in cleaned)


def _tokenize_text_variants(tokenizer, text: str) -> set[int]:
    ids: set[int] = set()
    text = str(text or "").strip()
    if not text:
        return ids
    for variant in (text, f" {text}"):
        toks = tokenizer.encode(variant, add_special_tokens=False)
        ids |= {int(tok) for tok in toks if _keep_content_token(tokenizer, int(tok))}
    return ids


def _build_content_ep_sets(tokenizer, options: dict[str, str], correct: str, letter_token_sets: dict[str, set[int]]) -> dict[str, Any]:
    option_content_token_sets: dict[str, set[int]] = {}
    for letter in VALID_LETTERS:
        option_content_token_sets[letter] = _tokenize_text_variants(tokenizer, str(options.get(letter, "")).strip())

    correct_letter = str(correct).upper()
    wrong_letters = [letter for letter in VALID_LETTERS if letter != correct_letter]
    e_content_raw = set(option_content_token_sets.get(correct_letter, set()))
    p_content_all_wrong_raw: set[int] = set()
    for letter in wrong_letters:
        p_content_all_wrong_raw |= set(option_content_token_sets.get(letter, set()))

    return {
        "correct_letter": correct_letter,
        "wrong_letters": wrong_letters,
        "option_content_token_sets": option_content_token_sets,
        "E_content_raw": e_content_raw,
        "P_content_all_wrong_raw": p_content_all_wrong_raw,
        "E_letter": set(int(x) for x in letter_token_sets.get(correct_letter, set())),
        "P_letter_all_wrong": set(int(x) for letter in wrong_letters for x in letter_token_sets.get(letter, set())),
    }


def _materialize_ep_sets(ep_base: dict[str, Any], *, parametric_answer: str, p_set_mode: str, letter_token_sets: dict[str, set[int]]) -> dict[str, Any]:
    wrong_letters = [str(x) for x in ep_base.get("wrong_letters", [])]
    parametric_letter = str(parametric_answer).upper()
    if str(p_set_mode) == "param_only" and parametric_letter in wrong_letters:
        p_letters = [parametric_letter]
    else:
        p_letters = list(wrong_letters)

    option_content_token_sets = ep_base["option_content_token_sets"]
    e_content_raw = set(ep_base["E_content_raw"])
    p_content_raw: set[int] = set()
    for letter in p_letters:
        p_content_raw |= set(option_content_token_sets.get(letter, set()))
    content_overlap = set(e_content_raw) & set(p_content_raw)
    e_content = set(e_content_raw) - content_overlap
    p_content = set(p_content_raw) - content_overlap

    e_letter_raw = set(ep_base["E_letter"])
    p_letter_raw: set[int] = set()
    for letter in p_letters:
        p_letter_raw |= {int(x) for x in letter_token_sets.get(letter, set())}
    letter_overlap = set(e_letter_raw) & set(p_letter_raw)
    e_letter = set(e_letter_raw) - letter_overlap
    p_letter = set(p_letter_raw) - letter_overlap

    e_combined_raw = set(e_content) | set(e_letter)
    p_combined_raw = set(p_content) | set(p_letter)
    combined_overlap = set(e_combined_raw) & set(p_combined_raw)
    e_combined = set(e_combined_raw) - combined_overlap
    p_combined = set(p_combined_raw) - combined_overlap

    return {
        "p_letters": list(p_letters),
        "E_content": e_content,
        "P_content": p_content,
        "E_letter": e_letter,
        "P_letter": p_letter,
        "E_combined": e_combined,
        "P_combined": p_combined,
        "n_E_content_tokens": int(len(e_content)),
        "n_P_content_tokens": int(len(p_content)),
        "n_E_letter_tokens": int(len(e_letter)),
        "n_P_letter_tokens": int(len(p_letter)),
        "n_E_combined_tokens": int(len(e_combined)),
        "n_P_combined_tokens": int(len(p_combined)),
        "overlap_removed_count": int(len(content_overlap) + len(letter_overlap) + len(combined_overlap)),
    }

def _select_bootstrap_sets(medqa_ep_mode: str, ep_sets: dict[str, Any]) -> tuple[set[int], set[int], dict[str, Any]]:
    if str(medqa_ep_mode) == "letter_only":
        return set(ep_sets["E_letter"]), set(ep_sets["P_letter"]), {"bootstrap_space": "letter"}
    if str(medqa_ep_mode) in {"content_only", "combined"}:
        return set(ep_sets["E_content"]), set(ep_sets["P_content"]), {"bootstrap_space": "content"}
    raise ValueError(f"Unsupported medqa_ep_mode={medqa_ep_mode}")


def _select_intervention_sets(medqa_ep_mode: str, ep_sets: dict[str, Any]) -> tuple[set[int], set[int], dict[str, Any]]:
    if str(medqa_ep_mode) == "letter_only":
        return set(ep_sets["E_letter"]), set(ep_sets["P_letter"]), {"intervention_space": "letter"}
    if str(medqa_ep_mode) == "content_only":
        return set(ep_sets["E_content"]), set(ep_sets["P_content"]), {"intervention_space": "content"}
    if str(medqa_ep_mode) == "combined":
        return set(ep_sets["E_combined"]), set(ep_sets["P_combined"]), {"intervention_space": "combined"}
    raise ValueError(f"Unsupported medqa_ep_mode={medqa_ep_mode}")


def _mode_role(medqa_ep_mode: str) -> str:
    if str(medqa_ep_mode) == "letter_only":
        return "oracle"
    if str(medqa_ep_mode) == "content_only":
        return "mechanistic"
    if str(medqa_ep_mode) == "combined":
        return "practical"
    return "unknown"


def _run_single_config(
    *,
    model,
    tokenizer,
    device: torch.device,
    conflicts: list[dict[str, Any]],
    letter_token_sets: dict[str, set[int]],
    args: argparse.Namespace,
    medqa_ep_mode: str,
    p_set_mode: str,
    bootstrap_mode: str,
    use_ov_steering: int,
    alpha_value: float,
    lambda_value: float,
    subset_keyset: set[tuple[str, str, str]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    wrong_diagnostics: list[dict[str, Any]] = []
    n_skipped_missing_fields = 0
    n_skipped_invalid_labels = 0
    n_skipped_empty_sets = 0
    n_skipped_missing_representative = 0
    n_filtered_by_ref_subset = 0
    n_filtered_by_eval_subset = 0
    debug_remaining = max(0, int(args.debug_wrong_examples))
    effective_use_ov = bool(int(use_ov_steering)) and str(medqa_ep_mode) in {"content_only", "combined"}

    for ex in tqdm(
        conflicts,
        desc=(
            f"MedQA robust eval ({medqa_ep_mode}, p={p_set_mode}, boot={bootstrap_mode}, "
            f"ov={int(effective_use_ov)}, alpha={alpha_value:g}, lambda={lambda_value:g})"
        ),
    ):
        question = str(ex.get("question", "")).strip()
        options = ex.get("options", {}) or {}
        context = str(ex.get("context", "")).strip()
        correct = str(ex.get("correct_answer", "")).strip().upper()
        param = str(ex.get("parametric_answer", "")).strip().upper()

        if subset_keyset is not None and _row_key(ex) not in subset_keyset:
            n_filtered_by_ref_subset += 1
            continue
        if not question or not isinstance(options, dict) or not context:
            n_skipped_missing_fields += 1
            continue
        if correct not in VALID_LETTERS or param not in VALID_LETTERS or param == correct:
            n_skipped_invalid_labels += 1
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

        ep_base = _build_content_ep_sets(tokenizer, options, correct, letter_token_sets)
        ep_sets = _materialize_ep_sets(
            ep_base,
            parametric_answer=param,
            p_set_mode=str(p_set_mode),
            letter_token_sets=letter_token_sets,
        )
        e_boot, p_boot, boot_meta = _select_bootstrap_sets(str(medqa_ep_mode), ep_sets)
        e_int, p_int, int_meta = _select_intervention_sets(str(medqa_ep_mode), ep_sets)
        if not e_boot or not p_boot or not e_int or not p_int:
            n_skipped_empty_sets += 1
            continue

        ev_id = _choose_representative(cached_logits, e_boot)
        par_id = _choose_representative(cached_logits, p_boot)
        if ev_id is None or par_id is None:
            n_skipped_missing_representative += 1
            continue

        letter_scores = _answer_scores(cached_logits, letter_token_sets)
        decision_margin = _decision_margin_from_scores(cached_logits, e_boot, p_boot)
        base_ids = _generate_ids(model, prompt_ids, int(args.max_new_tokens))
        base_letter = _decode_letter(base_ids, tokenizer, letter_token_sets)
        base_correct = bool(base_letter == correct)
        if str(args.eval_subset) == "baseline_wrong" and base_correct:
            n_filtered_by_eval_subset += 1
            continue

        gate = RunningMarginGate(
            model=model,
            tokenizer=tokenizer,
            evidence_set=e_boot,
            parametric_set=p_boot,
            target_layer=int(args.target_layer),
            target_head=int(args.target_head),
            tau_margin=float(args.tau_margin),
            alpha=float(alpha_value),
            lambda_max=float(lambda_value),
            first_token_only=bool(int(args.first_token_only)),
        )
        gate.setup_example(int(ev_id), int(par_id), device=device)
        if str(bootstrap_mode) == "decision_position":
            _configure_gate_bootstrap_from_margin(gate, decision_margin)
        else:
            gate.bootstrap_prefill(prompt_ids=prompt_ids, device=device, context_answer=None, parametric_answer=None)

        evidence_pos_bias = 0.0
        evidence_neg_bias = 0.0
        logits_processors: list[Any] = []
        if float(gate._lambda) > 0.0:
            evidence_pos_bias = float(max(0.0, float(args.evidence_gamma_ratio)) * float(gate._lambda))
            evidence_neg_bias = float(max(0.0, float(args.evidence_neg_bias_ratio)) * float(gate._lambda))
            if (evidence_pos_bias > 0.0 and e_int) or (evidence_neg_bias > 0.0 and p_int):
                logits_processors.append(
                    FirstStepTokenBiasProcessor(
                        p_token_ids=(set(p_int) if evidence_neg_bias > 0.0 else set()),
                        e_token_ids=(set(e_int) if evidence_pos_bias > 0.0 else set()),
                        neg_bias=float(evidence_neg_bias),
                        pos_bias=float(evidence_pos_bias),
                    )
                )

        if effective_use_ov:
            gate.attach(device=device)
        try:
            ft_ids = _generate_ids(
                model,
                prompt_ids,
                int(args.max_new_tokens),
                logits_processors=logits_processors,
            )
        finally:
            if effective_use_ov:
                gate.detach()
        ft_letter = _decode_letter(ft_ids, tokenizer, letter_token_sets)
        gate_summary = gate.summary()

        ft_correct = bool(ft_letter == correct)
        ft_restored = bool((not base_correct) and ft_correct)
        ft_harmed = bool(base_correct and (not ft_correct))
        ov_gate_fired = bool(int(gate_summary.get("n_steps_steered", 0)) > 0)
        gate_fired = bool(ov_gate_fired or evidence_pos_bias > 0.0 or evidence_neg_bias > 0.0)
        row = {
            "question": question,
            "source": str(ex.get("source", "medqa")),
            "medqa_ep_mode": str(medqa_ep_mode),
            "mode_role": _mode_role(str(medqa_ep_mode)),
            "p_set_mode": str(p_set_mode),
            "bootstrap_mode": str(bootstrap_mode),
            "use_ov_steering": bool(int(use_ov_steering)),
            "effective_use_ov_steering": bool(effective_use_ov),
            "correct_answer": correct,
            "parametric_answer": param,
            "base_letter": base_letter,
            "ft_letter": ft_letter,
            "base_correct": bool(base_correct),
            "ft_correct": bool(ft_correct),
            "ft_restored": bool(ft_restored),
            "ft_harmed": bool(ft_harmed),
            "base_matches_parametric": bool(str(base_letter) == str(param)),
            "prefill_margin": float(gate_summary.get("prefill_margin", 0.0)),
            "prefill_lambda": float(gate_summary.get("prefill_lambda", 0.0)),
            "decision_margin_from_scores": float(decision_margin) if np.isfinite(decision_margin) else float("nan"),
            "n_steps_steered": int(gate_summary.get("n_steps_steered", 0)),
            "n_steps_total": int(gate_summary.get("n_steps_total", 0)),
            "frac_steered": float(gate_summary.get("frac_steered", 0.0)),
            "mean_lambda": float(gate_summary.get("mean_lambda", 0.0)),
            "gate_fired": bool(gate_fired),
            "ov_gate_fired": bool(ov_gate_fired),
            "evidence_pos_bias": float(evidence_pos_bias),
            "evidence_neg_bias": float(evidence_neg_bias),
            "alpha": float(alpha_value),
            "lambda_max": float(lambda_value),
            "boot_space": str(boot_meta.get("bootstrap_space", "")),
            "intervention_space": str(int_meta.get("intervention_space", "")),
            "p_letters": list(ep_sets["p_letters"]),
            "n_E_content_tokens": int(ep_sets["n_E_content_tokens"]),
            "n_P_content_tokens": int(ep_sets["n_P_content_tokens"]),
            "n_E_letter_tokens": int(ep_sets["n_E_letter_tokens"]),
            "n_P_letter_tokens": int(ep_sets["n_P_letter_tokens"]),
            "n_E_combined_tokens": int(ep_sets["n_E_combined_tokens"]),
            "n_P_combined_tokens": int(ep_sets["n_P_combined_tokens"]),
            "overlap_removed_count": int(ep_sets["overlap_removed_count"]),
            "answer_scores": letter_scores,
        }
        results.append(row)

        if not base_correct:
            wrong_diagnostics.append(dict(row))
            if debug_remaining > 0:
                debug_remaining -= 1
                print(
                    "MEDQA_WRONG_DEBUG "
                    f"mode={medqa_ep_mode} p={p_set_mode} boot={bootstrap_mode} ov={int(effective_use_ov)} "
                    f"alpha={alpha_value:g} lambda={lambda_value:g} "
                    f"base={base_letter} ft={ft_letter} correct={correct} param={param} "
                    f"prefill_margin={row['prefill_margin']:.4f} prefill_lambda={row['prefill_lambda']:.4f} "
                    f"decision_margin={row['decision_margin_from_scores']:.4f} gate_fired={row['gate_fired']}"
                )

    if not results:
        raise RuntimeError("No MedQA robust evaluation rows produced.")

    n = len(results)
    base_wrong = [r for r in results if not bool(r["base_correct"])]
    base_correct_rows = [r for r in results if bool(r["base_correct"])]
    wrong_with_prefill = [r for r in base_wrong if np.isfinite(float(r.get("prefill_margin", float("nan"))))]
    wrong_with_decision = [r for r in base_wrong if np.isfinite(float(r.get("decision_margin_from_scores", float("nan"))))]
    active_bias_rows = [r for r in results if float(r.get("evidence_pos_bias", 0.0)) > 0.0 or float(r.get("evidence_neg_bias", 0.0)) > 0.0]

    n_base_wrong = int(len(base_wrong))
    n_base_correct = int(len(base_correct_rows))
    n_restored = int(sum(1 for r in results if bool(r["ft_restored"])))
    n_harmed = int(sum(1 for r in results if bool(r["ft_harmed"])))

    metrics = {
        "task": "medqa_multiple_choice_robust",
        "answer_format": "single_letter_token",
        "medqa_ep_mode": str(medqa_ep_mode),
        "mode_role": _mode_role(str(medqa_ep_mode)),
        "use_ov_steering": bool(int(use_ov_steering)),
        "effective_use_ov_steering": bool(effective_use_ov),
        "p_set_mode": str(p_set_mode),
        "bootstrap_mode": str(bootstrap_mode),
        "tau_margin": float(args.tau_margin),
        "alpha": float(alpha_value),
        "lambda_max": float(lambda_value),
        "target_layer": int(args.target_layer),
        "target_head": int(args.target_head),
        "n_examples": int(n),
        "n_base_wrong": int(n_base_wrong),
        "n_skipped_missing_fields": int(n_skipped_missing_fields),
        "n_skipped_invalid_labels": int(n_skipped_invalid_labels),
        "n_skipped_empty_sets": int(n_skipped_empty_sets),
        "n_skipped_missing_representative": int(n_skipped_missing_representative),
        "n_filtered_by_ref_subset": int(n_filtered_by_ref_subset),
        "n_filtered_by_eval_subset": int(n_filtered_by_eval_subset),
        "baseline_accuracy": float(sum(1 for r in results if bool(r["base_correct"])) / max(n, 1)),
        "ft_accuracy": float(sum(1 for r in results if bool(r["ft_correct"])) / max(n, 1)),
        "ft_restoration_rate": float(n_restored / max(n_base_wrong, 1)),
        "ft_harm_rate": float(n_harmed / max(n_base_correct, 1)),
        "ft_restoration_improvement": float((n_restored / max(n_base_wrong, 1)) - (n_harmed / max(n_base_correct, 1))),
        "restoration_rate": float(n_restored / max(n_base_wrong, 1)),
        "harm_rate": float(n_harmed / max(n_base_correct, 1)),
        "restoration_improvement": float((n_restored / max(n_base_wrong, 1)) - (n_harmed / max(n_base_correct, 1))),
        "gate_fire_rate_base_wrong": float(sum(1 for r in base_wrong if bool(r["gate_fired"])) / max(n_base_wrong, 1)),
        "ov_gate_fire_rate_base_wrong": float(sum(1 for r in base_wrong if bool(r["ov_gate_fired"])) / max(n_base_wrong, 1)),
        "fraction_negative_prefill_margin_base_wrong": float(sum(1 for r in wrong_with_prefill if float(r["prefill_margin"]) < 0.0) / max(len(wrong_with_prefill), 1)),
        "fraction_negative_decision_margin_base_wrong": float(sum(1 for r in wrong_with_decision if float(r["decision_margin_from_scores"]) < 0.0) / max(len(wrong_with_decision), 1)),
        "mean_prefill_margin_base_wrong": _safe_mean([float(r["prefill_margin"]) for r in base_wrong]),
        "mean_decision_margin_base_wrong": _safe_mean([float(r["decision_margin_from_scores"]) for r in wrong_with_decision]),
        "mean_n_E_content_tokens": _safe_mean([float(r["n_E_content_tokens"]) for r in results]),
        "mean_n_P_content_tokens": _safe_mean([float(r["n_P_content_tokens"]) for r in results]),
        "mean_n_E_letter_tokens": _safe_mean([float(r["n_E_letter_tokens"]) for r in results]),
        "mean_n_P_letter_tokens": _safe_mean([float(r["n_P_letter_tokens"]) for r in results]),
        "mean_n_E_combined_tokens": _safe_mean([float(r["n_E_combined_tokens"]) for r in results]),
        "mean_n_P_combined_tokens": _safe_mean([float(r["n_P_combined_tokens"]) for r in results]),
        "mean_overlap_removed_count": _safe_mean([float(r["overlap_removed_count"]) for r in results]),
        "mean_token_bias_when_active": _safe_mean([float(r["evidence_pos_bias"]) for r in active_bias_rows]),
        "mean_negative_bias_when_active": _safe_mean([float(r["evidence_neg_bias"]) for r in active_bias_rows]),
        "mean_prefill_lambda": _safe_mean([float(r["prefill_lambda"]) for r in results]),
        "mean_frac_steered": _safe_mean([float(r["frac_steered"]) for r in results]),
        "n_rows_with_active_token_bias": int(len(active_bias_rows)),
        "token_bias_active_rate": float(len(active_bias_rows) / max(n, 1)),
    }

    wrong_match_param = [r for r in base_wrong if bool(r["base_matches_parametric"])]
    wrong_not_param = [r for r in base_wrong if not bool(r["base_matches_parametric"])]
    metrics["n_base_wrong_matches_parametric"] = int(len(wrong_match_param))
    metrics["n_base_wrong_not_parametric"] = int(len(wrong_not_param))
    metrics["restoration_rate_base_wrong_matches_parametric"] = float(sum(1 for r in wrong_match_param if bool(r["ft_restored"])) / max(len(wrong_match_param), 1))
    metrics["restoration_rate_base_wrong_not_parametric"] = float(sum(1 for r in wrong_not_param if bool(r["ft_restored"])) / max(len(wrong_not_param), 1))
    return metrics, results, wrong_diagnostics


def main() -> int:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conflicts = _load_jsonl((PROJECT_ROOT / args.conflict_path).resolve())
    if not conflicts:
        raise RuntimeError(f"No rows in conflict file: {args.conflict_path}")
    if int(args.n_examples) > 0:
        conflicts = conflicts[: int(args.n_examples)]

    lambda_grid = _parse_float_grid(args.lambda_values, float(args.lambda_max))
    alpha_grid = _parse_float_grid(args.alpha_values, float(args.alpha))
    medqa_modes = _resolve_modes(str(args.medqa_ep_mode), ["letter_only", "content_only", "combined"])
    p_set_modes = _resolve_modes(str(args.p_set_mode), ["param_only", "all_wrong"])
    bootstrap_modes = _resolve_modes(str(args.bootstrap_mode), ["last_prompt", "decision_position"])
    requested_ov_modes = _resolve_ov_modes(str(args.use_ov_steering))

    print(f"Loaded MedQA conflicts: {len(conflicts)}")
    print(f"MedQA EP modes: {medqa_modes}")
    print(f"P-set modes: {p_set_modes}")
    print(f"Bootstrap modes: {bootstrap_modes}")
    print(f"Alpha grid: {alpha_grid}")
    print(f"Lambda grid: {lambda_grid}")

    subset_keyset: set[tuple[str, str, str]] | None = None
    if str(args.subset_from_results).strip():
        subset_keyset = _load_wrong_case_keyset(str(args.subset_from_results))
        print(f"Loaded reference wrong-case subset size: {len(subset_keyset)}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = _build_letter_token_sets(tokenizer)

    sweep_rows: list[dict[str, Any]] = []
    config_rows: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path]] = []
    total_configs = 0
    for medqa_ep_mode in medqa_modes:
        ov_modes = [0] if str(medqa_ep_mode) == "letter_only" else list(requested_ov_modes)
        for p_set_mode in p_set_modes:
            for bootstrap_mode in bootstrap_modes:
                for ov_mode in ov_modes:
                    for alpha_value in alpha_grid:
                        for lambda_value in lambda_grid:
                            total_configs += 1
                            metrics, results, wrong_diagnostics = _run_single_config(
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                conflicts=conflicts,
                                letter_token_sets=letter_token_sets,
                                args=args,
                                medqa_ep_mode=str(medqa_ep_mode),
                                p_set_mode=str(p_set_mode),
                                bootstrap_mode=str(bootstrap_mode),
                                use_ov_steering=int(ov_mode),
                                alpha_value=float(alpha_value),
                                lambda_value=float(lambda_value),
                                subset_keyset=subset_keyset,
                            )
                            config_out_dir = (
                                out_dir
                                / f"ep_mode_{medqa_ep_mode}"
                                / f"p_set_{p_set_mode}"
                                / f"bootstrap_{bootstrap_mode}"
                                / f"ov_{int(metrics['effective_use_ov_steering'])}"
                                / f"alpha_{_slug(float(alpha_value))}"
                                / f"lambda_{_slug(float(lambda_value))}"
                            )
                            config_out_dir.mkdir(parents=True, exist_ok=True)
                            write_json(config_out_dir / "medqa_eval_metrics.json", metrics)
                            write_jsonl(config_out_dir / "medqa_eval_results.jsonl", results)
                            if int(args.write_wrong_diagnostics) == 1:
                                write_jsonl(config_out_dir / "medqa_wrong_case_diagnostics.jsonl", wrong_diagnostics)
                            config_rows.append((metrics, results, wrong_diagnostics, config_out_dir))
                            sweep_rows.append(
                                {
                                    "medqa_ep_mode": str(medqa_ep_mode),
                                    "mode_role": str(metrics["mode_role"]),
                                    "p_set_mode": str(p_set_mode),
                                    "bootstrap_mode": str(bootstrap_mode),
                                    "use_ov_steering": bool(int(ov_mode)),
                                    "effective_use_ov_steering": bool(metrics["effective_use_ov_steering"]),
                                    "alpha": float(alpha_value),
                                    "lambda_max": float(lambda_value),
                                    "baseline_accuracy": float(metrics["baseline_accuracy"]),
                                    "ft_accuracy": float(metrics["ft_accuracy"]),
                                    "ft_restoration_rate": float(metrics["ft_restoration_rate"]),
                                    "ft_harm_rate": float(metrics["ft_harm_rate"]),
                                    "ft_restoration_improvement": float(metrics["ft_restoration_improvement"]),
                                    "gate_fire_rate_base_wrong": float(metrics["gate_fire_rate_base_wrong"]),
                                    "fraction_negative_prefill_margin_base_wrong": float(metrics["fraction_negative_prefill_margin_base_wrong"]),
                                    "fraction_negative_decision_margin_base_wrong": float(metrics["fraction_negative_decision_margin_base_wrong"]),
                                    "mean_n_E_content_tokens": float(metrics["mean_n_E_content_tokens"]),
                                    "mean_n_P_content_tokens": float(metrics["mean_n_P_content_tokens"]),
                                    "mean_n_E_combined_tokens": float(metrics["mean_n_E_combined_tokens"]),
                                    "mean_n_P_combined_tokens": float(metrics["mean_n_P_combined_tokens"]),
                                    "mean_token_bias_when_active": float(metrics["mean_token_bias_when_active"]),
                                    "n_base_wrong": int(metrics["n_base_wrong"]),
                                    "out_dir": str(config_out_dir),
                                }
                            )
                            print(json.dumps(metrics, indent=2))
                            print(f"metrics={config_out_dir / 'medqa_eval_metrics.json'}")
                            print(f"rows={config_out_dir / 'medqa_eval_results.jsonl'}")
                            if int(args.write_wrong_diagnostics) == 1:
                                print(f"wrong_diagnostics={config_out_dir / 'medqa_wrong_case_diagnostics.jsonl'}")

    if total_configs == 1 and config_rows:
        metrics, results, wrong_diagnostics, _ = config_rows[0]
        write_json(out_dir / "medqa_eval_metrics.json", metrics)
        write_jsonl(out_dir / "medqa_eval_results.jsonl", results)
        if int(args.write_wrong_diagnostics) == 1:
            write_jsonl(out_dir / "medqa_wrong_case_diagnostics.jsonl", wrong_diagnostics)
        print(f"top_level_metrics={out_dir / 'medqa_eval_metrics.json'}")
        print(f"top_level_rows={out_dir / 'medqa_eval_results.jsonl'}")

    ranked = sorted(
        sweep_rows,
        key=lambda row: (float(row["ft_restoration_improvement"]), -float(row["ft_harm_rate"]), float(row["ft_accuracy"])),
        reverse=True,
    )
    ablation_payload = {
        "task": "medqa_multiple_choice_robust",
        "conflict_path": str((PROJECT_ROOT / args.conflict_path).resolve()),
        "n_examples_requested": int(args.n_examples),
        "n_examples_loaded": int(len(conflicts)),
        "medqa_ep_modes": medqa_modes,
        "p_set_modes": p_set_modes,
        "bootstrap_modes": bootstrap_modes,
        "requested_use_ov_modes": requested_ov_modes,
        "alpha_values": [float(x) for x in alpha_grid],
        "lambda_values": [float(x) for x in lambda_grid],
        "rows": sweep_rows,
        "best_by_restoration_improvement": ranked[0] if ranked else None,
    }
    write_json(out_dir / "medqa_eval_robust_ablation_summary.json", ablation_payload)
    print("Ablation summary:")
    print(json.dumps(ablation_payload, indent=2))
    print(f"ablation_summary={out_dir / 'medqa_eval_robust_ablation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
