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
_row_key = _BASE._row_key
_load_wrong_case_keyset = _BASE._load_wrong_case_keyset

VALID_LETTERS = ("A", "B", "C", "D")

EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "content_gate_letter_intervention": {
        "role": "mechanistic",
        "description": (
            "Bootstrap uses content margin and intervention uses letter tokens. "
            "This is the primary MedQA method."
        ),
        "bootstrap_E_key": "E_content",
        "bootstrap_P_key": "P_content",
        "interv_E_key": "E_letter",
        "interv_P_key": "P_letter",
        "default_lambda_values": [5.0, 10.0, 15.0, 20.0],
        "default_neg_bias_ratio_values": [0.0, 0.5, 1.0],
    },
    "letter_gate_letter_intervention": {
        "role": "label_direct",
        "description": (
            "Bootstrap uses letter margin and intervention uses letter tokens. "
            "This is the label-space comparison condition."
        ),
        "bootstrap_E_key": "E_letter",
        "bootstrap_P_key": "P_letter",
        "interv_E_key": "E_letter",
        "interv_P_key": "P_letter",
        "default_lambda_values": [5.0, 10.0, 15.0, 20.0],
        "default_neg_bias_ratio_values": [0.5],
    },
    "oracle_letter_unconstrained": {
        "role": "oracle",
        "description": (
            "Bootstrap uses letter margin and intervention uses letter tokens with "
            "an unconstrained lambda upper bound. This is the controllability ceiling."
        ),
        "bootstrap_E_key": "E_letter",
        "bootstrap_P_key": "P_letter",
        "interv_E_key": "E_letter",
        "interv_P_key": "P_letter",
        "default_lambda_values": [100.0],
        "default_neg_bias_ratio_values": [0.5],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final MedQA evidence-routing experiment.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final/exp_A")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_SPECS.keys()),
        default="content_gate_letter_intervention",
    )
    parser.add_argument("--n_examples", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--p_set_mode", choices=["all_wrong", "param_only"], default="all_wrong")
    parser.add_argument("--tau_margin", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=10.0)
    parser.add_argument("--lambda_values", default="")
    parser.add_argument("--neg_bias_ratio", type=float, default=0.5)
    parser.add_argument("--neg_bias_ratio_values", default="")
    parser.add_argument("--evidence_gamma_ratio", type=float, default=0.5)
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--eval_subset", choices=["all", "baseline_wrong"], default="all")
    parser.add_argument("--subset_from_results", default="")
    parser.add_argument("--debug_wrong_examples", type=int, default=0)
    parser.add_argument("--write_wrong_diagnostics", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_sanity_checks", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _parse_float_grid(raw: str, fallback: list[float]) -> list[float]:
    if not str(raw or "").strip():
        return [float(x) for x in fallback]
    out: list[float] = []
    seen: set[float] = set()
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        value = float(token)
        if value in seen:
            continue
        seen.add(value)
        out.append(float(value))
    return out or [float(x) for x in fallback]


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean([float(x) for x in values]))


def _clean_token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return (
        str(text)
        .replace("\u2581", " ")
        .replace("\u0120", " ")
        .replace("\u010A", " ")
        .strip()
    )


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

def _build_content_ep_sets(
    tokenizer,
    options: dict[str, str],
    correct: str,
    letter_token_sets: dict[str, set[int]],
) -> dict[str, Any]:
    option_content_token_sets: dict[str, set[int]] = {}
    for letter in VALID_LETTERS:
        option_text = str(options.get(letter, "")).strip()
        option_content_token_sets[letter] = _tokenize_text_variants(tokenizer, option_text)

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
        "P_letter_all_wrong": set(
            int(x) for letter in wrong_letters for x in letter_token_sets.get(letter, set())
        ),
    }


def _materialize_ep_sets(
    ep_base: dict[str, Any],
    *,
    parametric_answer: str,
    p_set_mode: str,
    letter_token_sets: dict[str, set[int]],
) -> dict[str, Any]:
    correct_letter = str(ep_base["correct_letter"])
    wrong_letters = [str(x) for x in ep_base["wrong_letters"]]
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

    return {
        "p_letters": list(p_letters),
        "E_content": e_content,
        "P_content": p_content,
        "E_letter": e_letter,
        "P_letter": p_letter,
        "n_E_content": int(len(e_content)),
        "n_P_content": int(len(p_content)),
        "n_E_letter": int(len(e_letter)),
        "n_P_letter": int(len(p_letter)),
        "overlap_removed_count": int(len(content_overlap) + len(letter_overlap)),
    }


def _run_config(
    *,
    model,
    tokenizer,
    device: torch.device,
    conflicts: list[dict[str, Any]],
    letter_token_sets: dict[str, set[int]],
    experiment_name: str,
    experiment_spec: dict[str, Any],
    p_set_mode: str,
    lambda_max: float,
    alpha: float,
    neg_bias_ratio: float,
    tau_margin: float,
    evidence_gamma_ratio: float,
    max_new_tokens: int,
    max_length: int,
    target_layer: int,
    target_head: int,
    subset_keyset: set[tuple[str, str, str]] | None,
    eval_subset: str,
    debug_wrong_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    wrong_diagnostics: list[dict[str, Any]] = []
    n_skipped_missing_fields = 0
    n_skipped_invalid_labels = 0
    n_skipped_empty_sets = 0
    n_skipped_missing_representative = 0
    n_filtered_by_ref_subset = 0
    n_filtered_by_eval_subset = 0
    debug_remaining = max(0, int(debug_wrong_examples))

    bootstrap_E_key = str(experiment_spec["bootstrap_E_key"])
    bootstrap_P_key = str(experiment_spec["bootstrap_P_key"])
    interv_E_key = str(experiment_spec["interv_E_key"])
    interv_P_key = str(experiment_spec["interv_P_key"])

    for ex in tqdm(
        conflicts,
        desc=(
            f"MedQA final ({experiment_name}, p={p_set_mode}, lambda={lambda_max:g}, "
            f"neg={neg_bias_ratio:g})"
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
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length)).to(device)
        prompt_ids = enc["input_ids"][0]

        with torch.inference_mode():
            cached_logits = model(
                prompt_ids.unsqueeze(0),
                use_cache=False,
                return_dict=True,
            ).logits[0, -1, :].float()

        ep_base = _build_content_ep_sets(tokenizer, options, correct, letter_token_sets)
        ep_sets = _materialize_ep_sets(
            ep_base,
            parametric_answer=param,
            p_set_mode=str(p_set_mode),
            letter_token_sets=letter_token_sets,
        )

        e_boot = set(ep_sets[bootstrap_E_key])
        p_boot = set(ep_sets[bootstrap_P_key])
        e_int = set(ep_sets[interv_E_key])
        p_int = set(ep_sets[interv_P_key])
        if not e_boot or not p_boot or not e_int or not p_int:
            n_skipped_empty_sets += 1
            continue

        ev_id = _choose_representative(cached_logits, e_boot)
        par_id = _choose_representative(cached_logits, p_boot)
        if ev_id is None or par_id is None:
            n_skipped_missing_representative += 1
            continue

        base_ids = _generate_ids(model, prompt_ids, int(max_new_tokens))
        base_letter = _decode_letter(base_ids, tokenizer, letter_token_sets)
        base_correct = bool(base_letter == correct)
        if str(eval_subset) == "baseline_wrong" and base_correct:
            n_filtered_by_eval_subset += 1
            continue

        gate = RunningMarginGate(
            model=model,
            tokenizer=tokenizer,
            evidence_set=e_boot,
            parametric_set=p_boot,
            target_layer=int(target_layer),
            target_head=int(target_head),
            tau_margin=float(tau_margin),
            alpha=float(alpha),
            lambda_max=float(lambda_max),
            first_token_only=True,
        )
        gate.setup_example(int(ev_id), int(par_id), device=device)
        gate.bootstrap_prefill(prompt_ids=prompt_ids, device=device, context_answer=None, parametric_answer=None)
        gate_summary = gate.summary()
        prefill_margin = float(gate_summary.get("prefill_margin", 0.0))
        gate_lambda = float(gate_summary.get("prefill_lambda", 0.0))
        conflict_detected = bool(np.isfinite(prefill_margin) and prefill_margin < float(tau_margin) and gate_lambda > 0.0)

        lambda_pos = float(max(0.0, float(evidence_gamma_ratio)) * gate_lambda) if conflict_detected else 0.0
        lambda_neg = float(max(0.0, float(neg_bias_ratio)) * lambda_pos) if conflict_detected else 0.0
        logits_processors: list[Any] = []
        if (lambda_pos > 0.0 and e_int) or (lambda_neg > 0.0 and p_int):
            logits_processors.append(
                FirstStepTokenBiasProcessor(
                    e_token_ids=(set(e_int) if lambda_pos > 0.0 else set()),
                    p_token_ids=(set(p_int) if lambda_neg > 0.0 else set()),
                    pos_bias=float(lambda_pos),
                    neg_bias=float(lambda_neg),
                )
            )

        ft_ids = _generate_ids(model, prompt_ids, int(max_new_tokens), logits_processors=logits_processors)
        ft_letter = _decode_letter(ft_ids, tokenizer, letter_token_sets)
        ft_correct = bool(ft_letter == correct)
        ft_restored = bool((not base_correct) and ft_correct)
        ft_harmed = bool(base_correct and (not ft_correct))
        invalid_ft_letter = bool(ft_letter not in VALID_LETTERS)
        invalid_base_letter = bool(base_letter not in VALID_LETTERS)

        row = {
            "config_name": str(experiment_name),
            "experiment_role": str(experiment_spec["role"]),
            "question": question,
            "source": str(ex.get("source", "medqa")),
            "correct": correct,
            "parametric": param,
            "base_letter": base_letter,
            "ft_letter": ft_letter,
            "base_correct": bool(base_correct),
            "ft_correct": bool(ft_correct),
            "ft_restored": bool(ft_restored),
            "ft_harmed": bool(ft_harmed),
            "base_matches_parametric": bool(base_letter == param),
            "conflict_detected": bool(conflict_detected),
            "prefill_margin": float(prefill_margin),
            "gate_lambda": float(gate_lambda),
            "lambda_pos_applied": float(lambda_pos),
            "lambda_neg_applied": float(lambda_neg),
            "n_E_content": int(ep_sets["n_E_content"]),
            "n_P_content": int(ep_sets["n_P_content"]),
            "n_E_letter": int(ep_sets["n_E_letter"]),
            "n_P_letter": int(ep_sets["n_P_letter"]),
            "overlap_removed_count": int(ep_sets["overlap_removed_count"]),
            "lambda_max": float(lambda_max),
            "neg_bias_ratio": float(neg_bias_ratio),
            "bootstrap_space": bootstrap_E_key,
            "intervention_space": interv_E_key,
            "invalid_base_letter": bool(invalid_base_letter),
            "invalid_ft_letter": bool(invalid_ft_letter),
        }
        results.append(row)

        if (not base_correct) or ft_harmed or invalid_ft_letter:
            diag = dict(row)
            diag["context"] = context
            diag["options"] = options
            wrong_diagnostics.append(diag)
            if debug_remaining > 0 and not base_correct:
                debug_remaining -= 1
                print(
                    "MEDQA_FINAL_DEBUG "
                    f"exp={experiment_name} lambda={lambda_max:g} neg={neg_bias_ratio:g} "
                    f"base={base_letter} ft={ft_letter} correct={correct} param={param} "
                    f"margin={prefill_margin:.4f} gate_lambda={gate_lambda:.4f} "
                    f"lambda_pos={lambda_pos:.4f} lambda_neg={lambda_neg:.4f}"
                )

    if not results:
        raise RuntimeError(f"No MedQA rows produced for {experiment_name} (lambda={lambda_max:g}, neg={neg_bias_ratio:g}).")

    n = len(results)
    base_wrong = [r for r in results if not bool(r["base_correct"])]
    base_correct_rows = [r for r in results if bool(r["base_correct"])]
    n_base_wrong = int(len(base_wrong))
    n_base_correct = int(len(base_correct_rows))
    n_restored = int(sum(1 for r in results if bool(r["ft_restored"])))
    n_harmed = int(sum(1 for r in results if bool(r["ft_harmed"])))
    n_conflict_detected = int(sum(1 for r in results if bool(r["conflict_detected"])))
    wrong_neg_margin = [r for r in base_wrong if float(r["prefill_margin"]) < float(tau_margin)]
    wrong_match_param = [r for r in base_wrong if bool(r["base_matches_parametric"])]
    wrong_not_param = [r for r in base_wrong if not bool(r["base_matches_parametric"])]
    invalid_ft = [r for r in results if bool(r["invalid_ft_letter"])]
    invalid_base = [r for r in results if bool(r["invalid_base_letter"])]
    conflict_rows = [r for r in results if bool(r["conflict_detected"])]

    metrics = {
        "config_name": str(experiment_name),
        "experiment_role": str(experiment_spec["role"]),
        "experiment_description": str(experiment_spec["description"]),
        "bootstrap_space": bootstrap_E_key,
        "intervention_space": interv_E_key,
        "p_set_mode": str(p_set_mode),
        "n_examples": int(n),
        "n_base_wrong": int(n_base_wrong),
        "baseline_accuracy": float(sum(1 for r in results if bool(r["base_correct"])) / max(n, 1)),
        "ft_accuracy": float(sum(1 for r in results if bool(r["ft_correct"])) / max(n, 1)),
        "restoration_rate": float(n_restored / max(n_base_wrong, 1)),
        "harm_rate": float(n_harmed / max(n_base_correct, 1)),
        "stability_rate": float(sum(1 for r in base_correct_rows if not bool(r["ft_harmed"])) / max(n_base_correct, 1)),
        "restoration_improvement": float((n_restored / max(n_base_wrong, 1)) - (n_harmed / max(n_base_correct, 1))),
        "gate_fire_rate_base_wrong": float(sum(1 for r in base_wrong if bool(r["conflict_detected"])) / max(n_base_wrong, 1)),
        "gate_fire_rate_base_correct": float(sum(1 for r in base_correct_rows if bool(r["conflict_detected"])) / max(n_base_correct, 1)),
        "fraction_negative_margin_base_wrong": float(len(wrong_neg_margin) / max(n_base_wrong, 1)),
        "mean_prefill_margin_base_wrong": _safe_mean([float(r["prefill_margin"]) for r in base_wrong]),
        "mean_prefill_margin_base_correct": _safe_mean([float(r["prefill_margin"]) for r in base_correct_rows]),
        "n_base_wrong_matches_param": int(len(wrong_match_param)),
        "restoration_rate_param_wrong": float(sum(1 for r in wrong_match_param if bool(r["ft_restored"])) / max(len(wrong_match_param), 1)),
        "restoration_rate_other_wrong": float(sum(1 for r in wrong_not_param if bool(r["ft_restored"])) / max(len(wrong_not_param), 1)),
        "mean_lambda_pos_when_conflict": _safe_mean([float(r["lambda_pos_applied"]) for r in conflict_rows]),
        "mean_lambda_neg_when_conflict": _safe_mean([float(r["lambda_neg_applied"]) for r in conflict_rows]),
        "n_conflict_detected": int(n_conflict_detected),
        "conflict_detection_rate": float(n_conflict_detected / max(n, 1)),
        "n_invalid_base_letter": int(len(invalid_base)),
        "n_invalid_ft_letter": int(len(invalid_ft)),
        "invalid_ft_letter_rate": float(len(invalid_ft) / max(n, 1)),
        "ft_matches_base_rate": float(sum(1 for r in results if r["ft_letter"] == r["base_letter"]) / max(n, 1)),
        "lambda_max": float(lambda_max),
        "neg_bias_ratio": float(neg_bias_ratio),
        "tau_margin": float(tau_margin),
        "alpha": float(alpha),
        "evidence_gamma_ratio": float(evidence_gamma_ratio),
        "target_layer": int(target_layer),
        "target_head": int(target_head),
        "n_skipped_missing_fields": int(n_skipped_missing_fields),
        "n_skipped_invalid_labels": int(n_skipped_invalid_labels),
        "n_skipped_empty_sets": int(n_skipped_empty_sets),
        "n_skipped_missing_representative": int(n_skipped_missing_representative),
        "n_filtered_by_ref_subset": int(n_filtered_by_ref_subset),
        "n_filtered_by_eval_subset": int(n_filtered_by_eval_subset),
        "mean_n_E_content": _safe_mean([float(r["n_E_content"]) for r in results]),
        "mean_n_P_content": _safe_mean([float(r["n_P_content"]) for r in results]),
        "mean_n_E_letter": _safe_mean([float(r["n_E_letter"]) for r in results]),
        "mean_n_P_letter": _safe_mean([float(r["n_P_letter"]) for r in results]),
        "mean_overlap_removed_count": _safe_mean([float(r["overlap_removed_count"]) for r in results]),
    }
    return metrics, results, wrong_diagnostics


def _build_sanity_checks(ablation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    zero_rows = [row for row in ablation_rows if abs(float(row.get("lambda_max", 0.0))) < 1e-9]
    if zero_rows:
        passed = all(
            abs(float(row.get("ft_accuracy", 0.0)) - float(row.get("baseline_accuracy", 0.0))) < 1e-9
            and abs(1.0 - float(row.get("ft_matches_base_rate", 0.0))) < 1e-9
            for row in zero_rows
        )
        checks.append({"name": "lambda_zero_matches_baseline", "passed": bool(passed)})

    suppression_rows = [row for row in ablation_rows if float(row.get("neg_bias_ratio", 0.0)) > 0.0]
    if suppression_rows:
        passed = all(
            float(row.get("mean_lambda_neg_when_conflict", 0.0)) > 0.0 or int(row.get("n_conflict_detected", 0)) == 0
            for row in suppression_rows
        )
        checks.append({"name": "suppression_active_when_enabled", "passed": bool(passed)})

    gate_rows = [row for row in ablation_rows if int(row.get("n_base_wrong", 0)) > 0]
    if gate_rows:
        passed = all(
            float(row.get("gate_fire_rate_base_wrong", 0.0)) > float(row.get("gate_fire_rate_base_correct", 0.0))
            for row in gate_rows
        )
        checks.append({"name": "gate_prefers_wrong_examples", "passed": bool(passed)})

    if ablation_rows:
        passed = all(float(row.get("invalid_ft_letter_rate", 0.0)) <= 0.02 for row in ablation_rows)
        checks.append({"name": "letter_only_intervention_keeps_letter_outputs", "passed": bool(passed)})

    lambda10_rows = [
        row for row in ablation_rows
        if abs(float(row.get("lambda_max", 0.0)) - 10.0) < 1e-9 and abs(float(row.get("neg_bias_ratio", 0.0)) - 0.5) < 1e-9
    ]
    if lambda10_rows:
        passed = all(
            float(row.get("restoration_improvement", 0.0)) > 0.0 and float(row.get("harm_rate", 1.0)) < 0.02
            for row in lambda10_rows
        )
        checks.append({"name": "lambda10_positive_restoration_low_harm", "passed": bool(passed)})

    payload = {
        "all_passed": bool(all(bool(check["passed"]) for check in checks)) if checks else True,
        "checks": checks,
        "failed_checks": [check for check in checks if not bool(check["passed"])],
    }
    return payload


def main() -> int:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conflicts = _load_jsonl((PROJECT_ROOT / args.conflict_path).resolve())
    if not conflicts:
        raise RuntimeError(f"No rows in conflict file: {args.conflict_path}")
    if int(args.n_examples) > 0:
        conflicts = conflicts[: int(args.n_examples)]

    subset_keyset: set[tuple[str, str, str]] | None = None
    if str(args.subset_from_results).strip():
        subset_keyset = _load_wrong_case_keyset(str(args.subset_from_results))
        print(f"Loaded reference wrong-case subset size: {len(subset_keyset)}")

    experiment_name = str(args.experiment)
    experiment_spec = dict(EXPERIMENT_SPECS[experiment_name])
    lambda_grid = _parse_float_grid(str(args.lambda_values), list(experiment_spec["default_lambda_values"]))
    neg_bias_grid = _parse_float_grid(str(args.neg_bias_ratio_values), list(experiment_spec["default_neg_bias_ratio_values"]))

    print(f"Loaded MedQA conflicts: {len(conflicts)}")
    print(f"Experiment: {experiment_name}")
    print(f"Lambda grid: {lambda_grid}")
    print(f"Neg-bias grid: {neg_bias_grid}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = _build_letter_token_sets(tokenizer)

    sweep_rows: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_results: list[dict[str, Any]] | None = None
    best_wrong: list[dict[str, Any]] | None = None

    for neg_bias_ratio in neg_bias_grid:
        for lambda_value in lambda_grid:
            metrics, results, wrong_diagnostics = _run_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                conflicts=conflicts,
                letter_token_sets=letter_token_sets,
                experiment_name=experiment_name,
                experiment_spec=experiment_spec,
                p_set_mode=str(args.p_set_mode),
                lambda_max=float(lambda_value),
                alpha=float(args.alpha),
                neg_bias_ratio=float(neg_bias_ratio),
                tau_margin=float(args.tau_margin),
                evidence_gamma_ratio=float(args.evidence_gamma_ratio),
                max_new_tokens=int(args.max_new_tokens),
                max_length=int(args.max_length),
                target_layer=int(args.target_layer),
                target_head=int(args.target_head),
                subset_keyset=subset_keyset,
                eval_subset=str(args.eval_subset),
                debug_wrong_examples=int(args.debug_wrong_examples),
            )
            config_out_dir = out_dir / f"neg_bias_{_slug(float(neg_bias_ratio))}" / f"lambda_{_slug(float(lambda_value))}"
            config_out_dir.mkdir(parents=True, exist_ok=True)
            write_json(config_out_dir / "medqa_eval_metrics.json", metrics)
            write_jsonl(config_out_dir / "medqa_eval_results.jsonl", results)
            if int(args.write_wrong_diagnostics) == 1:
                write_jsonl(config_out_dir / "medqa_wrong_diagnostics.jsonl", wrong_diagnostics)

            sweep_row = {
                "config_name": experiment_name,
                "experiment_role": str(experiment_spec["role"]),
                "bootstrap_space": str(experiment_spec["bootstrap_E_key"]),
                "intervention_space": str(experiment_spec["interv_E_key"]),
                "p_set_mode": str(args.p_set_mode),
                "lambda_max": float(lambda_value),
                "neg_bias_ratio": float(neg_bias_ratio),
                "baseline_accuracy": float(metrics["baseline_accuracy"]),
                "ft_accuracy": float(metrics["ft_accuracy"]),
                "restoration_rate": float(metrics["restoration_rate"]),
                "harm_rate": float(metrics["harm_rate"]),
                "stability_rate": float(metrics["stability_rate"]),
                "restoration_improvement": float(metrics["restoration_improvement"]),
                "gate_fire_rate_base_wrong": float(metrics["gate_fire_rate_base_wrong"]),
                "gate_fire_rate_base_correct": float(metrics["gate_fire_rate_base_correct"]),
                "fraction_negative_margin_base_wrong": float(metrics["fraction_negative_margin_base_wrong"]),
                "mean_prefill_margin_base_wrong": float(metrics["mean_prefill_margin_base_wrong"]),
                "ft_matches_base_rate": float(metrics["ft_matches_base_rate"]),
                "invalid_ft_letter_rate": float(metrics["invalid_ft_letter_rate"]),
                "mean_lambda_pos_when_conflict": float(metrics["mean_lambda_pos_when_conflict"]),
                "mean_lambda_neg_when_conflict": float(metrics["mean_lambda_neg_when_conflict"]),
                "n_base_wrong": int(metrics["n_base_wrong"]),
                "n_conflict_detected": int(metrics["n_conflict_detected"]),
                "out_dir": str(config_out_dir),
            }
            sweep_rows.append(sweep_row)

            if best_metrics is None:
                best_metrics, best_results, best_wrong = metrics, results, wrong_diagnostics
            else:
                current_key = (
                    float(metrics["restoration_improvement"]),
                    -float(metrics["harm_rate"]),
                    float(metrics["ft_accuracy"]),
                )
                best_key = (
                    float(best_metrics["restoration_improvement"]),
                    -float(best_metrics["harm_rate"]),
                    float(best_metrics["ft_accuracy"]),
                )
                if current_key > best_key:
                    best_metrics, best_results, best_wrong = metrics, results, wrong_diagnostics

            print(json.dumps(metrics, indent=2))
            print(f"metrics={config_out_dir / 'medqa_eval_metrics.json'}")
            print(f"rows={config_out_dir / 'medqa_eval_results.jsonl'}")
            if int(args.write_wrong_diagnostics) == 1:
                print(f"wrong_diagnostics={config_out_dir / 'medqa_wrong_diagnostics.jsonl'}")

    if best_metrics is None or best_results is None or best_wrong is None:
        raise RuntimeError(f"No completed MedQA sweep rows for {experiment_name}")

    ablation_payload = {
        "experiment": experiment_name,
        "role": str(experiment_spec["role"]),
        "description": str(experiment_spec["description"]),
        "conflict_path": str((PROJECT_ROOT / args.conflict_path).resolve()),
        "n_examples_requested": int(args.n_examples),
        "n_examples_loaded": int(len(conflicts)),
        "p_set_mode": str(args.p_set_mode),
        "tau_margin": float(args.tau_margin),
        "alpha": float(args.alpha),
        "evidence_gamma_ratio": float(args.evidence_gamma_ratio),
        "lambda_values": [float(x) for x in lambda_grid],
        "neg_bias_ratio_values": [float(x) for x in neg_bias_grid],
        "rows": sweep_rows,
        "best_by_restoration_improvement": best_metrics,
    }
    write_json(out_dir / "medqa_eval_ablation.json", ablation_payload)
    write_json(out_dir / "medqa_eval_ablation_summary.json", ablation_payload)
    write_json(out_dir / "medqa_eval_metrics_best.json", best_metrics)
    write_json(out_dir / "medqa_eval_metrics.json", best_metrics)
    write_jsonl(out_dir / "medqa_eval_results.jsonl", best_results)
    write_jsonl(out_dir / "medqa_eval_results_best.jsonl", best_results)
    if int(args.write_wrong_diagnostics) == 1:
        write_jsonl(out_dir / "medqa_wrong_diagnostics.jsonl", best_wrong)
        write_jsonl(out_dir / "medqa_wrong_diagnostics_best.jsonl", best_wrong)

    if int(args.run_sanity_checks) == 1:
        sanity_payload = _build_sanity_checks(sweep_rows)
        write_json(out_dir / "sanity_checks.json", sanity_payload)
        print("Sanity checks:")
        print(json.dumps(sanity_payload, indent=2))
        print(f"sanity_checks={out_dir / 'sanity_checks.json'}")

    print("Ablation summary:")
    print(json.dumps(ablation_payload, indent=2))
    print(f"ablation={out_dir / 'medqa_eval_ablation.json'}")
    print(f"best_metrics={out_dir / 'medqa_eval_metrics_best.json'}")
    print(f"best_rows={out_dir / 'medqa_eval_results.jsonl'}")
    if int(args.write_wrong_diagnostics) == 1:
        print(f"best_wrong={out_dir / 'medqa_wrong_diagnostics.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
