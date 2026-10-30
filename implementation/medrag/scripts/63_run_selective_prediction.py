#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
CONFLICTQA_ROOT = REPO_ROOT / "conflictqa"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))
sys.path.insert(0, str(CONFLICTQA_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from conflictqa_pipeline.utils import clean_generated_answer, normalize_entity, prefill_logits, read_jsonl
from medrag.final_paper import (
    aurc_from_curve,
    auc_rank,
    build_cross_model_metadata,
    load_medqa_cross_model_inputs,
    risk_coverage_curve,
    write_markdown_section,
)
from medrag.medqa_v3 import build_letter_token_sets, decode_letter, set_global_seed
from medrag.utils.io import read_json, write_json


POPQA_BASE_PATH = CONFLICTQA_ROOT / "scripts" / "63_run_popqa_intervention.py"


def _load_popqa_base():
    spec = importlib.util.spec_from_file_location("popqa_selective_base", POPQA_BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load PopQA base script: {POPQA_BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


POPQA_BASE = _load_popqa_base()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective prediction analysis for MedQA and PopQA.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--medqa_conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--medqa_results_path", default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl")
    parser.add_argument("--medqa_matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--medqa_label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--medqa_cache_dir", default="cache/selective_prediction/medqa")
    parser.add_argument("--popqa_results_path", default="../conflictqa/results/popqa/full_3141/results.jsonl")
    parser.add_argument("--popqa_summary_path", default="../conflictqa/results/popqa/full_3141/summary.json")
    parser.add_argument("--out_dir", default="results/selective_prediction")
    parser.add_argument("--figures_dir", default="results/selective_prediction/figures")
    parser.add_argument("--self_consistency_samples", type=int, default=5)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sample_top_p", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--medqa_max_new_tokens", type=int, default=5)
    parser.add_argument("--popqa_max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _resolve_pad_token_id(model, tokenizer) -> int:
    for candidate in (
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None
        if candidate is not None:
            return int(candidate)
    return 0


def _sample_ids(
    *,
    model,
    prompt_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[int]:
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model.generate(
            prompt_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=_resolve_pad_token_id(model, tokenizer=None),
        )
    return [int(x) for x in out[0, prompt_ids.shape[0] :].detach().cpu().tolist()]


def _letter_distribution(logits: torch.Tensor, letter_token_sets: dict[str, set[int]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    vocab = int(logits.numel())
    for letter, token_ids in sorted(letter_token_sets.items()):
        valid = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab]
        if not valid:
            continue
        scores[str(letter)] = float(torch.logsumexp(logits[valid].float(), dim=0).item())
    if not scores:
        return {}
    ordered = sorted(scores.keys())
    arr = torch.as_tensor([scores[key] for key in ordered], dtype=torch.float32)
    probs = torch.softmax(arr, dim=0).detach().cpu().tolist()
    return {key: float(prob) for key, prob in zip(ordered, probs)}


def _entropy_from_probs(probs: list[float]) -> float:
    if not probs:
        return 0.0
    return float(-sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs))


def _agreement_ratio(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(str(label) for label in labels if str(label))
    if not counts:
        return 0.0
    return float(max(counts.values()) / max(len(labels), 1))


def _popqa_prompt(tokenizer, question: str, context: str) -> str:
    return str(POPQA_BASE._build_c2_prompt(tokenizer, question, context))


def _curve_with_thresholds(scores: list[float], correct: list[bool]) -> dict[str, Any]:
    if len(scores) != len(correct):
        raise ValueError("scores and correct must have the same length.")
    pairs = [(float(score), bool(ok)) for score, ok in zip(scores, correct) if math.isfinite(float(score))]
    if not pairs:
        return {"rows": [], "aurc": 1.0}
    unique = sorted({score for score, _ in pairs})
    thresholds = [unique[-1] + 1e-6] + [0.5 * (unique[i] + unique[i + 1]) for i in range(len(unique) - 2, -1, -1)] + [unique[0] - 1e-6]
    rows: list[dict[str, float]] = []
    for tau in thresholds:
        kept = [(score, ok) for score, ok in pairs if float(score) >= float(tau)]
        coverage = float(len(kept) / len(pairs))
        accuracy = float(sum(1 for _, ok in kept if ok) / max(len(kept), 1))
        risk = float(1.0 - accuracy)
        rows.append({"threshold": float(tau), "coverage": coverage, "accuracy": accuracy, "risk": risk})
    curve = [{"coverage": float(row["coverage"]), "risk": float(row["risk"]), "accuracy": float(row["accuracy"])} for row in rows]
    aurc = aurc_from_curve(curve)
    return {"rows": rows, "aurc": float(aurc)}


def _select_knee(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"threshold": 0.0, "coverage": 0.0, "accuracy": 0.0, "risk": 1.0, "distance_to_ideal": math.sqrt(2.0)}
    best = min(
        rows,
        key=lambda row: ((1.0 - float(row["coverage"])) ** 2 + float(row["risk"]) ** 2, -float(row["coverage"]), float(row["threshold"])),
    )
    return dict(best) | {"distance_to_ideal": float(math.sqrt((1.0 - float(best["coverage"])) ** 2 + float(best["risk"]) ** 2))}


def _plot_rc_curves(payload: dict[str, Any], out_path: Path) -> None:
    datasets = ["medqa", "popqa"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        methods = payload[dataset]["methods"]
        for method_name, method_payload in methods.items():
            rows = method_payload["curve"]
            xs = [float(row["coverage"]) for row in rows]
            ys = [float(row["risk"]) for row in rows]
            ax.plot(xs, ys, label=f"{method_name} (AURC={float(method_payload['aurc']):.3f})")
        ax.set_title(dataset.upper())
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_three_mode(payload: dict[str, Any], out_path: Path) -> None:
    summary = payload["selected_operating_point"]
    breakdown = [
        float(summary["answer_rate"]),
        float(summary["intervene_rate"]),
        float(summary["abstain_rate"]),
    ]
    metrics = [
        float(summary["selective_accuracy"]),
        float(summary["restoration_rate"]),
        float(summary["harm_rate"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(["answer", "intervene", "abstain"], breakdown, color=["#4c78a8", "#f58518", "#bab0ab"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Three-mode routing")
    axes[0].set_ylabel("Rate")
    axes[1].bar(["selective acc", "restoration", "harm"], metrics, color=["#54a24b", "#e45756", "#72b7b2"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Quality metrics")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_medqa_payload(args: argparse.Namespace, model, tokenizer, device: torch.device) -> dict[str, Any]:
    shared = load_medqa_cross_model_inputs(
        project_root=PROJECT_ROOT,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=args.medqa_conflict_path,
        matched_subset_path=args.medqa_matched_subset_path,
        label_audit_path=args.medqa_label_audit_path,
        cache_dir=args.medqa_cache_dir,
        target_layer=27,
        max_length=int(args.max_length),
        max_new_tokens=int(args.medqa_max_new_tokens),
        prompt_variant="vanilla",
        p_set_mode="all_wrong",
    )
    result_rows = {
        str(row["item_id"]): row
        for row in read_jsonl((PROJECT_ROOT / args.medqa_results_path).resolve())
        if bool(row.get("in_matched_subset", False)) and not bool(row.get("skipped", False))
    }
    medqa_rows: list[dict[str, Any]] = []
    for rec in shared.prepared_records:
        row = result_rows.get(str(rec.item_id))
        if row is None:
            continue
        probs = _letter_distribution(rec.cached_logits, shared.letter_token_sets)
        entropy = _entropy_from_probs(list(probs.values()))
        maxprob = max(probs.values()) if probs else 0.0
        samples: list[str] = []
        for _ in range(int(args.self_consistency_samples)):
            ids = _sample_ids(
                model=model,
                prompt_ids=rec.prompt_ids,
                device=device,
                max_new_tokens=int(args.medqa_max_new_tokens),
                temperature=float(args.sample_temperature),
                top_p=float(args.sample_top_p),
            )
            label = decode_letter(ids, tokenizer, shared.letter_token_sets) or ""
            samples.append(str(label))
        medqa_rows.append(
            {
                "item_id": str(rec.item_id),
                "margin": float(row["prefill_margin"]),
                "correct": bool(row["base_correct"]),
                "maxprob": float(maxprob),
                "entropy_confidence": float(-entropy),
                "self_consistency": float(_agreement_ratio(samples)),
            }
        )
    return {"rows": medqa_rows}


def _build_popqa_payload(args: argparse.Namespace, model, tokenizer, device: torch.device) -> dict[str, Any]:
    rows = read_jsonl((PROJECT_ROOT / args.popqa_results_path).resolve())
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt = _popqa_prompt(tokenizer, str(row["question"]), str(row["context"]))
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length))
        prompt_ids = encoded["input_ids"][0].detach().cpu().long()
        logits = prefill_logits(model=model, prompt_ids=prompt_ids, device=device)
        probs = torch.softmax(logits.float(), dim=0)
        maxprob = float(torch.max(probs).item())
        entropy = float((-probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        samples: list[str] = []
        for _ in range(int(args.self_consistency_samples)):
            ids = _sample_ids(
                model=model,
                prompt_ids=prompt_ids,
                device=device,
                max_new_tokens=int(args.popqa_max_new_tokens),
                temperature=float(args.sample_temperature),
                top_p=float(args.sample_top_p),
            )
            text = clean_generated_answer(tokenizer.decode(ids, skip_special_tokens=True))
            samples.append(normalize_entity(text))
        out_rows.append(
            {
                "item_id": str(row["item_id"]),
                "margin": float(row["prefill_margin"]),
                "correct": bool(row["c2_context_faithful"]),
                "maxprob": float(maxprob),
                "entropy_confidence": float(-entropy),
                "self_consistency": float(_agreement_ratio(samples)),
                "c2_context_faithful": bool(row["c2_context_faithful"]),
                "c2_parametric": bool(row["c2_parametric"]),
                "c3_context_faithful": bool(row["c3_context_faithful"]),
            }
        )
    return {"rows": out_rows}


def _dataset_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [bool(row["correct"]) for row in rows]
    methods: dict[str, Any] = {}
    for method_name, score_key in (
        ("margin", "margin"),
        ("maxprob", "maxprob"),
        ("entropy", "entropy_confidence"),
        ("self_consistency", "self_consistency"),
    ):
        scores = [float(row[score_key]) for row in rows]
        sweep = _curve_with_thresholds(scores, correct)
        methods[method_name] = {
            "aurc": float(sweep["aurc"]),
            "curve": sweep["rows"],
            "knee": _select_knee(sweep["rows"]),
        }
    return methods


def _three_mode_operation(popqa_rows: list[dict[str, Any]], tau_high: float) -> dict[str, Any]:
    sweep = _curve_with_thresholds([float(row["margin"]) for row in popqa_rows], [bool(row["correct"]) for row in popqa_rows])
    candidate_rows = [row for row in sweep["rows"] if float(row["threshold"]) < float(tau_high)]
    if not candidate_rows:
        candidate_rows = list(sweep["rows"])

    operation_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        tau_low = float(candidate["threshold"])
        answer = 0
        intervene = 0
        abstain = 0
        faithful_answered = 0
        c2_parametric_total = sum(1 for row in popqa_rows if bool(row["c2_parametric"]))
        c2_faithful_total = sum(1 for row in popqa_rows if bool(row["c2_context_faithful"]))
        restored = 0
        harmed = 0
        for row in popqa_rows:
            margin = float(row["margin"])
            if margin < tau_low:
                abstain += 1
                continue
            if margin < float(tau_high):
                intervene += 1
                faithful = bool(row["c3_context_faithful"])
                if faithful:
                    faithful_answered += 1
                if bool(row["c2_parametric"]) and faithful:
                    restored += 1
                if bool(row["c2_context_faithful"]) and (not faithful):
                    harmed += 1
            else:
                answer += 1
                faithful = bool(row["c2_context_faithful"])
                if faithful:
                    faithful_answered += 1
        answered = answer + intervene
        accuracy = float(faithful_answered / max(answered, 1))
        risk = float(1.0 - accuracy)
        coverage = float(answered / max(len(popqa_rows), 1))
        operation_rows.append(
            {
                "tau_low": float(tau_low),
                "tau_high": float(tau_high),
                "answer_rate": float(answer / max(len(popqa_rows), 1)),
                "intervene_rate": float(intervene / max(len(popqa_rows), 1)),
                "abstain_rate": float(abstain / max(len(popqa_rows), 1)),
                "coverage": coverage,
                "selective_accuracy": accuracy,
                "risk": risk,
                "restoration_rate": float(restored / max(c2_parametric_total, 1)),
                "harm_rate": float(harmed / max(c2_faithful_total, 1)),
            }
        )
    selected = min(
        operation_rows,
        key=lambda row: ((1.0 - float(row["coverage"])) ** 2 + float(row["risk"]) ** 2, -float(row["coverage"]), float(row["tau_low"])),
    )
    return {"rows": operation_rows, "selected_operating_point": selected}


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    medqa = _build_medqa_payload(args, model, tokenizer, device)
    popqa = _build_popqa_payload(args, model, tokenizer, device)

    methods_payload = {
        "medqa": {"methods": _dataset_methods(medqa["rows"]), "n_items": int(len(medqa["rows"]))},
        "popqa": {"methods": _dataset_methods(popqa["rows"]), "n_items": int(len(popqa["rows"]))},
    }
    aurc_payload = {
        dataset: {
            method: {
                "aurc": float(payload["aurc"]),
                "knee": payload["knee"],
                "curve": payload["curve"],
            }
            for method, payload in dataset_payload["methods"].items()
        }
        for dataset, dataset_payload in methods_payload.items()
    }
    aurc_summary = {
        dataset: {method: float(payload["aurc"]) for method, payload in dataset_payload["methods"].items()}
        for dataset, dataset_payload in methods_payload.items()
    }
    aurc_output = {
        "aurc": aurc_summary,
        "details": aurc_payload,
        "metadata": build_cross_model_metadata(
            args=args,
            input_files=[
                (PROJECT_ROOT / args.medqa_conflict_path).resolve(),
                (PROJECT_ROOT / args.medqa_results_path).resolve(),
                (PROJECT_ROOT / args.medqa_matched_subset_path).resolve(),
                (PROJECT_ROOT / args.medqa_label_audit_path).resolve(),
                (PROJECT_ROOT / args.popqa_results_path).resolve(),
                (PROJECT_ROOT / args.popqa_summary_path).resolve(),
            ],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "aurc_comparison.json", aurc_output)

    popqa_summary = read_json((PROJECT_ROOT / args.popqa_summary_path).resolve())
    tau_high = float(popqa_summary["selected_tau"])
    three_mode = _three_mode_operation(popqa["rows"], tau_high)
    three_mode["metadata"] = aurc_output["metadata"]
    write_json(out_dir / "three_mode_operation.json", three_mode)

    _plot_rc_curves(methods_payload, figures_dir / "selective_prediction_rc_curve.png")
    _plot_three_mode(three_mode, figures_dir / "three_mode_operation.png")

    lines = [
        "## AURC",
        "",
        "| Dataset | Margin | MaxProb | Entropy | Self-consistency |",
        "|---|---:|---:|---:|---:|",
        f"| MedQA | {aurc_summary['medqa']['margin']:.4f} | {aurc_summary['medqa']['maxprob']:.4f} | {aurc_summary['medqa']['entropy']:.4f} | {aurc_summary['medqa']['self_consistency']:.4f} |",
        f"| PopQA | {aurc_summary['popqa']['margin']:.4f} | {aurc_summary['popqa']['maxprob']:.4f} | {aurc_summary['popqa']['entropy']:.4f} | {aurc_summary['popqa']['self_consistency']:.4f} |",
        "",
        "## Three-mode PopQA operating point",
        f"- tau_high: `{tau_high:.6f}`",
        f"- tau_low: `{float(three_mode['selected_operating_point']['tau_low']):.6f}`",
        f"- answer rate: `{100.0 * float(three_mode['selected_operating_point']['answer_rate']):.1f}%`",
        f"- intervene rate: `{100.0 * float(three_mode['selected_operating_point']['intervene_rate']):.1f}%`",
        f"- abstain rate: `{100.0 * float(three_mode['selected_operating_point']['abstain_rate']):.1f}%`",
        f"- selective accuracy: `{100.0 * float(three_mode['selected_operating_point']['selective_accuracy']):.1f}%`",
        f"- restoration rate: `{100.0 * float(three_mode['selected_operating_point']['restoration_rate']):.1f}%`",
        f"- harm rate: `{100.0 * float(three_mode['selected_operating_point']['harm_rate']):.1f}%`",
        "",
        "## Figures",
        f"- `{(figures_dir / 'selective_prediction_rc_curve.png').relative_to(PROJECT_ROOT)}`",
        f"- `{(figures_dir / 'three_mode_operation.png').relative_to(PROJECT_ROOT)}`",
    ]
    write_markdown_section(out_dir / "SECTION.md", "Selective Prediction", lines)

    print(json.dumps({"aurc_comparison": str(out_dir / 'aurc_comparison.json'), "three_mode_operation": str(out_dir / 'three_mode_operation.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
