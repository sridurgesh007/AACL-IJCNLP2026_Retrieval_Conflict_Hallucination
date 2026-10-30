from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from evidence_aware.ag_erha.track_b import build_evidence_set, build_parametric_set
from evidence_aware.ag_erha.utils import (
    load_cache_rows,
    normalize_example_rows,
    prepare_examples,
    read_json,
    read_jsonl,
    seed_everything,
    write_json,
    write_jsonl,
)
from evidence_aware.mg_erha.gating import MarginGatedHeadAmplifier, decode_with_margin_gate
from evidence_aware.phase2.common import PreparedExample
from evidence_aware.phase2.modeling import load_model_and_tokenizer, to_tensors


@dataclass(frozen=True)
class MGERHAConfig:
    model_id: str
    cache_path: Path
    phase2_main_eval_dir: Path
    track_b_margin_path: Path
    ag_track_a_dir: Path
    out_dir: Path
    tau_margin_grid: list[float]
    sanity_taus: list[float]
    beta: float
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    sanity_tolerance: float
    init_prev_margin: float
    precompute_missing_sets: bool
    strict_track_b_sets: bool
    first_token_only: bool
    bootstrap_first_token_margin: bool
    target_layer_idx: int
    target_head_idx: int


def _parse_float_list(value: str) -> list[float]:
    out = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise ValueError(f"Expected non-empty float list, got: {value}")
    return out


def parse_mg_erha_config(args: Any) -> MGERHAConfig:
    tau_grid = _parse_float_list(args.tau_margin_grid)
    sanity_taus = _parse_float_list(args.sanity_taus)
    if len(sanity_taus) < 2:
        raise ValueError("sanity_taus must contain at least two boundary values.")
    return MGERHAConfig(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase2_main_eval_dir=Path(args.phase2_main_eval_dir),
        track_b_margin_path=Path(args.track_b_margin_path),
        ag_track_a_dir=Path(args.ag_track_a_dir),
        out_dir=Path(args.out_dir),
        tau_margin_grid=tau_grid,
        sanity_taus=sanity_taus,
        beta=float(args.beta),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        sanity_tolerance=float(args.sanity_tolerance),
        init_prev_margin=float(args.init_prev_margin),
        precompute_missing_sets=bool(int(args.precompute_missing_sets)),
        strict_track_b_sets=bool(int(args.strict_track_b_sets)),
        first_token_only=bool(int(args.first_token_only)),
        bootstrap_first_token_margin=bool(int(args.bootstrap_first_token_margin)),
        target_layer_idx=int(getattr(args, "target_layer_idx", 27)),
        target_head_idx=int(getattr(args, "target_head_idx", 20)),
    )


def _metric_as_rate(payload: dict[str, Any], key: str) -> float:
    if key in payload:
        val = float(payload[key])
    elif key == "stability_rate" and "stability" in payload:
        val = float(payload["stability"])
    elif key == "accuracy" and "n_current_faithful" in payload and "n_total" in payload:
        val = float(payload["n_current_faithful"]) / max(float(payload["n_total"]), 1.0)
    else:
        raise KeyError(f"Missing metric '{key}' in payload keys={sorted(payload.keys())}")
    return val / 100.0 if val > 1.0 else val


def load_baseline_label_map(main_eval_dir: Path) -> dict[int, dict[str, Any]]:
    baseline_examples = main_eval_dir / "baseline_examples.jsonl"
    if not baseline_examples.exists():
        raise FileNotFoundError(f"Missing baseline examples map: {baseline_examples}")
    rows = read_jsonl(baseline_examples)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(row["id"])] = {"label": str(row.get("label", "other"))}
    return out


def _load_eval_examples(
    cache_path: Path,
    tokenizer,
    max_length: int,
    baseline_map: dict[int, dict[str, Any]],
) -> list[PreparedExample]:
    rows = normalize_example_rows(load_cache_rows(cache_path))
    eval_ids = set(int(x) for x in baseline_map.keys())
    eval_rows = [r for r in rows if int(r.id) in eval_ids]
    prepared = prepare_examples(eval_rows, tokenizer, max_length=max_length)
    prepared_ids = {x.id for x in prepared}
    missing = sorted(eval_ids - prepared_ids)
    if missing:
        raise RuntimeError(
            "Prepared eval examples do not match baseline id set. "
            f"Missing prepared ids count={len(missing)} example={missing[:10]}"
        )
    # Keep baseline order for reproducibility.
    order = {idx: pos for pos, idx in enumerate(baseline_map.keys())}
    prepared.sort(key=lambda x: order.get(x.id, 10**18))
    return prepared


def _load_track_b_sets(track_b_margin_path: Path) -> dict[int, dict[str, set[int]]]:
    if not track_b_margin_path.exists():
        return {}
    rows = read_jsonl(track_b_margin_path)
    out: dict[int, dict[str, set[int]]] = {}
    for row in rows:
        if "example_id" not in row:
            continue
        if "evidence_set_token_ids" not in row or "parametric_set_token_ids" not in row:
            continue
        example_id = int(row["example_id"])
        e_set = {int(x) for x in row.get("evidence_set_token_ids", [])}
        p_set = {int(x) for x in row.get("parametric_set_token_ids", [])}
        if not e_set or not p_set:
            continue
        out[example_id] = {"evidence_set": e_set, "parametric_set": p_set}
    return out


def _precompute_sets_for_missing_ids(
    model,
    tokenizer,
    device: torch.device,
    examples: list[PreparedExample],
    existing_sets: dict[int, dict[str, set[int]]],
) -> dict[int, dict[str, set[int]]]:
    out = dict(existing_sets)
    for ex in examples:
        if ex.id in out:
            continue
        evidence_set = set(build_evidence_set(tokenizer, ex.answer_liquid))
        if not evidence_set:
            evidence_set = {int(ex.evidence_token_id)}
        input_ids, attention_mask = to_tensors(ex, device=device)
        with torch.inference_mode():
            base_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        base_logits = base_out.logits[0, -1, :].float()
        baseline_top = torch.topk(base_logits, k=min(20, int(base_logits.numel()))).indices.tolist()
        parametric_set = set(build_parametric_set([int(t) for t in baseline_top], evidence_set, k=5))
        parametric_set = {int(x) for x in parametric_set if int(x) not in evidence_set}
        if not parametric_set:
            parametric_set = {int(ex.parametric_token_id)}
        out[ex.id] = {"evidence_set": evidence_set, "parametric_set": parametric_set}
    return out


def _write_set_map(path: Path, set_map: dict[int, dict[str, set[int]]], track_b_ids: set[int]) -> None:
    rows = []
    for example_id, payload in sorted(set_map.items(), key=lambda x: x[0]):
        rows.append(
            {
                "example_id": int(example_id),
                "source": ("track_b" if example_id in track_b_ids else "precomputed"),
                "evidence_set_token_ids": sorted(int(x) for x in payload["evidence_set"]),
                "parametric_set_token_ids": sorted(int(x) for x in payload["parametric_set"]),
            }
        )
    write_jsonl(path, rows)


def _compute_set_diagnostics(
    *,
    eval_examples: list[PreparedExample],
    set_map: dict[int, dict[str, set[int]]],
    track_b_ids: set[int],
    vocab_size: int,
) -> dict[str, Any]:
    eval_ids = [int(ex.id) for ex in eval_examples]
    missing_ids = [eid for eid in eval_ids if eid not in set_map]
    empty_evidence_ids: list[int] = []
    empty_parametric_ids: list[int] = []
    invalid_evidence_ids: list[int] = []
    invalid_parametric_ids: list[int] = []

    for eid in eval_ids:
        payload = set_map.get(eid)
        if payload is None:
            continue
        e_set = set(int(x) for x in payload.get("evidence_set", set()))
        p_set = set(int(x) for x in payload.get("parametric_set", set()))
        if not e_set:
            empty_evidence_ids.append(eid)
        if not p_set:
            empty_parametric_ids.append(eid)
        if e_set and any((tid < 0 or tid >= int(vocab_size)) for tid in e_set):
            invalid_evidence_ids.append(eid)
        if p_set and any((tid < 0 or tid >= int(vocab_size)) for tid in p_set):
            invalid_parametric_ids.append(eid)

    return {
        "n_eval_examples": int(len(eval_ids)),
        "n_track_b_sets_available_for_eval": int(len(track_b_ids & set(eval_ids))),
        "n_precomputed_missing_sets": int(len([eid for eid in eval_ids if eid not in track_b_ids])),
        "n_missing_set_rows": int(len(missing_ids)),
        "n_empty_evidence_sets": int(len(empty_evidence_ids)),
        "n_empty_parametric_sets": int(len(empty_parametric_ids)),
        "n_invalid_evidence_sets": int(len(invalid_evidence_ids)),
        "n_invalid_parametric_sets": int(len(invalid_parametric_ids)),
        "sample_missing_ids": missing_ids[:10],
        "sample_empty_evidence_ids": empty_evidence_ids[:10],
        "sample_empty_parametric_ids": empty_parametric_ids[:10],
        "sample_invalid_evidence_ids": invalid_evidence_ids[:10],
        "sample_invalid_parametric_ids": invalid_parametric_ids[:10],
    }


def _compute_rates(
    results: dict[str, Any],
    baseline_map: dict[int, dict[str, Any]],
    eval_ids: list[int],
) -> None:
    baseline_labels = [baseline_map.get(i, {"label": "other"})["label"] for i in eval_ids]
    n_baseline_hall = sum(1 for x in baseline_labels if x == "hallucinated")
    n_baseline_faith = sum(1 for x in baseline_labels if x == "faithful")
    results["accuracy"] = float(results["n_faithful"] / max(results["n_total"], 1))
    results["restoration_rate"] = float(results["n_restored"] / max(n_baseline_hall, 1))
    results["harm_rate"] = float(results["n_harmed"] / max(n_baseline_faith, 1))
    results["stability_rate"] = float(results["n_stable"] / max(n_baseline_faith, 1))
    results["restoration_improvement"] = float(results["restoration_rate"] - results["harm_rate"])


def _baseline_artifact_consistency_check(
    baseline_map: dict[int, dict[str, Any]],
    baseline_metrics: dict[str, Any],
    tol: float,
) -> dict[str, Any]:
    labels = [str(v.get("label", "other")) for v in baseline_map.values()]
    n_total = len(labels)
    n_faithful = sum(1 for x in labels if x == "faithful")
    mapped_accuracy = float(n_faithful / max(n_total, 1))
    baseline_accuracy = float(_metric_as_rate(baseline_metrics, "accuracy"))
    diff = abs(mapped_accuracy - baseline_accuracy)
    return {
        "n_total_from_baseline_examples": int(n_total),
        "n_faithful_from_baseline_examples": int(n_faithful),
        "accuracy_from_baseline_examples": mapped_accuracy,
        "accuracy_from_baseline_results_json": baseline_accuracy,
        "abs_diff": float(diff),
        "tolerance": float(tol),
        "passed": bool(diff <= tol),
    }


def run_single_tau_margin(
    model,
    tokenizer,
    device: torch.device,
    eval_examples: list[PreparedExample],
    baseline_map: dict[int, dict[str, Any]],
    sets_by_id: dict[int, dict[str, set[int]]],
    tau_margin: float,
    beta: float,
    max_new_tokens: int,
    init_prev_margin: float,
    first_token_only: bool,
    bootstrap_first_token_margin: bool,
    target_layer_idx: int,
    target_head_idx: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = MarginGatedHeadAmplifier(
        model=model,
        evidence_set=set(),
        parametric_set=set(),
        tau_margin=float(tau_margin),
        beta=float(beta),
        init_prev_margin=float(init_prev_margin),
        first_token_only=bool(first_token_only),
        bootstrap_first_token_margin=bool(bootstrap_first_token_margin),
        target_layer_idx=int(target_layer_idx),
        target_head_idx=int(target_head_idx),
    )
    gate.attach()

    results = {
        "tau_margin": float(tau_margin),
        "beta": float(beta),
        "n_total": 0,
        "n_faithful": 0,
        "n_hallucinated": 0,
        "n_other": 0,
        "n_restored": 0,
        "n_harmed": 0,
        "n_stable": 0,
        "n_gate_fired": 0,
        "n_gate_blocked": 0,
        "margin_error_count": 0,
        "n_examples_with_nan_margin": 0,
        "gate_fired_and_restored": 0,
        "gate_fired_and_harmed": 0,
        "gate_stats_by_label": {"faithful": [], "hallucinated": []},
    }
    gate_rows: list[dict[str, Any]] = []

    try:
        for ex in eval_examples:
            sets = sets_by_id.get(ex.id)
            if sets is None:
                raise RuntimeError(f"Missing evidence/parametric sets for example id={ex.id}")
            e_set = set(int(x) for x in sets["evidence_set"])
            p_set = set(int(x) for x in sets["parametric_set"])
            if not e_set or not p_set:
                raise RuntimeError(f"Invalid evidence/parametric sets for example id={ex.id}")

            gate.update_sets(e_set, p_set)
            gate.reset_log()
            out = decode_with_margin_gate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                example=ex,
                max_new_tokens=max_new_tokens,
                gate=gate,
            )
            baseline_label = baseline_map.get(ex.id, {"label": "other"})["label"]

            results["n_total"] += 1
            results[f"n_{out.label}"] += 1
            results["n_gate_fired"] += int(out.gate_fired)
            results["n_gate_blocked"] += int(out.gate_blocked)

            if baseline_label == "hallucinated" and out.label == "faithful":
                results["n_restored"] += 1
                if out.gate_fired > 0:
                    results["gate_fired_and_restored"] += 1
            if baseline_label == "faithful" and out.label == "hallucinated":
                results["n_harmed"] += 1
                if out.gate_fired > 0:
                    results["gate_fired_and_harmed"] += 1
            if baseline_label == "faithful" and out.label == "faithful":
                results["n_stable"] += 1

            summary = gate.gate_summary()
            results["margin_error_count"] += int(summary.get("margin_errors", 0))
            if not np.isfinite(float(summary.get("mean_margin", float("nan")))):
                results["n_examples_with_nan_margin"] += 1
            if baseline_label in ("faithful", "hallucinated"):
                results["gate_stats_by_label"][baseline_label].append(summary)

            gate_rows.append(
                {
                    "example_id": int(ex.id),
                    "tau_margin": float(tau_margin),
                    "beta": float(beta),
                    "baseline_label": str(baseline_label),
                    "label": str(out.label),
                    "gate_fired": int(out.gate_fired),
                    "gate_blocked": int(out.gate_blocked),
                    "gate_fired_frac": float(out.gate_fired_frac),
                    "mean_margin": float(out.mean_margin),
                    "fired_and_restored": bool(baseline_label == "hallucinated" and out.label == "faithful" and out.gate_fired > 0),
                    "fired_and_harmed": bool(baseline_label == "faithful" and out.label == "hallucinated" and out.gate_fired > 0),
                    "margin_steps_collected": int(summary.get("margin_steps_collected", 0)),
                    "margin_error_count": int(summary.get("margin_errors", 0)),
                    "first_margin_error": str(summary.get("first_margin_error", "")),
                }
            )
    finally:
        gate.detach()

    _compute_rates(results, baseline_map=baseline_map, eval_ids=[e.id for e in eval_examples])

    for label in ("faithful", "hallucinated"):
        stats = results["gate_stats_by_label"][label]
        if stats:
            results[f"mean_fired_frac_{label}"] = float(np.mean([float(s["fired_frac"]) for s in stats]))
            valid_margins = [float(s["mean_margin"]) for s in stats if np.isfinite(float(s["mean_margin"]))]
            results[f"mean_margin_{label}"] = float(np.mean(valid_margins)) if valid_margins else float("nan")
        else:
            results[f"mean_fired_frac_{label}"] = float("nan")
            results[f"mean_margin_{label}"] = float("nan")
    return results, gate_rows


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _choose_best_tau(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    strong = [r for r in rows if float(r["stability_rate"]) >= 0.92 and float(r["restoration_improvement"]) >= 0.02]
    if strong:
        return max(strong, key=lambda r: (float(r["restoration_improvement"]), float(r["stability_rate"]))), "STRONG"
    moderate = [r for r in rows if float(r["stability_rate"]) >= 0.88 and float(r["restoration_improvement"]) >= 0.01]
    if moderate:
        return max(moderate, key=lambda r: (float(r["restoration_improvement"]), float(r["stability_rate"]))), "MODERATE"
    fallback = max(rows, key=lambda r: (float(r["restoration_improvement"]), float(r["stability_rate"])))
    return fallback, "GATE_STILL_INSUFFICIENT"


def _load_ag_best_metrics(ag_track_a_dir: Path) -> dict[str, Any] | None:
    summary_path = ag_track_a_dir / "ag_erha_sweep_summary.json"
    if not summary_path.exists():
        return None
    rows = read_json(summary_path)
    if not isinstance(rows, list) or not rows:
        return None
    # AG sweep contains sanity taus 0 and 1; exclude those for best-tau selection.
    candidates = [r for r in rows if float(r.get("tau", 999.0)) not in (0.0, 1.0)]
    if not candidates:
        candidates = rows
    return max(candidates, key=lambda r: (float(r.get("restoration_improvement", 0.0)), float(r.get("stability_rate", 0.0))))


def _plot_mg1_stability_vs_tau(out_dir: Path, rows: list[dict[str, Any]], baseline_stability: float) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = [float(r["tau_margin"]) for r in rows]
    ys = [float(r["stability_rate"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(xs, ys, marker="o", linewidth=2.0, label="MG-ERHA stability")
    ax.axhline(float(baseline_stability), linestyle="--", linewidth=1.2, color="black", label="baseline stability")
    ax.axhline(0.92, linestyle="--", linewidth=1.2, color="orange", label="target 0.92")
    ax.set_title("MG1: Stability vs tau_margin")
    ax.set_xlabel("tau_margin")
    ax.set_ylabel("stability_rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stability_vs_tau_margin.png", dpi=240)
    plt.close(fig)


def _plot_mg2_restoration_improvement(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = [float(r["tau_margin"]) for r in rows]
    restoration = [float(r["restoration_rate"]) for r in rows]
    harm = [float(r["harm_rate"]) for r in rows]
    improvement = [float(r["restoration_improvement"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(xs, improvement, marker="o", linewidth=2.0, label="restoration_improvement")
    ax.plot(xs, restoration, marker="s", linewidth=1.8, label="restoration_rate")
    ax.plot(xs, harm, marker="^", linewidth=1.8, label="harm_rate")
    ax.axhline(0.02, linestyle="--", linewidth=1.2, color="orange", label="target +0.02")
    ax.set_title("MG2: Restoration Improvement vs tau_margin")
    ax.set_xlabel("tau_margin")
    ax.set_ylabel("rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "restoration_improvement_vs_tau_margin.png", dpi=240)
    plt.close(fig)


def _plot_mg3_fire_rate_by_label(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = [float(r["tau_margin"]) for r in rows]
    hall = [float(r.get("mean_fired_frac_hallucinated", np.nan)) for r in rows]
    faith = [float(r.get("mean_fired_frac_faithful", np.nan)) for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(xs, hall, marker="o", linewidth=2.0, label="baseline_hallucinated")
    ax.plot(xs, faith, marker="s", linewidth=2.0, label="baseline_faithful")
    ax.set_title("MG3: Gate Fire Rate by Baseline Label vs tau_margin")
    ax.set_xlabel("tau_margin")
    ax.set_ylabel("mean gate_fired_frac")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "gate_fire_rate_by_label_vs_tau.png", dpi=240)
    plt.close(fig)


def _plot_mg4_margin_distribution(out_dir: Path, gate_rows_by_tau: dict[float, list[dict[str, Any]]], tau_for_plot: float) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = gate_rows_by_tau.get(float(tau_for_plot), [])
    faithful = [float(r["mean_margin"]) for r in rows if r["baseline_label"] == "faithful" and np.isfinite(float(r["mean_margin"]))]
    hall = [float(r["mean_margin"]) for r in rows if r["baseline_label"] == "hallucinated" and np.isfinite(float(r["mean_margin"]))]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.hist(faithful, bins=40, alpha=0.5, label="baseline_faithful")
    ax.hist(hall, bins=40, alpha=0.5, label="baseline_hallucinated")
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_title(f"MG4: Margin Distribution by Baseline Label (tau_margin={tau_for_plot})")
    ax.set_xlabel("mean per-step margin")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "margin_distribution_by_label.png", dpi=240)
    plt.close(fig)


def _plot_mg5_comparison(
    out_dir: Path,
    baseline_metrics: dict[str, Any],
    unconditional_metrics: dict[str, Any] | None,
    ag_best_metrics: dict[str, Any] | None,
    mg_best_metrics: dict[str, Any],
    dola_metrics: dict[str, Any] | None,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    default_metrics = {"accuracy": 0.0, "restoration_rate": 0.0, "stability_rate": 0.0}
    comparison = [
        ("baseline", baseline_metrics or default_metrics),
        ("erha_unconditional_beta5", unconditional_metrics or default_metrics),
        ("ag_erha_best", ag_best_metrics or default_metrics),
        ("mg_erha_best", mg_best_metrics),
        ("dola", dola_metrics or default_metrics),
    ]
    labels = [x[0] for x in comparison]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width, [float(m.get("accuracy", 0.0)) for _, m in comparison], width, label="accuracy")
    ax.bar(x, [float(m.get("restoration_rate", 0.0)) for _, m in comparison], width, label="restoration_rate")
    ax.bar(x + width, [float(m.get("stability_rate", 0.0)) for _, m in comparison], width, label="stability_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("MG5: Final Comparison (baseline / ERHA / AG / MG / DoLa)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_bar_final.png", dpi=240)
    plt.close(fig)


def _run_sanity_checks(
    summary_by_tau: dict[float, dict[str, Any]],
    sanity_taus: list[float],
    baseline_metrics: dict[str, Any],
    unconditional_metrics: dict[str, Any] | None,
    tol: float,
    first_token_only: bool,
) -> dict[str, Any]:
    if unconditional_metrics is None and not first_token_only:
        raise RuntimeError(
            "Missing unconditional ERHA metrics file: "
            "results/phase2/main_eval/erha_results_beta_5.0.json required for MG sanity checks."
        )

    low = float(min(sanity_taus))
    high = float(max(sanity_taus))
    if low not in summary_by_tau or high not in summary_by_tau:
        raise RuntimeError(f"Sanity tau runs missing. required=[{low},{high}] got={sorted(summary_by_tau.keys())}")

    low_row = summary_by_tau[low]
    high_row = summary_by_tau[high]

    checks = []
    pairs = [
        ("low_matches_baseline_accuracy", _metric_as_rate(low_row, "accuracy"), _metric_as_rate(baseline_metrics, "accuracy")),
        ("low_matches_baseline_stability", _metric_as_rate(low_row, "stability_rate"), _metric_as_rate(baseline_metrics, "stability_rate")),
    ]
    if not first_token_only:
        # Per equation Gate fires when M < tau:
        # low (e.g., -999) -> never fire -> baseline
        # high (e.g., +999) -> always fire -> unconditional ERHA
        pairs.extend(
            [
                (
                    "high_matches_unconditional_accuracy",
                    _metric_as_rate(high_row, "accuracy"),
                    _metric_as_rate(unconditional_metrics, "accuracy"),
                ),
                (
                    "high_matches_unconditional_stability",
                    _metric_as_rate(high_row, "stability_rate"),
                    _metric_as_rate(unconditional_metrics, "stability_rate"),
                ),
            ]
        )
    for name, got, ref in pairs:
        diff = abs(float(got) - float(ref))
        checks.append({"name": name, "got": float(got), "ref": float(ref), "abs_diff": float(diff), "passed": bool(diff <= tol)})
    all_passed = all(bool(c["passed"]) for c in checks)
    return {
        "tolerance": float(tol),
        "expected_mapping": (
            {"low_tau": "baseline", "high_tau": "unconditional_erha_beta5"}
            if not first_token_only
            else {"low_tau": "baseline", "high_tau": "first_token_only_always_fire"}
        ),
        "first_token_only_mode": bool(first_token_only),
        "all_passed": bool(all_passed),
        "checks": checks,
        "failed_checks": [c for c in checks if not c["passed"]],
    }


def _check_gate_discrimination(gate_rows_by_tau: dict[float, list[dict[str, Any]]], tau: float) -> dict[str, Any]:
    rows = gate_rows_by_tau.get(float(tau), [])
    hall = [float(r["gate_fired_frac"]) for r in rows if r["baseline_label"] == "hallucinated"]
    faith = [float(r["gate_fired_frac"]) for r in rows if r["baseline_label"] == "faithful"]
    hall_mean = float(np.mean(hall)) if hall else float("nan")
    faith_mean = float(np.mean(faith)) if faith else float("nan")
    passed = bool(np.isfinite(hall_mean) and np.isfinite(faith_mean) and hall_mean > faith_mean)
    return {
        "tau_margin": float(tau),
        "mean_fired_frac_hallucinated": hall_mean,
        "mean_fired_frac_faithful": faith_mean,
        "passed": passed,
    }


def _check_margin_sign_sanity(gate_rows_by_tau: dict[float, list[dict[str, Any]]], tau: float) -> dict[str, Any]:
    rows = gate_rows_by_tau.get(float(tau), [])
    hall = [float(r["mean_margin"]) for r in rows if r["baseline_label"] == "hallucinated" and np.isfinite(float(r["mean_margin"]))][:20]
    faith = [float(r["mean_margin"]) for r in rows if r["baseline_label"] == "faithful" and np.isfinite(float(r["mean_margin"]))][:20]
    hall_mean = float(np.mean(hall)) if hall else float("nan")
    faith_mean = float(np.mean(faith)) if faith else float("nan")
    passed = bool(np.isfinite(hall_mean) and np.isfinite(faith_mean) and faith_mean > 0.0 and hall_mean < 0.0)
    return {
        "tau_margin": float(tau),
        "first20_mean_margin_hallucinated": hall_mean,
        "first20_mean_margin_faithful": faith_mean,
        "passed": passed,
    }


def run_mg_erha_pipeline(config: MGERHAConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(
        model_id=config.model_id,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )

    baseline_map = load_baseline_label_map(config.phase2_main_eval_dir)
    eval_examples = _load_eval_examples(
        cache_path=config.cache_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        baseline_map=baseline_map,
    )

    track_b_sets = _load_track_b_sets(config.track_b_margin_path)
    track_b_ids = set(track_b_sets.keys())
    eval_ids = {ex.id for ex in eval_examples}
    missing_ids = sorted(eval_ids - track_b_ids)

    if missing_ids and config.strict_track_b_sets and not config.precompute_missing_sets:
        raise RuntimeError(
            "Missing track-b precomputed token sets for eval ids while strict_track_b_sets=1 and precompute_missing_sets=0. "
            f"missing_count={len(missing_ids)} sample={missing_ids[:10]}"
        )

    if missing_ids and config.precompute_missing_sets:
        sets_by_id = _precompute_sets_for_missing_ids(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=eval_examples,
            existing_sets=track_b_sets,
        )
    else:
        sets_by_id = dict(track_b_sets)

    # Final safety: every eval example must have non-empty sets before generation.
    final_missing = [ex.id for ex in eval_examples if ex.id not in sets_by_id]
    if final_missing:
        raise RuntimeError(
            "MG-ERHA requires precomputed evidence/parametric sets for all eval examples. "
            f"missing_count={len(final_missing)} sample={final_missing[:10]}"
        )
    _write_set_map(config.out_dir / "precomputed_sets.jsonl", sets_by_id, track_b_ids=track_b_ids)
    set_diagnostics = _compute_set_diagnostics(
        eval_examples=eval_examples,
        set_map=sets_by_id,
        track_b_ids=track_b_ids,
        vocab_size=int(model.lm_head.out_features),
    )
    write_json(config.out_dir / "set_diagnostics.json", set_diagnostics)
    if (
        int(set_diagnostics["n_missing_set_rows"]) > 0
        or int(set_diagnostics["n_empty_evidence_sets"]) > 0
        or int(set_diagnostics["n_empty_parametric_sets"]) > 0
        or int(set_diagnostics["n_invalid_evidence_sets"]) > 0
        or int(set_diagnostics["n_invalid_parametric_sets"]) > 0
    ):
        raise RuntimeError(
            "MG-ERHA set diagnostics failed: missing/empty/invalid token-set rows detected. "
            f"See {config.out_dir / 'set_diagnostics.json'}"
        )

    baseline_metrics = read_json(config.phase2_main_eval_dir / "baseline_results.json")
    unconditional_metrics = _load_optional_json(config.phase2_main_eval_dir / "erha_results_beta_5.0.json")
    dola_metrics = _load_optional_json(config.phase2_main_eval_dir / "dola_results.json")
    ag_best_metrics = _load_ag_best_metrics(config.ag_track_a_dir)

    baseline_consistency = _baseline_artifact_consistency_check(
        baseline_map=baseline_map,
        baseline_metrics=baseline_metrics,
        tol=float(config.sanity_tolerance),
    )
    write_json(config.out_dir / "baseline_artifact_consistency.json", baseline_consistency)
    if not baseline_consistency["passed"]:
        raise RuntimeError(
            "Phase-2 baseline artifacts are inconsistent: baseline_examples.jsonl and baseline_results.json "
            "do not match. Re-run Phase-2 main_eval to regenerate a consistent artifact set. "
            f"See {config.out_dir / 'baseline_artifact_consistency.json'}"
        )

    tau_values = list(dict.fromkeys([float(x) for x in config.sanity_taus + config.tau_margin_grid]))
    tau_results: list[dict[str, Any]] = []
    gate_rows_by_tau: dict[float, list[dict[str, Any]]] = {}

    for tau in tau_values:
        result, gate_rows = run_single_tau_margin(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_examples=eval_examples,
            baseline_map=baseline_map,
            sets_by_id=sets_by_id,
            tau_margin=float(tau),
            beta=float(config.beta),
            max_new_tokens=int(config.max_new_tokens),
            init_prev_margin=float(config.init_prev_margin),
            first_token_only=bool(config.first_token_only),
            bootstrap_first_token_margin=bool(config.bootstrap_first_token_margin),
            target_layer_idx=int(config.target_layer_idx),
            target_head_idx=int(config.target_head_idx),
        )
        tau_results.append(result)
        gate_rows_by_tau[float(tau)] = gate_rows
        write_json(config.out_dir / f"mg_erha_tau_{tau:.1f}.json", result)
        write_jsonl(config.out_dir / f"gate_stats_tau_{tau:.1f}.jsonl", gate_rows)

    tau_results_sorted = sorted(tau_results, key=lambda x: float(x["tau_margin"]))
    write_json(config.out_dir / "mg_erha_sweep_summary.json", tau_results_sorted)

    summary_by_tau = {float(r["tau_margin"]): r for r in tau_results_sorted}
    sanity = _run_sanity_checks(
        summary_by_tau=summary_by_tau,
        sanity_taus=config.sanity_taus,
        baseline_metrics=baseline_metrics,
        unconditional_metrics=unconditional_metrics,
        tol=float(config.sanity_tolerance),
        first_token_only=bool(config.first_token_only),
    )
    write_json(config.out_dir / "sanity_checks.json", sanity)
    if not sanity["all_passed"]:
        raise RuntimeError(f"MG-ERHA sanity checks failed. See {config.out_dir / 'sanity_checks.json'}")

    gate_discrimination = _check_gate_discrimination(gate_rows_by_tau, tau=0.0)
    margin_sign_sanity = _check_margin_sign_sanity(gate_rows_by_tau, tau=0.0)
    write_json(config.out_dir / "gate_discrimination_tau_0.json", gate_discrimination)
    write_json(config.out_dir / "margin_sign_sanity_tau_0.json", margin_sign_sanity)

    non_sanity_rows = [r for r in tau_results_sorted if float(r["tau_margin"]) not in set(config.sanity_taus)]
    if not non_sanity_rows:
        non_sanity_rows = list(tau_results_sorted)
    best_row, verdict = _choose_best_tau(non_sanity_rows)
    best_tau = float(best_row["tau_margin"])
    _plot_mg1_stability_vs_tau(config.out_dir, tau_results_sorted, baseline_stability=float(_metric_as_rate(baseline_metrics, "stability_rate")))
    _plot_mg2_restoration_improvement(config.out_dir, tau_results_sorted)
    _plot_mg3_fire_rate_by_label(config.out_dir, tau_results_sorted)
    _plot_mg4_margin_distribution(config.out_dir, gate_rows_by_tau, tau_for_plot=best_tau)
    _plot_mg5_comparison(
        out_dir=config.out_dir,
        baseline_metrics=baseline_metrics,
        unconditional_metrics=unconditional_metrics,
        ag_best_metrics=ag_best_metrics,
        mg_best_metrics=best_row,
        dola_metrics=dola_metrics,
    )

    best_hall_fired = float(best_row.get("mean_fired_frac_hallucinated", np.nan))
    best_faith_fired = float(best_row.get("mean_fired_frac_faithful", np.nan))
    gate_discriminates = bool(np.isfinite(best_hall_fired) and np.isfinite(best_faith_fired) and best_hall_fired > best_faith_fired)

    verdict_lines = [
        f"BEST_TAU_MARGIN: {best_tau}",
        f"STABILITY_AT_BEST: {float(best_row['stability_rate'])}",
        f"RESTORATION_IMPROVEMENT_AT_BEST: {float(best_row['restoration_improvement'])}",
        f"HARM_RATE_AT_BEST: {float(best_row['harm_rate'])}",
        f"RESTORATION_RATE_AT_BEST: {float(best_row['restoration_rate'])}",
        f"GATE_DISCRIMINATES: {'yes' if gate_discriminates else 'no'}",
        f"MEAN_FIRED_FRAC_HALLUCINATED_AT_BEST: {best_hall_fired}",
        f"MEAN_FIRED_FRAC_FAITHFUL_AT_BEST: {best_faith_fired}",
        f"VERDICT: {verdict}",
    ]
    (config.out_dir / "mg_erha_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    final_summary = {
        "best_tau_margin": best_tau,
        "best_row": best_row,
        "verdict": verdict,
        "baseline_artifact_consistency": baseline_consistency,
        "gate_discrimination_tau_0": gate_discrimination,
        "margin_sign_sanity_tau_0": margin_sign_sanity,
        "sanity_checks": sanity,
        "num_eval_examples": len(eval_examples),
        "track_b_sets_available_for_eval": int(len(track_b_ids & eval_ids)),
        "precomputed_missing_sets_count": int(len(missing_ids)),
        "set_diagnostics": set_diagnostics,
        "mode": {
            "first_token_only": bool(config.first_token_only),
            "bootstrap_first_token_margin": bool(config.bootstrap_first_token_margin),
            "target_layer_idx": int(config.target_layer_idx),
            "target_head_idx": int(config.target_head_idx),
        },
        "outputs": {
            "sweep_summary": str(config.out_dir / "mg_erha_sweep_summary.json"),
            "verdict_txt": str(config.out_dir / "mg_erha_verdict.txt"),
            "plots_dir": str(config.out_dir / "plots"),
        },
    }
    write_json(config.out_dir / "final_summary.json", final_summary)
    return final_summary
