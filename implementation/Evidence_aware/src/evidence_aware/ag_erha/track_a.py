from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evidence_aware.ag_erha.gating import AttentionGatedERHA, decode_with_gate
from evidence_aware.ag_erha.utils import evidence_positions_with_fallback, read_json, read_jsonl, write_json, write_jsonl
from evidence_aware.phase2.common import PreparedExample


def load_baseline_label_map(main_eval_dir: Path) -> dict[int, dict[str, Any]]:
    baseline_examples = main_eval_dir / "baseline_examples.jsonl"
    if not baseline_examples.exists():
        raise FileNotFoundError(f"Missing baseline examples map: {baseline_examples}")
    rows = read_jsonl(baseline_examples)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(row["id"])] = {"label": str(row.get("label", "other"))}
    return out


def _compute_primary_metrics(results: dict[str, Any], baseline_map: dict[int, dict[str, Any]], eval_ids: list[int]) -> None:
    baseline_labels = [baseline_map.get(i, {"label": "other"})["label"] for i in eval_ids]
    n_baseline_hall = sum(1 for x in baseline_labels if x == "hallucinated")
    n_baseline_faith = sum(1 for x in baseline_labels if x == "faithful")
    results["accuracy"] = results["n_faithful"] / max(results["n_total"], 1)
    results["restoration_rate"] = results["n_restored"] / max(n_baseline_hall, 1)
    results["harm_rate"] = results["n_harmed"] / max(n_baseline_faith, 1)
    results["stability_rate"] = results["n_stable"] / max(n_baseline_faith, 1)
    results["restoration_improvement"] = results["restoration_rate"] - results["harm_rate"]


def _metric_as_rate(payload: dict[str, Any], key: str) -> float:
    if key in payload:
        val = float(payload[key])
    elif key == "stability_rate" and "stability" in payload:
        val = float(payload["stability"])
    elif key == "accuracy" and "n_current_faithful" in payload and "n_total" in payload:
        val = float(payload["n_current_faithful"]) / max(float(payload["n_total"]), 1.0)
    else:
        raise KeyError(f"Missing metric '{key}' in payload keys={sorted(payload.keys())}")
    # Accept either [0,1] or [0,100] convention.
    return val / 100.0 if val > 1.0 else val


def _run_sanity_checks(
    tau0: dict[str, Any],
    tau1: dict[str, Any],
    baseline_metrics: dict[str, Any],
    unconditional_metrics: dict[str, Any] | None,
    tol: float = 1e-3,
) -> dict[str, Any]:
    if unconditional_metrics is None:
        raise RuntimeError(
            "Missing unconditional ERHA metrics file (expected erha_results_beta_5.0.json) "
            "required for tau=0.0 sanity check."
        )

    checks = []
    # Check 1: tau=0.0 must match unconditional ERHA beta=5.0
    for metric in ("accuracy", "stability_rate"):
        got = _metric_as_rate(tau0, metric)
        ref = _metric_as_rate(unconditional_metrics, metric)
        diff = abs(got - ref)
        checks.append(
            {
                "name": f"tau0_matches_unconditional_{metric}",
                "got": got,
                "ref": ref,
                "abs_diff": diff,
                "passed": diff <= tol,
            }
        )

    # Check 2: tau=1.0 must match baseline
    for metric in ("accuracy", "stability_rate"):
        got = _metric_as_rate(tau1, metric)
        ref = _metric_as_rate(baseline_metrics, metric)
        diff = abs(got - ref)
        checks.append(
            {
                "name": f"tau1_matches_baseline_{metric}",
                "got": got,
                "ref": ref,
                "abs_diff": diff,
                "passed": diff <= tol,
            }
        )

    all_pass = all(bool(c["passed"]) for c in checks)
    failed = [c for c in checks if not c["passed"]]
    return {"tolerance": tol, "all_passed": all_pass, "checks": checks, "failed_checks": failed}


def run_single_tau(
    model,
    tokenizer,
    device,
    eval_examples: list[PreparedExample],
    tau: float,
    beta: float,
    baseline_map: dict[int, dict[str, Any]],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = AttentionGatedERHA(model=model, evidence_positions=[], tau=tau, beta=beta)
    gate.attach()
    results = {
        "tau": float(tau),
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
        "gate_fired_and_restored": 0,
        "gate_fired_and_harmed": 0,
        "n_evidence_fallback_used": 0,
    }
    gate_rows: list[dict[str, Any]] = []

    try:
        for ex in eval_examples:
            evidence_positions, fallback_used = evidence_positions_with_fallback(tokenizer, ex.input_ids, ex.evidence)
            gate.update_evidence_positions(evidence_positions)
            gate.reset_log()
            out = decode_with_gate(
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
            results["n_gate_fired"] += out.gate_fired
            results["n_gate_blocked"] += out.gate_blocked
            if fallback_used:
                results["n_evidence_fallback_used"] += 1
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

            gate_rows.append(
                {
                    "example_id": ex.id,
                    "tau": float(tau),
                    "beta": float(beta),
                    "baseline_label": baseline_label,
                    "label": out.label,
                    "gate_fired": out.gate_fired,
                    "gate_blocked": out.gate_blocked,
                    "gate_fired_frac": out.gate_fired_frac,
                    "fallback_used": bool(fallback_used),
                }
            )
    finally:
        gate.detach()

    _compute_primary_metrics(results, baseline_map=baseline_map, eval_ids=[e.id for e in eval_examples])
    return results, gate_rows


def _best_tau_rule(results: list[dict[str, Any]]) -> tuple[float | None, str, str]:
    rule1 = [r for r in results if r["stability_rate"] >= 0.92 and r["restoration_improvement"] >= 0.02]
    if rule1:
        best = max(rule1, key=lambda r: r["restoration_improvement"])
        return float(best["tau"]), "STRONG", "proceed_to_gate_upgrade"
    rule2 = [r for r in results if r["stability_rate"] >= 0.88 and r["restoration_improvement"] >= 0.015]
    if rule2:
        best = max(rule2, key=lambda r: r["restoration_improvement"])
        return float(best["tau"]), "MARGINAL", "track_b_only"
    return None, "GATE_INSUFFICIENT", "track_b_only"


def _gate_discriminates(gate_rows_by_tau: dict[float, list[dict[str, Any]]], tau_ref: float | None) -> bool:
    if not gate_rows_by_tau:
        return False
    key = tau_ref if tau_ref is not None else sorted(gate_rows_by_tau.keys())[0]
    rows = gate_rows_by_tau.get(key, [])
    hall = [r["gate_fired_frac"] for r in rows if r["baseline_label"] == "hallucinated"]
    faith = [r["gate_fired_frac"] for r in rows if r["baseline_label"] == "faithful"]
    if not hall or not faith:
        return False
    return float(np.mean(hall)) > float(np.mean(faith))


def generate_track_a_plots(
    out_dir: Path,
    tau_results: list[dict[str, Any]],
    gate_rows_by_tau: dict[float, list[dict[str, Any]]],
    baseline_metrics: dict[str, Any],
    unconditional_metrics: dict[str, Any] | None,
    dola_metrics: dict[str, Any] | None,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(tau_results, key=lambda x: float(x["tau"]))
    xs = [float(r["tau"]) for r in sorted_rows]
    stability = [float(r["stability_rate"]) for r in sorted_rows]
    improvement = [float(r["restoration_improvement"]) for r in sorted_rows]
    restoration = [float(r["restoration_rate"]) for r in sorted_rows]
    harm = [float(r["harm_rate"]) for r in sorted_rows]
    accuracy = [float(r["accuracy"]) for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(xs, stability, marker="o", linewidth=2.0)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="baseline stability")
    ax.axhline(0.92, linestyle="--", linewidth=1.2, color="orange", label="target 0.92")
    ax.set_title("Stability rate vs gate threshold tau (beta=5, L27H20)")
    ax.set_xlabel("tau")
    ax.set_ylabel("stability_rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stability_vs_tau.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(xs, improvement, marker="o", linewidth=2.0, label="restoration_improvement")
    ax.plot(xs, restoration, marker="s", linewidth=1.8, label="restoration_rate")
    ax.plot(xs, harm, marker="^", linewidth=1.8, label="harm_rate")
    ax.axhline(0.02, linestyle="--", linewidth=1.2, color="orange", label="target 0.02")
    ax.set_title("Restoration improvement vs tau")
    ax.set_xlabel("tau")
    ax.set_ylabel("rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "restoration_improvement_vs_tau.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(xs, accuracy, marker="o", linewidth=2.0)
    ax.axhline(float(baseline_metrics["accuracy"]), linestyle="--", linewidth=1.2, label="baseline accuracy")
    ax.set_title("Accuracy vs tau")
    ax.set_xlabel("tau")
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_tau.png", dpi=220)
    plt.close(fig)

    hall_mean, faith_mean = [], []
    for tau in xs:
        rows = gate_rows_by_tau.get(tau, [])
        hall = [r["gate_fired_frac"] for r in rows if r["baseline_label"] == "hallucinated"]
        faith = [r["gate_fired_frac"] for r in rows if r["baseline_label"] == "faithful"]
        hall_mean.append(float(np.mean(hall)) if hall else 0.0)
        faith_mean.append(float(np.mean(faith)) if faith else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(xs, hall_mean, marker="o", linewidth=2.0, label="baseline_hallucinated")
    ax.plot(xs, faith_mean, marker="s", linewidth=2.0, label="baseline_faithful")
    ax.set_title("Gate activation rate by baseline label")
    ax.set_xlabel("tau")
    ax.set_ylabel("mean gate_fired_frac")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "gate_fire_rate_vs_tau.png", dpi=220)
    plt.close(fig)

    best_tau, _, _ = _best_tau_rule(sorted_rows)
    best_row = None
    if best_tau is not None:
        best_row = next((r for r in sorted_rows if float(r["tau"]) == float(best_tau)), None)
    if best_row is None and sorted_rows:
        best_row = max(sorted_rows, key=lambda r: r["restoration_improvement"])

    labels = ["baseline", "erha_unconditional_beta5", "ag_erha_best_tau", "dola"]
    metrics_source = [
        baseline_metrics,
        unconditional_metrics or {"accuracy": 0.0, "restoration_rate": 0.0, "stability_rate": 0.0},
        best_row or {"accuracy": 0.0, "restoration_rate": 0.0, "stability_rate": 0.0},
        dola_metrics or {"accuracy": 0.0, "restoration_rate": 0.0, "stability_rate": 0.0},
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, [float(m["accuracy"]) for m in metrics_source], width, label="accuracy")
    ax.bar(x, [float(m["restoration_rate"]) for m in metrics_source], width, label="restoration_rate")
    ax.bar(x + width, [float(m["stability_rate"]) for m in metrics_source], width, label="stability_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Baseline vs Unconditional ERHA vs AG-ERHA (best tau) vs DoLa")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_bar_tau_best.png", dpi=220)
    plt.close(fig)


def run_track_a(
    model,
    tokenizer,
    device,
    eval_examples: list[PreparedExample],
    tau_grid: list[float],
    beta: float,
    max_new_tokens: int,
    phase2_main_eval_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_map = load_baseline_label_map(phase2_main_eval_dir)
    baseline_metrics = read_json(phase2_main_eval_dir / "baseline_results.json")
    unconditional_path = phase2_main_eval_dir / "erha_results_beta_5.0.json"
    unconditional_metrics = read_json(unconditional_path) if unconditional_path.exists() else None
    dola_path = phase2_main_eval_dir / "dola_results.json"
    dola_metrics = read_json(dola_path) if dola_path.exists() else None

    tau_results: list[dict[str, Any]] = []
    gate_rows_by_tau: dict[float, list[dict[str, Any]]] = {}
    # Step 1: boundary sanity checks first, before any sweep.
    for tau in [0.0, 1.0]:
        result, gate_rows = run_single_tau(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_examples=eval_examples,
            tau=tau,
            beta=beta,
            baseline_map=baseline_map,
            max_new_tokens=max_new_tokens,
        )
        tau_results.append(result)
        gate_rows_by_tau[float(tau)] = gate_rows
        write_json(out_dir / f"ag_erha_tau_{tau:.2f}.json", result)
        write_jsonl(out_dir / f"gate_stats_tau_{tau:.2f}.jsonl", gate_rows)
        if tau == 0.0:
            write_json(out_dir / "sanity_tau_0.0.json", result)
        if tau == 1.0:
            write_json(out_dir / "sanity_tau_1.0.json", result)

    tau0 = next(r for r in tau_results if float(r["tau"]) == 0.0)
    tau1 = next(r for r in tau_results if float(r["tau"]) == 1.0)
    sanity_summary = _run_sanity_checks(
        tau0=tau0,
        tau1=tau1,
        baseline_metrics=baseline_metrics,
        unconditional_metrics=unconditional_metrics,
        tol=1e-3,
    )
    write_json(out_dir / "sanity_checks.json", sanity_summary)
    if not sanity_summary["all_passed"]:
        raise RuntimeError(
            "Track A sanity checks failed. "
            f"See {out_dir / 'sanity_checks.json'} for metric diffs."
        )

    # Step 2: full tau sweep after sanity passes.
    sweep_taus = sorted(set(float(x) for x in tau_grid if float(x) not in (0.0, 1.0)))
    for tau in sweep_taus:
        result, gate_rows = run_single_tau(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_examples=eval_examples,
            tau=tau,
            beta=beta,
            baseline_map=baseline_map,
            max_new_tokens=max_new_tokens,
        )
        tau_results.append(result)
        gate_rows_by_tau[float(tau)] = gate_rows
        write_json(out_dir / f"ag_erha_tau_{tau:.2f}.json", result)
        write_jsonl(out_dir / f"gate_stats_tau_{tau:.2f}.jsonl", gate_rows)

    tau_results_sorted = sorted(tau_results, key=lambda x: float(x["tau"]))
    write_json(out_dir / "ag_erha_sweep_summary.json", tau_results_sorted)
    generate_track_a_plots(
        out_dir=out_dir,
        tau_results=tau_results_sorted,
        gate_rows_by_tau=gate_rows_by_tau,
        baseline_metrics=baseline_metrics,
        unconditional_metrics=unconditional_metrics,
        dola_metrics=dola_metrics,
    )

    best_tau, verdict, next_step = _best_tau_rule([r for r in tau_results_sorted if r["tau"] not in (0.0, 1.0)])
    if best_tau is not None:
        best_row = next(r for r in tau_results_sorted if float(r["tau"]) == float(best_tau))
    else:
        best_row = max([r for r in tau_results_sorted if r["tau"] not in (0.0, 1.0)], key=lambda r: r["restoration_improvement"])
    gate_discriminates = _gate_discriminates(gate_rows_by_tau, best_tau)

    verdict_lines = [
        f"BEST_TAU: {best_tau if best_tau is not None else 'none'}",
        f"STABILITY_AT_BEST: {best_row['stability_rate']}",
        f"RESTORATION_IMPROVEMENT_AT_BEST: {best_row['restoration_improvement']}",
        f"GATE_DISCRIMINATES: {'yes' if gate_discriminates else 'no'}",
        f"VERDICT: {verdict}",
        f"RECOMMENDED_NEXT: {next_step}",
    ]
    (out_dir / "track_a_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    return {
        "tau_results": tau_results_sorted,
        "best_tau": best_tau,
        "best_row": best_row,
        "gate_discriminates": gate_discriminates,
        "verdict": verdict,
        "recommended_next": next_step,
    }
