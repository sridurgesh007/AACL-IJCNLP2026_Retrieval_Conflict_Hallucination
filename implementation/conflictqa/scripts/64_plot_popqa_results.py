#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PopQA conflict intervention results.")
    parser.add_argument("--results_dir", default="results/popqa")
    parser.add_argument("--figures_dir", default="figures")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _err(values: list[float], intervals: list[list[float]]) -> list[list[float]]:
    lo = [max(0.0, value - float(bounds[0])) for value, bounds in zip(values, intervals)]
    hi = [max(0.0, float(bounds[1]) - value) for value, bounds in zip(values, intervals)]
    return [lo, hi]


def plot_main(summary: dict[str, Any], out_path: Path) -> str:
    conditions = summary["conditions"]
    keys = ["C1", "C2", "C3", "C4", "C5"]
    labels = {
        "C1": "No context",
        "C2": "Short context",
        "C3": "Mechanism",
        "C4": "Authoritative",
        "C5": "Injection",
    }
    colors = ["#8f8f8f", "#2b8cbe", "#31a354", "#d95f02", "#756bb1"]
    metrics = [
        ("faithfulness_rate", "faithfulness_rate_ci95_wilson", "Faithfulness rate"),
        ("restoration_rate", "restoration_rate_ci95_wilson", "Restoration rate vs C2 parametric"),
        ("harm_rate", "harm_rate_ci95_wilson", "Harm rate vs C2 faithful"),
    ]
    selected_lambda_pos = float(summary.get("selected_lambda_pos", 0.0) or 0.0)
    selected_lambda_neg = float(summary.get("selected_lambda_neg", 0.0) or 0.0)
    intervention_mode = str(summary.get("intervention_mode", "suppression"))
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    for ax, (metric, ci_key, title) in zip(axes, metrics):
        values = [float(conditions[key][metric]) for key in keys]
        intervals = [conditions[key][ci_key] for key in keys]
        x = list(range(len(keys)))
        bars = ax.bar(x, values, yerr=_err(values, intervals), capsize=4, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, [labels[key] for key in keys])
        ax.set_ylim(0.0, min(1.0, max(values) + 0.20))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "PopQA conflict intervention "
        f"(pilot_auc={float(summary['pilot']['margin_auc']):.3f}, full_auc={float(summary['margin_auc_full']):.3f}, "
        f"tau={float(summary['selected_tau']):.3f}, mode={intervention_mode}, "
        f"lambda_pos={selected_lambda_pos:.1f}, lambda_neg={selected_lambda_neg:.1f})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def plot_margin_distribution(pilot: dict[str, Any], out_path: Path) -> str:
    faithful = [float(value) for value in pilot.get("faithful_margins", [])]
    parametric = [float(value) for value in pilot.get("parametric_margins", [])]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(parametric, bins=20, alpha=0.60, color="#de2d26", label=f"Parametric (n={len(parametric)})")
    ax.hist(faithful, bins=20, alpha=0.60, color="#2b8cbe", label=f"Faithful (n={len(faithful)})")
    if pilot.get("selected_tau") is not None:
        ax.axvline(float(pilot["selected_tau"]), color="black", linestyle="--", linewidth=1.0, label=f"Selected tau={float(pilot['selected_tau']):.3f}")
    ax.set_xlabel("Prefill margin log P(E) - log P(P)")
    ax.set_ylabel("count")
    ax.set_title(f"Pilot margin distribution (AUC={float(pilot['margin_auc']):.3f})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def plot_stratification(summary: dict[str, Any], out_path: Path) -> str:
    strat = summary["stratification"]
    s_pop_rows = strat["s_pop"]["rows"]
    prob_rows = strat["llm_answer_prob"]["rows"]
    prop_rows = strat["prop"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    for ax, rows, title in [
        (axes[0], s_pop_rows, "Restoration by subject popularity"),
        (axes[1], prob_rows, "Restoration by no-context confidence"),
    ]:
        labels = [str(row["label"]) for row in rows]
        values = [float(row["restoration_rate"]) for row in rows]
        intervals = [row["restoration_rate_ci95_wilson"] for row in rows]
        x = list(range(len(rows)))
        bars = ax.bar(x, values, yerr=_err(values, intervals), capsize=4, color="#31a354", edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, min(1.0, max(values) + 0.18 if values else 0.2))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    labels = [str(row["prop"]) for row in prop_rows]
    values = [float(row["restoration_rate"]) for row in prop_rows]
    intervals = [row["restoration_rate_ci95_wilson"] for row in prop_rows]
    x = list(range(len(prop_rows)))
    bars = ax.bar(x, values, yerr=_err(values, intervals), capsize=4, color="#756bb1", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0.0, min(1.0, max(values) + 0.18 if values else 0.2))
    ax.set_title("Restoration by relation type")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    pilot = _read_json(results_dir / "pilot_margin_auc.json")
    summary = _read_json(results_dir / "summary.json")
    lambda_sweep = _read_json(results_dir / "lambda_sweep.json")

    results_plot = plot_main(summary, figures_dir / "popqa_results.png")
    margin_plot = plot_margin_distribution(pilot, figures_dir / "popqa_margin_distribution.png")
    strat_plot = plot_stratification(summary, figures_dir / "popqa_stratification.png")

    plot_summary = {
        "pilot_margin_auc_json": str((results_dir / "pilot_margin_auc.json").resolve()),
        "lambda_sweep_json": str((results_dir / "lambda_sweep.json").resolve()),
        "summary_json": str((results_dir / "summary.json").resolve()),
        "results_jsonl": str((results_dir / "results.jsonl").resolve()),
        "results_figure": results_plot,
        "margin_distribution_figure": margin_plot,
        "stratification_figure": strat_plot,
    }
    _write_json(results_dir / "plot_summary.json", plot_summary)
    _write_json(figures_dir / "popqa_plot_summary.json", plot_summary)
    print(json.dumps(plot_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
