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
    parser = argparse.ArgumentParser(description="Plot MedQA cross-model C1-C5 prompt-baseline comparison.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--figures_dir", default="")
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


def _available_condition_keys(summary: dict[str, Any]) -> list[str]:
    conditions = summary.get("conditions", {})
    keys = [key for key in ("C1", "C2", "C3", "C4", "C5") if key in conditions]
    missing = [key for key in ("C1", "C2", "C3", "C4", "C5") if key not in conditions]
    if missing:
        raise RuntimeError(f"Missing conditions in summary.json: {', '.join(missing)}")
    return keys


def _condition_values(
    conditions: dict[str, Any],
    keys: list[str],
    metric: str,
    ci_metric: str,
) -> tuple[list[float], list[list[float]]]:
    values = [float(conditions[key][metric]) for key in keys]
    cis = [list(conditions[key][ci_metric]) for key in keys]
    return values, cis


def plot_accuracy(summary: dict[str, Any], out_path: Path) -> str:
    conditions = summary["conditions"]
    keys = _available_condition_keys(summary)
    labels = {
        "C1": "C1\nNo context",
        "C2": "C2\nTrust prompt",
        "C3": "C3\nMechanism",
        "C4": "C4\nAuthoritative",
        "C5": "C5\nExplicit answer",
    }
    colors = ["#8f8f8f", "#2b8cbe", "#31a354", "#fdae61", "#d7191c"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    specs = [
        ("condition_accuracy", "condition_accuracy_ci95_wilson", "Matched-subset accuracy"),
        ("condition_accuracy_clean_subset", "condition_accuracy_clean_subset_ci95_wilson", "Clean-subset accuracy"),
    ]
    x = list(range(len(keys)))
    for ax, (metric, ci_metric, title) in zip(axes, specs):
        values, cis = _condition_values(conditions, keys, metric, ci_metric)
        bars = ax.bar(x, values, yerr=_err(values, cis), capsize=4, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, [labels[key] for key in keys])
        ax.set_ylim(0.0, min(1.0, max(values) + 0.15 if values else 0.2))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{summary.get('model_name', 'model')} MedQA C1-C5 accuracy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def plot_transition(summary: dict[str, Any], out_path: Path) -> str:
    conditions = summary["conditions"]
    keys = _available_condition_keys(summary)
    labels = {
        "C1": "C1\nNo context",
        "C2": "C2\nTrust prompt",
        "C3": "C3\nMechanism",
        "C4": "C4\nAuthoritative",
        "C5": "C5\nExplicit answer",
    }
    colors = ["#8f8f8f", "#2b8cbe", "#31a354", "#fdae61", "#d7191c"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    specs = [
        ("restoration_rate", "restoration_rate_ci95_wilson", "Restoration vs C1-wrong"),
        ("harm_rate", "harm_rate_ci95_wilson", "Harm vs C1-correct"),
    ]
    x = list(range(len(keys)))
    for ax, (metric, ci_metric, title) in zip(axes, specs):
        values, cis = _condition_values(conditions, keys, metric, ci_metric)
        bars = ax.bar(x, values, yerr=_err(values, cis), capsize=4, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, [labels[key] for key in keys])
        ax.set_ylim(0.0, min(1.0, max(values) + 0.15 if values else 0.2))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{summary.get('model_name', 'model')} MedQA prompt-baseline comparison")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve() if args.figures_dir else (results_dir / "figures").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(results_dir / "summary.json")
    accuracy_plot = plot_accuracy(summary, figures_dir / "c12345_accuracy.png")
    transition_plot = plot_transition(summary, figures_dir / "c12345_restoration_harm.png")
    plot_summary = {
        "summary_json": str((results_dir / "summary.json").resolve()),
        "accuracy_figure": accuracy_plot,
        "restoration_harm_figure": transition_plot,
        "conditions": ["C1", "C2", "C3", "C4", "C5"],
    }
    _write_json(results_dir / "plot_summary_c45.json", plot_summary)
    _write_json(figures_dir / "medqa_c45_plot_summary.json", plot_summary)
    print(json.dumps(plot_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
