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
    parser = argparse.ArgumentParser(description="Plot CounterFact C2-C5 cross-model prompt-baseline comparison.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--figures_dir", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _value(condition: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in condition:
            return float(condition[key])
    return float(default)


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    figures_dir = (PROJECT_ROOT / (args.figures_dir or str(Path(args.results_dir) / "figures"))).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_json(results_dir / "summary.json")
    conditions = summary["conditions"]
    keys = ["C2", "C3", "C4", "C5"]
    missing = [key for key in keys if key not in conditions]
    if missing:
        raise RuntimeError(f"Missing conditions in summary.json: {', '.join(missing)}")

    labels = ["C2\nEvidence", "C3\nMechanism", "C4\nAuthoritative", "C5\nExplicit answer"]
    faithful_values = [
        _value(conditions["C2"], "faithful_rate"),
        _value(conditions["C3"], "ft_faithful_rate"),
        _value(conditions["C4"], "condition_faithful_rate", "ft_faithful_rate"),
        _value(conditions["C5"], "condition_faithful_rate", "ft_faithful_rate"),
    ]
    restoration_values = [
        0.0,
        _value(conditions["C3"], "restoration_rate"),
        _value(conditions["C4"], "restoration_rate"),
        _value(conditions["C5"], "restoration_rate"),
    ]
    harm_values = [
        0.0,
        _value(conditions["C3"], "harm_rate"),
        _value(conditions["C4"], "harm_rate"),
        _value(conditions["C5"], "harm_rate"),
    ]

    colors = ["#8f8f8f", "#31a354", "#fdae61", "#d7191c"]
    x = list(range(len(keys)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    bars = axes[0].bar(x, faithful_values, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, min(1.0, max(faithful_values) + 0.2))
    axes[0].set_ylabel("faithful rate")
    axes[0].set_title("Final faithful outputs")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, faithful_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    width = 0.38
    left = [i - width / 2.0 for i in x]
    right = [i + width / 2.0 for i in x]
    bars_r = axes[1].bar(left, restoration_values, width=width, color="#31a354", edgecolor="black", linewidth=0.8, label="restoration")
    bars_h = axes[1].bar(right, harm_values, width=width, color="#756bb1", edgecolor="black", linewidth=0.8, label="harm")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, min(1.0, max(restoration_values + harm_values) + 0.2))
    axes[1].set_ylabel("rate vs C2")
    axes[1].set_title("Restoration and harm")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in list(zip(bars_r, restoration_values)) + list(zip(bars_h, harm_values)):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"CounterFact {summary.get('model_name', summary.get('model_id'))} C2-C5 comparison")
    figure_path = figures_dir / "counterfact_c2345_comparison.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    plot_summary = {
        "summary_json": str((results_dir / "summary.json").resolve()),
        "figure": str(figure_path.resolve()),
        "conditions": keys,
    }
    _write_json(results_dir / "plot_summary_c45.json", plot_summary)
    _write_json(figures_dir / "counterfact_c45_plot_summary.json", plot_summary)
    print(json.dumps(plot_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
