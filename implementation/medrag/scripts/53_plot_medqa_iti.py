#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medrag.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ITI comparison figures for MedQA.")
    parser.add_argument("--results_root", default="results/medqa_final_v3/baselines/iti")
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/baselines/iti/figures")
    parser.add_argument("--title_prefix", default="MedQA")
    return parser.parse_args()


def _condition_sort_key(condition_id: str) -> tuple[int, str]:
    if condition_id.startswith("C") and condition_id[1:].isdigit():
        return int(condition_id[1:]), str(condition_id)
    return 999, str(condition_id)


def _stars(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "n.s."


def _color(condition_id: str) -> str:
    palette = {
        "C1": "#9b9b9b",
        "C2": "#d2b48c",
        "C3": "#c49c94",
        "C4": "#3f7fba",
        "C5": "#7a5195",
        "C6": "#3ca370",
        "C7": "#de8f05",
    }
    return palette.get(str(condition_id), "#666666")


def plot_killtest(comparison: dict[str, Any], figures_dir: Path, title_prefix: str) -> str:
    rows = sorted(list(comparison.get("table_rows", [])), key=lambda row: _condition_sort_key(str(row.get("condition_id", ""))))
    if not rows:
        raise RuntimeError("comparison.json does not contain table_rows.")
    labels = [str(row.get("condition_id", "")) for row in rows]
    values = [float(row.get("restoration_rate_clean_subset", 0.0)) for row in rows]
    yerr_lo = [max(0.0, float(row.get("restoration_rate_clean_subset", 0.0)) - float(row.get("restoration_rate_clean_subset_ci95_wilson", [0.0, 0.0])[0])) for row in rows]
    yerr_hi = [max(0.0, float(row.get("restoration_rate_clean_subset_ci95_wilson", [0.0, 0.0])[1]) - float(row.get("restoration_rate_clean_subset", 0.0))) for row in rows]
    colors = [_color(label) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(rows)))
    bars = ax.bar(
        x,
        values,
        color=colors,
        yerr=[yerr_lo, yerr_hi],
        capsize=4,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, max(max(values) + 0.12, 0.15))
    ax.set_ylabel("Clean restoration rate")
    ax.set_title(f"{title_prefix}: Kill-test with ITI")
    ax.grid(axis="y", alpha=0.25)

    for bar, row in zip(bars, rows):
        value = float(row.get("restoration_rate_clean_subset", 0.0))
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    index_by_id = {str(row.get("condition_id", "")): idx for idx, row in enumerate(rows)}
    for candidate_id in ("C6", "C7"):
        if candidate_id not in index_by_id or "C4" not in index_by_id:
            continue
        candidate_row = rows[index_by_id[candidate_id]]
        mcnemar = candidate_row.get("mcnemar_vs_C4", {})
        p_value = float(mcnemar.get("p_value", 1.0))
        star_text = _stars(p_value)
        x1 = index_by_id["C4"]
        x2 = index_by_id[candidate_id]
        y = max(values[x1] + yerr_hi[x1], values[x2] + yerr_hi[x2]) + (0.05 if candidate_id == "C6" else 0.09)
        ax.plot([x1, x1, x2, x2], [y - 0.01, y, y, y - 0.01], color="black", linewidth=0.8)
        ax.text((x1 + x2) / 2.0, y + 0.004, star_text, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out_path = figures_dir / "killtest_with_iti.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path.resolve())


def plot_sweep(aggregate: dict[str, Any], figures_dir: Path, title_prefix: str) -> str:
    option_a = list(aggregate.get("medqa_option_a_cells", []))
    option_b = list(aggregate.get("truthfulqa_option_b_cells", []))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if option_a:
        markers = {"every_position": "o", "last_position_only": "s"}
        for position_mode in sorted({str(row.get("position_mode")) for row in option_a}):
            mode_rows = [row for row in option_a if str(row.get("position_mode")) == position_mode]
            mode_rows = sorted(mode_rows, key=lambda row: (int(row.get("selected_k", 0)), float(row.get("alpha", 0.0))))
            ax.scatter(
                [float(row.get("alpha", 0.0)) for row in mode_rows],
                [float(row.get("restoration_rate_clean_subset", 0.0)) for row in mode_rows],
                label=position_mode,
                marker=markers.get(position_mode, "o"),
                s=55,
                alpha=0.85,
            )
        ax.set_xlabel("alpha")
        ax.set_ylabel("Clean restoration rate")
        ax.set_title("Domain-matched ITI sweep")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Option A not run", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    if option_b:
        ordered = sorted(option_b, key=lambda row: str(row.get("position_mode", "")))
        x = list(range(len(ordered)))
        vals = [float(row.get("restoration_rate_clean_subset", 0.0)) for row in ordered]
        lo = [max(0.0, val - float(row.get("restoration_rate_clean_subset_ci95_wilson", [0.0, 0.0])[0])) for val, row in zip(vals, ordered)]
        hi = [max(0.0, float(row.get("restoration_rate_clean_subset_ci95_wilson", [0.0, 0.0])[1]) - val) for val, row in zip(vals, ordered)]
        ax.bar(x, vals, yerr=[lo, hi], capsize=4, color=["#4c78a8", "#f58518"][: len(ordered)], edgecolor="black")
        ax.set_xticks(x, [str(row.get("position_mode", "")) for row in ordered], rotation=15, ha="right")
        ax.set_ylabel("Clean restoration rate")
        ax.set_title("TruthfulQA-transfer ITI")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Option B not run", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle(f"{title_prefix}: ITI sweep summary")
    fig.tight_layout()
    out_path = figures_dir / "iti_sweep.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path.resolve())


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    aggregate = read_json(results_root / "aggregate.json")
    comparison = read_json(results_root / "comparison.json")
    killtest_path = plot_killtest(comparison, figures_dir, str(args.title_prefix))
    sweep_path = plot_sweep(aggregate, figures_dir, str(args.title_prefix))

    summary = {
        "killtest_with_iti": killtest_path,
        "iti_sweep": sweep_path,
    }
    write_json(figures_dir / "plot_summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
