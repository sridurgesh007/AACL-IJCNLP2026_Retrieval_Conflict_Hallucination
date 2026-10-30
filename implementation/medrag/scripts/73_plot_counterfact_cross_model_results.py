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
    parser = argparse.ArgumentParser(description="Plot CounterFact cross-model intervention results.")
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


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    figures_dir = (PROJECT_ROOT / (args.figures_dir or str(Path(args.results_dir) / "figures"))).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_json(results_dir / "summary.json")
    c2 = summary["conditions"]["C2"]
    c3 = summary["conditions"]["C3"]

    labels = ["C2 faithful", "C2 hallucinated", "C3 restored", "C3 harmed"]
    values = [
        float(c2["faithful_rate"]),
        float(c2["hallucinated_rate"]),
        float(c3["restoration_rate"]),
        float(c3["harm_rate"]),
    ]
    colors = ["#2b8cbe", "#de2d26", "#31a354", "#756bb1"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(values)), labels, rotation=20, ha="right")
    ax.set_ylim(0.0, min(1.0, max(values) + 0.2))
    ax.set_ylabel("rate")
    ax.set_title(
        f"CounterFact {summary.get('model_name', summary.get('model_id'))}: "
        f"lambda_pos={float(summary['selected_lambda_pos']):.1f}, lambda_neg={float(summary['selected_lambda_neg']):.1f}"
    )
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    figure_path = figures_dir / "counterfact_cross_model_results.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    plot_summary = {
        "summary_json": str((results_dir / "summary.json").resolve()),
        "results_jsonl": str((results_dir / "results.jsonl").resolve()),
        "figure": str(figure_path.resolve()),
    }
    _write_json(results_dir / "plot_summary.json", plot_summary)
    _write_json(figures_dir / "counterfact_plot_summary.json", plot_summary)
    print(json.dumps(plot_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
