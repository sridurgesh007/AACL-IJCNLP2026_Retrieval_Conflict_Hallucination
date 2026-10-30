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
    parser = argparse.ArgumentParser(description="Plot fixed ClashEval intervention results.")
    parser.add_argument("--results_dir", default="results/clasheval_intervention_fixed")
    parser.add_argument("--figures_dir", default="", help="Defaults to <results_dir>/figures.")
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


def plot_fixed_results(summary: dict[str, Any], out_path: Path) -> str:
    c_values = [
        float(summary["c1_faithfulness"]),
        float(summary["c2_faithfulness"]),
        float(summary["c3_faithfulness"]),
    ]
    c_intervals = [
        summary["c1_faithfulness_ci95_wilson"],
        summary["c2_faithfulness_ci95_wilson"],
        summary["c3_faithfulness_ci95_wilson"],
    ]
    effect_labels = [
        f"Restoration on C2 hallucinated\nn={int(summary['n_c2_hallucinated'])}",
        f"Harm on C2 faithful\nn={int(summary['n_c2_faithful'])}",
    ]
    effect_values = [
        float(summary["restoration_on_c2_hallucinated"]),
        float(summary["harm_rate"]),
    ]
    effect_intervals = [
        summary["restoration_on_c2_hallucinated_ci95_wilson"],
        summary["harm_rate_ci95_wilson"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    ax = axes[0]
    x = [0, 1, 2]
    bars = ax.bar(
        x,
        c_values,
        yerr=[
            [max(0.0, val - float(ci[0])) for val, ci in zip(c_values, c_intervals)],
            [max(0.0, float(ci[1]) - val) for val, ci in zip(c_values, c_intervals)],
        ],
        capsize=4,
        color=["#8f8f8f", "#2b8cbe", "#31a354"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x, ["C1\nFull context", "C2\nShort prompt", "C3\nShort + mechanism"])
    ax.set_ylabel("faithfulness rate")
    ax.set_ylim(0.0, min(1.0, max(c_values) + 0.18))
    ax.set_title("Overall faithfulness")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, c_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    x = [0, 1]
    bars = ax.bar(
        x,
        effect_values,
        yerr=[
            [max(0.0, val - float(ci[0])) for val, ci in zip(effect_values, effect_intervals)],
            [max(0.0, float(ci[1]) - val) for val, ci in zip(effect_values, effect_intervals)],
        ],
        capsize=4,
        color=["#31a354", "#de2d26"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x, effect_labels)
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, min(1.0, max(effect_values) + 0.18))
    ax.set_title("Residual-set effect")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, effect_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "ClashEval fixed intervention "
        f"(tau={float(summary['tau']):.1f}, lambda_neg={float(summary['lambda_neg']):.1f}, "
        f"gate_fire_rate={float(summary['gate_fire_rate']):.3f})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve() if str(args.figures_dir).strip() else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(results_dir / "summary.json")
    figure_path = plot_fixed_results(summary, figures_dir / "clasheval_fixed_results.png")

    finding_lines = [
        f"n_examples={int(summary['n_examples'])}",
        f"c1_faithfulness={float(summary['c1_faithfulness']):.6f}",
        f"c2_faithfulness={float(summary['c2_faithfulness']):.6f}",
        f"c3_faithfulness={float(summary['c3_faithfulness']):.6f}",
        f"restoration_on_c2_hallucinated={float(summary['restoration_on_c2_hallucinated']):.6f}",
        f"harm_rate={float(summary['harm_rate']):.6f}",
        f"gate_fire_rate={float(summary['gate_fire_rate']):.6f}",
        f"mcnemar_c2_vs_c3={summary['mcnemar_c2_vs_c3']}",
        f"mechanism_mode={summary['mechanism_mode']}",
    ]
    finding_path = figures_dir / "clasheval_fixed_results.txt"
    finding_path.write_text("\n".join(finding_lines) + "\n", encoding="utf-8")

    plot_summary = {
        "summary_json": str((results_dir / "summary.json").resolve()),
        "results_jsonl": str((results_dir / "results.jsonl").resolve()),
        "figure": figure_path,
        "finding": str(finding_path.resolve()),
    }
    _write_json(results_dir / "plot_summary.json", plot_summary)
    print(json.dumps(plot_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
