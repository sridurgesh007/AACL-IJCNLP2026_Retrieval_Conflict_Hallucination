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
    parser = argparse.ArgumentParser(description="Plot ClashEval v3 intervention results.")
    parser.add_argument("--results_dir", default="results/clasheval_intervention_v3")
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


def plot_results(summary: dict[str, Any], sweep: dict[str, Any], out_path: Path) -> str:
    faith_values = [
        float(summary["c1_faithfulness"]),
        float(summary["c2_faithfulness"]),
        float(summary["c3_faithfulness"]),
    ]
    faith_intervals = [
        summary["c1_faithfulness_ci95_wilson"],
        summary["c2_faithfulness_ci95_wilson"],
        summary["c3_faithfulness_ci95_wilson"],
    ]
    lambda_rows = list(sweep["rows"])
    lambdas = [float(row["lambda_neg"]) for row in lambda_rows]
    restoration = [float(row["restoration_on_c2_hallucinated"]) for row in lambda_rows]
    harm = [float(row["harm_rate"]) for row in lambda_rows]
    youden = [float(row["youden_j"]) for row in lambda_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    bars = ax.bar(
        [0, 1, 2],
        faith_values,
        yerr=[
            [max(0.0, value - float(bounds[0])) for value, bounds in zip(faith_values, faith_intervals)],
            [max(0.0, float(bounds[1]) - value) for value, bounds in zip(faith_values, faith_intervals)],
        ],
        capsize=4,
        color=["#8f8f8f", "#2b8cbe", "#31a354"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks([0, 1, 2], ["C1\nFull context", "C2\nShort prompt", "C3\nSelected mechanism"])
    ax.set_ylabel("faithfulness rate")
    ax.set_ylim(0.0, min(1.0, max(faith_values) + 0.18))
    ax.set_title("Overall faithfulness")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, faith_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ax.plot(lambdas, restoration, marker="o", color="#31a354", linewidth=2.0, label="Restoration")
    ax.plot(lambdas, harm, marker="o", color="#de2d26", linewidth=2.0, label="Harm")
    ax.plot(lambdas, youden, marker="o", color="#756bb1", linewidth=2.0, label="Youden J")
    selected_lambda = float(summary["selected_lambda_neg"])
    ax.axvline(selected_lambda, color="black", linestyle="--", linewidth=1.0, label=f"Selected lambda={selected_lambda:g}")
    ax.set_xlabel("lambda_neg")
    ax.set_ylabel("rate")
    ax.set_title(f"Lambda sweep at tau={float(summary['selected_tau']):.3f}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "ClashEval v3 "
        f"(pilot_auc={float(summary['pilot_margin_auc']):.3f}, gate_fire={float(summary['gate_fire_rate']):.3f})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def plot_margin_distribution(pilot: dict[str, Any], out_path: Path) -> str:
    faithful = [float(value) for value in pilot.get("faithful_margins", [])]
    hallucinated = [float(value) for value in pilot.get("hallucinated_margins", [])]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(faithful, bins=20, alpha=0.60, label=f"Faithful (n={len(faithful)})", color="#2b8cbe")
    ax.hist(hallucinated, bins=20, alpha=0.60, label=f"Hallucinated (n={len(hallucinated)})", color="#de2d26")
    selected_tau = pilot.get("selected_tau")
    if selected_tau is not None:
        ax.axvline(float(selected_tau), color="black", linestyle="--", linewidth=1.0, label=f"Selected tau={float(selected_tau):.3f}")
    ax.set_xlabel("prefill margin")
    ax.set_ylabel("count")
    auc = pilot.get("margin_auc")
    ax.set_title(f"Pilot margin distribution (AUC={float(auc):.3f})" if auc is not None else "Pilot margin distribution")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
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

    pilot = _read_json(results_dir / "pilot_margin_auc.json")
    sweep = _read_json(results_dir / "lambda_sweep.json")
    summary = _read_json(results_dir / "summary.json")

    results_figure = plot_results(summary, sweep, figures_dir / "clasheval_v3_results.png")
    margin_figure = plot_margin_distribution(pilot, figures_dir / "clasheval_v3_margin_distribution.png")

    finding_text = (
        f"pilot_margin_auc={pilot.get('margin_auc')}\n"
        f"selected_tau={summary.get('selected_tau')}\n"
        f"selected_lambda_neg={summary.get('selected_lambda_neg')}\n"
        f"c1_faithfulness={summary.get('c1_faithfulness')}\n"
        f"c2_faithfulness={summary.get('c2_faithfulness')}\n"
        f"c3_faithfulness={summary.get('c3_faithfulness')}\n"
        f"restoration_on_c2_hallucinated={summary.get('restoration_on_c2_hallucinated')}\n"
        f"harm_rate={summary.get('harm_rate')}\n"
        f"gate_fire_rate={summary.get('gate_fire_rate')}\n"
        f"margin_auc_full={summary.get('margin_auc_full')}\n"
    )
    finding_path = figures_dir / "clasheval_v3_results.txt"
    finding_path.write_text(finding_text, encoding="utf-8")

    plot_summary = {
        "pilot_margin_auc_json": str((results_dir / "pilot_margin_auc.json").resolve()),
        "lambda_sweep_json": str((results_dir / "lambda_sweep.json").resolve()),
        "summary_json": str((results_dir / "summary.json").resolve()),
        "results_jsonl": str((results_dir / "results.jsonl").resolve()),
        "results_figure": results_figure,
        "margin_figure": margin_figure,
        "finding": str(finding_path.resolve()),
    }
    _write_json(results_dir / "plot_summary.json", plot_summary)
    print(json.dumps(plot_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
