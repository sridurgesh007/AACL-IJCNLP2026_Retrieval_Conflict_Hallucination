#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robust MedQA ablations.")
    parser.add_argument("--ablation_summary", default="results/medqa_eval_robust/medqa_eval_robust_ablation_summary.json")
    parser.add_argument("--out_dir", default="results/medqa_eval_robust/plots")
    parser.add_argument("--title_prefix", default="MedQA Robust")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected dict json at {path}")
    return payload


def _config_label(row: dict[str, Any]) -> str:
    return (
        f"ep={row.get('medqa_ep_mode','?')} | p={row.get('p_set_mode','?')} | "
        f"boot={row.get('bootstrap_mode','?')} | ov={int(bool(row.get('effective_use_ov_steering', False)))} | "
        f"a={float(row.get('alpha', 0.0)):g}"
    )


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_config_label(row), []).append(row)
    for label in grouped:
        grouped[label] = sorted(grouped[label], key=lambda r: float(r.get("lambda_max", 0.0)))
    return grouped


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda r: (
            float(r.get("ft_restoration_improvement", 0.0)),
            -float(r.get("ft_harm_rate", 0.0)),
            float(r.get("ft_accuracy", 0.0)),
        ),
        reverse=True,
    )
    return ranked[0]


def _plot_restoration_vs_lambda(grouped: dict[str, list[dict[str, Any]]], out_dir: Path, title_prefix: str) -> None:
    plt.figure(figsize=(11, 6))
    for label, rows in grouped.items():
        x = [float(r["lambda_max"]) for r in rows]
        y = [float(r["ft_restoration_improvement"]) for r in rows]
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("lambda_max")
    plt.ylabel("Restoration improvement")
    plt.title(f"{title_prefix}: Restoration Improvement vs lambda")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "restoration_improvement_vs_lambda.png", dpi=180)
    plt.close()


def _plot_restoration_harm(grouped: dict[str, list[dict[str, Any]]], out_dir: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, rows in grouped.items():
        x = [float(r["lambda_max"]) for r in rows]
        restoration = [float(r["ft_restoration_rate"]) for r in rows]
        harm = [float(r["ft_harm_rate"]) for r in rows]
        axes[0].plot(x, restoration, marker="o", linewidth=2, label=label)
        axes[1].plot(x, harm, marker="o", linewidth=2, label=label)
    axes[0].set_title("Restoration Rate")
    axes[1].set_title("Harm Rate")
    for ax in axes:
        ax.set_xlabel("lambda_max")
        ax.set_ylabel("Rate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title_prefix}: Restoration and Harm vs lambda", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "restoration_and_harm_vs_lambda.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_accuracy_by_ep_mode(rows: list[dict[str, Any]], out_dir: Path, title_prefix: str) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("medqa_ep_mode", "unknown")), []).append(row)
    best_by_mode = {mode: _best_row(mode_rows) for mode, mode_rows in by_mode.items()}
    modes = sorted(best_by_mode.keys())
    baseline = [float(best_by_mode[m]["baseline_accuracy"]) for m in modes]
    ft = [float(best_by_mode[m]["ft_accuracy"]) for m in modes]
    x = np.arange(len(modes))
    w = 0.35
    plt.figure(figsize=(9, 5))
    b1 = plt.bar(x - w / 2, baseline, width=w, color="#7a7a7a", label="baseline_accuracy")
    b2 = plt.bar(x + w / 2, ft, width=w, color="#2f855a", label="ft_accuracy")
    plt.xticks(x, modes)
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}: Best-Lambda Accuracy by EP Mode")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    for bars in (b1, b2):
        for bar in bars:
            val = float(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "best_accuracy_by_ep_mode.png", dpi=180)
    plt.close()
    return best_by_mode


def _plot_best_restoration(best_by_mode: dict[str, dict[str, Any]], out_dir: Path, title_prefix: str) -> None:
    modes = sorted(best_by_mode.keys())
    improvements = [float(best_by_mode[m]["ft_restoration_improvement"]) for m in modes]
    gate_fire = [float(best_by_mode[m]["gate_fire_rate_base_wrong"]) for m in modes]
    x = np.arange(len(modes))
    w = 0.35
    plt.figure(figsize=(10, 5))
    b1 = plt.bar(x - w / 2, improvements, width=w, color="#2b6cb0", label="restoration_improvement")
    b2 = plt.bar(x + w / 2, gate_fire, width=w, color="#d67c2f", label="gate_fire_rate_base_wrong")
    plt.xticks(x, modes)
    plt.ylabel("Value")
    plt.title(f"{title_prefix}: Best Configuration Summary by EP Mode")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    for bars in (b1, b2):
        for bar in bars:
            val = float(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "best_mode_summary.png", dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    ablation_summary = (PROJECT_ROOT / args.ablation_summary).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(ablation_summary)
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    if not rows:
        raise RuntimeError(f"No rows found in {ablation_summary}")

    grouped = _group_rows(rows)
    _plot_restoration_vs_lambda(grouped, out_dir, str(args.title_prefix))
    _plot_restoration_harm(grouped, out_dir, str(args.title_prefix))
    best_by_mode = _plot_best_accuracy_by_ep_mode(rows, out_dir, str(args.title_prefix))
    _plot_best_restoration(best_by_mode, out_dir, str(args.title_prefix))

    best_global = _best_row(rows)
    report = {
        "ablation_summary": str(ablation_summary),
        "out_dir": str(out_dir),
        "plot_files": [
            "restoration_improvement_vs_lambda.png",
            "restoration_and_harm_vs_lambda.png",
            "best_accuracy_by_ep_mode.png",
            "best_mode_summary.png",
        ],
        "best_global": best_global,
        "best_by_ep_mode": best_by_mode,
        "config_labels": sorted(grouped.keys()),
    }
    with (out_dir / "plot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"plots_dir={out_dir}")
    print(f"plot_summary={out_dir / 'plot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
