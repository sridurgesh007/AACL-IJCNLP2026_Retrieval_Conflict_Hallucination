#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from medrag.utils.io import git_commit_hash, sha256_file

SECTION_START = "<!-- SUBTHRESHOLD_DECOMP_START -->"
SECTION_END = "<!-- SUBTHRESHOLD_DECOMP_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MedQA sub-threshold decomposition sweep and update report.")
    parser.add_argument("--results_root", default="results/medqa_final_v3")
    parser.add_argument("--subthreshold_dir", default="results/medqa_final_v3/decomposition_subthreshold")
    parser.add_argument("--base_decomposition_dir", default="results/medqa_final_v3/decomposition")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/decomposition_subthreshold/figures")
    parser.add_argument("--report_path", default="results/medqa_final_v3/REPORT.md")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_metadata(*, args: argparse.Namespace, wall_clock_seconds: float, source_files: list[Path]) -> dict[str, Any]:
    return {
        "seed": 1234,
        "git_commit_hash": git_commit_hash(PROJECT_ROOT),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path.resolve()) for path in source_files],
        "input_file_sha256": {str(path.resolve()): sha256_file(path) for path in source_files},
    }


def _err_from_ci(value: float, ci: list[float]) -> tuple[float, float]:
    return max(0.0, float(value) - float(ci[0])), max(0.0, float(ci[1]) - float(value))


def _mode_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return sorted([row for row in rows if str(row["mode"]) == mode], key=lambda x: float(x["lambda"]))


def _plot_subthreshold_marginals(aggregate: dict[str, Any], out_dir: Path) -> None:
    rows = aggregate["rows"]
    promo = _mode_rows(rows, "pure_promotion")
    supp = _mode_rows(rows, "pure_suppression")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True)
    series = [
        ("pure promotion", promo, "#1f77b4"),
        ("pure suppression", supp, "#d62728"),
    ]
    for ax, key, title in (
        (axes[0], "restoration_rate", "Matched Subset"),
        (axes[1], "restoration_rate_clean_subset", "Clean Subset"),
    ):
        for label, subset, color in series:
            x = [float(row["lambda"]) for row in subset]
            y = [float(row[key]) for row in subset]
            ci_key = f"{key}_ci95_wilson"
            yerr = np.array([_err_from_ci(float(row[key]), row[ci_key]) for row in subset]).T
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, color=color, label=label)
        ax.axvspan(2.45, 2.55, color="gray", alpha=0.15)
        ax.set_xlabel("lambda")
        ax.set_ylabel("Restoration Rate")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("MedQA Sub-threshold Decomposition Marginals")
    fig.tight_layout()
    fig.savefig(out_dir / "subthreshold_marginals.png", dpi=180)
    plt.close(fig)


def _plot_subthreshold_separation(separation: dict[str, Any], out_dir: Path) -> None:
    rows = separation["by_lambda"]
    x = [float(row["lambda"]) for row in rows]
    y = [int(row["n_symmetric_difference"]) for row in rows]
    y_clean = [int(row["n_symmetric_difference_clean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(x, y, marker="o", linewidth=2, label="matched subset")
    ax.plot(x, y_clean, marker="s", linewidth=2, linestyle="--", label="clean subset")
    for row in rows:
        ax.text(float(row["lambda"]), int(row["n_symmetric_difference"]) + 0.1, str(int(row["n_symmetric_difference"])), fontsize=8, ha="center")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Symmetric Difference Size")
    ax.set_title(f"MedQA Sub-threshold Separation (Outcome {separation['outcome']})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "subthreshold_separation.png", dpi=180)
    plt.close(fig)


def _plot_decomposition_combined(base_decomposition: dict[str, Any], aggregate: dict[str, Any], out_dir: Path) -> None:
    base_rows = base_decomposition["rows"]
    lps = sorted({float(row["lambda_pos"]) for row in base_rows})
    lns = sorted({float(row["lambda_neg"]) for row in base_rows})
    grid = np.zeros((len(lns), len(lps)), dtype=float)
    for row in base_rows:
        i = lns.index(float(row["lambda_neg"]))
        j = lps.index(float(row["lambda_pos"]))
        grid[i, j] = float(row["restoration_rate"])

    promo = _mode_rows(aggregate["rows"], "pure_promotion")
    supp = _mode_rows(aggregate["rows"], "pure_suppression")
    bal = _mode_rows(aggregate["rows"], "balanced")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    im = axes[0].imshow(grid, cmap="viridis", aspect="auto")
    axes[0].set_xticks(np.arange(len(lps)), labels=[f"{x:g}" for x in lps])
    axes[0].set_yticks(np.arange(len(lns)), labels=[f"{x:g}" for x in lns])
    axes[0].set_xlabel("lambda_pos")
    axes[0].set_ylabel("lambda_neg")
    axes[0].set_title("v3 Supra-threshold Heatmap")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for label, subset, key, color, style in (
        ("promotion", promo, "restoration_rate", "#1f77b4", "-"),
        ("suppression", supp, "restoration_rate", "#d62728", "-"),
        ("balanced", bal, "restoration_rate", "#2ca02c", ":"),
    ):
        x = [float(row["lambda"]) for row in subset]
        y = [float(row[key]) for row in subset]
        yerr = np.array([_err_from_ci(float(row[key]), row[f"{key}_ci95_wilson"]) for row in subset]).T
        axes[1].errorbar(x, y, yerr=yerr, marker="o", linewidth=2, linestyle=style, color=color, label=label)
    axes[1].axvspan(2.45, 2.55, color="gray", alpha=0.15)
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("Restoration Rate")
    axes[1].set_title("Sub-threshold Sweep")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("MedQA Decomposition: Supra-threshold and Sub-threshold Regimes")
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition_combined.png", dpi=180)
    plt.close(fig)


def _build_report_section(separation: dict[str, Any], aggregate: dict[str, Any]) -> str:
    rows = separation["by_lambda"]
    peak = max(rows, key=lambda row: int(row["n_symmetric_difference"]))
    outcome = str(separation["outcome"])
    justification = str(separation["outcome_justification"])
    peak_line = (
        f"Peak symmetric difference is {int(peak['n_symmetric_difference'])} at λ={float(peak['lambda']):g}; "
        f"clean-subset peak is {int(peak['n_symmetric_difference_clean'])}."
    )
    balanced_best = max(_mode_rows(aggregate["rows"], "balanced"), key=lambda row: float(row["restoration_rate"]))
    text = f"""{SECTION_START}
## Sub-threshold Decomposition

Outcome {outcome}. {justification} {peak_line}

![Sub-threshold Marginals](decomposition_subthreshold/figures/subthreshold_marginals.png)

![Sub-threshold Separation](decomposition_subthreshold/figures/subthreshold_separation.png)

![Combined Decomposition Figure](decomposition_subthreshold/figures/decomposition_combined.png)

Balanced mode reaches restoration {float(balanced_best['restoration_rate']):.3f} at λ={float(balanced_best['lambda']):g}, with matched-subset Wilson CI {balanced_best['restoration_rate_ci95_wilson']}.
{SECTION_END}
"""
    return text


def _update_report(report_path: Path, section: str) -> None:
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
    else:
        text = "# MedQA Final v3 Report\n"
    if SECTION_START in text and SECTION_END in text:
        start = text.index(SECTION_START)
        end = text.index(SECTION_END) + len(SECTION_END)
        updated = text[:start] + section + text[end:]
    else:
        updated = text.rstrip() + "\n\n" + section + "\n"
    report_path.write_text(updated, encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    sub_dir = (PROJECT_ROOT / args.subthreshold_dir).resolve()
    base_dir = (PROJECT_ROOT / args.base_decomposition_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    report_path = (PROJECT_ROOT / args.report_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    aggregate_path = sub_dir / "aggregate.json"
    separation_path = sub_dir / "separation.json"
    base_path = base_dir / "aggregate.json"

    aggregate = _load_json(aggregate_path)
    separation = _load_json(separation_path)
    base_decomposition = _load_json(base_path)

    _plot_subthreshold_marginals(aggregate, out_dir)
    _plot_subthreshold_separation(separation, out_dir)
    _plot_decomposition_combined(base_decomposition, aggregate, out_dir)

    section = _build_report_section(separation, aggregate)
    _update_report(report_path, section)

    metadata = _build_metadata(
        args=args,
        wall_clock_seconds=float(time.perf_counter() - started),
        source_files=[aggregate_path, separation_path, base_path],
    )
    summary = {
        "figures": [
            str(out_dir / "subthreshold_marginals.png"),
            str(out_dir / "subthreshold_separation.png"),
            str(out_dir / "decomposition_combined.png"),
        ],
        "outcome": str(separation["outcome"]),
        "outcome_justification": str(separation["outcome_justification"]),
        "report_path": str(report_path),
        "metadata": metadata,
    }
    _write_json(out_dir / "plot_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"plot_summary={out_dir / 'plot_summary.json'}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
