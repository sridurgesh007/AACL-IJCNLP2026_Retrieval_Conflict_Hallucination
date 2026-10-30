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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and report MedQA final v3 experiments.")
    parser.add_argument("--results_root", default="results/medqa_final_v3")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/figures")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return payload


def _build_metadata(*, args: argparse.Namespace, wall_clock_seconds: float, source_files: list[Path]) -> dict[str, Any]:
    return {
        "seed": 0,
        "git_commit_hash": git_commit_hash(PROJECT_ROOT),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path.resolve()) for path in source_files],
        "input_file_sha256": {str(path.resolve()): sha256_file(path) for path in source_files},
    }


def _err_from_ci(value: float, ci: list[float]) -> tuple[float, float]:
    return max(0.0, float(value) - float(ci[0])), max(0.0, float(ci[1]) - float(value))


def _plot_decomposition_heatmap(aggregate: dict[str, Any], out_dir: Path) -> None:
    rows = aggregate["rows"]
    lps = sorted({float(r["lambda_pos"]) for r in rows})
    lns = sorted({float(r["lambda_neg"]) for r in rows})
    grid = np.zeros((len(lns), len(lps)), dtype=float)
    annotations: dict[tuple[int, int], str] = {}
    for row in rows:
        i = lns.index(float(row["lambda_neg"]))
        j = lps.index(float(row["lambda_pos"]))
        value = float(row["restoration_rate"])
        ci = row["restoration_rate_ci95_wilson"]
        grid[i, j] = value
        annotations[(i, j)] = f"{value:.2f}\n[{float(ci[0]):.2f},{float(ci[1]):.2f}]"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(lps)), labels=[f"{x:g}" for x in lps])
    ax.set_yticks(np.arange(len(lns)), labels=[f"{x:g}" for x in lns])
    ax.set_xlabel("lambda_pos")
    ax.set_ylabel("lambda_neg")
    ax.set_title("MedQA v3: Decomposition Heatmap")
    for (i, j), text in annotations.items():
        ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Restoration Rate")
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_decomposition_marginals(aggregate: dict[str, Any], out_dir: Path) -> None:
    rows = aggregate["rows"]
    pure_promo = sorted([r for r in rows if abs(float(r["lambda_neg"])) < 1e-9], key=lambda r: float(r["lambda_pos"]))
    pure_suppress = sorted([r for r in rows if abs(float(r["lambda_pos"])) < 1e-9], key=lambda r: float(r["lambda_neg"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, subset, xlabel, key in (
        (axes[0], pure_promo, "lambda_pos", "lambda_pos"),
        (axes[1], pure_suppress, "lambda_neg", "lambda_neg"),
    ):
        x = [float(r[key]) for r in subset]
        y = [float(r["restoration_rate"]) for r in subset]
        yerr = np.array([_err_from_ci(float(r["restoration_rate"]), r["restoration_rate_ci95_wilson"]) for r in subset]).T
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Restoration Rate")
        ax.grid(alpha=0.3)
    axes[0].set_title("Pure Promotion")
    axes[1].set_title("Pure Suppression")
    fig.suptitle("MedQA v3: Decomposition Marginals")
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition_marginals.png", dpi=180)
    plt.close(fig)


def _plot_decomposition_diagonal(aggregate: dict[str, Any], out_dir: Path) -> None:
    rows = aggregate["rows"]
    diagonal = sorted([r for r in rows if abs(float(r["lambda_pos"]) - float(r["lambda_neg"])) < 1e-9], key=lambda r: float(r["lambda_pos"]))
    pure_promo = sorted([r for r in rows if abs(float(r["lambda_neg"])) < 1e-9], key=lambda r: float(r["lambda_pos"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    x1 = [float(r["lambda_pos"]) for r in diagonal]
    y1 = [float(r["restoration_rate"]) for r in diagonal]
    x2 = [float(r["lambda_pos"]) for r in pure_promo]
    y2 = [float(r["restoration_rate"]) for r in pure_promo]
    ax.plot(x1, y1, marker="o", linewidth=2, label="lambda_pos == lambda_neg")
    ax.plot(x2, y2, marker="s", linewidth=2, label="pure promotion")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Restoration Rate")
    ax.set_title("MedQA v3: Diagonal vs Pure Promotion")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition_diagonal.png", dpi=180)
    plt.close(fig)


def _plot_pr_curve(aggregate: dict[str, Any], out_dir: Path) -> None:
    rows = aggregate["pr_curve"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for bootstrap_space in sorted({str(r["bootstrap_space"]) for r in rows}):
        subset = sorted([r for r in rows if str(r["bootstrap_space"]) == bootstrap_space], key=lambda r: float(r["tau"]))
        xs = [float(r["gate_fire_correct"]) for r in subset]
        ys = [float(r["restoration_rate"]) for r in subset]
        xerr = np.array([_err_from_ci(float(r["gate_fire_correct"]), r["gate_fire_correct_ci"]) for r in subset]).T
        yerr = np.array([_err_from_ci(float(r["restoration_rate"]), r["restoration_rate_ci"]) for r in subset]).T
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker="o", linewidth=2, label=bootstrap_space)
        for r, x, y in zip(subset, xs, ys):
            ax.text(x, y, f"tau={float(r['tau']):g}", fontsize=8)
    ax.set_xlabel("gate_fire_rate_base_correct")
    ax.set_ylabel("restoration_rate")
    ax.set_title("MedQA v3: PR Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=180)
    plt.close(fig)


def _sig_star(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _plot_killtest(comparison: dict[str, Any], out_dir: Path) -> None:
    labels = ["C1", "C2", "C3", "C4", "C5"]
    values = [float(comparison[f"accuracy_{label}"]) for label in labels]
    cis = [comparison[f"accuracy_{label}_ci95_wilson"] for label in labels]
    yerr = np.array([_err_from_ci(v, ci) for v, ci in zip(values, cis)]).T
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(np.arange(len(labels)), values, yerr=yerr, capsize=5, color=["#7a7a7a", "#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("MedQA v3: Prompting Kill-Test")
    ax.grid(axis="y", alpha=0.3)
    comparisons = [
        (3, "mcnemar_C4_vs_C2"),
        (3, "mcnemar_C4_vs_C3"),
        (4, "mcnemar_C5_vs_C4"),
        (4, "mcnemar_C5_vs_C2"),
    ]
    for idx, key in comparisons:
        p = float(comparison[key]["p_value"])
        ax.text(idx, values[idx] + 0.05, _sig_star(p), ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "killtest.png", dpi=180)
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_tables(
    results_root: Path,
    decomposition: dict[str, Any],
    pr_sweep: dict[str, Any],
    killtest: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tables = {
        "decomposition_winners": {
            "table_name": "decomposition_winners",
            "row": decomposition["best_decomposition_cell"],
            "metadata": metadata,
        },
        "pr_operating_points": {
            "table_name": "pr_operating_points",
            "row": pr_sweep["headline_operating_point"],
            "metadata": metadata,
        },
        "killtest_conditions": {
            "table_name": "killtest_conditions",
            "row": killtest,
            "metadata": metadata,
        },
    }
    _write_json(results_root / "decomposition_table.json", tables["decomposition_winners"])
    _write_json(results_root / "pr_table.json", tables["pr_operating_points"])
    _write_json(results_root / "killtest_table.json", tables["killtest_conditions"])
    paper = {
        "table_name": "medqa_paper_table",
        "decomposition_best": decomposition["best_decomposition_cell"],
        "headline_operating_point": pr_sweep["headline_operating_point"],
        "killtest": killtest,
        "metadata": metadata,
    }
    _write_json(results_root / "medqa_paper_table.json", paper)
    return {
        "decomposition_winners": decomposition["best_decomposition_cell"],
        "pr_operating_points": pr_sweep["headline_operating_point"],
        "killtest_conditions": killtest,
    }


def _write_report(results_root: Path, tables: dict[str, Any]) -> None:
    report = f"""# MedQA Final v3 Report

## Claim
The single-token counterfactual mechanism at L27/H20 transfers to multi-option categorical decisions under retrieval conflict.

## Figures
- `figures/decomposition_heatmap.png`
- `figures/pr_curve.png`
- `figures/killtest.png`

## Decomposition Winner
```json
{json.dumps(tables['decomposition_winners'], indent=2)}
```

## Headline Operating Point
```json
{json.dumps(tables['pr_operating_points'], indent=2)}
```

## Kill-Test Comparison
```json
{json.dumps(tables['killtest_conditions'], indent=2)}
```

## Label Audit
Results are reported on the matched subset and on the clean audited subset where available.

## Limitations
- Single dataset
- Single model
- Fixed target layer/head
- Conflict construction is MedQA-specific

## Next Round
Highest leverage next step: cross-model validation or RAGTruth transfer after verifying this MedQA decomposition and kill-test.
"""
    (results_root / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    decomposition_path = results_root / "decomposition" / "aggregate.json"
    pr_sweep_path = results_root / "pr_sweep" / "aggregate.json"
    killtest_path = results_root / "killtest" / "comparison.json"

    decomposition = _load_json(decomposition_path)
    pr_sweep = _load_json(pr_sweep_path)
    killtest = _load_json(killtest_path)

    _plot_decomposition_heatmap(decomposition, out_dir)
    _plot_decomposition_marginals(decomposition, out_dir)
    _plot_decomposition_diagonal(decomposition, out_dir)
    _plot_pr_curve(pr_sweep, out_dir)
    _plot_killtest(killtest, out_dir)
    metadata = _build_metadata(
        args=args,
        wall_clock_seconds=(time.perf_counter() - started),
        source_files=[decomposition_path, pr_sweep_path, killtest_path],
    )
    tables = _build_tables(results_root, decomposition, pr_sweep, killtest, metadata=metadata)
    _write_report(results_root, tables)

    summary = {
        "figures": [
            str(out_dir / "decomposition_heatmap.png"),
            str(out_dir / "decomposition_marginals.png"),
            str(out_dir / "decomposition_diagonal.png"),
            str(out_dir / "pr_curve.png"),
            str(out_dir / "killtest.png"),
        ],
        "paper_table": str(results_root / "medqa_paper_table.json"),
        "report": str(results_root / "REPORT.md"),
        "metadata": metadata,
    }
    _write_json(out_dir / "plot_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"plot_summary={out_dir / 'plot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
