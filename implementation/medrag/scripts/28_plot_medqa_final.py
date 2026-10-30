#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ORDER = ["exp_A", "exp_B", "exp_C"]
EXPERIMENT_LABELS = {
    "sanity": "Sanity",
    "exp_A": "Experiment A",
    "exp_B": "Experiment B",
    "exp_C": "Experiment C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final MedQA experiment outputs.")
    parser.add_argument("--results_root", default="results/medqa_final")
    parser.add_argument("--out_dir", default="results/medqa_final/plots")
    parser.add_argument("--title_prefix", default="MedQA Final")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _load_bundle(results_root: Path, name: str) -> dict[str, Any] | None:
    exp_dir = results_root / name
    if not exp_dir.exists():
        return None
    ablation = _load_json(exp_dir / "medqa_eval_ablation.json")
    best = _load_json(exp_dir / "medqa_eval_metrics_best.json")
    sanity = _load_json(exp_dir / "sanity_checks.json")
    if ablation is None and best is None and sanity is None:
        return None
    return {
        "name": name,
        "label": EXPERIMENT_LABELS.get(name, name),
        "dir": str(exp_dir),
        "ablation": ablation,
        "best": best,
        "sanity": sanity,
    }


def _plot_exp_a_metric(exp_a: dict[str, Any], field: str, out_path: Path, ylabel: str, title: str) -> None:
    ablation = exp_a.get("ablation") or {}
    rows = [row for row in ablation.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row.get("neg_bias_ratio", 0.0)), []).append(row)
    plt.figure(figsize=(9, 5))
    for neg_ratio, neg_rows in sorted(grouped.items(), key=lambda x: x[0]):
        neg_rows = sorted(neg_rows, key=lambda row: float(row.get("lambda_max", 0.0)))
        x = [float(row.get("lambda_max", 0.0)) for row in neg_rows]
        y = [float(row.get(field, 0.0)) for row in neg_rows]
        plt.plot(x, y, marker="o", linewidth=2, label=f"neg_ratio={neg_ratio:g}")
    plt.xlabel("lambda_max")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_experiment_best_comparison(bundles: list[dict[str, Any]], out_path: Path, title_prefix: str) -> None:
    bundles = [bundle for bundle in bundles if isinstance(bundle.get("best"), dict)]
    if not bundles:
        return
    labels = [bundle["label"] for bundle in bundles]
    restoration = [float(bundle["best"].get("restoration_improvement", 0.0)) for bundle in bundles]
    harm = [float(bundle["best"].get("harm_rate", 0.0)) for bundle in bundles]
    stability = [float(bundle["best"].get("stability_rate", 0.0)) for bundle in bundles]
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(10, 5))
    b1 = plt.bar(x - w, restoration, width=w, label="restoration_improvement", color="#2b6cb0")
    b2 = plt.bar(x, harm, width=w, label="harm_rate", color="#c05621")
    b3 = plt.bar(x + w, stability, width=w, label="stability_rate", color="#2f855a")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title(f"{title_prefix}: Best Configuration Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    for bars in (b1, b2, b3):
        for bar in bars:
            val = float(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_gate_diagnostics(bundles: list[dict[str, Any]], out_path: Path, title_prefix: str) -> None:
    bundles = [bundle for bundle in bundles if isinstance(bundle.get("best"), dict)]
    if not bundles:
        return
    labels = [bundle["label"] for bundle in bundles]
    wrong = [float(bundle["best"].get("gate_fire_rate_base_wrong", 0.0)) for bundle in bundles]
    correct = [float(bundle["best"].get("gate_fire_rate_base_correct", 0.0)) for bundle in bundles]
    neg = [float(bundle["best"].get("fraction_negative_margin_base_wrong", 0.0)) for bundle in bundles]
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - w, wrong, width=w, label="gate_fire_wrong", color="#2b6cb0")
    plt.bar(x, correct, width=w, label="gate_fire_correct", color="#c05621")
    plt.bar(x + w, neg, width=w, label="fraction_negative_margin_wrong", color="#2f855a")
    plt.xticks(x, labels)
    plt.ylabel("Rate")
    plt.title(f"{title_prefix}: Gate Diagnostics for Best Configs")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _build_paper_table(exp_a: dict[str, Any] | None, exp_b: dict[str, Any] | None, exp_c: dict[str, Any] | None) -> dict[str, Any]:
    primary = (exp_a or exp_b or exp_c or {}).get("best") or {}
    paper = {
        "baseline_accuracy": float(primary.get("baseline_accuracy", 0.0)),
        "exp_A_best": (exp_a or {}).get("best"),
        "exp_B_best": (exp_b or {}).get("best"),
        "exp_C_best": (exp_c or {}).get("best"),
        "exp_A_restoration": float(((exp_a or {}).get("best") or {}).get("restoration_rate", 0.0)),
        "exp_A_harm_rate": float(((exp_a or {}).get("best") or {}).get("harm_rate", 0.0)),
        "exp_A_restoration_improvement": float(((exp_a or {}).get("best") or {}).get("restoration_improvement", 0.0)),
        "exp_B_restoration": float(((exp_b or {}).get("best") or {}).get("restoration_rate", 0.0)),
        "exp_B_harm_rate": float(((exp_b or {}).get("best") or {}).get("harm_rate", 0.0)),
        "exp_B_restoration_improvement": float(((exp_b or {}).get("best") or {}).get("restoration_improvement", 0.0)),
        "exp_C_restoration": float(((exp_c or {}).get("best") or {}).get("restoration_rate", 0.0)),
        "exp_C_harm_rate": float(((exp_c or {}).get("best") or {}).get("harm_rate", 0.0)),
        "exp_C_restoration_improvement": float(((exp_c or {}).get("best") or {}).get("restoration_improvement", 0.0)),
        "gate_fire_rate_wrong": float(((exp_a or {}).get("best") or {}).get("gate_fire_rate_base_wrong", 0.0)),
        "fraction_neg_margin_wrong": float(((exp_a or {}).get("best") or {}).get("fraction_negative_margin_base_wrong", 0.0)),
        "best_lambda_max": float(((exp_a or {}).get("best") or {}).get("lambda_max", 0.0)),
        "best_neg_bias_ratio": float(((exp_a or {}).get("best") or {}).get("neg_bias_ratio", 0.0)),
    }
    return paper


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity = _load_bundle(results_root, "sanity")
    exp_a = _load_bundle(results_root, "exp_A")
    exp_b = _load_bundle(results_root, "exp_B")
    exp_c = _load_bundle(results_root, "exp_C")
    bundles = [bundle for bundle in [exp_a, exp_b, exp_c] if bundle is not None]

    if exp_a is not None:
        _plot_exp_a_metric(exp_a, "restoration_improvement", out_dir / "exp_a_restoration_improvement_vs_lambda.png", "Restoration improvement", f"{args.title_prefix}: Experiment A Restoration vs lambda")
        _plot_exp_a_metric(exp_a, "harm_rate", out_dir / "exp_a_harm_rate_vs_lambda.png", "Harm rate", f"{args.title_prefix}: Experiment A Harm vs lambda")
    _plot_experiment_best_comparison(bundles, out_dir / "experiment_best_comparison.png", str(args.title_prefix))
    _plot_gate_diagnostics(bundles, out_dir / "gate_diagnostics.png", str(args.title_prefix))

    paper_table = _build_paper_table(exp_a, exp_b, exp_c)
    with (results_root / "medqa_paper_table.json").open("w", encoding="utf-8") as f:
        json.dump(paper_table, f, indent=2)

    report = {
        "results_root": str(results_root),
        "out_dir": str(out_dir),
        "plot_files": [
            "exp_a_restoration_improvement_vs_lambda.png",
            "exp_a_harm_rate_vs_lambda.png",
            "experiment_best_comparison.png",
            "gate_diagnostics.png",
        ],
        "sanity": sanity,
        "exp_A": exp_a,
        "exp_B": exp_b,
        "exp_C": exp_c,
        "paper_table": paper_table,
    }
    with (out_dir / "plot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"plots_dir={out_dir}")
    print(f"plot_summary={out_dir / 'plot_summary.json'}")
    print(f"paper_table={results_root / 'medqa_paper_table.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
