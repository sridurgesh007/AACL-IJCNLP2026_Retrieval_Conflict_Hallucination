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
    parser = argparse.ArgumentParser(description="Plot MedQA Path A ablation (p-set/bootstrap modes).")
    parser.add_argument("--ablation_summary", default="results/medqa_eval/medqa_eval_ablation_summary.json")
    parser.add_argument("--out_dir", default="results/medqa_eval/plots")
    parser.add_argument("--title_prefix", default="MedQA Path A")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object json at {path}")
    return payload


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        e_mode = str(row.get("evidence_aligned_mode", "")).strip()
        p_mode = str(row.get("p_set_mode", "unknown"))
        b_mode = str(row.get("bootstrap_mode", "")).strip()
        parts: list[str] = []
        if e_mode:
            parts.append(f"ev={e_mode}")
        parts.append(f"p={p_mode}")
        if b_mode:
            parts.append(f"boot={b_mode}")
        mode = " | ".join(parts)
        grouped.setdefault(mode, []).append(row)
    for mode in grouped:
        grouped[mode] = sorted(grouped[mode], key=lambda r: float(r.get("lambda_max", 0.0)))
    return grouped


def _load_metrics_for_row(row: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out_dir = str(row.get("out_dir", "")).strip()
    if not out_dir:
        return {}
    if out_dir in cache:
        return cache[out_dir]
    metrics_path = Path(out_dir) / "medqa_eval_metrics.json"
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            cache[out_dir] = payload
            return payload
    except Exception:
        pass
    cache[out_dir] = {}
    return {}


def _resolve_field(row: dict[str, Any], field: str, cache: dict[str, dict[str, Any]]) -> float:
    if field in row and row[field] is not None:
        return float(row[field])
    metrics = _load_metrics_for_row(row, cache)
    if field in metrics and metrics[field] is not None:
        return float(metrics[field])
    return float("nan")


def _plot_restoration_improvement(
    grouped: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    title_prefix: str,
) -> None:
    plt.figure(figsize=(9, 5))
    for mode, rows in grouped.items():
        x = [float(r["lambda_max"]) for r in rows]
        y = [float(r["ft_restoration_improvement"]) for r in rows]
        plt.plot(x, y, marker="o", linewidth=2, label=mode)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("lambda_max")
    plt.ylabel("Restoration Improvement")
    plt.title(f"{title_prefix}: Restoration Improvement vs lambda")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "restoration_improvement_vs_lambda.png", dpi=180)
    plt.close()


def _plot_restoration_harm(
    grouped: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mode, rows in grouped.items():
        x = [float(r["lambda_max"]) for r in rows]
        restoration = [float(r["ft_restoration_rate"]) for r in rows]
        harm = [float(r["ft_harm_rate"]) for r in rows]
        axes[0].plot(x, restoration, marker="o", linewidth=2, label=mode)
        axes[1].plot(x, harm, marker="o", linewidth=2, label=mode)

    axes[0].set_title("Restoration Rate")
    axes[0].set_xlabel("lambda_max")
    axes[0].set_ylabel("Rate")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Harm Rate")
    axes[1].set_xlabel("lambda_max")
    axes[1].set_ylabel("Rate")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{title_prefix}: Restoration/Harm vs lambda", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "restoration_and_harm_vs_lambda.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_best_mode(
    grouped: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    title_prefix: str,
) -> dict[str, Any]:
    best_by_mode: dict[str, dict[str, Any]] = {}
    for mode, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda r: (float(r["ft_restoration_improvement"]), -float(r["ft_harm_rate"])),
            reverse=True,
        )
        best_by_mode[mode] = ranked[0]

    modes = sorted(best_by_mode.keys())
    baseline_acc = [float(best_by_mode[m]["baseline_accuracy"]) for m in modes]
    ft_acc = [float(best_by_mode[m]["ft_accuracy"]) for m in modes]

    x = np.arange(len(modes))
    w = 0.35
    plt.figure(figsize=(9, 5))
    b1 = plt.bar(x - w / 2, baseline_acc, width=w, label="baseline_accuracy", color="#7a7a7a")
    b2 = plt.bar(x + w / 2, ft_acc, width=w, label="ft_accuracy", color="#49a35c")
    plt.xticks(x, modes)
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}: Best-Lambda Accuracy by P-set Mode")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    for bars in (b1, b2):
        for bar in bars:
            h = float(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "best_lambda_accuracy_by_mode.png", dpi=180)
    plt.close()
    return best_by_mode


def _plot_wrong_subset(
    best_by_mode: dict[str, dict[str, Any]],
    out_dir: Path,
    title_prefix: str,
    metrics_cache: dict[str, dict[str, Any]],
) -> None:
    modes = sorted(best_by_mode.keys())
    match = [
        _resolve_field(best_by_mode[m], "restoration_rate_base_wrong_matches_parametric", metrics_cache)
        for m in modes
    ]
    not_match = [
        _resolve_field(best_by_mode[m], "restoration_rate_base_wrong_not_parametric", metrics_cache)
        for m in modes
    ]

    x = np.arange(len(modes))
    w = 0.35
    plt.figure(figsize=(10, 5))
    b1 = plt.bar(x - w / 2, match, width=w, label="wrong cases where base==parametric", color="#2b6cb0")
    b2 = plt.bar(x + w / 2, not_match, width=w, label="wrong cases where base!=parametric", color="#d67c2f")
    plt.xticks(x, modes)
    plt.ylabel("Restoration Rate")
    plt.title(f"{title_prefix}: Wrong-Subset Restoration by Mode (Best lambda)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = float(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "wrong_subset_restoration_by_mode.png", dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    ablation_summary = (PROJECT_ROOT / args.ablation_summary).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(ablation_summary)
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No rows found in {ablation_summary}")
    grouped = _group_rows([r for r in rows if isinstance(r, dict)])
    if not grouped:
        raise RuntimeError(f"Could not group rows by p_set_mode in {ablation_summary}")

    metrics_cache: dict[str, dict[str, Any]] = {}
    _plot_restoration_improvement(grouped=grouped, out_dir=out_dir, title_prefix=str(args.title_prefix))
    _plot_restoration_harm(grouped=grouped, out_dir=out_dir, title_prefix=str(args.title_prefix))
    best_by_mode = _plot_accuracy_best_mode(grouped=grouped, out_dir=out_dir, title_prefix=str(args.title_prefix))
    _plot_wrong_subset(
        best_by_mode=best_by_mode,
        out_dir=out_dir,
        title_prefix=str(args.title_prefix),
        metrics_cache=metrics_cache,
    )

    best_global = payload.get("best_by_restoration_improvement")
    if not isinstance(best_global, dict):
        ranked_all = sorted(
            [r for r in rows if isinstance(r, dict)],
            key=lambda r: (float(r["ft_restoration_improvement"]), -float(r["ft_harm_rate"])),
            reverse=True,
        )
        best_global = ranked_all[0] if ranked_all else None

    report = {
        "ablation_summary": str(ablation_summary),
        "out_dir": str(out_dir),
        "plot_files": [
            "restoration_improvement_vs_lambda.png",
            "restoration_and_harm_vs_lambda.png",
            "best_lambda_accuracy_by_mode.png",
            "wrong_subset_restoration_by_mode.png",
        ],
        "best_by_mode": best_by_mode,
        "best_global": best_global,
    }
    with (out_dir / "plot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"plots_dir={out_dir}")
    print(f"plot_summary={out_dir / 'plot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
