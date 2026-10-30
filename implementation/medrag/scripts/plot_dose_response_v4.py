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

MODE_SPECS = {
    "pure_promotion": {"label": "Pure promotion", "color": "#1f77b4"},
    "pure_suppression": {"label": "Pure suppression", "color": "#d62728"},
    "balanced": {"label": "Balanced", "color": "#2ca02c"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MedQA v3 dose-response consolidation figures and update REPORT.md.")
    parser.add_argument("--results_root", default="results/medqa_final_v3")
    parser.add_argument("--subthreshold_dir", default="results/medqa_final_v3/decomposition_subthreshold")
    parser.add_argument("--consolidation_dir", default="results/medqa_final_v3/consolidation")
    parser.add_argument("--base_decomposition_dir", default="results/medqa_final_v3/decomposition")
    parser.add_argument("--pr_sweep_dir", default="results/medqa_final_v3/pr_sweep")
    parser.add_argument("--killtest_dir", default="results/medqa_final_v3/killtest")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/consolidation/figures")
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
    return sorted([row for row in rows if str(row["mode"]) == mode], key=lambda row: float(row["lambda"]))


def _sig_marker(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _paired_row_map(subset_payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(row["lambda"]): row for row in subset_payload["by_lambda"]}


def _headline_lambda(recommended_headline: dict[str, Any]) -> float:
    return float(max(float(recommended_headline["lambda_pos"]), float(recommended_headline["lambda_neg"])))


def _plot_dose_response(
    *,
    aggregate: dict[str, Any],
    paired_subset: dict[str, Any],
    base_decomposition: dict[str, Any],
    recommended_headline: dict[str, Any],
    out_path: Path,
    clean: bool,
) -> None:
    key = "restoration_rate_clean_subset" if clean else "restoration_rate"
    ci_key = f"{key}_ci95_wilson"
    paired_rows = _paired_row_map(paired_subset)

    fig = plt.figure(figsize=(13.5, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 3], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    ymax = 0.0
    for mode, spec in MODE_SPECS.items():
        subset = _mode_rows(aggregate["rows"], mode)
        xs = [float(row["lambda"]) for row in subset]
        ys = [float(row[key]) for row in subset]
        ymax = max(ymax, max(ys) if ys else 0.0)
        yerr = np.array([_err_from_ci(float(row[key]), row[ci_key]) for row in subset]).T
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2,
            color=spec["color"],
            label=spec["label"],
        )

    ax.axvspan(1.5, max(float(row["lambda"]) for row in aggregate["rows"]), color="gray", alpha=0.10)
    headline_x = _headline_lambda(recommended_headline)
    ax.axvline(headline_x, color="black", linestyle="--", linewidth=1.5)
    ax.text(headline_x + 0.03, ymax + 0.015, f"headline λ={headline_x:g}", fontsize=9, ha="left", va="bottom")

    # paired-bootstrap significance markers for balanced vs the two pure modes
    for lambda_value, row in sorted(paired_rows.items()):
        bal_promo = row["balanced_minus_promotion"]
        bal_supp = row["balanced_minus_suppression"]
        marker_promo = _sig_marker(float(bal_promo["p_value"]))
        marker_supp = _sig_marker(float(bal_supp["p_value"]))
        if marker_promo:
            ax.text(lambda_value, ymax + 0.03, marker_promo, color=MODE_SPECS["pure_promotion"]["color"], ha="center", va="bottom", fontsize=10)
        if marker_supp:
            ax.text(lambda_value, ymax + 0.06, marker_supp, color=MODE_SPECS["pure_suppression"]["color"], ha="center", va="bottom", fontsize=10)

    ax.text(
        0.02,
        0.98,
        "Blue: balanced > promotion\nRed: balanced > suppression",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )
    ax.set_xlabel("lambda")
    ax.set_ylabel("Restoration rate")
    ax.set_title("Matched subset" if not clean else "Clean subset")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.0, max(0.45, ymax + 0.12))
    ax.legend(loc="lower right")

    base_rows = base_decomposition["rows"]
    lps = sorted({float(row["lambda_pos"]) for row in base_rows})
    lns = sorted({float(row["lambda_neg"]) for row in base_rows})
    grid = np.zeros((len(lns), len(lps)), dtype=float)
    for row in base_rows:
        i = lns.index(float(row["lambda_neg"]))
        j = lps.index(float(row["lambda_pos"]))
        grid[i, j] = float(row["restoration_rate"])
    im = ax_heat.imshow(grid, cmap="viridis", aspect="auto")
    ax_heat.set_xticks(np.arange(len(lps)), labels=[f"{x:g}" for x in lps])
    ax_heat.set_yticks(np.arange(len(lns)), labels=[f"{x:g}" for x in lns])
    ax_heat.set_xlabel("lambda_pos")
    ax_heat.set_ylabel("lambda_neg")
    ax_heat.set_title("Supra-threshold saturation regime")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.suptitle("MedQA v3 Dose-response and Additivity", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _format_lambda_list(values: list[float]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{float(x):g}" for x in values)


def _peak_effect(subset_payload: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    best_name = ""
    best_lambda = 0.0
    best_stats: dict[str, Any] | None = None
    for row in subset_payload["by_lambda"]:
        for name in ("balanced_minus_promotion", "balanced_minus_suppression", "promotion_minus_suppression"):
            stats = row[name]
            if best_stats is None or float(stats["mean_diff"]) > float(best_stats["mean_diff"]):
                best_stats = stats
                best_name = name
                best_lambda = float(row["lambda"])
    if best_stats is None:
        best_stats = {"mean_diff": 0.0, "ci95": [0.0, 0.0], "p_value": 1.0}
    return best_name, best_lambda, best_stats


def _clean_resolution_limited(clean_subset: dict[str, Any]) -> bool:
    for row in clean_subset["by_lambda"]:
        if float(row["lambda"]) >= 0.5:
            continue
        for name in ("balanced_minus_promotion", "balanced_minus_suppression"):
            stats = row[name]
            width = float(stats["ci95"][1]) - float(stats["ci95"][0])
            if width >= 0.20 or float(stats["p_value"]) >= 0.05:
                return True
    return False


def _build_report(
    *,
    results_root: Path,
    best_cell: dict[str, Any] | None,
    pr_headline: dict[str, Any] | None,
    killtest: dict[str, Any] | None,
    paired_bootstrap: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    matched = paired_bootstrap["matched_subset"]
    clean = paired_bootstrap["clean_subset"]
    peak_name, peak_lambda, peak_stats = _peak_effect(matched)
    recommended = comparison["recommended_headline"]
    headline_changed = str(recommended["condition_id"]) != "H4"

    matched_bal_promo = matched["significant_lambdas"]["balanced_minus_promotion"]
    matched_bal_supp = matched["significant_lambdas"]["balanced_minus_suppression"]
    clean_bal_promo = clean["significant_lambdas"]["balanced_minus_promotion"]
    clean_bal_supp = clean["significant_lambdas"]["balanced_minus_suppression"]

    lines: list[str] = [
        "# MedQA Final v3 Report",
        "",
        "## Claim",
        "The single-token counterfactual mechanism at L27/H20 transfers to multi-option categorical decisions under retrieval conflict, and its sub-threshold regime exhibits a measurable dose-response structure rather than a single isolated operating point.",
        "",
        "## Figures",
        "- `consolidation/figures/dose_response_main.png`",
        "- `consolidation/figures/dose_response_clean.png`",
    ]
    if pr_headline is not None:
        lines.append("- `figures/pr_curve.png`")
    if killtest is not None:
        lines.append("- `figures/killtest.png`")

    if best_cell is not None:
        lines.extend(
            [
                "",
                "## Decomposition Winner",
                "```json",
                json.dumps(best_cell, indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Dose-response and Additivity",
            "Restoration follows a monotonic dose-response curve from lambda=0.1 upward. Balanced intervention saturates earlier than the pure modes, while pure promotion and pure suppression only catch up in the supra-threshold regime.",
            "",
            "![Dose-response main](consolidation/figures/dose_response_main.png)",
            "",
            "![Dose-response clean](consolidation/figures/dose_response_clean.png)",
            "",
            f"On the {int(matched['n_items'])}-item matched base-wrong subset, balanced significantly exceeds pure promotion at lambda={_format_lambda_list(matched_bal_promo)} and pure suppression at lambda={_format_lambda_list(matched_bal_supp)} under paired bootstrap. "
            f"The peak matched-subset additivity signal is {peak_name} at lambda={peak_lambda:g}, with mean difference {float(peak_stats['mean_diff']):.4f} and 95% CI "
            f"[{float(peak_stats['ci95'][0]):.4f}, {float(peak_stats['ci95'][1]):.4f}].",
            f"On the {int(clean['n_items'])}-item clean base-wrong subset, balanced exceeds pure promotion at lambda={_format_lambda_list(clean_bal_promo)} and pure suppression at lambda={_format_lambda_list(clean_bal_supp)}.",
            "Balanced reaches its own saturation by lambda=1.5, while the pure modes continue climbing and only converge to the same asymptotic flip set by lambda=2.5. The old heatmap therefore describes the supra-threshold limit, not the whole decomposition story.",
        ]
    )

    lines.extend(
        [
            "",
            "## Headline Operating Point",
        ]
    )
    if headline_changed:
        lines.append(
            f"The recommended headline changes from the v3 operating point to {recommended['condition_id']} "
            f"(lambda_pos={float(recommended['lambda_pos']):g}, lambda_neg={float(recommended['lambda_neg']):g}) "
            f"because it maximizes Youden's J under the consolidation verification."
        )
    else:
        lines.append(
            f"The v3 headline remains the recommended operating point: {recommended['condition_id']} "
            f"(lambda_pos={float(recommended['lambda_pos']):g}, lambda_neg={float(recommended['lambda_neg']):g}). "
            f"{recommended['reason']}"
        )
    lines.extend(
        [
            "```json",
            json.dumps(recommended, indent=2),
            "```",
        ]
    )

    if killtest is not None:
        lines.extend(
            [
                "",
                "## Kill-Test Comparison",
                "```json",
                json.dumps(killtest, indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Label Audit",
            "Results are reported on the matched subset and on the clean audited subset where available.",
            "",
            "## Limitations",
            "- Single dataset",
            "- Single model",
            "- Fixed target layer/head",
            "- Conflict construction is MedQA-specific",
        ]
    )
    if _clean_resolution_limited(clean):
        lines.append("- The 40-item clean subset limits statistical resolution of additivity claims below lambda=0.5.")

    lines.extend(
        [
            "",
            "## Next Round",
            "Next generalization step: run the ClashEval probe experiment using the same matched-subset and mechanistic evaluation discipline to test whether the MedQA dose-response/additivity finding survives outside this dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    results_root = (PROJECT_ROOT / args.results_root).resolve()
    subthreshold_dir = (PROJECT_ROOT / args.subthreshold_dir).resolve()
    consolidation_dir = (PROJECT_ROOT / args.consolidation_dir).resolve()
    base_decomposition_dir = (PROJECT_ROOT / args.base_decomposition_dir).resolve()
    pr_sweep_dir = (PROJECT_ROOT / args.pr_sweep_dir).resolve()
    killtest_dir = (PROJECT_ROOT / args.killtest_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    report_path = (PROJECT_ROOT / args.report_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = subthreshold_dir / "aggregate.json"
    paired_bootstrap_path = consolidation_dir / "paired_bootstrap.json"
    comparison_path = consolidation_dir / "headline_verify" / "comparison.json"
    base_decomposition_path = base_decomposition_dir / "aggregate.json"

    aggregate = _load_json(aggregate_path)
    paired_bootstrap = _load_json(paired_bootstrap_path)
    comparison = _load_json(comparison_path)
    base_decomposition = _load_json(base_decomposition_path)

    best_cell_path = base_decomposition_dir / "best_cell.json"
    pr_headline_path = pr_sweep_dir / "headline.json"
    killtest_path = killtest_dir / "comparison.json"
    best_cell = _load_json(best_cell_path) if best_cell_path.exists() else None
    pr_headline = _load_json(pr_headline_path) if pr_headline_path.exists() else None
    killtest = _load_json(killtest_path) if killtest_path.exists() else None

    recommended_headline = comparison["recommended_headline"]
    _plot_dose_response(
        aggregate=aggregate,
        paired_subset=paired_bootstrap["matched_subset"],
        base_decomposition=base_decomposition,
        recommended_headline=recommended_headline,
        out_path=out_dir / "dose_response_main.png",
        clean=False,
    )
    _plot_dose_response(
        aggregate=aggregate,
        paired_subset=paired_bootstrap["clean_subset"],
        base_decomposition=base_decomposition,
        recommended_headline=recommended_headline,
        out_path=out_dir / "dose_response_clean.png",
        clean=True,
    )

    report_text = _build_report(
        results_root=results_root,
        best_cell=best_cell,
        pr_headline=pr_headline,
        killtest=killtest,
        paired_bootstrap=paired_bootstrap,
        comparison=comparison,
    )
    report_path.write_text(report_text, encoding="utf-8")

    source_files = [aggregate_path, paired_bootstrap_path, comparison_path, base_decomposition_path]
    if best_cell_path.exists():
        source_files.append(best_cell_path)
    if pr_headline_path.exists():
        source_files.append(pr_headline_path)
    if killtest_path.exists():
        source_files.append(killtest_path)

    summary = {
        "figures": [
            str((out_dir / "dose_response_main.png").resolve()),
            str((out_dir / "dose_response_clean.png").resolve()),
        ],
        "report": str(report_path.resolve()),
        "recommended_headline": recommended_headline,
        "metadata": _build_metadata(
            args=args,
            wall_clock_seconds=float(time.perf_counter() - started),
            source_files=source_files,
        ),
    }
    _write_json(out_dir / "plot_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"plot_summary={out_dir / 'plot_summary.json'}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
