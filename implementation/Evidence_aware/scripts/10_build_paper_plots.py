from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CORRECT_TABLE: list[dict[str, Any]] = [
    {
        "method": "baseline",
        "accuracy": 0.3055,
        "restoration_rate": 0.0,
        "stability_rate": 1.0,
        "harm_rate": 0.0,
        "restoration_improvement": 0.0,
        "source": "results/phase2/main_eval/baseline_results.json",
    },
    {
        "method": "dola",
        "accuracy": 0.109,
        "restoration_rate": 0.034,
        "stability_rate": 0.259,
        "harm_rate": 0.016,
        "restoration_improvement": 0.016,
        "source": "results/dola/dola_results.json",
    },
    {
        "method": "erha_unconditional_beta5",
        "accuracy": 0.290,
        "restoration_rate": 0.068,
        "stability_rate": 0.851,
        "harm_rate": 0.000,
        "restoration_improvement": 0.068,
        "source": "results/phase2/main_eval/erha_unconditional_beta5.json",
    },
    {
        "method": "ag_erha_best",
        "accuracy": 0.304,
        "restoration_rate": 0.0,
        "stability_rate": 1.0,
        "harm_rate": 0.0,
        "restoration_improvement": 0.0,
        "source": "results/ag_erha/ag_erha_best.json",
    },
    {
        "method": "mg_erha_single_head",
        "accuracy": 0.319,
        "restoration_rate": 0.057,
        "stability_rate": 0.987,
        "harm_rate": 0.000,
        "restoration_improvement": 0.057,
        "source": "results/mg_erha_first_token/final_summary.json",
    },
    {
        "method": "true_multi_head_best",
        "accuracy": 0.334,
        "restoration_rate": 0.061,
        "stability_rate": 0.995,
        "harm_rate": 0.000,
        "restoration_improvement": 0.061,
        "source": "results/true_multi_head/experiment_a_sweep_summary.json",
        "config": "multi_all8_b3p0 tau=0.0",
    },
    {
        "method": "ld_steering_clip10_alpha1_tau0",
        "accuracy": 0.344,
        "restoration_rate": 0.172,
        "stability_rate": 0.997,
        "harm_rate": 0.000,
        "restoration_improvement": 0.172,
        "lambda_max": 10.0,
        "alpha": 1.0,
        "tau_margin": 0.0,
        "clip_fraction": 0.407,
        "mean_lambda": 4.47,
        "source": "results/true_multi_head/exp_b_corrected/experiment_b_sweep_summary.json",
        "note": "Primary reported steering result",
    },
    {
        "method": "ld_steering_clip20_alpha1_tau0",
        "accuracy": 0.370,
        "restoration_rate": 0.319,
        "stability_rate": 0.995,
        "harm_rate": 0.000,
        "restoration_improvement": 0.319,
        "lambda_max": 20.0,
        "alpha": 1.0,
        "tau_margin": 0.0,
        "clip_fraction": 0.325,
        "mean_lambda": 8.14,
        "source": "results/true_multi_head/exp_b_corrected/experiment_b_sweep_summary.json",
        "note": "Upper bound - clip boundary effect, report with caveat",
    },
]

METHOD_ORDER = [
    "baseline",
    "dola",
    "erha_unconditional_beta5",
    "ag_erha_best",
    "mg_erha_single_head",
    "true_multi_head_best",
    "ld_steering_clip10_alpha1_tau0",
]

METHOD_LABELS = [
    "Baseline",
    "DoLa",
    "ERHA\nunconditional",
    "AG-ERHA",
    "MG-ERHA\nsingle-head",
    "True\nmulti-head",
    "LD Steering\nclip10 alpha=1",
]

COPY_SOURCES: dict[str, list[str]] = {
    "fig1_track_b_margin_trajectory.png": [
        "results/track_b/plots/margin_trajectory_by_layer.png",
        "../track_b/plots/margin_trajectory_by_layer.png",
        "track_b/plots/margin_trajectory_by_layer.png",
    ],
    "fig2_phase1_ld_heatmap.png": [
        "results/phase1_diagnostics/plots/head_ld_heatmap_delta.png",
        "results/phase1_output/plots/head_ld_heatmap_delta.png",
        "../phase1_output/plots/head_ld_heatmap_delta.png",
    ],
    "fig3_mg_erha_gate_discrimination.png": [
        "results/mg_erha_first_token/plots/gate_fire_rate_by_label.png",
        "../mg_erha_first_token/plots/gate_fire_rate_by_label.png",
    ],
    "fig4_mg_erha_margin_distribution.png": [
        "results/mg_erha_first_token/plots/margin_distribution_by_label.png",
        "../mg_erha_first_token/plots/margin_distribution_by_label.png",
    ],
    "fig6_steering_restoration_vs_tau.png": [
        "results/true_multi_head/exp_b_corrected/plots/restoration_vs_tau_steering.png",
        "../true_multi_head/exp_b_corrected/plots/restoration_vs_tau_steering.png",
        "true_multi_head/exp_b_corrected/plots/restoration_vs_tau_steering.png",
    ],
    "fig7_steering_stability_vs_tau.png": [
        "results/true_multi_head/exp_b_corrected/plots/stability_vs_tau_steering.png",
        "../true_multi_head/exp_b_corrected/plots/stability_vs_tau_steering.png",
        "true_multi_head/exp_b_corrected/plots/stability_vs_tau_steering.png",
    ],
    "fig10_exp_a_per_layer_gate_fire.png": [
        "results/true_multi_head/plots/per_layer_gate_fire_rate_exp_a.png",
        "../true_multi_head/plots/per_layer_gate_fire_rate_exp_a.png",
        "true_multi_head/plots/per_layer_gate_fire_rate_exp_a.png",
    ],
    "fig11_cross_model_comparison.png": [
        "results/generalization/final_cross_model_comparison.png",
        "results/generalization/cross_model_comparison_table.png",
        "../generalization/final_cross_model_comparison.png",
        "../generalization/cross_model_comparison_table.png",
        "generalization/final_cross_model_comparison.png",
        "generalization/cross_model_comparison_table.png",
    ],
}


def _resolve_existing(project_dir: Path, candidates: list[str]) -> Path | None:
    for cand in candidates:
        path = (project_dir / cand).resolve()
        if path.exists():
            return path
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_fig5(out_path: Path, table_rows: list[dict[str, Any]]) -> None:
    table = {str(r["method"]): r for r in table_rows}
    restoration = [float(table[m]["restoration_rate"]) for m in METHOD_ORDER]
    stability = [float(table[m]["stability_rate"]) for m in METHOD_ORDER]
    improvement = [float(table[m]["restoration_improvement"]) for m in METHOD_ORDER]

    x = np.arange(len(METHOD_ORDER))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.bar(x - w, restoration, w, label="restoration_rate", color="#4C72B0")
    ax.bar(x, stability, w, label="stability_rate", color="#DD8452")
    ax.bar(x + w, improvement, w, label="restoration_improvement", color="#55A868")
    ax.axhline(0.92, color="gold", linestyle="--", linewidth=1.2, label="stability target")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Comparison Across Methods (Llama 3.1 8B, CounterFact)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _load_gate_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Gate stats file is empty: {path}")
    return rows


def _build_fig8(out_path: Path, gate_rows: list[dict[str, Any]]) -> None:
    hall_lambdas: list[float] = []
    faith_lambdas: list[float] = []
    for ex in gate_rows:
        lam = float(ex.get("mean_lambda_when_steered", 0.0))
        label = str(ex.get("baseline_label", ""))
        if label == "hallucinated":
            hall_lambdas.append(lam)
        elif label == "faithful":
            faith_lambdas.append(lam)

    bins = np.linspace(0.0, 12.0, 30)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.hist(faith_lambdas, bins=bins, alpha=0.6, label="faithful", color="#DD8452")
    ax.hist(hall_lambdas, bins=bins, alpha=0.6, label="hallucinated", color="#4C72B0")
    ax.axvline(10.0, color="red", linestyle="--", linewidth=1.2, label="lambda_max=10 (clip)")
    ax.set_xlim(0.0, 12.0)
    ax.set_xlabel("mean lambda when steered")
    ax.set_ylabel("count")
    ax.set_title("LD Steering: Lambda Distribution by Label (L27H20, clip10, alpha=1.0, tau=0.0)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _build_fig9(out_path: Path, gate_rows: list[dict[str, Any]]) -> None:
    from collections import defaultdict

    layer_hall: dict[int, list[float]] = defaultdict(list)
    layer_faith: dict[int, list[float]] = defaultdict(list)

    for ex in gate_rows:
        label = str(ex.get("baseline_label", ""))
        per_layer = ex.get("per_layer_lambda_clipped")
        if not isinstance(per_layer, dict) or not per_layer:
            per_layer = ex.get("per_layer_lambda", {})
        if not isinstance(per_layer, dict):
            continue
        for l_key, lam_val in per_layer.items():
            lam = float(lam_val)
            if abs(lam) <= 1e-8:
                continue
            layer_idx = int(l_key)
            if label == "hallucinated":
                layer_hall[layer_idx].append(lam)
            elif label == "faithful":
                layer_faith[layer_idx].append(lam)

    layers = sorted(set(layer_hall.keys()) | set(layer_faith.keys()))
    if not layers:
        raise RuntimeError("No non-zero per-layer lambda values found for fig9.")
    hall_means = [float(np.mean(layer_hall[l])) if layer_hall[l] else 0.0 for l in layers]
    faith_means = [float(np.mean(layer_faith[l])) if layer_faith[l] else 0.0 for l in layers]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(layers, hall_means, "o-", label="hallucinated", color="#4C72B0", linewidth=2)
    ax.plot(layers, faith_means, "s-", label="faithful", color="#DD8452", linewidth=2)
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.0, label="lambda_max=10")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean lambda (non-zero examples)")
    ax.set_title("Per-layer Lambda by Label (L27H20, clip10, alpha=1.0, tau=0.0)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corrected final paper plots from final_correction.md")
    parser.add_argument("--out_dir", default="results/final_correction/paper_plots")
    parser.add_argument(
        "--clip10_gate_stats_path",
        default="",
        help="Optional explicit path to gate_stats_steering_L27H20_clip10p0_tau_0.jsonl",
    )
    parser.add_argument("--strict_copy", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    out_dir = (project_dir / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "combined_comparison_table.json"
    _write_json(table_path, CORRECT_TABLE)

    fig5_path = out_dir / "fig5_method_comparison_bar.png"
    _build_fig5(fig5_path, CORRECT_TABLE)

    gate_candidates = []
    if str(args.clip10_gate_stats_path).strip():
        gate_candidates.append(str(args.clip10_gate_stats_path))
    gate_candidates.extend(
        [
            "results/true_multi_head/exp_b_corrected/gate_stats/gate_stats_steering_L27H20_clip10p0_tau_0.jsonl",
            "results/true_multi_head/exp_b_corrected/gate_stats/gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl",
            "../true_multi_head/exp_b_corrected/gate_stats/gate_stats_steering_L27H20_clip10p0_tau_0.jsonl",
            "../true_multi_head/exp_b_corrected/gate_stats/gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl",
            "true_multi_head/exp_b_corrected/gate_stats/gate_stats_steering_L27H20_clip10p0_tau_0.jsonl",
            "true_multi_head/exp_b_corrected/gate_stats/gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl",
        ]
    )
    gate_path = _resolve_existing(project_dir, gate_candidates)
    if gate_path is None:
        raise FileNotFoundError(
            "Could not resolve clip10 gate stats JSONL. Checked candidates: "
            + ", ".join(gate_candidates)
        )

    gate_rows = _load_gate_rows(gate_path)
    fig8_path = out_dir / "fig8_steering_lambda_distribution.png"
    fig9_path = out_dir / "fig9_per_layer_lambda_by_label.png"
    _build_fig8(fig8_path, gate_rows)
    _build_fig9(fig9_path, gate_rows)

    copied: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for target_name, candidates in COPY_SOURCES.items():
        source_path = _resolve_existing(project_dir, candidates)
        target_path = out_dir / target_name
        if source_path is None:
            missing[target_name] = candidates
            continue
        shutil.copy2(source_path, target_path)
        copied[target_name] = str(source_path)

    expected = {
        "fig1_track_b_margin_trajectory.png",
        "fig2_phase1_ld_heatmap.png",
        "fig3_mg_erha_gate_discrimination.png",
        "fig4_mg_erha_margin_distribution.png",
        "fig5_method_comparison_bar.png",
        "fig6_steering_restoration_vs_tau.png",
        "fig7_steering_stability_vs_tau.png",
        "fig8_steering_lambda_distribution.png",
        "fig9_per_layer_lambda_by_label.png",
        "fig10_exp_a_per_layer_gate_fire.png",
        "fig11_cross_model_comparison.png",
    }
    present = {p.name for p in out_dir.glob("*.png")}
    missing_expected = sorted(expected - present)

    summary = {
        "out_dir": str(out_dir),
        "combined_table": str(table_path),
        "clip10_gate_stats_used": str(gate_path),
        "copied_figures": copied,
        "copy_missing_sources": missing,
        "present_figures": sorted(present),
        "missing_expected_figures": missing_expected,
        "strict_copy": bool(int(args.strict_copy)),
    }
    summary_path = out_dir / "build_summary.json"
    _write_json(summary_path, summary)

    if bool(int(args.strict_copy)) and missing_expected:
        raise RuntimeError(
            "Missing expected paper plots in strict mode: " + ", ".join(missing_expected)
        )

    print(json.dumps(summary, indent=2))
    print(f"paper_plots={out_dir}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
