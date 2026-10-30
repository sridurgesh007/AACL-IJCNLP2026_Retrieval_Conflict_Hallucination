#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medrag.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update REPORT.md with MedQA-specific causal localization findings.")
    parser.add_argument("--report_path", default="results/medqa_final_v3/REPORT.md")
    parser.add_argument("--localization_root", default="results/medqa_final_v3/localization_v2")
    return parser.parse_args()


def _replace_or_append(report_text: str, new_section: str) -> str:
    pattern = re.compile(
        r"(^## MedQA-specific causal localization\n.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(report_text):
        return pattern.sub(new_section + "\n", report_text)
    if report_text and not report_text.endswith("\n"):
        report_text += "\n"
    if not report_text.strip():
        report_text = "# MedQA Final V3 Report\n\n"
    return report_text + "\n" + new_section + "\n"


def _fmt_rate(value: float, ci: list[float]) -> str:
    return f"{float(value):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _selected_patch_summary(patching: dict[str, Any], cell_name: str) -> dict[str, Any]:
    for row in patching.get("candidate_summaries", []):
        if str(row.get("cell_name")) == str(cell_name):
            return row
    raise RuntimeError(f"Could not find patching summary for {cell_name}.")


def _section_text(
    *,
    localization: dict[str, Any],
    patching: dict[str, Any],
    intervention: dict[str, Any],
) -> str:
    top5 = ", ".join(
        f"L{int(row['layer'])}/H{int(row['head'])} (rank {int(row['rank'])}, {float(row['mean_contribution']):.3f})"
        for row in localization["top_5_heads"]
    )
    selected_interv = intervention["selected_candidate"]
    selected_cell = str(selected_interv["cell_name"])
    selected_patch = _selected_patch_summary(patching, selected_cell)
    headline_summary = selected_interv["headline_summary"]
    balanced_summary = selected_interv["balanced_summary"]
    baseline = intervention["reference_h4"]

    return (
        "## MedQA-specific causal localization\n\n"
        "### 1. Head importance on MedQA decision prompts\n"
        f"Top-5 heads by MedQA letter-margin contribution: {top5}. "
        f"L27/H20 rank = {int(localization['L27_H20_rank'])}, mean contribution = {float(localization['L27_H20_mean_contribution']):.3f}. "
        "Figure: `results/medqa_final_v3/localization_v2/figures/head_importance_heatmap.png`.\n\n"
        "### 2. Causal patching at the MedQA-localized head\n"
        f"Top-level patching decision: {patching['decision']['text']} "
        f"For {selected_cell}, main patching flip rate = {_fmt_rate(float(selected_patch['main_flip_rate']), selected_patch['main_flip_rate_ci95_wilson'])}. "
        f"Existing L27/H20 null main flip rate = {_fmt_rate(float(patching['existing_l27_h20_null']['main_flip_rate']), patching['existing_l27_h20_null']['main_flip_rate_ci95_wilson'])}. "
        "Figure: `results/medqa_final_v3/localization_v2/figures/patching_comparison_bars.png`.\n\n"
        "### 3. Intervention effectiveness at the new head\n"
        f"Headline config at {selected_cell}: restoration = {_fmt_rate(float(headline_summary['restoration_rate']), headline_summary['restoration_rate_ci95_wilson'])}, "
        f"harm = {_fmt_rate(float(headline_summary['harm_rate']), headline_summary['harm_rate_ci95_wilson'])}, "
        f"ft_accuracy = {_fmt_rate(float(headline_summary['ft_accuracy']), headline_summary['ft_accuracy_ci95_wilson'])}. "
        f"Balanced config at {selected_cell}: restoration = {_fmt_rate(float(balanced_summary['restoration_rate']), balanced_summary['restoration_rate_ci95_wilson'])}. "
        f"Reference H4 at L27/H20: restoration = {_fmt_rate(float(baseline['restoration_rate']), baseline['restoration_rate_ci95_wilson'])}, "
        f"harm = {_fmt_rate(float(baseline['harm_rate']), baseline['harm_rate_ci95_wilson'])}, "
        f"ft_accuracy = {_fmt_rate(float(baseline['ft_accuracy']), baseline['ft_accuracy_ci95_wilson'])}. "
        "Figure: `results/medqa_final_v3/localization_v2/figures/intervention_comparison_bars.png`.\n\n"
        "### 4. Framing decision\n"
        f"Option {intervention['framing_decision']['option']}: {intervention['framing_decision']['reason']}\n"
    )


def main() -> int:
    args = parse_args()
    report_path = (PROJECT_ROOT / args.report_path).resolve()
    localization_root = (PROJECT_ROOT / args.localization_root).resolve()

    localization = read_json(localization_root / "head_importance.json")
    patching = read_json(localization_root / "patching_at_new_head" / "summary.json")
    intervention = read_json(localization_root / "intervention_at_new_head" / "summary.json")

    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    section = _section_text(
        localization=localization,
        patching=patching,
        intervention=intervention,
    )
    updated = _replace_or_append(existing, section)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(updated, encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
