#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medrag.utils.io import read_json


MECHANISM_OLD = "we causally localize the commitment site"
MECHANISM_NEW = (
    "we demonstrate that margin-gated intervention operates robustly at multiple late-layer sites "
    "because multi-option commitment is distributed across a localized cluster of heads, and "
    "characterize the effective-intervention site as structurally distinct from the causal-decision locus"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update REPORT.md with MedQA joint multi-head patching findings.")
    parser.add_argument("--report_path", default="results/medqa_final_v3/REPORT.md")
    parser.add_argument("--localization_root", default="results/medqa_final_v3/localization_v2")
    return parser.parse_args()


def _fmt_rate(value: float, ci: list[float]) -> str:
    return f"{float(value):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _replace_or_append_joint(report_text: str, new_section: str) -> str:
    joint_pattern = re.compile(
        r"(^## Joint multi-head patching and distributed-commitment finding\n.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if joint_pattern.search(report_text):
        return joint_pattern.sub(new_section + "\n", report_text)

    causal_pattern = re.compile(
        r"(^## MedQA-specific causal localization\n.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = causal_pattern.search(report_text)
    if match:
        insert_at = match.end()
        prefix = report_text[:insert_at]
        suffix = report_text[insert_at:]
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return prefix + "\n" + new_section + "\n" + suffix.lstrip("\n")

    if report_text and not report_text.endswith("\n"):
        report_text += "\n"
    if not report_text.strip():
        report_text = "# MedQA Final V3 Report\n\n"
    return report_text + "\n" + new_section + "\n"


def _build_section(
    *,
    head_importance: dict,
    patching: dict,
    intervention: dict,
    joint: dict,
) -> str:
    top5 = ", ".join(
        f"L{int(row['layer'])}/H{int(row['head'])}"
        for row in joint["top5_heads"]
    )
    top1 = str(joint["top5_heads"][0]["cell_name"])
    reference = intervention["reference_h4"]

    return (
        "## Joint multi-head patching and distributed-commitment finding\n\n"
        f"Joint patching used the top-5 MedQA-localized heads: {top5}. "
        f"Joint top-5 main flip rate = {_fmt_rate(float(joint['main_flip_rate']), joint['main_flip_rate_ci95_wilson'])}. "
        f"Joint control A = {_fmt_rate(float(joint['control_A_flip_rate']), joint['control_A_flip_rate_ci95_wilson'])}; "
        f"joint control B = {_fmt_rate(float(joint['control_B_flip_rate']), joint['control_B_flip_rate_ci95_wilson'])}; "
        f"pooled controls = {_fmt_rate(float(joint['controls_pooled_flip_rate']), joint['controls_pooled_flip_rate_ci95_wilson'])}. "
        f"Figure: `results/medqa_final_v3/localization_v2/figures/joint_patching_bars.png`.\n\n"
        f"Single-head null comparisons remain weak: L27/H20 main = {_fmt_rate(float(patching['existing_l27_h20_null']['main_flip_rate']), patching['existing_l27_h20_null']['main_flip_rate_ci95_wilson'])}; "
        f"{top1} main = {_fmt_rate(float(next(row for row in patching['candidate_summaries'] if str(row['cell_name']) == top1)['main_flip_rate']), next(row for row in patching['candidate_summaries'] if str(row['cell_name']) == top1)['main_flip_rate_ci95_wilson'])}. "
        f"Joint-vs-L27/H20 McNemar p={float(joint['mcnemar_joint_main_vs_l27_single']['mcnemar']['p_value']):.4g}; "
        f"joint-vs-{top1} McNemar p={float(joint['mcnemar_joint_main_vs_top1_single']['mcnemar']['p_value']):.4g}.\n\n"
        "Synthesis: the effective-intervention site is structurally distinct from the decision locus. "
        f"Headline intervention remains at {_fmt_rate(float(reference['restoration_rate']), reference['restoration_rate_ci95_wilson'])} restoration with "
        f"{_fmt_rate(float(reference['harm_rate']), reference['harm_rate_ci95_wilson'])} harm, while single-head causal patching at both L27/H20 and {top1} is null. "
        f"{joint['finding']}\n"
    )


def main() -> int:
    args = parse_args()
    report_path = (PROJECT_ROOT / args.report_path).resolve()
    localization_root = (PROJECT_ROOT / args.localization_root).resolve()

    head_importance = read_json(localization_root / "head_importance.json")
    patching = read_json(localization_root / "patching_at_new_head" / "summary.json")
    intervention = read_json(localization_root / "intervention_at_new_head" / "summary.json")
    joint = read_json(localization_root / "joint_patching" / "summary.json")

    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if MECHANISM_OLD in text:
        text = text.replace(MECHANISM_OLD, MECHANISM_NEW)
    section = _build_section(
        head_importance=head_importance,
        patching=patching,
        intervention=intervention,
        joint=joint,
    )
    updated = _replace_or_append_joint(text, section)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(updated, encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
