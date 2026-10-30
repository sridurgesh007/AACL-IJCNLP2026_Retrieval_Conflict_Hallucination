#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medrag.utils.io import read_json


MECHANISM_DISTRIBUTED = (
    "we demonstrate that margin-gated intervention operates robustly at multiple late-layer sites "
    "because multi-option commitment is distributed across a localized cluster of heads, and "
    "characterize the effective-intervention site as structurally distinct from the causal-decision locus"
)

MECHANISM_ACTIVATION = (
    "we demonstrate that joint patching of a localized late-layer head cluster substantially alters MedQA "
    "letter commitment, though the mechanism is more consistent with aggregate activation influence than strict "
    "decision substitution, and characterize the effective-intervention site as structurally distinct from the "
    "causal-decision locus"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update REPORT.md with MedQA joint-patching disruption control.")
    parser.add_argument("--report_path", default="results/medqa_final_v3/REPORT.md")
    parser.add_argument("--localization_root", default="results/medqa_final_v3/localization_v2")
    return parser.parse_args()


def _fmt_rate(value: float, ci: list[float]) -> str:
    return f"{float(value):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _build_subsection(joint: dict, disruption: dict) -> str:
    category = str(disruption.get("finding_category", "intermediate"))
    if category == "H1":
        interpretation = (
            "Interpretation: the disruption control supports the decision-carrying reading. "
            "Joint patching substitutes the commitment rather than merely destabilizing it, so the distributed-commitment claim stands as stated."
        )
    elif category == "H2":
        interpretation = (
            "Interpretation: the disruption control is more consistent with aggregate activation influence than strict decision substitution. "
            "The MedQA mechanism claim should be reframed as an activation-level intervention on the localized commitment cluster."
        )
    else:
        interpretation = (
            "Interpretation: the disruption control gives partial support for decision substitution but does not fully rule out aggregate activation effects. "
            "The distributed-commitment claim should be retained with this caveat."
        )

    return (
        "### Disruption control on joint patching\n\n"
        f"On mismatched-letter source->dest pairs, flip-to-source = {_fmt_rate(float(disruption['flip_to_source_rate']), disruption['flip_to_source_rate_ci95_wilson'])}; "
        f"flip-to-dest-correct = {_fmt_rate(float(disruption['flip_to_dest_correct_rate']), disruption['flip_to_dest_correct_rate_ci95_wilson'])}; "
        f"flip-to-third-letter = {_fmt_rate(float(disruption['flip_to_third_letter_rate']), disruption['flip_to_third_letter_rate_ci95_wilson'])}; "
        f"unchanged = {_fmt_rate(float(disruption['unchanged_rate']), disruption['unchanged_rate_ci95_wilson'])}. "
        f"The matched-letter joint-patch reference remains {_fmt_rate(float(joint['main_flip_rate']), joint['main_flip_rate_ci95_wilson'])}. "
        f"McNemar for disruption flip-to-source vs matched-letter main: p={float(disruption['mcnemar_flip_to_source_vs_matched_letter_main']['mcnemar']['p_value']):.4g}. "
        "Figure: `results/medqa_final_v3/localization_v2/figures/joint_patching_disruption_bars.png`.\n\n"
        f"{disruption['finding']}\n\n"
        f"{interpretation}\n"
    )


def _replace_or_append_subsection(report_text: str, subsection: str) -> str:
    joint_pattern = re.compile(
        r"(^## Joint multi-head patching and distributed-commitment finding\n)(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = joint_pattern.search(report_text)
    if not match:
        if report_text and not report_text.endswith("\n"):
            report_text += "\n"
        if not report_text.strip():
            report_text = "# MedQA Final V3 Report\n\n"
        report_text += "## Joint multi-head patching and distributed-commitment finding\n\n"
        report_text += subsection + "\n"
        return report_text

    header = match.group(1)
    body = match.group(2).strip()
    subsection_pattern = re.compile(
        r"(^### Disruption control on joint patching\n.*?)(?=^### |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if subsection_pattern.search(body):
        body = subsection_pattern.sub(subsection + "\n", body).strip()
    else:
        body = (body + "\n\n" + subsection).strip()
    replacement = header + body + "\n"
    return report_text[: match.start()] + replacement + report_text[match.end() :]


def main() -> int:
    args = parse_args()
    report_path = (PROJECT_ROOT / args.report_path).resolve()
    localization_root = (PROJECT_ROOT / args.localization_root).resolve()

    joint = read_json(localization_root / "joint_patching" / "summary.json")
    disruption = read_json(localization_root / "joint_patching_disruption" / "summary.json")

    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    subsection = _build_subsection(joint, disruption)
    updated = _replace_or_append_subsection(text, subsection)

    category = str(disruption.get("finding_category", "intermediate"))
    if category == "H2":
        updated = updated.replace(MECHANISM_DISTRIBUTED, MECHANISM_ACTIVATION)
    else:
        updated = updated.replace(MECHANISM_ACTIVATION, MECHANISM_DISTRIBUTED)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(updated, encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
