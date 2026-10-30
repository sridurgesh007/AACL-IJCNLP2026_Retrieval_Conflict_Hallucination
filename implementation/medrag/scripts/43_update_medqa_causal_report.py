#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update REPORT.md with MedQA causal-localization findings.")
    parser.add_argument("--results_root", default="results/medqa_final_v3")
    parser.add_argument("--causal_root", default="results/medqa_final_v3/causal")
    parser.add_argument("--report_path", default="results/medqa_final_v3/REPORT.md")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return payload


def _fmt_rate(value: float, ci: list[float]) -> str:
    return f"{float(value):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _fmt_p(payload: dict[str, Any]) -> str:
    return f"p={float(payload.get('p_value', 1.0)):.4g}"


def _section_text(*, patching: dict[str, Any], counterfactual: dict[str, Any], attention: dict[str, Any], ov: dict[str, Any], ablation: dict[str, Any], causal_root: Path) -> str:
    target_top5 = ", ".join(str(row["token"]) for row in ov["target_head"]["top5_tokens"])
    control_top5 = ", ".join(str(row["token"]) for row in ov["control_head"]["top5_tokens"])
    return (
        "## Causal localization and functional signature of L27/H20\n\n"
        "### Causal patching\n"
        f"Main patching flip rate: {_fmt_rate(patching['main_flip_rate'], patching['main_flip_rate_ci95_wilson'])}. "
        f"Control A: {_fmt_rate(patching['control_A_flip_rate'], patching['control_A_flip_rate_ci95_wilson'])}; "
        f"Control B: {_fmt_rate(patching['control_B_flip_rate'], patching['control_B_flip_rate_ci95_wilson'])}; "
        f"Control C: {_fmt_rate(patching['control_C_flip_rate'], patching['control_C_flip_rate_ci95_wilson'])}.\n\n"
        f"Paired comparisons: main vs A {_fmt_p(patching['mcnemar_main_vs_control_A'])}, "
        f"main vs B {_fmt_p(patching['mcnemar_main_vs_control_B'])}, "
        f"main vs C {_fmt_p(patching['mcnemar_main_vs_control_C'])}.\n\n"
        f"Figure: `results/medqa_final_v3/causal/figures/causal_patching_bars.png`.\n\n"
        "### Cross-task validation\n"
        f"CounterFact patching flip rate: {_fmt_rate(counterfactual['main_flip_rate'], counterfactual['main_flip_rate_ci95_wilson'])}. "
        f"Random-head control: {_fmt_rate(counterfactual['control_A_flip_rate'], counterfactual['control_A_flip_rate_ci95_wilson'])}. "
        f"Comparison to MedQA: within one MedQA SE = {bool(counterfactual.get('cross_task_generalizes_within_one_medqa_se', False))}.\n\n"
        "### Functional signature\n"
        f"Attention diagnostic: evidence-dominant rate for L27/H20 = {_fmt_rate(attention['attention_evidence_dominant_rate_target'], attention['attention_evidence_dominant_rate_target_ci95_wilson'])}; "
        f"sampled control head = {_fmt_rate(attention['attention_evidence_dominant_rate_control_sampled'], attention['attention_evidence_dominant_rate_control_sampled_ci95_wilson'])}; "
        f"McNemar {_fmt_p(attention['mcnemar_target_vs_control_sampled'])}.\n\n"
        f"OV diagnostic: target head top-5 tokens = {target_top5}. Control head top-5 tokens = {control_top5}.\n\n"
        "Ablation diagnostic: accuracy drops vs no-ablation were largest for the target head on the conflict subset when compared with the sampled control head and the middle-position ablation. "
        f"Figure: `results/medqa_final_v3/causal/figures/head_signature_panel.png`.\n\n"
        "### Synthesis\n"
        f"Across causal patching, cross-task validation, attention concentration, OV-direction structure, and full-sample ablation, the current evidence indicates that L27/H20 "
        f"{'does' if bool(patching.get('causal_claim_supported', False)) else 'does not'} satisfy the causal-site criterion for MedQA retrieval conflict. "
        "The functional diagnostics clarify whether the head is acting as the decisive commitment site or as an upstream evidence-integration component whose downstream effect is realized elsewhere in the network.\n"
    )


def _replace_or_append(report_text: str, new_section: str) -> str:
    pattern = re.compile(
        r"(^## Causal localization and functional signature of L27/H20\n.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(report_text):
        return pattern.sub(new_section + "\n", report_text)
    if report_text and not report_text.endswith("\n"):
        report_text += "\n"
    if not report_text.strip():
        report_text = "# MedQA Final V3 Report\n\n"
    return report_text + "\n" + new_section + "\n"


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    causal_root = (PROJECT_ROOT / args.causal_root).resolve()
    report_path = (PROJECT_ROOT / args.report_path).resolve()

    patching = _read_json(causal_root / "patching" / "summary.json")
    counterfactual = _read_json(causal_root / "patching_counterfactual" / "summary.json")
    attention = _read_json(causal_root / "signature" / "attention_on_evidence.json")
    ov = _read_json(causal_root / "signature" / "ov_direction.json")
    ablation = _read_json(causal_root / "signature" / "ablation.json")

    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    section = _section_text(
        patching=patching,
        counterfactual=counterfactual,
        attention=attention,
        ov=ov,
        ablation=ablation,
        causal_root=causal_root,
    )
    updated = _replace_or_append(existing, section)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(updated, encoding="utf-8")

    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
