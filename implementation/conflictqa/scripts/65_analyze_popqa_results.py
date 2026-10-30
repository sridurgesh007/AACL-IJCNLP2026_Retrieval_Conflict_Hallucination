#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-hoc PopQA C4/C5 and stratification analysis from existing results.")
    parser.add_argument("--results_dir", default="results/popqa")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--previous_summary_path", default="")
    parser.add_argument("--low_n_threshold", type=int, default=5)
    parser.add_argument("--max_failure_examples", type=int, default=5)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _wilson_ci(k: int, n: int) -> list[float]:
    import math
    from statistics import NormalDist

    if int(n) <= 0:
        return [0.0, 0.0]
    z = float(NormalDist().inv_cdf(0.975))
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) / float(n)) + (z * z) / (4.0 * n * n)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mcnemar_from_paired(base_correct: list[bool], cur_correct: list[bool]) -> dict[str, Any]:
    if len(base_correct) != len(cur_correct):
        raise ValueError("Paired vectors must have same length.")
    b = 0
    c = 0
    for left, right in zip(base_correct, cur_correct):
        if bool(left) and not bool(right):
            b += 1
        elif (not bool(left)) and bool(right):
            c += 1
    total = b + c
    if total <= 0:
        return {"b": int(b), "c": int(c), "n_discordant": 0, "p_value": 1.0, "method": "degenerate"}
    tail = 0.0
    for i in range(0, min(b, c) + 1):
        tail += math.comb(total, i) * (0.5**total)
    return {
        "b": int(b),
        "c": int(c),
        "n_discordant": int(total),
        "p_value": float(min(1.0, 2.0 * tail)),
        "method": "exact_binomial",
    }


def _mean(values: Iterable[float]) -> float | None:
    seq = [float(value) for value in values]
    if not seq:
        return None
    return float(sum(seq) / len(seq))


def _std(values: Iterable[float]) -> float | None:
    seq = [float(value) for value in values]
    if len(seq) <= 1:
        return 0.0 if seq else None
    mu = float(sum(seq) / len(seq))
    var = float(sum((value - mu) ** 2 for value in seq) / len(seq))
    return float(math.sqrt(var))


def _restore_harm_summary(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    c2_wrong = [row for row in rows if bool(row["c2_parametric"])]
    c2_right = [row for row in rows if bool(row["c2_context_faithful"])]
    restored = sum(1 for row in c2_wrong if bool(row["c3_context_faithful"]))
    harmed = sum(1 for row in c2_right if not bool(row["c3_context_faithful"]))
    return int(len(c2_wrong)), int(len(c2_right)), int(restored), int(harmed)


def _condition_vectors(rows: list[dict[str, Any]], condition: str) -> list[bool]:
    if condition == "C2":
        return [bool(row["c2_context_faithful"]) for row in rows]
    if condition == "C3":
        return [bool(row["c3_context_faithful"]) for row in rows]
    if condition == "C4":
        return [str(row["c4_classification"]) == "context_faithful" for row in rows]
    if condition == "C5":
        return [bool(row.get("c5_context_faithful", str(row.get("c5_classification", "")) == "context_faithful")) for row in rows]
    raise ValueError(f"Unsupported condition {condition}")


def _build_c4_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    c4 = dict(summary["conditions"]["C4"])
    c2_vec = _condition_vectors(rows, "C2")
    c3_vec = _condition_vectors(rows, "C3")
    c4_vec = _condition_vectors(rows, "C4")
    c4["mcnemar_c2_vs_c4"] = _mcnemar_from_paired(c2_vec, c4_vec)
    c4["mcnemar_c3_vs_c4"] = _mcnemar_from_paired(c3_vec, c4_vec)
    c4["c3_faithfulness_rate"] = float(sum(1 for value in c3_vec if value) / max(len(c3_vec), 1))
    return c4


def _build_c5_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    c5 = dict(summary["conditions"]["C5"])
    c2_vec = _condition_vectors(rows, "C2")
    c3_vec = _condition_vectors(rows, "C3")
    c5_vec = _condition_vectors(rows, "C5")
    c5["mcnemar_c2_vs_c5"] = _mcnemar_from_paired(c2_vec, c5_vec)
    c5["mcnemar_c3_vs_c5"] = _mcnemar_from_paired(c3_vec, c5_vec)
    return c5


def _prop_rows(rows: list[dict[str, Any]], low_n_threshold: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prop"]), []).append(row)
    out: list[dict[str, Any]] = []
    for prop, group_rows in sorted(grouped.items()):
        c2_wrong = [row for row in group_rows if bool(row["c2_parametric"])]
        c2_right = [row for row in group_rows if bool(row["c2_context_faithful"])]
        restored = sum(1 for row in c2_wrong if bool(row["c3_context_faithful"]))
        harmed = sum(1 for row in c2_right if not bool(row["c3_context_faithful"]))
        out.append(
            {
                "prop": prop,
                "prop_key": prop.replace(" ", "_"),
                "n_items": int(len(group_rows)),
                "n_c2_hallucinated": int(len(c2_wrong)),
                "n_c3_restored": int(restored),
                "restoration_rate": float(restored / max(len(c2_wrong), 1)),
                "restoration_rate_ci95_wilson": _wilson_ci(restored, len(c2_wrong)),
                "harm_rate": float(harmed / max(len(c2_right), 1)),
                "harm_rate_ci95_wilson": _wilson_ci(harmed, len(c2_right)),
                "low_n": bool(len(c2_wrong) < int(low_n_threshold)),
            }
        )
    return out


def _bucket_prob(value: float) -> tuple[str, float, float | None]:
    score = float(value)
    if score < 0.45:
        return ("Q1: 0.30-0.45", 0.30, 0.45)
    if score < 0.55:
        return ("Q2: 0.45-0.55", 0.45, 0.55)
    if score < 0.70:
        return ("Q3: 0.55-0.70", 0.55, 0.70)
    return ("Q4: 0.70+", 0.70, None)


def _prob_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["Q1: 0.30-0.45", "Q2: 0.45-0.55", "Q3: 0.55-0.70", "Q4: 0.70+"]
    grouped = {key: [] for key in order}
    for row in rows:
        grouped[_bucket_prob(float(row["llm_answer_prob"]))[0]].append(row)
    out: list[dict[str, Any]] = []
    for key in order:
        group_rows = grouped[key]
        c2_wrong = [row for row in group_rows if bool(row["c2_parametric"])]
        c2_right = [row for row in group_rows if bool(row["c2_context_faithful"])]
        restored = sum(1 for row in c2_wrong if bool(row["c3_context_faithful"]))
        harmed = sum(1 for row in c2_right if not bool(row["c3_context_faithful"]))
        margins = [float(row["prefill_margin"]) for row in group_rows]
        out.append(
            {
                "label": key,
                "n_items": int(len(group_rows)),
                "n_c2_hallucinated": int(len(c2_wrong)),
                "prefill_margin_mean": _mean(margins),
                "prefill_margin_std": _std(margins),
                "restoration_rate": float(restored / max(len(c2_wrong), 1)),
                "restoration_rate_ci95_wilson": _wilson_ci(restored, len(c2_wrong)),
                "harm_rate": float(harmed / max(len(c2_right), 1)),
                "harm_rate_ci95_wilson": _wilson_ci(harmed, len(c2_right)),
            }
        )
    return out


def _bucket_popularity(value: float) -> str:
    score = float(value)
    if score < 500.0:
        return "Low: s_pop < 500"
    if score <= 5000.0:
        return "Medium: 500-5000"
    return "High: >5000"


def _popularity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["Low: s_pop < 500", "Medium: 500-5000", "High: >5000"]
    grouped = {key: [] for key in order}
    for row in rows:
        grouped[_bucket_popularity(float(row["s_pop"]))].append(row)
    out: list[dict[str, Any]] = []
    for key in order:
        group_rows = grouped[key]
        c2_wrong = [row for row in group_rows if bool(row["c2_parametric"])]
        c2_right = [row for row in group_rows if bool(row["c2_context_faithful"])]
        restored = sum(1 for row in c2_wrong if bool(row["c3_context_faithful"]))
        harmed = sum(1 for row in c2_right if not bool(row["c3_context_faithful"]))
        hallucinated_margins = [float(row["prefill_margin"]) for row in c2_wrong]
        out.append(
            {
                "label": key,
                "n_items": int(len(group_rows)),
                "n_c2_hallucinated": int(len(c2_wrong)),
                "restoration_rate": float(restored / max(len(c2_wrong), 1)),
                "restoration_rate_ci95_wilson": _wilson_ci(restored, len(c2_wrong)),
                "harm_rate": float(harmed / max(len(c2_right), 1)),
                "harm_rate_ci95_wilson": _wilson_ci(harmed, len(c2_right)),
                "mean_prefill_margin_hallucinated": _mean(hallucinated_margins),
            }
        )
    return out


def _failure_mode(rows: list[dict[str, Any]], max_examples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures = [row for row in rows if bool(row["c3_gate_fired"]) and str(row["c3_classification"]) == "parametric"]
    margins = [float(row["prefill_margin"]) for row in failures]
    confidences = [float(row["llm_answer_prob"]) for row in failures]
    prop_counts = Counter(str(row["prop"]) for row in failures)
    examples = []
    for row in sorted(failures, key=lambda item: (float(item["prefill_margin"]), -float(item["llm_answer_prob"])))[: int(max_examples)]:
        examples.append(
            {
                "item_id": row["item_id"],
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "parametric_answer": row["parametric_answer"],
                "prefill_margin": float(row["prefill_margin"]),
                "llm_answer_prob": float(row["llm_answer_prob"]),
                "prop": row["prop"],
                "c3_generated": row["c3_generated_answer"],
            }
        )
    summary = {
        "n_gate_fired_c3_parametric": int(len(failures)),
        "mean_prefill_margin": _mean(margins),
        "mean_llm_answer_prob": _mean(confidences),
        "prop_counts": dict(prop_counts.most_common()),
    }
    return summary, examples


def _stars(p_value: float | None) -> str:
    if p_value is None:
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _err(values: list[float], intervals: list[list[float]]) -> list[list[float]]:
    low = [max(0.0, value - float(bounds[0])) for value, bounds in zip(values, intervals)]
    high = [max(0.0, float(bounds[1]) - value) for value, bounds in zip(values, intervals)]
    return [low, high]


def _annotate_sig(ax, x1: float, x2: float, y: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2.0, y + 0.0125, text, ha="center", va="bottom", fontsize=9)


def _plot_killtest_full(summary: dict[str, Any], c4_summary: dict[str, Any], c5_summary: dict[str, Any], out_path: Path) -> str:
    conditions = summary["conditions"]
    order = ["C1", "C2", "C3", "C4", "C5"]
    labels = ["C1", "C2", "C3", "C4", "C5"]
    faith = [float(conditions[key]["faithfulness_rate"]) for key in order]
    faith_ci = [conditions[key]["faithfulness_rate_ci95_wilson"] for key in order]
    restore = [0.0, float(conditions["C2"]["restoration_rate"]), float(conditions["C3"]["restoration_rate"]), float(conditions["C4"]["restoration_rate"]), float(conditions["C5"]["restoration_rate"])]
    restore_ci = [[0.0, 0.0], conditions["C2"]["restoration_rate_ci95_wilson"], conditions["C3"]["restoration_rate_ci95_wilson"], conditions["C4"]["restoration_rate_ci95_wilson"], conditions["C5"]["restoration_rate_ci95_wilson"]]
    harm = [0.0, float(conditions["C2"]["harm_rate"]), float(conditions["C3"]["harm_rate"]), float(conditions["C4"]["harm_rate"]), float(conditions["C5"]["harm_rate"])]
    harm_ci = [[0.0, 0.0], conditions["C2"]["harm_rate_ci95_wilson"], conditions["C3"]["harm_rate_ci95_wilson"], conditions["C4"]["harm_rate_ci95_wilson"], conditions["C5"]["harm_rate_ci95_wilson"]]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    colors = ["#8f8f8f", "#2b8cbe", "#31a354", "#d95f02", "#756bb1"]

    x = list(range(len(order)))
    bars = axes[0].bar(x, faith, yerr=_err(faith, faith_ci), capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, min(1.05, max(faith) + 0.12))
    axes[0].set_title("Faithfulness across PopQA kill-test conditions")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, faith):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    p_c2_c3 = float(summary["mcnemar_c2_vs_c3"]["p_value"])
    p_c3_c4 = float(c4_summary["mcnemar_c3_vs_c4"]["p_value"])
    p_c3_c5 = float(c5_summary["mcnemar_c3_vs_c5"]["p_value"])
    y_max = max(faith) + 0.03
    _annotate_sig(axes[0], 1, 2, y_max, f"C2 vs C3 {_stars(p_c2_c3)}")
    _annotate_sig(axes[0], 2, 3, y_max + 0.05, f"C3 vs C4 {_stars(p_c3_c4)}")
    _annotate_sig(axes[0], 2, 4, y_max + 0.10, f"C3 vs C5 {_stars(p_c3_c5)}")

    width = 0.35
    axes[1].bar([pos - width / 2.0 for pos in x], restore, width=width, yerr=_err(restore, restore_ci), capsize=4, color="#31a354", edgecolor="black", linewidth=0.8, label="Restoration")
    axes[1].bar([pos + width / 2.0 for pos in x], harm, width=width, yerr=_err(harm, harm_ci), capsize=4, color="#de2d26", edgecolor="black", linewidth=0.8, label="Harm")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, min(1.05, max(restore + harm) + 0.15))
    axes[1].set_title("Restoration / harm relative to C2")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("PopQA kill-test full comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def _plot_by_prop(prop_rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [row["prop_key"] for row in prop_rows]
    values = [float(row["restoration_rate"]) for row in prop_rows]
    intervals = [row["restoration_rate_ci95_wilson"] for row in prop_rows]
    colors = ["#bdbdbd" if bool(row["low_n"]) else "#756bb1" for row in prop_rows]

    fig, ax = plt.subplots(figsize=(14.5, 5.0))
    x = list(range(len(prop_rows)))
    bars = ax.bar(x, values, yerr=_err(values, intervals), capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0.0, min(1.0, max(values) + 0.18 if values else 0.2))
    ax.set_title("C3 restoration by relation type")
    ax.set_ylabel("restoration_rate")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, prop_rows):
        suffix = " low-N" if bool(row["low_n"]) else ""
        ax.text(bar.get_x() + bar.get_width() / 2.0, float(row["restoration_rate"]) + 0.015, f"{float(row['restoration_rate']):.2f}{suffix}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def _plot_by_prob(prob_rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [row["label"] for row in prob_rows]
    restoration = [float(row["restoration_rate"]) for row in prob_rows]
    restoration_ci = [row["restoration_rate_ci95_wilson"] for row in prob_rows]
    harm = [float(row["harm_rate"]) for row in prob_rows]
    harm_ci = [row["harm_rate_ci95_wilson"] for row in prob_rows]
    x = list(range(len(prob_rows)))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.errorbar(x, restoration, yerr=_err(restoration, restoration_ci), marker="o", linewidth=2.0, capsize=4, color="#31a354", label="Restoration")
    ax.errorbar(x, harm, yerr=_err(harm, harm_ci), marker="s", linewidth=2.0, capsize=4, color="#de2d26", label="Harm")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, min(1.0, max(restoration + harm) + 0.15))
    ax.set_ylabel("rate")
    ax.set_title("Mechanism performance by parametric confidence bucket")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def _plot_by_popularity(pop_rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [row["label"] for row in pop_rows]
    restoration = [float(row["restoration_rate"]) for row in pop_rows]
    restoration_ci = [row["restoration_rate_ci95_wilson"] for row in pop_rows]
    harm = [float(row["harm_rate"]) for row in pop_rows]
    harm_ci = [row["harm_rate_ci95_wilson"] for row in pop_rows]
    x = list(range(len(pop_rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar([pos - width / 2.0 for pos in x], restoration, width=width, yerr=_err(restoration, restoration_ci), capsize=4, color="#31a354", edgecolor="black", linewidth=0.8, label="Restoration")
    ax.bar([pos + width / 2.0 for pos in x], harm, width=width, yerr=_err(harm, harm_ci), capsize=4, color="#de2d26", edgecolor="black", linewidth=0.8, label="Harm")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, min(1.0, max(restoration + harm) + 0.18))
    ax.set_ylabel("rate")
    ax.set_title("Mechanism performance by entity popularity bucket")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path.resolve())


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(results_dir / "summary.json")
    results = _read_jsonl(results_dir / "results.jsonl")
    if not results:
        raise RuntimeError(f"No PopQA results rows found in {results_dir / 'results.jsonl'}")
    required_result_keys = {"c4_classification", "c5_classification", "c3_gate_fired", "c3_generated_answer", "c2_parametric", "c2_context_faithful"}
    missing = sorted(key for key in required_result_keys if key not in results[0])
    if missing:
        raise RuntimeError(f"Existing results.jsonl is missing required fields for analysis: {missing}")
    if not {"C4", "C5"}.issubset(set(summary.get("conditions", {}).keys())):
        raise RuntimeError("Existing summary.json does not contain C4/C5. Re-run PopQA evaluation with the updated script wiring.")

    c4_summary = _build_c4_summary(summary, results)
    c5_summary = _build_c5_summary(summary, results)
    prop_rows = _prop_rows(results, int(args.low_n_threshold))
    prob_rows = _prob_rows(results)
    popularity_rows = _popularity_rows(results)
    failure_summary, failure_examples = _failure_mode(results, int(args.max_failure_examples))

    c2_vec = _condition_vectors(results, "C2")
    c3_vec = _condition_vectors(results, "C3")
    c4_vec = _condition_vectors(results, "C4")
    comparison_table = [
        {"condition": "C1: No context", "faithfulness": float(summary["conditions"]["C1"]["faithfulness_rate"]), "restoration": None, "harm": None, "mcnemar_vs_c3": None},
        {"condition": "C2: Short context", "faithfulness": float(summary["conditions"]["C2"]["faithfulness_rate"]), "restoration": float(summary["conditions"]["C2"]["restoration_rate"]), "harm": float(summary["conditions"]["C2"]["harm_rate"]), "mcnemar_vs_c3": _mcnemar_from_paired(c3_vec, c2_vec)},
        {"condition": "C3: Mechanism", "faithfulness": float(summary["conditions"]["C3"]["faithfulness_rate"]), "restoration": float(summary["conditions"]["C3"]["restoration_rate"]), "harm": float(summary["conditions"]["C3"]["harm_rate"]), "mcnemar_vs_c3": {"p_value": 1.0, "method": "identity"}},
        {"condition": "C4: Authoritative prompt", "faithfulness": float(summary["conditions"]["C4"]["faithfulness_rate"]), "restoration": float(summary["conditions"]["C4"]["restoration_rate"]), "harm": float(summary["conditions"]["C4"]["harm_rate"]), "mcnemar_vs_c3": c4_summary["mcnemar_c3_vs_c4"]},
        {"condition": "C5: Explicit injection", "faithfulness": float(summary["conditions"]["C5"]["faithfulness_rate"]), "restoration": float(summary["conditions"]["C5"]["restoration_rate"]), "harm": float(summary["conditions"]["C5"]["harm_rate"]), "mcnemar_vs_c3": c5_summary["mcnemar_c3_vs_c5"]},
    ]
    old_vs_new_c2 = None
    previous_summary_path = str(args.previous_summary_path).strip()
    if previous_summary_path:
        previous_summary = _read_json((PROJECT_ROOT / previous_summary_path).resolve())
        old_vs_new_c2 = {
            "previous_summary_path": str((PROJECT_ROOT / previous_summary_path).resolve()),
            "old_c2_faithfulness": float(previous_summary["conditions"]["C2"]["faithfulness_rate"]),
            "new_c2_faithfulness": float(summary["conditions"]["C2"]["faithfulness_rate"]),
        }

    stratification = {
        "by_relation_type": prop_rows,
        "by_parametric_confidence": prob_rows,
        "by_popularity": popularity_rows,
        "failure_mode": failure_summary,
        "comparison_table": comparison_table,
        "old_vs_new_c2": old_vs_new_c2,
    }

    c4_path = results_dir / "c4_summary.json"
    c5_path = results_dir / "c5_summary.json"
    strat_path = results_dir / "stratification.json"
    failure_path = results_dir / "failure_examples.json"
    _write_json(c4_path, c4_summary)
    _write_json(c5_path, c5_summary)
    _write_json(strat_path, stratification)
    _write_json(failure_path, failure_examples)

    killtest_fig = _plot_killtest_full(summary, c4_summary, c5_summary, figures_dir / "popqa_killtest_full.png")
    prop_fig = _plot_by_prop(prop_rows, figures_dir / "popqa_stratification_by_prop.png")
    prob_fig = _plot_by_prob(prob_rows, figures_dir / "popqa_stratification_by_prob.png")
    popularity_fig = _plot_by_popularity(popularity_rows, figures_dir / "popqa_stratification_by_popularity.png")

    analysis_summary = {
        "c4_summary_json": str(c4_path.resolve()),
        "c5_summary_json": str(c5_path.resolve()),
        "stratification_json": str(strat_path.resolve()),
        "failure_examples_json": str(failure_path.resolve()),
        "figures": {
            "popqa_killtest_full": killtest_fig,
            "popqa_stratification_by_prop": prop_fig,
            "popqa_stratification_by_prob": prob_fig,
            "popqa_stratification_by_popularity": popularity_fig,
        },
    }
    _write_json(results_dir / "analysis_summary.json", analysis_summary)
    print(json.dumps(analysis_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
