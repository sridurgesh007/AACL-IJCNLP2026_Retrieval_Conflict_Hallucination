from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from medrag.causal_localization import build_metadata, read_jsonl
from medrag.utils.io import read_json
from medrag.utils.stats import mcnemar_from_paired


def mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = float(sum(float(x) for x in values) / float(n))
    if n == 1:
        return mean, 0.0, mean, mean
    var = float(sum((float(x) - mean) ** 2 for x in values) / float(max(n - 1, 1)))
    std = math.sqrt(max(var, 0.0))
    half = 1.96 * (std / math.sqrt(float(n)))
    return mean, std, float(mean - half), float(mean + half)


def flatten_head_contributions(per_layer: dict[int, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in sorted(int(k) for k in per_layer.keys()):
        values = list(per_layer[int(layer)])
        for head, value in enumerate(values):
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "mean_contribution": float(value),
                }
            )
    rows.sort(key=lambda row: (float(row["mean_contribution"]), -int(row["layer"]), -int(row["head"])), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def choose_localization_candidates(head_rows: list[dict[str, Any]], *, ref_layer: int = 27, ref_head: int = 20) -> dict[str, Any]:
    if not head_rows:
        raise RuntimeError("No head rows available for candidate selection.")
    top5 = [dict(row) for row in head_rows[:5]]
    top1 = dict(top5[0])
    l27_row = next((dict(row) for row in head_rows if int(row["layer"]) == int(ref_layer) and int(row["head"]) == int(ref_head)), None)
    if l27_row is None:
        raise RuntimeError(f"Reference head L{ref_layer}/H{ref_head} missing from localization scores.")

    ambiguous = False
    if len(top5) >= 5:
        denom = max(abs(float(top1["mean_contribution"])), 1e-8)
        spread_ratio = abs(float(top5[-1]["mean_contribution"]) - float(top1["mean_contribution"])) / denom
        ambiguous = bool(spread_ratio <= 0.10)

    if int(top1["layer"]) == int(ref_layer) and int(top1["head"]) == int(ref_head):
        candidates = [top1]
        mode = "confirmed_l27_h20"
        decision_text = (
            "L27/H20 is confirmed as top-localized head on MedQA; patching null is therefore a finding "
            "about the head, not a miscalibration of the site."
        )
    elif ambiguous:
        candidates = [dict(row) for row in head_rows[:3]]
        mode = "top3_ambiguous"
        decision_text = (
            "No clear winner: the top-5 heads are within 10% of one another in mean contribution. "
            "Proceeding with the top-3 heads for causal patching."
        )
    else:
        candidates = [top1]
        mode = "new_top1"
        decision_text = (
            f"A new MedQA-localized head outranks L{ref_layer}/H{ref_head}; proceeding with "
            f"L{int(top1['layer'])}/H{int(top1['head'])} for causal patching."
        )

    return {
        "mode": str(mode),
        "candidate_heads": candidates,
        "top_5_heads": top5,
        "reference_head": l27_row,
        "decision_text": decision_text,
    }


def load_head_importance(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def read_jsonl_by_key(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            out[value] = row
    return out


def pairwise_correctness_comparison(
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
) -> dict[str, Any]:
    shared = sorted(set(left_rows.keys()) & set(right_rows.keys()))
    left_flags = [bool(left_rows[item_id].get(left_key, False)) for item_id in shared]
    right_flags = [bool(right_rows[item_id].get(right_key, False)) for item_id in shared]
    return {
        "n_items": int(len(shared)),
        "mcnemar": mcnemar_from_paired(left_flags, right_flags),
        "shared_item_ids": shared,
    }


__all__ = [
    "build_metadata",
    "choose_localization_candidates",
    "flatten_head_contributions",
    "load_head_importance",
    "mean_ci95",
    "pairwise_correctness_comparison",
    "read_jsonl",
    "read_jsonl_by_key",
]
