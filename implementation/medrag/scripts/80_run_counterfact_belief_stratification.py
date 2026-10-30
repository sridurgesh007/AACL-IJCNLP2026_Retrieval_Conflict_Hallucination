#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


REFUSAL_ARGMAX = {"I", "There", "Unfortunately", "Sorry", "As", "Without", "It"}
REFUSAL_PATTERNS = [
    "i don't have",
    "i couldn't find",
    "i cannot",
    "i can't",
    "i do not have",
    "not enough information",
    "i'm not aware",
    "i am not aware",
    "i don't know",
    "unable to",
    "i'm sorry",
    "no information",
    "i need more",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Belief-stratified re-analysis of CounterFact OV steering for rebuttal."
    )
    parser.add_argument(
        "--premise_path",
        default="auto",
        help="Premise-check v2 per_item.jsonl. Use 'auto' to resolve known locations.",
    )
    parser.add_argument(
        "--ov_results_path",
        default="auto",
        help=(
            "CounterFact OV-steering per-item JSONL. Use 'auto' for the prior L27/H20 "
            "gate-stats path; pass an explicit path if the file is stored elsewhere."
        ),
    )
    parser.add_argument("--out_dir", default="results/rebuttal/belief_stratification")
    parser.add_argument("--strict_join", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_examples", type=int, default=6)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}; got {type(row).__name__}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_path(value: str, candidates: list[Path]) -> Path:
    if str(value).strip().lower() != "auto":
        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Explicit path does not exist: {path}")
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve auto path. Checked:\n{checked}")


def _premise_candidates() -> list[Path]:
    return [
        (PROJECT_ROOT / "results/rebuttal/premise_check_v2/per_item.jsonl").resolve(),
        (REPO_ROOT / "premise_check_v2/per_item.jsonl").resolve(),
        (REPO_ROOT / "results/rebuttal/premise_check_v2/per_item.jsonl").resolve(),
    ]


def _ov_candidates() -> list[Path]:
    fixed = [
        (
            REPO_ROOT
            / "true_multi_head/exp_b_corrected/gate_stats/"
            "gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl"
        ).resolve(),
        (
            REPO_ROOT
            / "true_multi_head/exp_b_corrected/gate_stats/"
            "gate_stats_steering_L27H20_clip10p0_tau_0.jsonl"
        ).resolve(),
        (
            REPO_ROOT
            / "Evidence_aware/results/true_multi_head/exp_b_corrected/gate_stats/"
            "gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl"
        ).resolve(),
        (
            REPO_ROOT
            / "Evidence_aware/results/true_multi_head/exp_b_corrected/gate_stats/"
            "gate_stats_steering_L27H20_clip10p0_tau_0.jsonl"
        ).resolve(),
    ]
    discovered = sorted(
        REPO_ROOT.glob("**/gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl")
    ) + sorted(REPO_ROOT.glob("**/gate_stats_steering_L27H20_clip10p0_tau_0.jsonl"))
    out: list[Path] = []
    seen: set[str] = set()
    for path in fixed + discovered:
        key = str(path.resolve())
        if key not in seen:
            out.append(path.resolve())
            seen.add(key)
    return out


def _id_value(row: dict[str, Any]) -> Any:
    for key in ("item_id", "example_id", "id", "source_index", "idx"):
        if key in row:
            return row[key]
    raise KeyError("row has no item_id/example_id/id/source_index/idx")


def _id_type(rows: list[dict[str, Any]]) -> str:
    types = sorted({type(_id_value(row)).__name__ for row in rows[: min(len(rows), 50)]})
    return "|".join(types) if types else "empty"


def _norm_id(value: Any) -> str:
    return str(value)


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def _lower_label(value: Any) -> str:
    return str(value if value is not None else "").strip().lower()


def _is_base_hallucinated(row: dict[str, Any]) -> bool:
    explicit = _first_present(
        row,
        (
            "base_hallucinated",
            "baseline_hallucinated",
            "c2_hallucinated",
            "base_is_hallucinated",
        ),
    )
    if explicit is not None:
        return bool(explicit)
    label = _lower_label(
        _first_present(
            row,
            (
                "base_label",
                "baseline_label",
                "c2_label",
                "base_classification",
                "baseline_classification",
            ),
        )
    )
    return label in {"hallucinated", "parametric", "wrong", "base_wrong"}


def _is_base_faithful(row: dict[str, Any]) -> bool:
    explicit = _first_present(
        row,
        ("base_faithful", "baseline_faithful", "c2_faithful", "base_is_faithful"),
    )
    if explicit is not None:
        return bool(explicit)
    label = _lower_label(
        _first_present(
            row,
            (
                "base_label",
                "baseline_label",
                "c2_label",
                "base_classification",
                "baseline_classification",
            ),
        )
    )
    return label in {"faithful", "context_faithful", "correct"}


def _is_restored(row: dict[str, Any]) -> bool:
    explicit = _first_present(row, ("restored", "is_restored", "ft_restored", "c3_restored"))
    if explicit is not None:
        return bool(explicit)
    ft_label = _lower_label(
        _first_present(row, ("ft_label", "steered_label", "c3_label", "intervention_label", "label"))
    )
    return _is_base_hallucinated(row) and ft_label in {"faithful", "context_faithful", "correct"}


def _is_harmed(row: dict[str, Any]) -> bool:
    explicit = _first_present(row, ("harmed", "is_harmed", "ft_harmed", "c3_harmed"))
    if explicit is not None:
        return bool(explicit)
    ft_label = _lower_label(
        _first_present(row, ("ft_label", "steered_label", "c3_label", "intervention_label", "label"))
    )
    return _is_base_faithful(row) and ft_label not in {"faithful", "context_faithful", "correct"}


def _gate_fired(row: dict[str, Any]) -> bool:
    explicit = _first_present(row, ("gate_fired", "intervened", "steered", "gate"))
    if explicit is not None:
        return bool(explicit)
    for key in ("n_layers_steered", "n_fired_layers"):
        if key in row:
            try:
                return int(row[key]) > 0
            except (TypeError, ValueError):
                return False
    return False


def _prefill_margin(row: dict[str, Any]) -> float | None:
    value = _first_present(row, ("prefill_margin", "margin", "c2_margin", "baseline_margin"))
    if value is None:
        per_layer = row.get("per_layer_margin")
        if isinstance(per_layer, dict):
            value = _first_present(per_layer, ("27", 27, "L27H20", "L27"))
    if value is None:
        per_layer = row.get("per_layer")
        if isinstance(per_layer, dict):
            layer_payload = _first_present(per_layer, ("27", 27))
            if isinstance(layer_payload, dict):
                value = _first_present(layer_payload, ("margin", "prefill_margin"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_text(row: dict[str, Any]) -> str:
    return str(_first_present(row, ("query", "question", "prompt"), ""))


def _as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(row: dict[str, Any], key: str, default: int = 10**9) -> int:
    value = row.get(key)
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_refusal_argmax(row: dict[str, Any]) -> bool:
    return str(row.get("argmax_token", "")).strip() in REFUSAL_ARGMAX


def _is_refusal_text(row: dict[str, Any]) -> bool:
    text = str(row.get("c1_generation", "")).casefold()
    return any(pattern in text for pattern in REFUSAL_PATTERNS)


def _assign_stratum(
    row: dict[str, Any],
    *,
    asserted_rank_cutoff: int = 1,
    refusal_rank_cutoff: int = 5,
    refusal_p_cutoff: float = 0.02,
    no_belief_rank_cutoff: int = 20,
    no_belief_p_cutoff: float = 0.005,
) -> str:
    rank = _as_int(row, "rank_target_true")
    prob = _as_float(row, "p_target_true")
    if rank <= int(asserted_rank_cutoff):
        return "asserted"
    if _is_refusal_argmax(row) and rank <= int(refusal_rank_cutoff) and prob >= float(refusal_p_cutoff):
        return "refusal_suppressed"
    if rank > int(no_belief_rank_cutoff) or prob < float(no_belief_p_cutoff):
        return "no_belief"
    return "ambiguous"


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if n <= 0:
        return {"low": None, "high": None}
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) / n) + (z * z / (4.0 * n * n))) / denom
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    sorted_pairs = sorted(pairs, key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            if sorted_pairs[idx][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    base_h = [row for row in rows if _is_base_hallucinated(row["ov"])]
    base_f = [row for row in rows if _is_base_faithful(row["ov"])]
    restored = [row for row in base_h if _is_restored(row["ov"])]
    harmed = [row for row in base_f if _is_harmed(row["ov"])]
    gate = [row for row in rows if _gate_fired(row["ov"])]
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        margin = _prefill_margin(row["ov"])
        if margin is None:
            continue
        if _is_base_faithful(row["ov"]):
            labels.append(1)
            scores.append(float(margin))
        elif _is_base_hallucinated(row["ov"]):
            labels.append(0)
            scores.append(float(margin))
    restoration_n = len(base_h)
    harm_n = len(base_f)
    restoration_k = len(restored)
    harm_k = len(harmed)
    ld_ratio = _ld_ratio(rows)
    return {
        "n": n,
        "n_base_hallucinated": restoration_n,
        "n_base_faithful": harm_n,
        "n_restored": restoration_k,
        "n_harmed": harm_k,
        "restoration": restoration_k / restoration_n if restoration_n else None,
        "restoration_wilson95": _wilson(restoration_k, restoration_n),
        "harm": harm_k / harm_n if harm_n else None,
        "harm_wilson95": _wilson(harm_k, harm_n),
        "n_gate_fired": len(gate),
        "gate_fire_rate": len(gate) / n if n else None,
        "prefill_margin_auc": _auc(scores, labels),
        "L27H20_LD_ratio": ld_ratio["ratio"],
        "L27H20_LD_ratio_status": ld_ratio["status"],
    }


def _ld_value(row: dict[str, Any]) -> float | None:
    keys = (
        "L27H20_logit_difference",
        "L27H20_ld",
        "l27h20_logit_difference",
        "l27h20_ld",
        "head_L27H20_ld",
        "ld_L27H20",
    )
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _ld_ratio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    faithful: list[float] = []
    hallucinated: list[float] = []
    for row in rows:
        value = _ld_value(row["ov"])
        if value is None:
            continue
        if _is_base_faithful(row["ov"]):
            faithful.append(abs(float(value)))
        elif _is_base_hallucinated(row["ov"]):
            hallucinated.append(abs(float(value)))
    if not faithful and not hallucinated:
        return {"ratio": None, "status": "not_available_in_ov_results"}
    if not faithful or not hallucinated:
        return {"ratio": None, "status": "insufficient_labels_for_ratio"}
    denom = statistics.mean(hallucinated)
    if denom == 0.0:
        return {"ratio": None, "status": "zero_hallucinated_mean"}
    return {"ratio": statistics.mean(faithful) / denom, "status": "computed"}


def _quartile_edges(values: list[float]) -> tuple[float, float, float]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return (float("nan"), float("nan"), float("nan"))

    def q(p: float) -> float:
        if len(values) == 1:
            return values[0]
        pos = p * (len(values) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return values[int(pos)]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return (q(0.25), q(0.50), q(0.75))


def _quartile_label(value: float, edges: tuple[float, float, float]) -> str:
    q1, q2, q3 = edges
    if not math.isfinite(value):
        return "missing"
    if value <= q1:
        return "q1_lowest_belief"
    if value <= q2:
        return "q2"
    if value <= q3:
        return "q3"
    return "q4_highest_belief"


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> dict[str, float | None]:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    if row1 == 0 or row2 == 0 or col1 == 0 or col1 == total:
        return {"odds_ratio": None, "p": None}

    def hypergeom(x: int) -> float:
        return math.exp(
            math.lgamma(row1 + 1)
            - math.lgamma(x + 1)
            - math.lgamma(row1 - x + 1)
            + math.lgamma(row2 + 1)
            - math.lgamma(col1 - x + 1)
            - math.lgamma(row2 - (col1 - x) + 1)
            - (math.lgamma(total + 1) - math.lgamma(col1 + 1) - math.lgamma(total - col1 + 1))
        )

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    observed = hypergeom(a)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hypergeom(x)
        if px <= observed + 1e-12:
            p += px
    odds = None if b * c == 0 else (a * d) / (b * c)
    if odds is None and a * d > 0 and b * c == 0:
        odds = float("inf")
    return {"odds_ratio": odds, "p": min(1.0, p)}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return None
    return num / (denx * deny)


def _spearman_quartiles(quartile_summaries: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    keys = ["q1_lowest_belief", "q2", "q3", "q4_highest_belief"]
    ys: list[float] = []
    xs: list[float] = []
    for idx, key in enumerate(keys, start=1):
        value = quartile_summaries.get(key, {}).get("restoration")
        if value is None:
            continue
        xs.append(float(idx))
        ys.append(float(value))
    rho = _pearson(xs, ys)
    if rho is None:
        return {"rho": None, "p": None}
    count = 0
    extreme = 0
    for perm in set(itertools.permutations(ys)):
        r = _pearson(xs, list(perm))
        if r is None:
            continue
        count += 1
        if abs(r) >= abs(rho) - 1e-12:
            extreme += 1
    return {"rho": rho, "p": extreme / count if count else None}


def _restoration_count(rows: list[dict[str, Any]]) -> tuple[int, int]:
    base_h = [row for row in rows if _is_base_hallucinated(row["ov"])]
    restored = [row for row in base_h if _is_restored(row["ov"])]
    return len(restored), len(base_h)


def _validity_examples(rows: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    wanted_patterns = [
        "Michel Denisot",
        "Go Hyeon-jeong",
        "Anaal Nathrakh",
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in wanted_patterns:
        for row in rows:
            if pattern in str(row["premise"].get("query", "")):
                selected.append(_example_row(row))
                seen.add(row["item_id"])
                break
    for row in rows:
        if len(selected) >= max_examples:
            break
        if row["item_id"] in seen:
            continue
        premise = row["premise"]
        if _is_refusal_text(premise) or _is_refusal_argmax(premise) or row["stratum"] == "no_belief":
            selected.append(_example_row(row))
            seen.add(row["item_id"])
    return selected[:max_examples]


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    premise = row["premise"]
    ov = row["ov"]
    return {
        "item_id": row["item_id"],
        "stratum": row["stratum"],
        "query": premise.get("query", ""),
        "target_true": premise.get("target_true", ""),
        "target_new": premise.get("target_new", ""),
        "c1_generation": premise.get("c1_generation", ""),
        "rank_target_true": premise.get("rank_target_true"),
        "p_target_true": premise.get("p_target_true"),
        "argmax_token": premise.get("argmax_token"),
        "base_label": _first_present(ov, ("base_label", "baseline_label", "c2_label"), ""),
        "ft_label": _first_present(ov, ("ft_label", "steered_label", "c3_label"), ""),
        "restored": _is_restored(ov),
        "harmed": _is_harmed(ov),
    }


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("\n", " ")[:240])
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_join(
    premise_rows: list[dict[str, Any]], ov_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    premise_by_id = {_norm_id(_id_value(row)): row for row in premise_rows}
    ov_by_id = {_norm_id(_id_value(row)): row for row in ov_rows}
    premise_ids = set(premise_by_id)
    ov_ids = set(ov_by_id)
    joined_ids = sorted(premise_ids & ov_ids, key=lambda x: (len(x), x))
    joined = [
        {
            "item_id": item_id,
            "premise": premise_by_id[item_id],
            "ov": ov_by_id[item_id],
        }
        for item_id in joined_ids
    ]
    semantic_examples = []
    for row in joined[:5]:
        semantic_examples.append(
            {
                "item_id": row["item_id"],
                "premise_query": _query_text(row["premise"]),
                "ov_query": _query_text(row["ov"]),
                "premise_target_true": row["premise"].get("target_true"),
                "ov_correct_answer": _first_present(row["ov"], ("correct_answer", "answer_liquid"), ""),
                "ov_parametric_answer": _first_present(row["ov"], ("parametric_answer", "answer_solid"), ""),
            }
        )
    merge = {
        "n_premise": len(premise_rows),
        "n_ov_steering": len(ov_rows),
        "n_joined": len(joined),
        "n_premise_only": len(premise_ids - ov_ids),
        "n_ov_only": len(ov_ids - premise_ids),
        "id_type_premise": _id_type(premise_rows),
        "id_type_ov": _id_type(ov_rows),
        "normalization": "IDs are reported by raw type above, then joined on str(item_id).",
        "join_rate": len(joined) / max(len(premise_rows), 1),
        "semantic_examples": semantic_examples,
        "premise_only_sample": sorted(premise_ids - ov_ids)[:10],
        "ov_only_sample": sorted(ov_ids - premise_ids)[:10],
    }
    return joined, merge


def _freeze_rule(path: Path) -> dict[str, Any]:
    rule = {
        "frozen_at_utc": _now_iso(),
        "rule_status": "frozen_before_outcome_metrics",
        "refusal_argmax": sorted(REFUSAL_ARGMAX),
        "refusal_patterns": REFUSAL_PATTERNS,
        "strata": {
            "asserted": "rank_target_true == 1",
            "refusal_suppressed": "is_refusal_argmax AND rank_target_true <= 5 AND p_target_true >= 0.02",
            "no_belief": "rank_target_true > 20 OR p_target_true < 0.005",
            "ambiguous": "everything else",
        },
        "priority_order": ["asserted", "refusal_suppressed", "no_belief", "ambiguous"],
        "continuous_alternative": "logit_margin_c1 quartiles, reported alongside frozen categorical strata",
    }
    write_json(path, rule)
    return rule


def _sensitivity(joined: list[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for asserted_rank in (1, 2):
        for refusal_rank in (4, 5, 6):
            for refusal_p in (0.015, 0.020, 0.025):
                for no_belief_rank in (19, 20, 21):
                    for no_belief_p in (0.0, 0.005, 0.010):
                        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                        for row in joined:
                            stratum = _assign_stratum(
                                row["premise"],
                                asserted_rank_cutoff=asserted_rank,
                                refusal_rank_cutoff=refusal_rank,
                                refusal_p_cutoff=refusal_p,
                                no_belief_rank_cutoff=no_belief_rank,
                                no_belief_p_cutoff=no_belief_p,
                            )
                            groups[stratum].append(row)
                        values = {
                            key: _summary_for_rows(groups.get(key, [])).get("restoration")
                            for key in ("asserted", "refusal_suppressed", "ambiguous", "no_belief")
                        }
                        ordered = [
                            values.get("no_belief"),
                            values.get("ambiguous"),
                            values.get("refusal_suppressed"),
                            values.get("asserted"),
                        ]
                        finite = [x for x in ordered if x is not None]
                        monotone = bool(finite == sorted(finite)) if finite else False
                        runs.append(
                            {
                                "asserted_rank_cutoff": asserted_rank,
                                "refusal_rank_cutoff": refusal_rank,
                                "refusal_p_cutoff": refusal_p,
                                "no_belief_rank_cutoff": no_belief_rank,
                                "no_belief_p_cutoff": no_belief_p,
                                "restoration": values,
                                "monotone_increasing": monotone,
                            }
                        )
    return {
        "n_settings": len(runs),
        "n_monotone_increasing": sum(1 for row in runs if row["monotone_increasing"]),
        "frac_monotone_increasing": sum(1 for row in runs if row["monotone_increasing"]) / max(len(runs), 1),
        "settings": runs,
    }


def main() -> int:
    args = parse_args()
    out_dir = (PROJECT_ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    premise_path = _resolve_path(str(args.premise_path), _premise_candidates())
    ov_path = _resolve_path(str(args.ov_results_path), _ov_candidates())

    premise_rows = _read_jsonl(premise_path)
    ov_rows = _read_jsonl(ov_path)
    joined, merge_check = _build_join(premise_rows, ov_rows)
    merge_check.update(
        {
            "premise_path": str(premise_path),
            "ov_results_path": str(ov_path),
            "created_at_utc": _now_iso(),
        }
    )
    write_json(out_dir / "merge_check.json", merge_check)

    if bool(int(args.strict_join)) and float(merge_check["join_rate"]) < 1.0:
        readme = (
            "# CounterFact belief stratification\n\n"
            "Stopped before outcome analysis because the premise-check rows did not fully join "
            "to the OV-steering rows.\n\n"
            f"- premise rows: {merge_check['n_premise']}\n"
            f"- OV rows: {merge_check['n_ov_steering']}\n"
            f"- joined rows: {merge_check['n_joined']}\n"
            f"- join rate: {merge_check['join_rate']:.4f}\n"
            f"- premise path: `{premise_path}`\n"
            f"- OV path: `{ov_path}`\n"
        )
        (out_dir / "README.md").write_text(readme, encoding="utf-8")
        print(json.dumps(merge_check, indent=2))
        print(f"STOP: join_rate={merge_check['join_rate']:.4f} < 1.0")
        return 2

    _freeze_rule(out_dir / "strata_rule.json")

    margins = [_as_float(row["premise"], "logit_margin_c1") for row in joined]
    quartile_edges = _quartile_edges(margins)

    refusal_argmax_count = 0
    refusal_text_count = 0
    refusal_agree_count = 0
    for row in joined:
        premise = row["premise"]
        stratum = _assign_stratum(premise)
        qlabel = _quartile_label(_as_float(premise, "logit_margin_c1"), quartile_edges)
        is_argmax = _is_refusal_argmax(premise)
        is_text = _is_refusal_text(premise)
        refusal_argmax_count += int(is_argmax)
        refusal_text_count += int(is_text)
        refusal_agree_count += int(is_argmax == is_text)
        row["stratum"] = stratum
        row["logit_margin_c1_quartile"] = qlabel
        row["is_refusal_argmax"] = is_argmax
        row["is_refusal_text"] = is_text

    flat_rows = []
    for row in joined:
        flat = {
            "item_id": row["item_id"],
            "stratum": row["stratum"],
            "logit_margin_c1_quartile": row["logit_margin_c1_quartile"],
            "is_refusal_argmax": row["is_refusal_argmax"],
            "is_refusal_text": row["is_refusal_text"],
            "query": row["premise"].get("query", ""),
            "target_true": row["premise"].get("target_true", ""),
            "target_new": row["premise"].get("target_new", ""),
            "rank_target_true": row["premise"].get("rank_target_true"),
            "p_target_true": row["premise"].get("p_target_true"),
            "logit_margin_c1": row["premise"].get("logit_margin_c1"),
            "argmax_token": row["premise"].get("argmax_token"),
            "c1_generation": row["premise"].get("c1_generation", ""),
            "ov_base_label": _first_present(row["ov"], ("base_label", "baseline_label", "c2_label"), ""),
            "ov_ft_label": _first_present(row["ov"], ("ft_label", "steered_label", "c3_label"), ""),
            "ov_prefill_margin": _prefill_margin(row["ov"]),
            "ov_gate_fired": _gate_fired(row["ov"]),
            "ov_restored": _is_restored(row["ov"]),
            "ov_harmed": _is_harmed(row["ov"]),
        }
        flat_rows.append(flat)
    write_jsonl(out_dir / "per_item_joined.jsonl", flat_rows)

    strata_order = ["asserted", "refusal_suppressed", "ambiguous", "no_belief"]
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_quartile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_stratum[row["stratum"]].append(row)
        by_quartile[row["logit_margin_c1_quartile"]].append(row)

    strata_table = {key: _summary_for_rows(by_stratum.get(key, [])) for key in strata_order}
    quartile_order = ["q1_lowest_belief", "q2", "q3", "q4_highest_belief", "missing"]
    quartile_table = {key: _summary_for_rows(by_quartile.get(key, [])) for key in quartile_order}
    quartile_table["_edges"] = {
        "q1": quartile_edges[0],
        "q2": quartile_edges[1],
        "q3": quartile_edges[2],
    }
    write_json(out_dir / "strata_table.json", strata_table)
    write_json(out_dir / "quartile_table.json", quartile_table)

    md_rows = []
    for key in strata_order:
        row = {"stratum": key}
        row.update(strata_table[key])
        md_rows.append(row)
    _write_md_table(
        out_dir / "strata_table.md",
        "Belief strata outcome table",
        md_rows,
        [
            "stratum",
            "n",
            "n_base_hallucinated",
            "n_base_faithful",
            "restoration",
            "harm",
            "n_gate_fired",
            "gate_fire_rate",
            "prefill_margin_auc",
            "L27H20_LD_ratio_status",
        ],
    )

    a_rest, a_base = _restoration_count(by_stratum.get("asserted", []))
    n_rest, n_base = _restoration_count(by_stratum.get("no_belief", []))
    fisher = _fisher_exact_two_sided(a_rest, a_base - a_rest, n_rest, n_base - n_rest)
    ordered_restoration = [
        strata_table["no_belief"].get("restoration"),
        strata_table["ambiguous"].get("restoration"),
        strata_table["refusal_suppressed"].get("restoration"),
        strata_table["asserted"].get("restoration"),
    ]
    finite_rest = [x for x in ordered_restoration if x is not None]
    discriminating = {
        "prediction_frozen_before_run": (
            "Evidence-routing repair predicts restoration increases with belief strength; "
            "token-editing predicts flat restoration across belief strata."
        ),
        "restoration_asserted": strata_table["asserted"].get("restoration"),
        "restoration_refusal_suppressed": strata_table["refusal_suppressed"].get("restoration"),
        "restoration_ambiguous": strata_table["ambiguous"].get("restoration"),
        "restoration_no_belief": strata_table["no_belief"].get("restoration"),
        "restoration_monotone_in_belief": bool(finite_rest == sorted(finite_rest)) if finite_rest else False,
        "fisher_exact_asserted_vs_no_belief": fisher,
        "spearman_restoration_vs_logit_margin_c1_quartile": _spearman_quartiles(
            {k: v for k, v in quartile_table.items() if not k.startswith("_")}
        ),
        "counts_asserted_vs_no_belief": {
            "asserted_restored": a_rest,
            "asserted_not_restored": a_base - a_rest,
            "no_belief_restored": n_rest,
            "no_belief_not_restored": n_base - n_rest,
        },
    }
    write_json(out_dir / "discriminating_test.json", discriminating)
    write_json(out_dir / "sensitivity_analysis.json", _sensitivity(joined))

    validity = {
        "n": len(joined),
        "frac_argmax_is_refusal": refusal_argmax_count / max(len(joined), 1),
        "frac_refusal_text": refusal_text_count / max(len(joined), 1),
        "refusal_argmax_text_agreement": refusal_agree_count / max(len(joined), 1),
        "refusal_argmax_text_discrepancy_count": len(joined) - refusal_agree_count,
        "frac_rank_1": sum(_as_int(row["premise"], "rank_target_true") == 1 for row in joined) / max(len(joined), 1),
        "frac_rank_over_20": sum(_as_int(row["premise"], "rank_target_true") > 20 for row in joined) / max(len(joined), 1),
        "frac_no_belief_stratum": len(by_stratum.get("no_belief", [])) / max(len(joined), 1),
        "stratum_counts": {key: len(by_stratum.get(key, [])) for key in strata_order},
        "rank_bucket_counts": dict(
            Counter(
                "rank_1"
                if _as_int(row["premise"], "rank_target_true") == 1
                else "rank_2_5"
                if _as_int(row["premise"], "rank_target_true") <= 5
                else "rank_6_20"
                if _as_int(row["premise"], "rank_target_true") <= 20
                else "rank_over_20"
                for row in joined
            )
        ),
        "refusal_with_high_belief": {
            "n": sum(
                (_is_refusal_argmax(row["premise"]) or _is_refusal_text(row["premise"]))
                and _as_int(row["premise"], "rank_target_true") <= 20
                for row in joined
            ),
            "note": "argmax or generated text is refusal-like but target_true ranks <= 20",
        },
    }
    write_json(out_dir / "counterfact_validity.json", validity)

    examples = _validity_examples(joined, int(args.max_examples))
    _write_md_table(
        out_dir / "validity_examples.md",
        "CounterFact validity examples",
        examples,
        [
            "item_id",
            "stratum",
            "query",
            "target_true",
            "target_new",
            "c1_generation",
            "rank_target_true",
            "p_target_true",
            "base_label",
            "ft_label",
            "restored",
        ],
    )

    readme = (
        "# CounterFact belief stratification\n\n"
        f"Join rate was {merge_check['join_rate']:.4f} "
        f"({merge_check['n_joined']}/{merge_check['n_premise']} premise rows) using `{ov_path}`.\n"
        f"Stratum sizes: {', '.join(f'{k}={len(by_stratum.get(k, []))}' for k in strata_order)}.\n"
        "Restoration by stratum: "
        + ", ".join(
            f"{k}={strata_table[k].get('restoration')}" for k in strata_order
        )
        + ".\n"
        f"Monotone-in-belief result: {discriminating['restoration_monotone_in_belief']}.\n"
        f"Refusal population confirmed: argmax-refusal fraction={validity['frac_argmax_is_refusal']:.4f}; "
        f"text-refusal fraction={validity['frac_refusal_text']:.4f}.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "merge_check": merge_check, "discriminating": discriminating}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
