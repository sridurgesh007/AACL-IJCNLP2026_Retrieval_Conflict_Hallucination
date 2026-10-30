#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.data import load_examples
from medrag.counterfact_cross_model import project_path
from medrag.utils.io import write_json


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
STRATA_RULE = {
    "priority_order": ["asserted", "refusal_suppressed", "no_belief", "ambiguous"],
    "asserted": {"rank_target_true_eq": 1},
    "refusal_suppressed": {
        "is_refusal_argmax": True,
        "rank_target_true_lte": 5,
        "p_target_true_gte": 0.02,
    },
    "no_belief": {"rank_target_true_gt": 20, "or_p_target_true_lt": 0.005},
    "ambiguous": "everything_else",
    "refusal_argmax": sorted(REFUSAL_ARGMAX),
}
# Baseline-label counts for the frozen L27/H20 tau=0.0 OV file.
# Note: 1077 is the other->other transition count, not baseline_label="other".
EXPECTED_OV_COUNTS = {"other": 1110, "faithful": 611, "hallucinated": 279}
EXPECTED_N = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-set CounterFact belief stratification and audit.")
    parser.add_argument("--premise_path", default="results/rebuttal/belief_verification_full/per_item.jsonl")
    parser.add_argument(
        "--ov_results_path",
        default=(
            "../Evidence_aware/results/true_multi_head/exp_b_corrected/gate_stats/"
            "gate_stats_steering_L27H20_clip10p0_tau_0p0.jsonl"
        ),
    )
    parser.add_argument(
        "--gate_stats_dir",
        default="../Evidence_aware/results/true_multi_head/exp_b_corrected/gate_stats",
    )
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument("--original_strata_rule_path", default="auto")
    parser.add_argument("--out_dir", default="results/rebuttal/belief_verification_full")
    parser.add_argument("--ov_id_key", default="example_id")
    parser.add_argument("--ov_base_label_key", default="baseline_label")
    parser.add_argument("--ov_ft_label_key", default="label")
    parser.add_argument("--semantic_join_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
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
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_md_table(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_keys(rows: list[dict[str, Any]], keys: list[str], name: str) -> None:
    missing: dict[str, int] = {}
    for key in keys:
        count = sum(1 for row in rows if key not in row)
        if count:
            missing[key] = count
    if missing:
        raise RuntimeError(f"{name} missing required keys: {missing}")


def _as_int(value: Any, default: int = 10**9) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_refusal_argmax(row: dict[str, Any]) -> bool:
    token = str(row.get("argmax_token", "")).replace("Ġ", " ").replace("▁", " ").strip()
    token = token.strip(".,:;!?\"'()[]{}")
    return token in REFUSAL_ARGMAX


def _is_refusal_text(row: dict[str, Any]) -> bool:
    text = str(row.get("c1_generation", "")).casefold()
    return any(pattern in text for pattern in REFUSAL_PATTERNS)


def _assign_stratum(row: dict[str, Any], *, no_belief_rank_cutoff: int = 20, no_belief_p_cutoff: float = 0.005) -> str:
    rank = _as_int(row.get("rank_target_true"))
    p_true = _as_float(row.get("p_target_true"))
    if rank == 1:
        return "asserted"
    if _is_refusal_argmax(row) and rank <= 5 and p_true >= 0.02:
        return "refusal_suppressed"
    if rank > int(no_belief_rank_cutoff) or p_true < float(no_belief_p_cutoff):
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


def _gate_fired(ov: dict[str, Any]) -> bool:
    return _as_int(ov.get("n_layers_steered"), 0) > 0


def _margin(ov: dict[str, Any]) -> float | None:
    per_layer = ov.get("per_layer_margin")
    if isinstance(per_layer, dict) and "27" in per_layer:
        return _as_float(per_layer["27"])
    return None


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> dict[str, float | None]:
    try:
        from scipy.stats import fisher_exact

        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return {"odds_ratio": None if math.isinf(float(odds)) else float(odds), "p": float(p)}
    except Exception:
        return {"odds_ratio": None, "p": None, "status": "scipy_unavailable"}


def _spearman(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(xs, ys)
        return {"rho": float(rho), "p": float(p)}
    except Exception:
        return {"rho": None, "p": None, "status": "scipy_unavailable"}


def _load_cache_by_id(cache_path: Path) -> dict[str, Any]:
    return {str(int(getattr(ex, "id"))): ex for ex in load_examples(cache_path)}


def _copy_or_write_rule(out_dir: Path, original_value: str) -> dict[str, Any]:
    out = out_dir / "strata_rule_copied.json"
    candidates = [
        PROJECT_ROOT / "results/rebuttal/belief_stratification_ov_ids/strata_rule.json",
        PROJECT_ROOT / "results/rebuttal/belief_stratification/strata_rule.json",
        REPO_ROOT / "belief_stratification_ov_ids/strata_rule.json",
    ]
    original = None
    if original_value.strip().lower() != "auto":
        original = project_path(PROJECT_ROOT, original_value)
    else:
        original = next((p for p in candidates if p.exists()), None)
    if original is not None and original.exists():
        shutil.copyfile(original, out)
        same = original.read_bytes() == out.read_bytes()
        return {"original_strata_rule_path": str(original), "strata_rule_copied_path": str(out), "byte_identical": bool(same)}
    write_json(out, STRATA_RULE)
    return {
        "original_strata_rule_path": None,
        "strata_rule_copied_path": str(out),
        "byte_identical": False,
        "status": "original_not_found_written_canonical_rule",
    }


def _verify_inputs(ov_rows: list[dict[str, Any]], premise_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    _require_keys(ov_rows, [args.ov_id_key, args.ov_base_label_key, args.ov_ft_label_key], "OV rows")
    _require_keys(premise_rows, ["item_id", "query", "rank_target_true", "p_target_true", "logit_margin_c1"], "premise rows")
    ov_ids = [str(row[args.ov_id_key]) for row in ov_rows]
    premise_ids = [str(row["item_id"]) for row in premise_rows]
    counts = Counter(str(row[args.ov_base_label_key]) for row in ov_rows)
    failures = []
    if len(ov_rows) != EXPECTED_N:
        failures.append(f"expected {EXPECTED_N} OV rows, got {len(ov_rows)}")
    if len(set(ov_ids)) != EXPECTED_N:
        failures.append(f"expected {EXPECTED_N} unique OV ids, got {len(set(ov_ids))}")
    if len(premise_rows) != EXPECTED_N:
        failures.append(f"expected {EXPECTED_N} premise rows, got {len(premise_rows)}")
    for key, expected in EXPECTED_OV_COUNTS.items():
        actual = int(counts.get(key, 0))
        if actual != expected:
            failures.append(f"baseline_label {key}: expected {expected}, got {actual}")
    if set(ov_ids) != set(premise_ids):
        failures.append(f"OV/premise id sets differ: ov_only={len(set(ov_ids)-set(premise_ids))} premise_only={len(set(premise_ids)-set(ov_ids))}")
    result = {
        "n_ov_rows": len(ov_rows),
        "n_premise_rows": len(premise_rows),
        "n_unique_ov_ids": len(set(ov_ids)),
        "n_unique_premise_ids": len(set(premise_ids)),
        "baseline_label_counts": dict(counts),
        "expected_baseline_label_counts": EXPECTED_OV_COUNTS,
        "passed": not failures,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("Input verification failed: " + "; ".join(failures))
    return result


def _semantic_join_check(joined: list[dict[str, Any]], cache_by_id: dict[str, Any], n_samples: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    sample = rng.sample(joined, min(n_samples, len(joined)))
    checks = []
    mismatches = []
    for row in sample:
        item_id = row["item_id"]
        ex = cache_by_id.get(item_id)
        cache_query = "" if ex is None else str(getattr(ex, "question", ""))
        premise_query = str(row["premise"]["query"])
        match = premise_query.strip() == cache_query.strip()
        item = {"item_id": item_id, "premise_query": premise_query, "cache_query": cache_query, "match": match}
        checks.append(item)
        if not match:
            mismatches.append(item)
    result = {"n_checked": len(checks), "n_mismatch": len(mismatches), "checks": checks, "passed": not mismatches}
    if mismatches:
        raise RuntimeError(f"Semantic join check failed for {len(mismatches)} sampled rows")
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    base_h = [r for r in rows if r["base_label"] == "hallucinated"]
    base_f = [r for r in rows if r["base_label"] == "faithful"]
    base_o = [r for r in rows if r["base_label"] == "other"]
    restored = [r for r in base_h if r["ft_label"] == "faithful"]
    harmed = [r for r in base_f if r["ft_label"] != "faithful"]
    gate = [r for r in rows if r["gate_fired"]]
    gate_h = [r for r in gate if r["base_label"] == "hallucinated"]
    return {
        "n": n,
        "n_base_hallucinated": len(base_h),
        "n_base_faithful": len(base_f),
        "n_base_other": len(base_o),
        "base_hallucination_rate": len(base_h) / n if n else None,
        "base_hallucination_wilson95": _wilson(len(base_h), n),
        "n_restored": len(restored),
        "restoration": len(restored) / len(base_h) if base_h else None,
        "n_harm_strict": len(harmed),
        "harm_strict": len(harmed) / len(base_f) if base_f else None,
        "harm_strict_wilson95": _wilson(len(harmed), len(base_f)),
        "n_gate_fired": len(gate),
        "gate_precision": len(gate_h) / len(gate) if gate else None,
    }


def _harm_correction(gate_stats_dir: Path, id_key: str, base_key: str, ft_key: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(gate_stats_dir.glob("gate_stats_steering_L27H20*.jsonl")):
        data = _read_jsonl(path)
        _require_keys(data, [id_key, base_key, ft_key], str(path))
        base_h = [r for r in data if r[base_key] == "hallucinated"]
        base_f = [r for r in data if r[base_key] == "faithful"]
        restored = [r for r in base_h if r[ft_key] == "faithful"]
        harm_strict = [r for r in base_f if r[ft_key] != "faithful"]
        harm_field = [r for r in base_f if bool(r.get("harmed", False))]
        rows.append(
            {
                "file": path.name,
                "alpha": data[0].get("alpha") if data else None,
                "lambda_max": data[0].get("lambda_max") if data else None,
                "tau_margin": data[0].get("tau_margin") if data else None,
                "n_total": len(data),
                "n_restored": len(restored),
                "n_base_hallucinated": len(base_h),
                "restoration": len(restored) / len(base_h) if base_h else None,
                "n_harm_strict": len(harm_strict),
                "n_base_faithful": len(base_f),
                "harm_strict": len(harm_strict) / len(base_f) if base_f else None,
                "n_harm_field": len(harm_field),
                "harm_field": len(harm_field) / len(base_f) if base_f else None,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    premise_path = project_path(PROJECT_ROOT, args.premise_path)
    ov_path = project_path(PROJECT_ROOT, args.ov_results_path)
    cache_path = project_path(PROJECT_ROOT, args.cache_path)
    gate_stats_dir = project_path(PROJECT_ROOT, args.gate_stats_dir)
    premise_rows = _read_jsonl(premise_path)
    ov_rows = _read_jsonl(ov_path)
    input_verification = _verify_inputs(ov_rows, premise_rows, args)
    input_verification |= {"premise_path": str(premise_path), "ov_results_path": str(ov_path), "created_at_utc": _now_iso()}
    write_json(out_dir / "input_verification_stratification.json", input_verification)

    rule_copy = _copy_or_write_rule(out_dir, str(args.original_strata_rule_path))
    write_json(out_dir / "strata_rule_copy_check.json", rule_copy)

    premise_by_id = {str(row["item_id"]): row for row in premise_rows}
    ov_by_id = {str(row[args.ov_id_key]): row for row in ov_rows}
    cache_by_id = _load_cache_by_id(cache_path)

    joined = []
    for item_id in sorted(premise_by_id, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        premise = premise_by_id[item_id]
        ov = ov_by_id[item_id]
        base_label = str(ov[args.ov_base_label_key])
        ft_label = str(ov[args.ov_ft_label_key])
        joined.append(
            {
                "item_id": item_id,
                "premise": premise,
                "ov": ov,
                "stratum": _assign_stratum(premise),
                "base_label": base_label,
                "ft_label": ft_label,
                "gate_fired": _gate_fired(ov),
                "ov_prefill_margin": _margin(ov),
            }
        )

    semantic_check = _semantic_join_check(joined, cache_by_id, int(args.semantic_join_samples), int(args.seed))
    write_json(out_dir / "semantic_join_check.json", semantic_check)

    flat = []
    for row in joined:
        premise = row["premise"]
        flat.append(
            {
                "item_id": row["item_id"],
                "stratum": row["stratum"],
                "query": premise.get("query"),
                "target_true": premise.get("target_true"),
                "target_new": premise.get("target_new"),
                "rank_target_true": premise.get("rank_target_true"),
                "p_target_true": premise.get("p_target_true"),
                "logit_margin_c1": premise.get("logit_margin_c1"),
                "is_refusal_argmax": _is_refusal_argmax(premise),
                "is_refusal_text": _is_refusal_text(premise),
                "ov_base_label": row["base_label"],
                "ov_ft_label": row["ft_label"],
                "ov_prefill_margin": row["ov_prefill_margin"],
                "ov_gate_fired": row["gate_fired"],
                "ov_restored": row["base_label"] == "hallucinated" and row["ft_label"] == "faithful",
                "ov_harm_strict": row["base_label"] == "faithful" and row["ft_label"] != "faithful",
                "ov_harm_field": bool(row["ov"].get("harmed", False)),
            }
        )
    _write_jsonl(out_dir / "per_item_joined.jsonl", flat)

    by_stratum = defaultdict(list)
    for row in joined:
        by_stratum[row["stratum"]].append(row)
    strata_order = ["asserted", "refusal_suppressed", "no_belief", "ambiguous"]
    strata_table = {key: _summary(by_stratum.get(key, [])) for key in strata_order}
    write_json(out_dir / "strata_table_full.json", strata_table)
    md_rows = [{"stratum": key, **strata_table[key]} for key in strata_order]
    _write_md_table(
        out_dir / "strata_table_full.md",
        md_rows,
        [
            "stratum",
            "n",
            "n_base_hallucinated",
            "n_base_faithful",
            "n_base_other",
            "base_hallucination_rate",
            "base_hallucination_wilson95",
            "restoration",
            "harm_strict",
            "n_gate_fired",
            "gate_precision",
        ],
        "Full CounterFact Belief Stratification",
    )

    asserted = by_stratum.get("asserted", [])
    no_belief = by_stratum.get("no_belief", [])
    a_h = sum(1 for r in asserted if r["base_label"] == "hallucinated")
    a_not = len(asserted) - a_h
    n_h = sum(1 for r in no_belief if r["base_label"] == "hallucinated")
    n_not = len(no_belief) - n_h
    hallucinated_rows = [r for r in joined if r["base_label"] == "hallucinated"]
    rank_values = [_as_float(r["premise"].get("rank_target_true")) for r in joined]
    margin_values = [_as_float(r["premise"].get("logit_margin_c1")) for r in joined]
    hall_labels = [1.0 if r["base_label"] == "hallucinated" else 0.0 for r in joined]
    discriminating = {
        "n_total": len(joined),
        "n_hallucinated_no_belief": n_h,
        "n_no_belief": len(no_belief),
        "categorical_claim_holds": n_h == 0,
        "fisher_exact_asserted_vs_no_belief": _fisher_exact_two_sided(a_h, a_not, n_h, n_not),
        "spearman_rank_vs_hallucinated": _spearman(rank_values, hall_labels),
        "spearman_logit_margin_c1_vs_hallucinated": _spearman(margin_values, hall_labels),
        "hallucination_rate_by_stratum": {
            key: strata_table[key]["base_hallucination_rate"] for key in strata_order
        },
        "min_p_target_true_among_hallucinated": min((_as_float(r["premise"].get("p_target_true")) for r in hallucinated_rows), default=None),
        "max_rank_target_true_among_hallucinated": max((_as_int(r["premise"].get("rank_target_true")) for r in hallucinated_rows), default=None),
        "counts_asserted_vs_no_belief": {
            "asserted_hallucinated": a_h,
            "asserted_not_hallucinated": a_not,
            "no_belief_hallucinated": n_h,
            "no_belief_not_hallucinated": n_not,
        },
    }
    write_json(out_dir / "discriminating_test_full.json", discriminating)

    harm_rows = _harm_correction(gate_stats_dir, str(args.ov_id_key), str(args.ov_base_label_key), str(args.ov_ft_label_key))
    write_json(out_dir / "harm_correction.json", {"rows": harm_rows})

    sensitivity = []
    for rank_cutoff in (10, 15, 20, 30, 50):
        for p_floor in (0.001, 0.005, 0.01):
            rows = []
            for row in joined:
                stratum = _assign_stratum(row["premise"], no_belief_rank_cutoff=rank_cutoff, no_belief_p_cutoff=p_floor)
                if stratum == "no_belief":
                    rows.append(row)
            sensitivity.append(
                {
                    "no_belief_rank_cutoff": rank_cutoff,
                    "p_target_true_floor": p_floor,
                    "n_no_belief": len(rows),
                    "n_hallucinated_no_belief": sum(1 for row in rows if row["base_label"] == "hallucinated"),
                }
            )
    write_json(out_dir / "sensitivity_full.json", {"rows": sensitivity})

    clip20_path = gate_stats_dir / "gate_stats_steering_L27H20_clip20p0_tau_0p0.jsonl"
    clip20_by_id = {}
    if clip20_path.exists():
        clip20_by_id = {str(row[args.ov_id_key]): row for row in _read_jsonl(clip20_path)}
    unrestored = []
    for row in asserted:
        if row["base_label"] == "hallucinated" and row["ft_label"] != "faithful":
            alt = clip20_by_id.get(row["item_id"])
            unrestored.append(
                {
                    "item_id": row["item_id"],
                    "query": row["premise"].get("query"),
                    "target_true": row["premise"].get("target_true"),
                    "p_target_true": row["premise"].get("p_target_true"),
                    "rank_target_true": row["premise"].get("rank_target_true"),
                    "ov_prefill_margin": row["ov_prefill_margin"],
                    "clip20_tau0_label": None if alt is None else alt.get(args.ov_ft_label_key),
                    "clip20_tau0_restored": None if alt is None else alt.get(args.ov_ft_label_key) == "faithful",
                }
            )
    _write_jsonl(out_dir / "unrestored_asserted.jsonl", unrestored)
    unrestored_summary = {
        "n": len(unrestored),
        "median_p_target_true": median([_as_float(r["p_target_true"]) for r in unrestored]) if unrestored else None,
        "median_ov_prefill_margin": median([_as_float(r["ov_prefill_margin"]) for r in unrestored]) if unrestored else None,
        "n_restored_at_clip20_tau0": sum(1 for r in unrestored if r["clip20_tau0_restored"] is True),
    }
    write_json(out_dir / "unrestored_asserted_summary.json", unrestored_summary)

    validity = {
        "n": len(joined),
        "frac_rank_1": sum(_as_int(row["premise"].get("rank_target_true")) == 1 for row in joined) / len(joined),
        "frac_rank_over_20": sum(_as_int(row["premise"].get("rank_target_true")) > 20 for row in joined) / len(joined),
        "frac_argmax_is_refusal": sum(_is_refusal_argmax(row["premise"]) for row in joined) / len(joined),
        "frac_no_belief": len(no_belief) / len(joined),
        "stratum_counts": {key: len(by_stratum.get(key, [])) for key in strata_order},
        "base_label_by_stratum": {
            key: dict(Counter(row["base_label"] for row in by_stratum.get(key, []))) for key in strata_order
        },
    }
    write_json(out_dir / "benchmark_validity_full.json", validity)

    readme = (
        "# Full CounterFact Belief Verification\n\n"
        f"Categorical claim holds: {discriminating['categorical_claim_holds']} "
        f"with n_hallucinated_no_belief={discriminating['n_hallucinated_no_belief']} / "
        f"{discriminating['n_no_belief']}.\n"
        f"Max rank_target_true among hallucinated items: {discriminating['max_rank_target_true_among_hallucinated']}.\n"
        f"Strict harm for the frozen tau=0/lambda_max=10 point is reported in `harm_correction.json`; "
        f"the historical `harmed` field is not used for strict harm.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
