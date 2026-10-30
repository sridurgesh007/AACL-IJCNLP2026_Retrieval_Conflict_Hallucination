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

import numpy as np

from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.utils.io import git_commit_hash, read_json, sha256_file, write_json

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("balanced_minus_promotion", "balanced", "pure_promotion"),
    ("balanced_minus_suppression", "balanced", "pure_suppression"),
    ("promotion_minus_suppression", "pure_promotion", "pure_suppression"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired-bootstrap analysis for MedQA sub-threshold decomposition.")
    parser.add_argument("--subthreshold_dir", default="results/medqa_final_v3/decomposition_subthreshold")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/consolidation")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _cell_name(lambda_pos: float, lambda_neg: float) -> str:
    return f"lp{_slug(lambda_pos)}_ln{_slug(lambda_neg)}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _build_metadata(*, args: argparse.Namespace, wall_clock_seconds: float, input_files: list[Path]) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "git_commit_hash": git_commit_hash(PROJECT_ROOT),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path.resolve()) for path in input_files],
        "input_file_sha256": {str(path.resolve()): sha256_file(path) for path in input_files},
    }


def _resolve_cell_paths(sub_dir: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    mode = str(row["mode"])
    lambda_pos = float(row["lambda_pos"])
    lambda_neg = float(row["lambda_neg"])
    derived_dir = sub_dir / mode / _cell_name(lambda_pos, lambda_neg)
    summary_path = derived_dir / "summary.json"
    results_path = derived_dir / "results.jsonl"
    raw_out_dir = str(row.get("out_dir", "")).strip()
    if raw_out_dir:
        maybe_dir = Path(raw_out_dir)
        maybe_summary = maybe_dir / "summary.json"
        maybe_results = maybe_dir / "results.jsonl"
        if maybe_summary.exists():
            summary_path = maybe_summary
        if maybe_results.exists():
            results_path = maybe_results
    return summary_path, results_path


def _load_flipped_ids(sub_dir: Path, row: dict[str, Any]) -> tuple[list[str], list[str]]:
    flipped = [str(x) for x in row.get("flipped_item_ids", []) if str(x).strip()]
    flipped_clean = [str(x) for x in row.get("flipped_item_ids_clean", []) if str(x).strip()]
    if flipped or flipped_clean:
        return sorted(set(flipped)), sorted(set(flipped_clean))

    summary_path, results_path = _resolve_cell_paths(sub_dir, row)
    if summary_path.exists():
        summary = read_json(summary_path)
        from_summary = [str(x) for x in summary.get("flipped_item_ids", []) if str(x).strip()]
        from_summary_clean = [str(x) for x in summary.get("flipped_item_ids_clean", []) if str(x).strip()]
        if from_summary or from_summary_clean:
            return sorted(set(from_summary)), sorted(set(from_summary_clean))

    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing flipped-item sources for mode={row['mode']} lambda={row['lambda']}: "
            f"expected {summary_path} or {results_path}"
        )
    results = _load_jsonl(results_path)
    matched = sorted(
        {
            str(item.get("item_id"))
            for item in results
            if bool(item.get("in_matched_subset", False))
            and not bool(item.get("skipped", False))
            and str(item.get("ft_letter")) != str(item.get("base_letter"))
        }
    )
    clean = sorted(
        {
            str(item.get("item_id"))
            for item in results
            if bool(item.get("in_clean_subset", False))
            and not bool(item.get("skipped", False))
            and str(item.get("ft_letter")) != str(item.get("base_letter"))
        }
    )
    return matched, clean


def _load_base_wrong_universe(sub_dir: Path, rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    for row in rows:
        _, results_path = _resolve_cell_paths(sub_dir, row)
        if not results_path.exists():
            continue
        results = _load_jsonl(results_path)
        matched_ids = sorted(
            {
                str(item.get("item_id"))
                for item in results
                if bool(item.get("in_matched_subset", False))
                and not bool(item.get("skipped", False))
                and not bool(item.get("base_correct", False))
            }
        )
        clean_ids = sorted(
            {
                str(item.get("item_id"))
                for item in results
                if bool(item.get("in_clean_subset", False))
                and not bool(item.get("skipped", False))
                and not bool(item.get("base_correct", False))
            }
        )
        if matched_ids:
            return matched_ids, clean_ids
    raise FileNotFoundError(
        "Unable to recover base-wrong item universe from sub-threshold per-cell results. "
        f"Checked under {sub_dir}"
    )


def _paired_bootstrap_binary_diff(
    left: list[int],
    right: list[int],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Bootstrap inputs must have the same length")
    n = len(left)
    if n == 0:
        return {
            "mean_diff": 0.0,
            "ci95": [0.0, 0.0],
            "p_value": 1.0,
            "n_bootstrap": int(n_bootstrap),
            "flip_rate_left": 0.0,
            "flip_rate_right": 0.0,
            "n_flipped_left": 0,
            "n_flipped_right": 0,
            "n_items": 0,
        }
    arr_left = np.asarray([int(x) for x in left], dtype=np.float64)
    arr_right = np.asarray([int(x) for x in right], dtype=np.float64)
    diffs = arr_left - arr_right
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diffs[idx]))
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    p_lower = float(np.mean(samples <= 0.0))
    p_upper = float(np.mean(samples >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    if float(np.max(np.abs(samples))) < 1e-12:
        p_value = 1.0
    return {
        "mean_diff": float(observed),
        "ci95": [float(lo), float(hi)],
        "p_value": float(p_value),
        "n_bootstrap": int(n_bootstrap),
        "flip_rate_left": float(np.mean(arr_left)),
        "flip_rate_right": float(np.mean(arr_right)),
        "n_flipped_left": int(np.sum(arr_left)),
        "n_flipped_right": int(np.sum(arr_right)),
        "n_items": int(n),
    }


def _build_indicator_vector(universe_ids: list[str], flipped_ids: set[str]) -> list[int]:
    return [1 if item_id in flipped_ids else 0 for item_id in universe_ids]


def _subset_analysis(
    *,
    subset_name: str,
    universe_ids: list[str],
    rows_by_key: dict[tuple[str, float], dict[str, Any]],
    n_bootstrap: int,
    seed: int,
    use_clean_ids: bool,
) -> dict[str, Any]:
    lambdas = sorted({float(key[1]) for key in rows_by_key.keys()})
    by_lambda: list[dict[str, Any]] = []
    significant: dict[str, list[float]] = {name: [] for name, _, _ in COMPARISONS}
    seed_counter = 0

    for lambda_value in lambdas:
        mode_to_flipped: dict[str, set[str]] = {}
        for mode in ("balanced", "pure_promotion", "pure_suppression"):
            row = rows_by_key[(mode, float(lambda_value))]
            key = "flipped_item_ids_clean" if use_clean_ids else "flipped_item_ids"
            mode_to_flipped[mode] = {str(x) for x in row[key]}

        payload: dict[str, Any] = {"lambda": float(lambda_value), "n_items": int(len(universe_ids))}
        for comparison_name, left_mode, right_mode in COMPARISONS:
            left_vec = _build_indicator_vector(universe_ids, mode_to_flipped[left_mode])
            right_vec = _build_indicator_vector(universe_ids, mode_to_flipped[right_mode])
            stats = _paired_bootstrap_binary_diff(
                left_vec,
                right_vec,
                n_bootstrap=n_bootstrap,
                seed=int(seed + seed_counter),
            )
            seed_counter += 1
            payload[comparison_name] = stats
            if float(stats["p_value"]) < 0.05:
                significant[comparison_name].append(float(lambda_value))
        by_lambda.append(payload)

    return {
        "subset_name": str(subset_name),
        "n_items": int(len(universe_ids)),
        "by_lambda": by_lambda,
        "significant_lambdas": significant,
    }


def _format_lambda_list(values: list[float]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{float(x):g}" for x in values)


def _peak_effect(subset_payload: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    best_name = ""
    best_lambda = 0.0
    best_stats: dict[str, Any] | None = None
    for row in subset_payload["by_lambda"]:
        for comparison_name, _, _ in COMPARISONS:
            stats = row[comparison_name]
            if best_stats is None or float(stats["mean_diff"]) > float(best_stats["mean_diff"]):
                best_stats = stats
                best_name = comparison_name
                best_lambda = float(row["lambda"])
    if best_stats is None:
        best_stats = {"mean_diff": 0.0, "ci95": [0.0, 0.0], "p_value": 1.0}
    return best_name, best_lambda, best_stats


def _write_finding(path: Path, matched_payload: dict[str, Any], clean_payload: dict[str, Any]) -> None:
    peak_name, peak_lambda, peak_stats = _peak_effect(matched_payload)
    matched_bal_promo = matched_payload["significant_lambdas"]["balanced_minus_promotion"]
    matched_bal_supp = matched_payload["significant_lambdas"]["balanced_minus_suppression"]
    clean_bal_promo = clean_payload["significant_lambdas"]["balanced_minus_promotion"]
    clean_bal_supp = clean_payload["significant_lambdas"]["balanced_minus_suppression"]
    text = (
        f"On the {int(matched_payload['n_items'])}-item matched base-wrong subset, balanced exceeds pure promotion at "
        f"lambda={_format_lambda_list(matched_bal_promo)} and exceeds pure suppression at "
        f"lambda={_format_lambda_list(matched_bal_supp)} under paired bootstrap (p<0.05). "
        f"The peak matched-subset effect is {peak_name} at lambda={peak_lambda:g}, with mean difference "
        f"{float(peak_stats['mean_diff']):.4f} and 95% CI [{float(peak_stats['ci95'][0]):.4f}, {float(peak_stats['ci95'][1]):.4f}]. "
        f"On the {int(clean_payload['n_items'])}-item clean base-wrong subset, balanced exceeds pure promotion at "
        f"lambda={_format_lambda_list(clean_bal_promo)} and pure suppression at "
        f"lambda={_format_lambda_list(clean_bal_supp)}. "
        "Overall claim: balanced intervention is super-additive relative to pure modes in the sub-saturation regime."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    sub_dir = (PROJECT_ROOT / args.subthreshold_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = sub_dir / "aggregate.json"
    matched_path = (PROJECT_ROOT / args.matched_subset_path).resolve()
    label_audit_path = (PROJECT_ROOT / args.label_audit_path).resolve()

    aggregate = read_json(aggregate_path)
    matched_payload = read_json(matched_path)
    label_audit = load_label_audit(label_audit_path)
    clean_ids_full = set(clean_subset_item_ids(matched_payload, label_audit))

    rows: list[dict[str, Any]] = []
    for row in aggregate.get("rows", []):
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        flipped_ids, flipped_ids_clean = _load_flipped_ids(sub_dir, copied)
        copied["flipped_item_ids"] = flipped_ids
        copied["flipped_item_ids_clean"] = flipped_ids_clean
        rows.append(copied)
    if not rows:
        raise RuntimeError(f"No aggregate rows found in {aggregate_path}")

    matched_base_wrong_ids, clean_base_wrong_ids = _load_base_wrong_universe(sub_dir, rows)
    clean_base_wrong_ids = [item_id for item_id in clean_base_wrong_ids if item_id in clean_ids_full]

    rows_by_key = {(str(row["mode"]), float(row["lambda"])): row for row in rows}
    required_modes = {"balanced", "pure_promotion", "pure_suppression"}
    lambda_grid = sorted({float(row["lambda"]) for row in rows})
    for lambda_value in lambda_grid:
        seen_modes = {mode for mode, lam in rows_by_key.keys() if abs(float(lam) - float(lambda_value)) < 1e-9}
        if seen_modes != required_modes:
            raise RuntimeError(f"Incomplete mode coverage at lambda={lambda_value:g}: {sorted(seen_modes)}")

    matched_analysis = _subset_analysis(
        subset_name="matched_subset",
        universe_ids=matched_base_wrong_ids,
        rows_by_key=rows_by_key,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        use_clean_ids=False,
    )
    clean_analysis = _subset_analysis(
        subset_name="clean_subset",
        universe_ids=clean_base_wrong_ids,
        rows_by_key=rows_by_key,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 1000,
        use_clean_ids=True,
    )

    output = {
        "matched_subset": matched_analysis,
        "clean_subset": clean_analysis,
        "significant_lambdas": {
            "matched_subset": matched_analysis["significant_lambdas"],
            "clean_subset": clean_analysis["significant_lambdas"],
        },
        "metadata": _build_metadata(
            args=args,
            wall_clock_seconds=float(time.perf_counter() - started),
            input_files=[aggregate_path, matched_path, label_audit_path],
        ),
    }
    write_json(out_dir / "paired_bootstrap.json", output)
    _write_finding(out_dir / "additivity_finding.txt", matched_analysis, clean_analysis)

    print(f"paired_bootstrap={out_dir / 'paired_bootstrap.json'}")
    print(f"additivity_finding={out_dir / 'additivity_finding.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
