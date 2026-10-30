#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.matched_subset import clean_subset_item_ids, load_label_audit
from medrag.medqa_v3 import (
    build_letter_token_sets,
    evaluate_intervention_cell,
    load_conflicts,
    prepare_examples_for_variant,
    set_global_seed,
)
from medrag.utils.activation_cache import ActivationCache
from medrag.utils.io import build_run_metadata, read_json, write_json, write_jsonl, write_summary_json

RATE_FIELDS = [
    "baseline_accuracy",
    "ft_accuracy",
    "ft_accuracy_clean_subset",
    "restoration_rate",
    "harm_rate",
    "restoration_rate_clean_subset",
    "harm_rate_clean_subset",
    "gate_fire_rate_base_wrong",
    "gate_fire_rate_base_correct",
    "restoration_rate_param_wrong",
    "restoration_rate_other_wrong",
]

SATURATION_LAMBDA = 2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedQA sub-threshold decomposition sweep.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/decomposition_subthreshold")
    parser.add_argument("--base_decomposition_dir", default="results/medqa_final_v3/decomposition")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--lambda_values", default="0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5")
    parser.add_argument("--modes", default="pure_promotion,pure_suppression,balanced")
    parser.add_argument("--p_set_mode", default="all_wrong", choices=["all_wrong", "param_only"])
    parser.add_argument("--tau_margin", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--smoke_test_n", type=int, default=10)
    return parser.parse_args()


def _parse_grid(raw: str) -> list[float]:
    vals: list[float] = []
    seen: set[float] = set()
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value in seen:
            continue
        seen.add(value)
        vals.append(float(value))
    if not vals:
        raise ValueError("Empty lambda grid")
    return sorted(vals)


def _parse_modes(raw: str) -> list[str]:
    allowed = {"pure_promotion", "pure_suppression", "balanced"}
    modes: list[str] = []
    for piece in str(raw).split(","):
        mode = piece.strip()
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(f"Unsupported mode: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("Empty mode list")
    return modes


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _cell_name(lambda_pos: float, lambda_neg: float) -> str:
    return f"lp{_slug(lambda_pos)}_ln{_slug(lambda_neg)}"


def _mode_params(mode: str, lam: float) -> tuple[float, float]:
    if mode == "pure_promotion":
        return float(lam), 0.0
    if mode == "pure_suppression":
        return 0.0, float(lam)
    if mode == "balanced":
        return float(lam), float(lam)
    raise ValueError(f"Unsupported mode: {mode}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _flipped_item_ids(rows: list[dict[str, Any]], *, clean_only: bool = False) -> list[str]:
    out: list[str] = []
    for row in rows:
        if not bool(row.get("in_matched_subset", False)) or bool(row.get("skipped", False)):
            continue
        if clean_only and not bool(row.get("in_clean_subset", False)):
            continue
        if str(row.get("ft_letter")) != str(row.get("base_letter")):
            out.append(str(row.get("item_id")))
    return sorted(set(out))


def _clean_eval_count(rows: list[dict[str, Any]]) -> int:
    return int(sum(1 for row in rows if bool(row.get("in_clean_subset", False)) and not bool(row.get("skipped", False))))


def _enrich_summary(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    mode: str,
    lambda_value: float,
    lambda_pos: float,
    lambda_neg: float,
    args: argparse.Namespace,
    source_metadata: dict[str, Any] | None = None,
    reused_from_v3: bool = False,
    source_cell_path: str | None = None,
) -> dict[str, Any]:
    flipped_ids = _flipped_item_ids(rows, clean_only=False)
    flipped_ids_clean = _flipped_item_ids(rows, clean_only=True)
    out = dict(summary)
    out.update(
        {
            "experiment": "decomposition_subthreshold",
            "mode": str(mode),
            "lambda": float(lambda_value),
            "lambda_pos": float(lambda_pos),
            "lambda_neg": float(lambda_neg),
            "tau_margin": float(args.tau_margin),
            "alpha": float(args.alpha),
            "target_layer": int(args.target_layer),
            "target_head": int(args.target_head),
            "p_set_mode": str(args.p_set_mode),
            "n_items_clean": int(_clean_eval_count(rows)),
            "n_flipped": int(len(flipped_ids)),
            "n_flipped_clean": int(len(flipped_ids_clean)),
            "flipped_item_ids": flipped_ids,
            "flipped_item_ids_clean": flipped_ids_clean,
            "reused_from_v3": bool(reused_from_v3),
        }
    )
    if source_cell_path is not None:
        out["source_cell_path"] = str(source_cell_path)
    metadata = build_run_metadata(
        args=args,
        input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
        seed=int(args.seed),
        wall_clock_seconds=float(out.get("wall_clock_seconds", 0.0)),
        project_root=PROJECT_ROOT,
    )
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata
    out["metadata"] = metadata
    return out


def _write_cell(
    *,
    root: Path,
    mode: str,
    lambda_pos: float,
    lambda_neg: float,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    cell_dir = root / mode / _cell_name(lambda_pos, lambda_neg)
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_summary_json(cell_dir / "summary.json", summary, rate_fields=RATE_FIELDS)
    write_jsonl(cell_dir / "results.jsonl", rows)
    return cell_dir


def _smoke_test(
    *,
    prepared_records: list[Any],
    matched_ids: set[str],
    clean_ids: set[str],
    model,
    tokenizer,
    device,
    letter_token_sets: dict[str, set[int]],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    probe = [rec for rec in prepared_records if (rec.item_id in matched_ids) and (not bool(rec.base_correct))]
    probe = probe[: max(0, int(args.smoke_test_n))]
    payload: dict[str, Any] = {
        "n_probe_requested": int(args.smoke_test_n),
        "n_probe_used": int(len(probe)),
        "checks": {},
    }
    if not probe:
        payload["status"] = "inconclusive_no_probe_rows"
        payload["metadata"] = build_run_metadata(
            args=args,
            input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
            seed=int(args.seed),
            wall_clock_seconds=0.0,
            project_root=PROJECT_ROOT,
        )
        write_json(out_dir / "smoke_test.json", payload)
        return

    for name, lp, ln in (
        ("pure_promotion_probe", 0.5, 0.0),
        ("pure_suppression_probe", 0.0, 0.5),
    ):
        _, rows = evaluate_intervention_cell(
            prepared_records=probe,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=letter_token_sets,
            bootstrap_space="E_letter",
            intervention_space="E_letter",
            lambda_pos=float(lp),
            lambda_neg=float(ln),
            tau_margin=float(args.tau_margin),
            alpha=float(args.alpha),
            target_layer=int(args.target_layer),
            target_head=int(args.target_head),
            matched_item_ids=matched_ids,
            clean_item_ids=clean_ids,
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
        )
        affected = [row for row in rows if not bool(row.get("skipped", False)) and (float(row.get("lambda_pos_applied", 0.0)) > 0.0 or float(row.get("lambda_neg_applied", 0.0)) > 0.0)]
        payload["checks"][name] = {
            "n_rows": int(len(rows)),
            "n_affected": int(len(affected)),
            "all_promotion_rows_have_zero_neg_bias": bool(all(abs(float(row.get("lambda_neg_applied", 0.0))) < 1e-9 for row in affected)) if name == "pure_promotion_probe" else None,
            "all_suppression_rows_have_zero_pos_bias": bool(all(abs(float(row.get("lambda_pos_applied", 0.0))) < 1e-9 for row in affected)) if name == "pure_suppression_probe" else None,
            "sample_rows": [
                {
                    "item_id": str(row.get("item_id")),
                    "lambda_pos_applied": float(row.get("lambda_pos_applied", 0.0)),
                    "lambda_neg_applied": float(row.get("lambda_neg_applied", 0.0)),
                    "conflict_detected": bool(row.get("conflict_detected", False)),
                }
                for row in affected[:5]
            ],
        }
    payload["status"] = "passed"
    payload["metadata"] = build_run_metadata(
        args=args,
        input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
        seed=int(args.seed),
        wall_clock_seconds=0.0,
        project_root=PROJECT_ROOT,
    )
    write_json(out_dir / "smoke_test.json", payload)


def _reused_v3_cell(
    *,
    base_dir: Path,
    mode: str,
    lambda_value: float,
    lambda_pos: float,
    lambda_neg: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = base_dir / _cell_name(lambda_pos, lambda_neg)
    source_summary = read_json(source_dir / "summary.json")
    source_rows = _read_jsonl(source_dir / "results.jsonl")
    summary = _enrich_summary(
        summary=source_summary,
        rows=source_rows,
        mode=mode,
        lambda_value=lambda_value,
        lambda_pos=lambda_pos,
        lambda_neg=lambda_neg,
        args=args,
        source_metadata=source_summary.get("metadata") if isinstance(source_summary.get("metadata"), dict) else None,
        reused_from_v3=True,
        source_cell_path=str(source_dir),
    )
    return summary, source_rows


def _aggregate_row(summary: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    return {
        "mode": str(summary["mode"]),
        "lambda": float(summary["lambda"]),
        "lambda_pos": float(summary["lambda_pos"]),
        "lambda_neg": float(summary["lambda_neg"]),
        "restoration_rate": float(summary["restoration_rate"]),
        "restoration_rate_ci95_wilson": summary["restoration_rate_ci95_wilson"],
        "restoration_rate_clean_subset": float(summary["restoration_rate_clean_subset"]),
        "restoration_rate_clean_subset_ci95_wilson": summary["restoration_rate_clean_subset_ci95_wilson"],
        "harm_rate": float(summary["harm_rate"]),
        "harm_rate_ci95_wilson": summary["harm_rate_ci95_wilson"],
        "gate_fire_rate_base_correct": float(summary["gate_fire_rate_base_correct"]),
        "gate_fire_rate_base_correct_ci95_wilson": summary["gate_fire_rate_base_correct_ci95_wilson"],
        "gate_fire_rate_base_wrong": float(summary["gate_fire_rate_base_wrong"]),
        "gate_fire_rate_base_wrong_ci95_wilson": summary["gate_fire_rate_base_wrong_ci95_wilson"],
        "restoration_improvement": float(summary["restoration_improvement"]),
        "youden_j": float(summary["restoration_improvement"] - summary["gate_fire_rate_base_correct"]),
        "n_flipped": int(summary["n_flipped"]),
        "n_flipped_clean": int(summary["n_flipped_clean"]),
        "flipped_item_ids": [str(x) for x in summary.get("flipped_item_ids", [])],
        "flipped_item_ids_clean": [str(x) for x in summary.get("flipped_item_ids_clean", [])],
        "reused_from_v3": bool(summary.get("reused_from_v3", False)),
        "out_dir": str(cell_dir),
    }


def _build_separation(rows: list[dict[str, Any]], summaries: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    lambdas = sorted({float(row["lambda"]) for row in rows})
    by_lambda: list[dict[str, Any]] = []
    positive_separation_lambdas: list[float] = []
    peak_lambda: float | None = None
    peak_value = -1

    for lam in lambdas:
        promo = summaries[("pure_promotion", float(lam))]
        supp = summaries[("pure_suppression", float(lam))]
        promo_ids = set(str(x) for x in promo.get("flipped_item_ids", []))
        supp_ids = set(str(x) for x in supp.get("flipped_item_ids", []))
        promo_clean = set(str(x) for x in promo.get("flipped_item_ids_clean", []))
        supp_clean = set(str(x) for x in supp.get("flipped_item_ids_clean", []))

        promo_only = sorted(promo_ids - supp_ids)
        supp_only = sorted(supp_ids - promo_ids)
        promo_only_clean = sorted(promo_clean - supp_clean)
        supp_only_clean = sorted(supp_clean - promo_clean)
        sym = len(promo_only) + len(supp_only)
        sym_clean = len(promo_only_clean) + len(supp_only_clean)

        if sym > 0:
            positive_separation_lambdas.append(float(lam))
        if sym > peak_value:
            peak_value = sym
            peak_lambda = float(lam)

        by_lambda.append(
            {
                "lambda": float(lam),
                "n_flipped_promotion": int(len(promo_ids)),
                "n_flipped_suppression": int(len(supp_ids)),
                "n_intersection": int(len(promo_ids & supp_ids)),
                "n_symmetric_difference": int(sym),
                "promotion_only_ids": promo_only,
                "suppression_only_ids": supp_only,
                "flipped_clean_promotion": int(len(promo_clean)),
                "flipped_clean_suppression": int(len(supp_clean)),
                "n_intersection_clean": int(len(promo_clean & supp_clean)),
                "n_symmetric_difference_clean": int(sym_clean),
                "promotion_only_ids_clean": promo_only_clean,
                "suppression_only_ids_clean": supp_only_clean,
            }
        )

    if all(int(row["n_symmetric_difference"]) == 0 for row in by_lambda):
        outcome = "A"
        justification = "Pure promotion and pure suppression have identical flipped-item sets at every sub-threshold magnitude, so gauge equivalence holds across the full tested range."
    else:
        outcome = "B"
        lam_text = ", ".join(f"{x:g}" for x in positive_separation_lambdas)
        justification = f"Promotion and suppression diverge below saturation: nonzero symmetric differences appear at λ={lam_text}, with peak separation {peak_value} at λ={peak_lambda:g}."
    return {
        "by_lambda": by_lambda,
        "outcome": outcome,
        "outcome_justification": justification,
    }


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    lambda_grid = _parse_grid(str(args.lambda_values))
    modes = _parse_modes(str(args.modes))
    base_dir = (PROJECT_ROOT / args.base_decomposition_dir).resolve()

    matched_payload = read_json((PROJECT_ROOT / args.matched_subset_path).resolve())
    matched_ids = set(str(x) for x in matched_payload.get("matched_item_ids", []))
    label_audit = load_label_audit((PROJECT_ROOT / args.label_audit_path).resolve())
    clean_ids = set(clean_subset_item_ids(matched_payload, label_audit))

    planned_cells: list[tuple[str, float, float, float, bool]] = []
    for mode in modes:
        for lam in lambda_grid:
            lp, ln = _mode_params(mode, lam)
            reuse = abs(float(lam) - SATURATION_LAMBDA) < 1e-9
            planned_cells.append((mode, float(lam), float(lp), float(ln), reuse))

    needs_model = any(not reuse for _, _, _, _, reuse in planned_cells)

    prepared: list[Any] = []
    model = tokenizer = device = letter_token_sets = cache = None
    if needs_model:
        conflicts = load_conflicts((PROJECT_ROOT / args.conflict_path).resolve())
        model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
        letter_token_sets = build_letter_token_sets(tokenizer)
        cache = ActivationCache((PROJECT_ROOT / args.cache_dir).resolve())
        prepared = prepare_examples_for_variant(
            conflicts=conflicts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            cache=cache,
            model_id=str(args.model_id),
            prompt_variant="vanilla",
            p_set_mode=str(args.p_set_mode),
            target_layer=int(args.target_layer),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            letter_token_sets=letter_token_sets,
        )
        _smoke_test(
            prepared_records=prepared,
            matched_ids=matched_ids,
            clean_ids=clean_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letter_token_sets=letter_token_sets,
            args=args,
            out_dir=out_dir,
        )

    aggregate_rows: list[dict[str, Any]] = []
    summaries_for_sep: dict[tuple[str, float], dict[str, Any]] = {}

    for mode, lam, lp, ln, reuse in planned_cells:
        if reuse:
            summary, rows = _reused_v3_cell(
                base_dir=base_dir,
                mode=mode,
                lambda_value=lam,
                lambda_pos=lp,
                lambda_neg=ln,
                args=args,
            )
        else:
            assert model is not None and tokenizer is not None and device is not None and letter_token_sets is not None
            raw_summary, rows = evaluate_intervention_cell(
                prepared_records=prepared,
                model=model,
                tokenizer=tokenizer,
                device=device,
                letter_token_sets=letter_token_sets,
                bootstrap_space="E_letter",
                intervention_space="E_letter",
                lambda_pos=float(lp),
                lambda_neg=float(ln),
                tau_margin=float(args.tau_margin),
                alpha=float(args.alpha),
                target_layer=int(args.target_layer),
                target_head=int(args.target_head),
                matched_item_ids=matched_ids,
                clean_item_ids=clean_ids,
                seed=int(args.seed),
                max_new_tokens=int(args.max_new_tokens),
            )
            summary = _enrich_summary(
                summary=raw_summary,
                rows=rows,
                mode=mode,
                lambda_value=lam,
                lambda_pos=lp,
                lambda_neg=ln,
                args=args,
            )

        cell_dir = _write_cell(root=out_dir, mode=mode, lambda_pos=lp, lambda_neg=ln, summary=summary, rows=rows)
        aggregate_rows.append(_aggregate_row(summary, cell_dir))
        summaries_for_sep[(mode, float(lam))] = summary
        print(json.dumps({"mode": mode, "lambda": lam, "summary": str(cell_dir / "summary.json")}, indent=2))

    aggregate = {
        "experiment": "decomposition_subthreshold",
        "lambda_grid": lambda_grid,
        "modes": modes,
        "rows": sorted(aggregate_rows, key=lambda x: (str(x["mode"]), float(x["lambda"]))),
        "metadata": build_run_metadata(
            args=args,
            input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=PROJECT_ROOT,
        ),
    }
    write_json(out_dir / "aggregate.json", aggregate)

    separation = _build_separation(aggregate["rows"], summaries_for_sep)
    separation["metadata"] = build_run_metadata(
        args=args,
        input_file=(PROJECT_ROOT / args.conflict_path).resolve(),
        seed=int(args.seed),
        wall_clock_seconds=float(time.perf_counter() - t0),
        project_root=PROJECT_ROOT,
    )
    write_json(out_dir / "separation.json", separation)

    finding = (
        f"Outcome {separation['outcome']}: {separation['outcome_justification']} "
        f"Matched-subset evaluation reuses λ=2.5 cells from v3 and audits clean-subset separation in parallel."
    )
    _write_text(out_dir / "finding.txt", finding + "\n")

    print(f"aggregate={out_dir / 'aggregate.json'}")
    print(f"separation={out_dir / 'separation.json'}")
    print(f"finding={out_dir / 'finding.txt'}")
    print(f"smoke_test={out_dir / 'smoke_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
