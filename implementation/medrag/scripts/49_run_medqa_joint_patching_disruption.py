#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import PatchSpec, cache_multihead_outputs, run_with_multihead_patch
from medrag.causal_localization import CausalItem, build_metadata, load_medqa_causal_inputs, predict_letter_from_logits
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl, write_summary_json
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


VALID_LETTERS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedQA joint-patching disruption control.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument(
        "--head_importance_path",
        default="results/medqa_final_v3/localization_v2/head_importance.json",
    )
    parser.add_argument("--pair_path", default="results/medqa_final_v3/causal/patching_pairs.json")
    parser.add_argument(
        "--matched_joint_results_path",
        default="results/medqa_final_v3/localization_v2/joint_patching/results.jsonl",
    )
    parser.add_argument(
        "--matched_joint_summary_path",
        default="results/medqa_final_v3/localization_v2/joint_patching/summary.json",
    )
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument(
        "--out_dir",
        default="results/medqa_final_v3/localization_v2/joint_patching_disruption",
    )
    parser.add_argument("--figures_dir", default="results/medqa_final_v3/localization_v2/figures")
    parser.add_argument("--top_k_heads", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _cell_name(layer: int, head: int) -> str:
    return f"L{int(layer)}_H{int(head)}"


def _load_top_heads(path: Path, top_k: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("top_5_heads", [])
    if not isinstance(rows, list) or len(rows) < int(top_k):
        raise RuntimeError(f"{path} does not contain at least {top_k} localized heads.")
    out: list[dict[str, Any]] = []
    for row in rows[: int(top_k)]:
        out.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "rank": int(row.get("rank", 0)),
                "mean_contribution": float(row.get("mean_contribution", 0.0)),
                "cell_name": _cell_name(int(row["layer"]), int(row["head"])),
            }
        )
    return out


def _rate(flags: list[bool]) -> tuple[float, list[float], int, int]:
    k = int(sum(1 for flag in flags if bool(flag)))
    n = int(len(flags))
    rate = float(k / max(n, 1))
    lo, hi = wilson_ci(k, n) if n > 0 else (0.0, 0.0)
    return float(rate), [float(lo), float(hi)], int(k), int(n)


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


def _load_existing_dest_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("pairs", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No pair rows found in {path}")
    return rows


def _build_mismatched_pairs(
    *,
    causal_items: list[CausalItem],
    h4_by_id: dict[str, dict[str, Any]],
    existing_dest_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    items_by_id = {str(item.item_id): item for item in causal_items}
    source_items = [
        item
        for item in causal_items
        if item.in_matched_subset
        and item.base_correct
        and (not item.conflict_detected)
        and str(item.correct) in VALID_LETTERS
    ]
    if not source_items:
        raise RuntimeError("No source items available for disruption control.")

    sources_by_letter: dict[str, list[CausalItem]] = {}
    for letter in VALID_LETTERS:
        bucket = [item for item in source_items if str(item.correct) == str(letter)]
        if bucket:
            sources_by_letter[str(letter)] = sorted(bucket, key=lambda item: (item.prompt_length, item.item_id))

    rng = random.Random(int(seed))
    letter_usage: dict[str, int] = {str(letter): 0 for letter in VALID_LETTERS}
    item_usage: dict[str, int] = {}
    out: list[dict[str, Any]] = []

    for index, row in enumerate(existing_dest_rows):
        dest_id = str(row.get("dest_id", "")).strip()
        if not dest_id:
            raise RuntimeError(f"Encountered patch row without dest_id at index={index}")
        if dest_id not in items_by_id:
            raise RuntimeError(f"Destination item {dest_id} is missing from prepared MedQA inputs.")

        dest = items_by_id[dest_id]
        dest_letter = str(dest.correct)
        eligible_letters = [letter for letter in VALID_LETTERS if letter in sources_by_letter and letter != dest_letter]
        if not eligible_letters:
            raise RuntimeError(f"No mismatched-letter source bucket available for dest_id={dest_id}")

        min_letter_usage = min(int(letter_usage.get(letter, 0)) for letter in eligible_letters)
        least_used_letters = [letter for letter in eligible_letters if int(letter_usage.get(letter, 0)) == int(min_letter_usage)]
        source_letter = str(rng.choice(least_used_letters))
        bucket = sources_by_letter[source_letter]

        min_item_usage = min(int(item_usage.get(item.item_id, 0)) for item in bucket)
        least_used_items = [item for item in bucket if int(item_usage.get(item.item_id, 0)) == int(min_item_usage)]
        min_prompt_diff = min(abs(int(item.prompt_length) - int(dest.prompt_length)) for item in least_used_items)
        closest_items = [
            item for item in least_used_items if abs(int(item.prompt_length) - int(dest.prompt_length)) == int(min_prompt_diff)
        ]
        source = rng.choice(closest_items)

        item_usage[source.item_id] = int(item_usage.get(source.item_id, 0) + 1)
        letter_usage[source_letter] = int(letter_usage.get(source_letter, 0) + 1)

        baseline_dest_answer = h4_by_id.get(dest_id, {}).get("base_letter")
        pair_id = f"{index:03d}_{source_letter}_to_{dest_letter}_{source.item_id[:8]}_{dest.item_id[:8]}"
        out.append(
            {
                "pair_id": str(pair_id),
                "dest_id": str(dest.item_id),
                "dest_correct_letter": str(dest_letter),
                "source_id": str(source.item_id),
                "source_correct_letter": str(source_letter),
                "baseline_dest_answer": None if baseline_dest_answer is None else str(baseline_dest_answer),
                "source_prompt_length": int(source.prompt_length),
                "dest_prompt_length": int(dest.prompt_length),
            }
        )

    return out


def _exclusive_outcome(
    *,
    patched_answer: str | None,
    source_correct: str,
    dest_correct: str,
    baseline_dest_answer: str | None,
) -> str:
    if patched_answer is not None and baseline_dest_answer is not None and str(patched_answer) == str(baseline_dest_answer):
        return "unchanged"
    if patched_answer is not None and str(patched_answer) == str(source_correct):
        return "flip_to_source_letter"
    if patched_answer is not None and str(patched_answer) == str(dest_correct):
        return "flip_to_dest_correct_letter"
    return "flip_to_third_letter"


def _plot_disruption_bars(
    *,
    matched_joint_summary: dict[str, Any],
    disruption_summary: dict[str, Any],
    out_path: Path,
) -> None:
    matched_main = float(matched_joint_summary["main_flip_rate"])
    matched_other = max(0.0, 1.0 - matched_main)

    source_rate = float(disruption_summary["flip_to_source_rate"])
    dest_rate = float(disruption_summary["flip_to_dest_correct_rate"])
    third_rate = float(disruption_summary["flip_to_third_letter_rate"])
    unchanged_rate = float(disruption_summary["unchanged_rate"])

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    x_positions = [0.0, 1.25]
    width = 0.62

    matched_bottom = 0.0
    ax.bar(x_positions[0], matched_main, width=width, color="#2ca25f", label="Matched-letter success")
    matched_bottom += matched_main
    ax.bar(x_positions[0], matched_other, width=width, bottom=matched_bottom, color="#bdbdbd", label="Matched-letter other")

    bottom = 0.0
    disruption_parts = [
        ("Flip to source", source_rate, "#2ca25f"),
        ("Flip to dest-correct", dest_rate, "#3182bd"),
        ("Flip to third letter", third_rate, "#f0ad4e"),
        ("Unchanged", unchanged_rate, "#7f7f7f"),
    ]
    for label, value, color in disruption_parts:
        ax.bar(x_positions[1], value, width=width, bottom=bottom, color=color, label=label)
        bottom += value

    ax.set_xticks(x_positions, ["Matched-letter joint patch", "Mismatched-letter disruption"])
    ax.set_ylabel("Outcome Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("MedQA Joint-Patch Disruption Control")

    matched_ci = matched_joint_summary["main_flip_rate_ci95_wilson"]
    source_ci = disruption_summary["flip_to_source_rate_ci95_wilson"]
    ax.text(
        x_positions[0],
        1.02,
        f"main={matched_main:.3f}\n[{float(matched_ci[0]):.3f}, {float(matched_ci[1]):.3f}]",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        x_positions[1],
        1.02,
        f"source={source_rate:.3f}\n[{float(source_ci[0]):.3f}, {float(source_ci[1]):.3f}]",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    head_importance_path = (PROJECT_ROOT / args.head_importance_path).resolve()
    pair_path = (PROJECT_ROOT / args.pair_path).resolve()
    matched_joint_results_path = (PROJECT_ROOT / args.matched_joint_results_path).resolve()
    matched_joint_summary_path = (PROJECT_ROOT / args.matched_joint_summary_path).resolve()

    top_heads = _load_top_heads(head_importance_path, int(args.top_k_heads))
    existing_pair_rows = _load_existing_dest_rows(pair_path)
    matched_joint_results = _read_jsonl(matched_joint_results_path)
    matched_joint_summary = read_json(matched_joint_summary_path)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    inputs = load_medqa_causal_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=PROJECT_ROOT / args.conflict_path,
        matched_subset_path=PROJECT_ROOT / args.matched_subset_path,
        label_audit_path=PROJECT_ROOT / args.label_audit_path,
        h4_results_path=PROJECT_ROOT / args.h4_results_path,
        cache_dir=PROJECT_ROOT / args.cache_dir,
        target_layer=max(int(row["layer"]) for row in top_heads),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )

    disruption_pairs = _build_mismatched_pairs(
        causal_items=inputs.causal_items,
        h4_by_id=inputs.h4_by_id,
        existing_dest_rows=existing_pair_rows,
        seed=int(args.seed),
    )
    write_json(out_dir / "pairs.json", {"pairs": disruption_pairs})

    main_specs = [PatchSpec(layer=int(row["layer"]), head=int(row["head"]), position=-1) for row in top_heads]
    results: list[dict[str, Any]] = []
    source_flags: list[bool] = []
    dest_flags: list[bool] = []
    third_flags: list[bool] = []
    unchanged_flags: list[bool] = []

    for pair in disruption_pairs:
        source = inputs.prepared_by_id[str(pair["source_id"])]
        dest = inputs.prepared_by_id[str(pair["dest_id"])]
        source_correct_letter = str(pair["source_correct_letter"])
        dest_correct_letter = str(pair["dest_correct_letter"])
        baseline_dest_answer = pair.get("baseline_dest_answer")

        source_patches = cache_multihead_outputs(
            model=model,
            prompt_ids=source.prompt_ids.to(device),
            patch_specs=main_specs,
        )
        patched_logits = run_with_multihead_patch(
            model=model,
            prompt_ids=dest.prompt_ids.to(device),
            patched_values=source_patches,
        )
        patched_answer, patched_scores = predict_letter_from_logits(patched_logits, inputs.letter_token_sets)
        outcome = _exclusive_outcome(
            patched_answer=patched_answer,
            source_correct=source_correct_letter,
            dest_correct=dest_correct_letter,
            baseline_dest_answer=(None if baseline_dest_answer is None else str(baseline_dest_answer)),
        )

        flipped_to_source = bool(outcome == "flip_to_source_letter")
        flipped_to_dest_correct = bool(outcome == "flip_to_dest_correct_letter")
        flipped_to_third = bool(outcome == "flip_to_third_letter")
        unchanged = bool(outcome == "unchanged")

        row = {
            "pair_id": str(pair["pair_id"]),
            "dest_id": str(pair["dest_id"]),
            "dest_correct_letter": str(dest_correct_letter),
            "source_id": str(pair["source_id"]),
            "source_correct_letter": str(source_correct_letter),
            "baseline_dest_answer": baseline_dest_answer,
            "patched_answer": patched_answer,
            "flipped_to_source_letter": bool(flipped_to_source),
            "flipped_to_dest_correct_letter": bool(flipped_to_dest_correct),
            "flipped_to_third_letter": bool(flipped_to_third),
            "unchanged": bool(unchanged),
            "outcome_category": str(outcome),
            "top5_heads_used": [
                {"layer": int(row["layer"]), "head": int(row["head"]), "rank": int(row["rank"])}
                for row in top_heads
            ],
            "patched_scores": patched_scores,
        }
        results.append(row)
        source_flags.append(bool(flipped_to_source))
        dest_flags.append(bool(flipped_to_dest_correct))
        third_flags.append(bool(flipped_to_third))
        unchanged_flags.append(bool(unchanged))

    source_rate, source_ci, source_k, n_pairs = _rate(source_flags)
    dest_rate, dest_ci, dest_k, _ = _rate(dest_flags)
    third_rate, third_ci, third_k, _ = _rate(third_flags)
    unchanged_rate, unchanged_ci, unchanged_k, _ = _rate(unchanged_flags)

    results_by_dest = {str(row["dest_id"]): row for row in results}
    matched_by_dest = {
        str(row.get("dest_id")): row
        for row in matched_joint_results
        if str(row.get("dest_id", "")).strip()
    }
    shared_dest_ids = [str(pair["dest_id"]) for pair in disruption_pairs if str(pair["dest_id"]) in matched_by_dest]
    mcnemar_source_vs_matched = mcnemar_from_paired(
        [bool(results_by_dest[dest_id]["flipped_to_source_letter"]) for dest_id in shared_dest_ids],
        [bool(matched_by_dest[dest_id].get("main_flip_indicator", False)) for dest_id in shared_dest_ids],
    )

    if float(source_rate) >= 0.60 and float(third_rate) <= 0.15:
        finding_category = "H1"
        finding_text = (
            "joint patch substitutes the commitment, not just disrupts it. "
            f"Flip-to-source rate = {source_rate:.3f} [{source_ci[0]:.3f}, {source_ci[1]:.3f}], "
            f"third-letter rate = {third_rate:.3f} [{third_ci[0]:.3f}, {third_ci[1]:.3f}]."
        )
    elif float(third_rate) > 0.30:
        finding_category = "H2"
        finding_text = (
            "joint patch is more consistent with activation-volume drag than strict decision substitution. "
            f"Flip-to-source rate = {source_rate:.3f} [{source_ci[0]:.3f}, {source_ci[1]:.3f}], "
            f"third-letter rate = {third_rate:.3f} [{third_ci[0]:.3f}, {third_ci[1]:.3f}]."
        )
    else:
        finding_category = "intermediate"
        finding_text = (
            "the disruption control gives partial support for decision substitution but does not cleanly rule out aggregate activation effects. "
            f"Flip-to-source rate = {source_rate:.3f} [{source_ci[0]:.3f}, {source_ci[1]:.3f}], "
            f"third-letter rate = {third_rate:.3f} [{third_ci[0]:.3f}, {third_ci[1]:.3f}]."
        )

    wall_clock_seconds = float(time.perf_counter() - started)
    metadata = build_metadata(
        args=args,
        input_files=[
            PROJECT_ROOT / args.conflict_path,
            PROJECT_ROOT / args.matched_subset_path,
            PROJECT_ROOT / args.label_audit_path,
            PROJECT_ROOT / args.h4_results_path,
            head_importance_path,
            pair_path,
            matched_joint_results_path,
            matched_joint_summary_path,
        ],
        seed=int(args.seed),
        wall_clock_seconds=wall_clock_seconds,
        project_root=PROJECT_ROOT,
    )

    summary = {
        "flip_to_source_rate": float(source_rate),
        "flip_to_source_rate_ci95_wilson": source_ci,
        "flip_to_source_count": int(source_k),
        "flip_to_dest_correct_rate": float(dest_rate),
        "flip_to_dest_correct_rate_ci95_wilson": dest_ci,
        "flip_to_dest_correct_count": int(dest_k),
        "flip_to_third_letter_rate": float(third_rate),
        "flip_to_third_letter_rate_ci95_wilson": third_ci,
        "flip_to_third_letter_count": int(third_k),
        "unchanged_rate": float(unchanged_rate),
        "unchanged_rate_ci95_wilson": unchanged_ci,
        "unchanged_count": int(unchanged_k),
        "n_pairs": int(n_pairs),
        "top5_heads": top_heads,
        "matched_letter_reference_main_flip_rate": float(matched_joint_summary["main_flip_rate"]),
        "matched_letter_reference_main_flip_rate_ci95_wilson": [
            float(matched_joint_summary["main_flip_rate_ci95_wilson"][0]),
            float(matched_joint_summary["main_flip_rate_ci95_wilson"][1]),
        ],
        "mcnemar_flip_to_source_vs_matched_letter_main": {
            "n_shared_dest_ids": int(len(shared_dest_ids)),
            "shared_dest_ids": shared_dest_ids,
            "mcnemar": mcnemar_source_vs_matched,
        },
        "finding_category": str(finding_category),
        "finding": str(finding_text),
        "metadata": metadata,
    }

    write_jsonl(out_dir / "results.jsonl", results)
    write_summary_json(
        out_dir / "summary.json",
        summary,
        rate_fields=[
            "flip_to_source_rate",
            "flip_to_dest_correct_rate",
            "flip_to_third_letter_rate",
            "unchanged_rate",
        ],
    )
    (out_dir / "finding.txt").write_text(str(finding_text) + "\n", encoding="utf-8")

    _plot_disruption_bars(
        matched_joint_summary=matched_joint_summary,
        disruption_summary=summary,
        out_path=figures_dir / "joint_patching_disruption_bars.png",
    )

    print(f"pairs={out_dir / 'pairs.json'}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"results={out_dir / 'results.jsonl'}")
    print(f"finding={out_dir / 'finding.txt'}")
    print(f"figure={figures_dir / 'joint_patching_disruption_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
