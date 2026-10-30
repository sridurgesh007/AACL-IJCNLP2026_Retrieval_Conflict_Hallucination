#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import PatchSpec, cache_multihead_outputs, run_with_multihead_patch
from medrag.causal_localization import CausalItem, build_metadata, load_medqa_causal_inputs, predict_letter_from_logits
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import read_json, write_json, write_jsonl
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


TOP5_HEADS = [(25, 15), (17, 24), (24, 24), (30, 27), (23, 6)]
VALID_LETTERS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched five-head causal controls for MedQA rebuttal.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--matched_subset_path", default="results/medqa_final_v3/matched_subset.json")
    parser.add_argument("--label_audit_path", default="results/medqa_final_v3/label_audit.json")
    parser.add_argument(
        "--h4_results_path",
        default="results/medqa_final_v3/consolidation/headline_verify/H4/results.jsonl",
    )
    parser.add_argument("--pair_path", default="results/medqa_final_v3/causal/patching_pairs.json")
    parser.add_argument("--head_importance_path", default="results/medqa_final_v3/localization_v2/head_importance.json")
    parser.add_argument("--cache_dir", default="cache/medqa_acts")
    parser.add_argument("--out_dir", default="results/rebuttal/matched_head_controls")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--n_random_draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_prediction(path: Path) -> None:
    if path.exists():
        return
    write_json(
        path,
        {
            "frozen_at_utc": _now_iso(),
            "predicted_top5_flip_rate": 0.886,
            "predicted_random5_same_layer_flip_rate": "< 0.30",
            "predicted_random5_late_layer_flip_rate": "< 0.20",
            "predicted_attribution_matched_flip_rate": "< 0.35",
            "prediction_recorded_before_execution": True,
            "note": "A random-5 flip rate above ~0.50 falsifies contribution #1.",
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _wilson(k: int, n: int) -> list[float]:
    lo, hi = wilson_ci(int(k), int(n)) if n else (0.0, 0.0)
    return [float(lo), float(hi)]


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list) or len(pairs) != 35:
        raise RuntimeError(f"Expected exactly 35 patching pairs in {path}; got {len(pairs) if isinstance(pairs, list) else 'non-list'}")
    return pairs


def _load_head_scores(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("per_head_mean_contribution")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} does not expose per_head_mean_contribution for all heads.")
    scores: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 3:
            continue
        layer, head, score = int(row[0]), int(row[1]), float(row[2])
        scores[(layer, head)] = {"layer": layer, "head": head, "score": score}
    if len(scores) != 1024:
        raise RuntimeError(f"Expected 1024 head scores from {path}; got {len(scores)}")
    return scores


def _head_specs(heads: list[tuple[int, int]]) -> list[PatchSpec]:
    return [PatchSpec(layer=int(layer), head=int(head), position=-1) for layer, head in heads]


def _arm_head_dict(heads: list[tuple[int, int]], scores: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(layer),
            "head": int(head),
            "score": float(scores.get((int(layer), int(head)), {}).get("score", float("nan"))),
        }
        for layer, head in heads
    ]


def _random5_same_layer(draw_idx: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(int(seed) + 100_003 * int(draw_idx) + 17)
    out: list[tuple[int, int]] = []
    for layer, top_head in TOP5_HEADS:
        candidates = [h for h in range(32) if h != int(top_head)]
        out.append((int(layer), int(rng.choice(candidates))))
    return out


def _random5_late_layer(draw_idx: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(int(seed) + 200_003 * int(draw_idx) + 29)
    excluded = set(TOP5_HEADS)
    pool = [(layer, head) for layer in range(17, 32) for head in range(32) if (layer, head) not in excluded]
    return [(int(layer), int(head)) for layer, head in rng.sample(pool, 5)]


def _attribution_matched(scores: dict[tuple[int, int], dict[str, Any]]) -> list[tuple[int, int]]:
    excluded = set(TOP5_HEADS)
    available = {key: float(value["score"]) for key, value in scores.items() if key not in excluded}
    out: list[tuple[int, int]] = []
    for top in TOP5_HEADS:
        target = float(scores[top]["score"])
        best = min(available, key=lambda key: (abs(float(available[key]) - target), key[0], key[1]))
        out.append(best)
        del available[best]
    return out


def _bottom5(scores: dict[tuple[int, int], dict[str, Any]]) -> list[tuple[int, int]]:
    ranked = sorted(scores.items(), key=lambda item: (float(item[1]["score"]), item[0][0], item[0][1]))
    return [(int(layer), int(head)) for (layer, head), _ in ranked[:5]]


def _build_arm_definitions(scores: dict[tuple[int, int], dict[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    arms: dict[str, Any] = {
        "top5": {"type": "deterministic", "heads": TOP5_HEADS},
        "attribution_matched_non_top5": {"type": "deterministic", "heads": _attribution_matched(scores)},
        "bottom5": {"type": "deterministic", "heads": _bottom5(scores)},
        "random5_same_layer": {"type": "random", "draws": []},
        "random5_late_layer": {"type": "random", "draws": []},
    }
    for draw_idx in range(int(draws)):
        arms["random5_same_layer"]["draws"].append({"draw_index": draw_idx, "heads": _random5_same_layer(draw_idx, seed)})
        arms["random5_late_layer"]["draws"].append({"draw_index": draw_idx, "heads": _random5_late_layer(draw_idx, seed)})
    for key, payload in arms.items():
        if "heads" in payload:
            payload["head_scores"] = _arm_head_dict(payload["heads"], scores)
        else:
            for draw in payload["draws"]:
                draw["head_scores"] = _arm_head_dict(draw["heads"], scores)
    return arms


def _rate_summary(flags: list[bool]) -> dict[str, Any]:
    k = int(sum(bool(x) for x in flags))
    n = int(len(flags))
    return {"n_flipped": k, "n": n, "flip_rate": float(k / max(n, 1)), "wilson95": _wilson(k, n)}


def _bootstrap_separation(top_rate: float, random_rates: list[float], seed: int, n_boot: int = 10000) -> dict[str, Any]:
    rng = random.Random(int(seed) + 777)
    vals = []
    for _ in range(int(n_boot)):
        sample = [rng.choice(random_rates) for _ in random_rates]
        vals.append(float(top_rate) - float(mean(sample)))
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return {"low": float(lo), "high": float(hi), "n_bootstrap": int(n_boot)}


def _load_medqa_inputs(args: argparse.Namespace, model, tokenizer, device):
    return load_medqa_causal_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(args.model_id),
        conflict_path=PROJECT_ROOT / args.conflict_path,
        matched_subset_path=PROJECT_ROOT / args.matched_subset_path,
        label_audit_path=PROJECT_ROOT / args.label_audit_path,
        h4_results_path=PROJECT_ROOT / args.h4_results_path,
        cache_dir=PROJECT_ROOT / args.cache_dir,
        target_layer=max(layer for layer, _ in TOP5_HEADS),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )


def _patch_arm(
    *,
    model,
    inputs,
    pairs: list[dict[str, Any]],
    heads: list[tuple[int, int]],
    arm_name: str,
    source_mode: str,
    mismatched_pairs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[bool]]:
    specs = _head_specs(heads)
    rows: list[dict[str, Any]] = []
    flags: list[bool] = []
    active_pairs = mismatched_pairs if source_mode == "mismatched" else pairs
    assert active_pairs is not None
    for pair in active_pairs:
        source = inputs.prepared_by_id[str(pair["source_id"])]
        dest = inputs.prepared_by_id[str(pair["dest_id"])]
        patches = cache_multihead_outputs(model=model, prompt_ids=source.prompt_ids.to(next(model.parameters()).device), patch_specs=specs)
        logits = run_with_multihead_patch(model=model, prompt_ids=dest.prompt_ids.to(next(model.parameters()).device), patched_values=patches)
        answer, scores = predict_letter_from_logits(logits, inputs.letter_token_sets)
        if source_mode == "matched":
            target = str(pair["shared_letter"])
            flipped = bool(answer == target)
            row = {
                "pair_id": str(pair["pair_id"]),
                "source_id": str(pair["source_id"]),
                "dest_id": str(pair["dest_id"]),
                "dest_base_letter": inputs.h4_by_id[str(pair["dest_id"])].get("base_letter"),
                "source_correct_letter": target,
                "post_patch_letter": answer,
                "flipped": bool(flipped),
                "scores": scores,
                "arm": arm_name,
            }
        else:
            source_letter = str(pair["source_correct_letter"])
            dest_letter = str(pair["dest_correct_letter"])
            baseline = None if pair.get("baseline_dest_answer") is None else str(pair.get("baseline_dest_answer"))
            outcome = _exclusive_disruption_outcome(
                post=answer,
                dest_base_letter=baseline,
                source_correct_letter=source_letter,
                dest_correct_letter=dest_letter,
            )
            unchanged = bool(outcome == "unchanged")
            flip_source = bool(outcome == "flip_to_source")
            flip_dest = bool(outcome == "flip_to_dest")
            flip_third = bool(outcome == "flip_to_third")
            row = {
                "pair_id": str(pair["pair_id"]),
                "source_id": str(pair["source_id"]),
                "dest_id": str(pair["dest_id"]),
                "dest_base_letter": baseline,
                "source_correct_letter": source_letter,
                "dest_correct_letter": dest_letter,
                "post_patch_letter": answer,
                "outcome": outcome,
                "flip_to_source": bool(flip_source),
                "flip_to_dest": bool(flip_dest),
                "flip_to_third": bool(flip_third),
                "unchanged": bool(unchanged),
                "scores": scores,
                "arm": arm_name,
            }
            flipped = flip_source
        rows.append(row)
        flags.append(bool(flipped))
    return rows, flags


def _build_mismatched_pairs(inputs, matched_pairs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    items_by_id = {str(item.item_id): item for item in inputs.causal_items}
    source_items = [
        item
        for item in inputs.causal_items
        if item.in_matched_subset and item.base_correct and (not item.conflict_detected) and str(item.correct) in VALID_LETTERS
    ]
    by_letter: dict[str, list[CausalItem]] = {
        letter: sorted([item for item in source_items if str(item.correct) == letter], key=lambda x: (x.prompt_length, x.item_id))
        for letter in VALID_LETTERS
    }
    usage: dict[str, int] = {}
    letter_usage: dict[str, int] = {str(letter): 0 for letter in VALID_LETTERS}
    out: list[dict[str, Any]] = []
    for idx, pair in enumerate(matched_pairs):
        dest_id = str(pair.get("dest_id", "")).strip()
        if not dest_id:
            raise RuntimeError(f"Encountered patch row without dest_id at index={idx}")
        if dest_id not in items_by_id:
            raise RuntimeError(f"Destination item {dest_id} is missing from prepared MedQA inputs.")
        dest = items_by_id[dest_id]
        dest_letter = str(dest.correct)
        eligible = [letter for letter in VALID_LETTERS if letter != dest_letter and by_letter.get(letter)]
        if not eligible:
            raise RuntimeError(f"No mismatched source available for dest {dest.item_id}")
        min_letter_usage = min(int(letter_usage.get(letter, 0)) for letter in eligible)
        least_used_letters = [letter for letter in eligible if int(letter_usage.get(letter, 0)) == int(min_letter_usage)]
        letter = str(rng.choice(least_used_letters))
        bucket = by_letter[letter]
        min_usage = min(int(usage.get(item.item_id, 0)) for item in bucket)
        candidates = [item for item in bucket if int(usage.get(item.item_id, 0)) == min_usage]
        min_prompt_diff = min(abs(int(item.prompt_length) - int(dest.prompt_length)) for item in candidates)
        closest = [item for item in candidates if abs(int(item.prompt_length) - int(dest.prompt_length)) == int(min_prompt_diff)]
        source = rng.choice(closest)
        usage[source.item_id] = int(usage.get(source.item_id, 0) + 1)
        letter_usage[letter] = int(letter_usage.get(letter, 0) + 1)
        out.append(
            {
                "pair_id": f"{idx:03d}_{letter}_to_{dest_letter}_{source.item_id[:8]}_{dest.item_id[:8]}",
                "source_id": str(source.item_id),
                "dest_id": str(dest.item_id),
                "source_correct_letter": str(source.correct),
                "dest_correct_letter": str(dest.correct),
                "baseline_dest_answer": inputs.h4_by_id[str(dest.item_id)].get("base_letter"),
                "source_prompt_length": int(source.prompt_length),
                "dest_prompt_length": int(dest.prompt_length),
            }
        )
    return out


def _verify_disruption_pairs(pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        raise RuntimeError("No disruption pairs were constructed.")
    bad = [
        pair
        for pair in pairs
        if str(pair.get("source_correct_letter")) == str(pair.get("dest_correct_letter"))
    ]
    if bad:
        raise RuntimeError(f"Disruption pair set is not mismatched-letter; first bad pair={bad[0]}")


def _exclusive_disruption_outcome(
    *,
    post: str | None,
    dest_base_letter: str | None,
    source_correct_letter: str,
    dest_correct_letter: str,
) -> str:
    if post is not None and dest_base_letter is not None and str(post) == str(dest_base_letter):
        return "unchanged"
    if post is not None and str(post) == str(source_correct_letter):
        return "flip_to_source"
    if post is not None and str(post) == str(dest_correct_letter):
        return "flip_to_dest"
    return "flip_to_third"


def _mcnemar(top_flags: list[bool], flags: list[bool]) -> dict[str, Any]:
    return mcnemar_from_paired([bool(x) for x in top_flags], [bool(x) for x in flags])


def _status(mean_random: float) -> str:
    if mean_random < 0.30:
        return "holds"
    if mean_random <= 0.50:
        return "weakened"
    return "falsified"


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_prediction(out_dir / "prediction.json")

    pair_path = (PROJECT_ROOT / args.pair_path).resolve()
    head_importance_path = (PROJECT_ROOT / args.head_importance_path).resolve()
    pairs = _load_pairs(pair_path)
    scores = _load_head_scores(head_importance_path)
    arm_defs = _build_arm_definitions(scores, int(args.seed), int(args.n_random_draws))
    write_json(out_dir / "head_scores.json", {"scores": [value for _, value in sorted(scores.items())]})
    write_json(out_dir / "arm_definitions.json", arm_defs)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    inputs = _load_medqa_inputs(args, model, tokenizer, device)
    mismatched_pairs = _build_mismatched_pairs(inputs, pairs, int(args.seed))
    _verify_disruption_pairs(mismatched_pairs)
    write_json(out_dir / "disruption_pairs.json", {"pairs": mismatched_pairs})

    provenance = {
        "patching_function_path": str((PROJECT_ROOT / "src/medrag/activation_patch.py").resolve()),
        "imported_functions": ["cache_multihead_outputs", "run_with_multihead_patch"],
        "patched_tensor": "model.model.layers[layer].self_attn.o_proj forward_pre_hook input slice for each head; post-attention/pre-W_O at final prefill position",
        "same_35_pairs_for_every_arm": True,
        "n_pairs": len(pairs),
    }
    write_json(out_dir / "provenance.json", provenance)
    disruption_provenance = {
        "pair_set": "mismatched-letter source-destination pairs built from matched joint-patching destinations",
        "pair_source": str(pair_path),
        "n_disruption_pairs": len(mismatched_pairs),
        "same_disruption_pairs_for_every_arm": True,
        "source_correct_letter_not_equal_dest_correct_letter": True,
        "expected_top5_flip_to_source": 22,
        "expected_top5_flip_to_third": 0,
        "expected_top5_unchanged": 13,
    }
    write_json(out_dir / "disruption_provenance.json", disruption_provenance)

    arm_flags: dict[str, list[bool]] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    disruption: dict[str, Any] = {}

    deterministic = ["top5", "attribution_matched_non_top5", "bottom5"]
    for arm in ["top5"]:
        heads = [(int(x[0]), int(x[1])) for x in arm_defs[arm]["heads"]]
        rows, flags = _patch_arm(model=model, inputs=inputs, pairs=pairs, heads=heads, arm_name=arm, source_mode="matched")
        arm_rows[arm] = rows
        arm_flags[arm] = flags
        write_jsonl(out_dir / f"arm_{arm}.jsonl", rows)
        d_rows, _ = _patch_arm(
            model=model,
            inputs=inputs,
            pairs=pairs,
            heads=heads,
            arm_name=arm,
            source_mode="mismatched",
            mismatched_pairs=mismatched_pairs,
        )
        write_jsonl(out_dir / f"disruption_arm_{arm}.jsonl", d_rows)
        disruption[arm] = _disruption_summary(d_rows)

    if sum(arm_flags["top5"]) != 31:
        write_json(out_dir / "provenance.json", provenance | {"top5_reproduced_31_of_35": False, "top5_n_flipped": sum(arm_flags["top5"])})
        raise RuntimeError(f"Top-5 did not reproduce 31/35; got {sum(arm_flags['top5'])}/35. STOP.")
    provenance["top5_reproduced_31_of_35"] = True
    provenance["top5_n_flipped"] = 31
    write_json(out_dir / "provenance.json", provenance)

    top5_disruption = disruption["top5"]
    top5_matches_legacy = bool(
        int(top5_disruption["flip_to_source"]) == 22
        and int(top5_disruption["flip_to_third"]) == 0
        and int(top5_disruption["unchanged"]) == 13
    )
    if not bool(top5_disruption.get("partition_check_passed", False)):
        raise RuntimeError(f"Top-5 disruption partition check failed: {top5_disruption}")
    disruption_provenance["top5_reproduced_expected_disruption"] = top5_matches_legacy
    disruption_provenance["top5_legacy_mismatch_action"] = (
        "continued; corrected mutually exclusive outcome accounting is authoritative"
        if not top5_matches_legacy
        else "none"
    )
    disruption_provenance["top5_observed"] = top5_disruption
    write_json(out_dir / "disruption_provenance.json", disruption_provenance)

    for arm in ["attribution_matched_non_top5", "bottom5"]:
        heads = [(int(x[0]), int(x[1])) for x in arm_defs[arm]["heads"]]
        rows, flags = _patch_arm(model=model, inputs=inputs, pairs=pairs, heads=heads, arm_name=arm, source_mode="matched")
        arm_rows[arm] = rows
        arm_flags[arm] = flags
        write_jsonl(out_dir / f"arm_{arm}.jsonl", rows)
        d_rows, _ = _patch_arm(
            model=model,
            inputs=inputs,
            pairs=pairs,
            heads=heads,
            arm_name=arm,
            source_mode="mismatched",
            mismatched_pairs=mismatched_pairs,
        )
        write_jsonl(out_dir / f"disruption_arm_{arm}.jsonl", d_rows)
        disruption[arm] = _disruption_summary(d_rows)

    for random_arm in ("random5_same_layer", "random5_late_layer"):
        arm_flags[random_arm] = []
        arm_rows[random_arm] = []
        disruption[random_arm] = {"draws": []}
        aggregate_rows: list[dict[str, Any]] = []
        aggregate_disruption_rows: list[dict[str, Any]] = []
        for draw in arm_defs[random_arm]["draws"]:
            draw_name = f"{random_arm}_draw{int(draw['draw_index']):02d}"
            heads = [(int(x[0]), int(x[1])) for x in draw["heads"]]
            rows, flags = _patch_arm(model=model, inputs=inputs, pairs=pairs, heads=heads, arm_name=draw_name, source_mode="matched")
            arm_rows[draw_name] = rows
            arm_flags[draw_name] = flags
            write_jsonl(out_dir / f"arm_{draw_name}.jsonl", rows)
            aggregate_rows.extend(rows)
            d_rows, _ = _patch_arm(
                model=model,
                inputs=inputs,
                pairs=pairs,
                heads=heads,
                arm_name=draw_name,
                source_mode="mismatched",
                mismatched_pairs=mismatched_pairs,
            )
            write_jsonl(out_dir / f"disruption_arm_{draw_name}.jsonl", d_rows)
            aggregate_disruption_rows.extend(d_rows)
            disruption[random_arm]["draws"].append({"draw_index": int(draw["draw_index"]), **_disruption_summary(d_rows)})
        write_jsonl(out_dir / f"arm_{random_arm}.jsonl", aggregate_rows)
        write_jsonl(out_dir / f"disruption_arm_{random_arm}.jsonl", aggregate_disruption_rows)
        disruption[random_arm]["aggregate"] = _random_disruption_aggregate(disruption[random_arm]["draws"])

    results_rows = []
    top_flags = arm_flags["top5"]
    top_rate = sum(top_flags) / len(top_flags)
    for arm in deterministic:
        flags = arm_flags[arm]
        rs = _rate_summary(flags)
        results_rows.append({"arm": arm, "heads": arm_defs[arm]["heads"], **rs, "mcnemar_p_vs_top5": None if arm == "top5" else _mcnemar(top_flags, flags)["p_value"]})
    random_distribution: dict[str, Any] = {}
    for random_arm in ("random5_same_layer", "random5_late_layer"):
        rates = []
        ps = []
        for draw in arm_defs[random_arm]["draws"]:
            draw_name = f"{random_arm}_draw{int(draw['draw_index']):02d}"
            flags = arm_flags[draw_name]
            rate = sum(flags) / len(flags)
            rates.append(float(rate))
            ps.append(float(_mcnemar(top_flags, flags)["p_value"]))
        random_distribution[random_arm] = {
            "flip_rates": rates,
            "mean": float(mean(rates)),
            "sd": float(pstdev(rates)),
            "min": float(min(rates)),
            "max": float(max(rates)),
            "mcnemar_p_values_vs_top5": ps,
        }
        results_rows.append(
            {
                "arm": random_arm,
                "heads": f"{len(rates)} independent draws",
                "n_flipped": f"mean={float(mean(rates)) * 35:.2f}",
                "n": 35,
                "flip_rate": float(mean(rates)),
                "wilson95": "per-draw Wilson CIs in random_draw_distribution.json",
                "mcnemar_p_vs_top5": f"min={min(ps):.6g}, max={max(ps):.6g}",
            }
        )
    same_rates = random_distribution["random5_same_layer"]["flip_rates"]
    best_idx = max(range(len(same_rates)), key=lambda i: same_rates[i])
    best_name = f"random5_same_layer_draw{best_idx:02d}"
    best_p = _mcnemar(top_flags, arm_flags[best_name])["p_value"]
    separation_ci = _bootstrap_separation(top_rate, same_rates, int(args.seed))
    separation = {
        "top5_flip_rate": float(top_rate),
        "random5_same_layer_mean": float(random_distribution["random5_same_layer"]["mean"]),
        "random5_same_layer_sd": float(random_distribution["random5_same_layer"]["sd"]),
        "random5_same_layer_max": float(random_distribution["random5_same_layer"]["max"]),
        "separation": float(top_rate - random_distribution["random5_same_layer"]["mean"]),
        "separation_bootstrap_ci95": [float(separation_ci["low"]), float(separation_ci["high"])],
        "mcnemar_top5_vs_best_random_draw_p": float(best_p),
        "best_random5_same_layer_draw": int(best_idx),
        "contribution_1_status": _status(float(random_distribution["random5_same_layer"]["mean"])),
    }

    write_json(out_dir / "random_draw_distribution.json", random_distribution)
    write_json(out_dir / "disruption_control.json", disruption)
    write_json(out_dir / "separation.json", separation)
    _write_results_table(out_dir / "results_table.md", results_rows)
    write_json(
        out_dir / "summary.json",
        {
            "results": results_rows,
            "random_draw_distribution": random_distribution,
            "separation": separation,
            "metadata": build_metadata(
                args=args,
                input_files=[
                    PROJECT_ROOT / args.conflict_path,
                    PROJECT_ROOT / args.matched_subset_path,
                    PROJECT_ROOT / args.label_audit_path,
                    PROJECT_ROOT / args.h4_results_path,
                    pair_path,
                    head_importance_path,
                ],
                seed=int(args.seed),
                wall_clock_seconds=float(time.perf_counter() - started),
                project_root=PROJECT_ROOT,
            ),
        },
    )
    readme = (
        "# Matched Five-Head Causal Controls\n\n"
        f"Top-5 reproduced 31/35: yes. "
        f"Mean random5_same_layer flip rate: {random_distribution['random5_same_layer']['mean']:.4f}. "
        f"Best random draw p vs top5: {float(best_p):.4g}. "
        f"Contribution #1 status: {separation['contribution_1_status']}. "
        "Disruption control is in `disruption_control.json` with source/third/unchanged rates per arm.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(readme)
    return 0


def _disruption_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    unchanged = 0
    source = 0
    dest = 0
    third = 0
    for row in rows:
        outcome = str(row.get("outcome", ""))
        if outcome == "unchanged":
            unchanged += 1
        elif outcome == "flip_to_source":
            source += 1
        elif outcome == "flip_to_dest":
            dest += 1
        elif outcome == "flip_to_third":
            third += 1
        else:
            raise RuntimeError(f"Unknown disruption outcome: {outcome!r}")
    total = int(unchanged + source + dest + third)
    if total != n:
        raise RuntimeError(f"outcomes do not partition: {unchanged}+{source}+{dest}+{third} != {n}")
    if min(unchanged, source, dest, third) < 0:
        raise RuntimeError("Negative disruption count encountered.")
    return {
        "n": n,
        "unchanged": unchanged,
        "unchanged_rate": unchanged / max(n, 1),
        "flip_to_source": source,
        "flip_to_source_rate": source / max(n, 1),
        "flip_to_dest": dest,
        "flip_to_dest_rate": dest / max(n, 1),
        "flip_to_third": third,
        "flip_to_third_rate": third / max(n, 1),
        "partition_check_passed": True,
    }


def _random_disruption_aggregate(draws: list[dict[str, Any]]) -> dict[str, Any]:
    if not draws:
        return {
            "n_draws": 0,
            "flip_to_third_rate_mean": 0.0,
            "flip_to_third_rate_sd": 0.0,
        }
    third_rates = [float(row["flip_to_third_rate"]) for row in draws]
    source_rates = [float(row["flip_to_source_rate"]) for row in draws]
    dest_rates = [float(row["flip_to_dest_rate"]) for row in draws]
    unchanged_rates = [float(row["unchanged_rate"]) for row in draws]
    return {
        "n_draws": int(len(draws)),
        "flip_to_third_rate_mean": float(mean(third_rates)),
        "flip_to_third_rate_sd": float(pstdev(third_rates)),
        "flip_to_source_rate_mean": float(mean(source_rates)),
        "flip_to_source_rate_sd": float(pstdev(source_rates)),
        "flip_to_dest_rate_mean": float(mean(dest_rates)),
        "flip_to_dest_rate_sd": float(pstdev(dest_rates)),
        "unchanged_rate_mean": float(mean(unchanged_rates)),
        "unchanged_rate_sd": float(pstdev(unchanged_rates)),
    }


def _write_results_table(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = ["arm", "heads", "n_flipped", "n", "flip_rate", "wilson95", "mcnemar_p_vs_top5"]
    lines = ["# Matched Head Control Results", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
