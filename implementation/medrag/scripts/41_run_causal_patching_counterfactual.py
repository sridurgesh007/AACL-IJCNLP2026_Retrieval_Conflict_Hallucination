#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.common import argmax_non_whitespace_token_id, classify_answer_token
from evidence_aware.phase2.data import load_examples, prepare_examples
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.activation_patch import cache_head_output, run_with_patch
from medrag.causal_localization import build_metadata
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_jsonl, write_summary_json
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-task causal patching on CounterFact single-token prompts.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument("--out_dir", default="results/medqa_final_v3/causal/patching_counterfactual")
    parser.add_argument("--medqa_summary_path", default="results/medqa_final_v3/causal/patching/summary.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--target_layer", type=int, default=27)
    parser.add_argument("--target_head", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _rate(flags: list[bool]) -> tuple[float, list[float], int, int]:
    k = int(sum(1 for flag in flags if bool(flag)))
    n = int(len(flags))
    rate = float(k / max(n, 1))
    ci = wilson_ci(k, n) if n > 0 else (0.0, 0.0)
    return float(rate), [float(ci[0]), float(ci[1])], int(k), int(n)


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    examples = load_examples((PROJECT_ROOT / args.cache_path).resolve())
    prepared = prepare_examples(examples, tokenizer, int(args.max_length))

    classified: list[dict[str, Any]] = []
    for ex in prepared:
        prompt_ids = torch.tensor(ex.input_ids, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(prompt_ids.unsqueeze(0), use_cache=False, return_dict=True).logits[0, -1, :].float()
        first_token = argmax_non_whitespace_token_id(logits, tokenizer)
        label = classify_answer_token(first_token, ex.evidence_token_id, ex.parametric_token_id)
        classified.append(
            {
                "example": ex,
                "base_label": str(label),
                "prompt_length": int(len(ex.input_ids)),
            }
        )

    faithful_by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    hallucinated_by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in classified:
        ex = row["example"]
        if row["base_label"] == "faithful":
            faithful_by_token[int(ex.evidence_token_id)].append(row)
        elif row["base_label"] == "hallucinated":
            hallucinated_by_token[int(ex.evidence_token_id)].append(row)

    control_heads = [int(h) for h in range(int(model.config.num_attention_heads)) if int(h) != int(args.target_head)]
    pairs: list[dict[str, Any]] = []
    for token_id in sorted(hallucinated_by_token.keys()):
        if token_id not in faithful_by_token:
            continue
        faithful_pool = sorted(
            faithful_by_token[token_id],
            key=lambda row: (int(row["prompt_length"]), int(row["example"].id)),
        )
        hallucinated_pool = sorted(
            hallucinated_by_token[token_id],
            key=lambda row: (int(row["prompt_length"]), int(row["example"].id)),
        )
        for dest in hallucinated_pool:
            source = min(
                faithful_pool,
                key=lambda row: (
                    abs(int(row["prompt_length"]) - int(dest["prompt_length"])),
                    int(row["example"].id),
                ),
            )
            pairs.append(
                {
                    "token_id": int(token_id),
                    "source": source,
                    "dest": dest,
                }
            )
    if len(pairs) < int(args.n_pairs):
        raise RuntimeError(
            f"Only {len(pairs)} counterfactual pairs share the same evidence token id; requested {args.n_pairs}."
        )
    pairs = pairs[: int(args.n_pairs)]

    main_flags: list[bool] = []
    control_flags: list[bool] = []
    pooled_control_flags: list[bool] = []
    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        source_ex = pair["source"]["example"]
        dest_ex = pair["dest"]["example"]
        source_prompt = torch.tensor(source_ex.input_ids, dtype=torch.long, device=device)
        dest_prompt = torch.tensor(dest_ex.input_ids, dtype=torch.long, device=device)

        patch_main = cache_head_output(
            model=model,
            prompt_ids=source_prompt,
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=-1,
        )
        patched_main = run_with_patch(
            model=model,
            prompt_ids=dest_prompt,
            layer=int(args.target_layer),
            head=int(args.target_head),
            position=-1,
            patched_value=patch_main,
        )
        main_first = argmax_non_whitespace_token_id(patched_main, tokenizer)
        main_flip = bool(int(main_first) == int(dest_ex.evidence_token_id)) if main_first is not None else False

        sampled_head = control_heads[pair_index % len(control_heads)]
        control_answers: dict[str, bool] = {}
        for head in control_heads:
            patch_control = cache_head_output(
                model=model,
                prompt_ids=source_prompt,
                layer=int(args.target_layer),
                head=int(head),
                position=-1,
            )
            patched_control = run_with_patch(
                model=model,
                prompt_ids=dest_prompt,
                layer=int(args.target_layer),
                head=int(head),
                position=-1,
                patched_value=patch_control,
            )
            first = argmax_non_whitespace_token_id(patched_control, tokenizer)
            flip = bool(int(first) == int(dest_ex.evidence_token_id)) if first is not None else False
            control_answers[str(head)] = bool(flip)
            pooled_control_flags.append(bool(flip))

        sampled_flip = bool(control_answers[str(sampled_head)])
        main_flags.append(bool(main_flip))
        control_flags.append(bool(sampled_flip))
        rows.append(
            {
                "pair_id": f"{pair_index:03d}_{int(dest_ex.id)}",
                "source_id": int(source_ex.id),
                "dest_id": int(dest_ex.id),
                "evidence_token_id": int(dest_ex.evidence_token_id),
                "parametric_token_id": int(dest_ex.parametric_token_id),
                "flip_indicator_main": bool(main_flip),
                "sampled_control_head": int(sampled_head),
                "flip_indicator_control_A_sampled": bool(sampled_flip),
                "flip_indicator_control_A_all_heads": control_answers,
            }
        )

    main_rate, main_ci, main_k, n_pairs = _rate(main_flags)
    control_rate, control_ci, control_k, _ = _rate(control_flags)
    pooled_rate, pooled_ci, pooled_k, pooled_n = _rate(pooled_control_flags)

    medqa_summary_path = (PROJECT_ROOT / args.medqa_summary_path).resolve()
    medqa_comparison: dict[str, Any] | None = None
    if medqa_summary_path.exists():
        medqa_summary = json.loads(medqa_summary_path.read_text(encoding="utf-8"))
        medqa_rate = float(medqa_summary["main_flip_rate"])
        medqa_n = int(medqa_summary["n_pairs"])
        medqa_se = math.sqrt(max(medqa_rate * (1.0 - medqa_rate), 1e-12) / max(medqa_n, 1))
        counterfact_se = math.sqrt(max(main_rate * (1.0 - main_rate), 1e-12) / max(n_pairs, 1))
        medqa_comparison = {
            "medqa_main_flip_rate": float(medqa_rate),
            "medqa_n_pairs": int(medqa_n),
            "medqa_standard_error": float(medqa_se),
            "counterfactual_standard_error": float(counterfact_se),
            "absolute_rate_gap": float(abs(main_rate - medqa_rate)),
            "within_one_medqa_standard_error": bool(abs(main_rate - medqa_rate) <= medqa_se),
        }

    metadata_inputs: list[Path] = [(PROJECT_ROOT / args.cache_path).resolve()]
    if medqa_summary_path.exists():
        metadata_inputs.append(medqa_summary_path)
    summary = {
        "n_pairs": int(n_pairs),
        "main_flip_rate": float(main_rate),
        "main_flip_rate_ci95_wilson": main_ci,
        "main_flips": int(main_k),
        "control_A_flip_rate": float(control_rate),
        "control_A_flip_rate_ci95_wilson": control_ci,
        "control_A_flips": int(control_k),
        "control_A_all_heads_pooled_flip_rate": float(pooled_rate),
        "control_A_all_heads_pooled_flip_rate_ci95_wilson": pooled_ci,
        "control_A_all_heads_total_runs": int(pooled_n),
        "control_A_all_heads_total_flips": int(pooled_k),
        "mcnemar_main_vs_control_A": mcnemar_from_paired(main_flags, control_flags),
        "medqa_comparison": medqa_comparison,
        "cross_task_generalizes_within_one_medqa_se": bool(
            medqa_comparison is not None and bool(medqa_comparison["within_one_medqa_standard_error"])
        ),
        "metadata": build_metadata(
            args=args,
            input_files=metadata_inputs,
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=PROJECT_ROOT,
        ),
    }
    write_summary_json(
        out_dir / "summary.json",
        summary,
        rate_fields=["main_flip_rate", "control_A_flip_rate", "control_A_all_heads_pooled_flip_rate"],
    )
    write_jsonl(out_dir / "results.jsonl", rows)
    print(f"summary={out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
