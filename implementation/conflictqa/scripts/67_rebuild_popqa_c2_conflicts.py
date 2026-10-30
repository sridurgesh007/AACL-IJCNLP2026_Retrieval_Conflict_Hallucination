#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer
from conflictqa_pipeline.utils import (
    build_metadata,
    build_user_chat_prompt,
    clean_generated_answer,
    entity_match,
    generate_short_answer,
    read_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a model-specific PopQA C2-resistant conflict set from C1 candidates."
    )
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--candidates_path", default="data/popqa_conflict_candidates_mistral.jsonl")
    parser.add_argument("--out_dir", default="results/cross_model/mistral_popqa_c2_rebuild")
    parser.add_argument("--min_first_token_prob", type=float, default=0.65)
    parser.add_argument("--target_items", type=int, default=500, help="0 means write all genuine conflicts.")
    parser.add_argument("--max_stage1_items", type=int, default=0, help="0 means score all stage-1 rows.")
    parser.add_argument("--batch_size", type=int, default=8, help="Progress chunk size; decoding remains deterministic.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _require_columns(rows: list[dict[str, Any]], required: list[str], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"No candidate rows found: {path}")
    available = set(rows[0].keys())
    missing = [column for column in required if column not in available]
    if missing:
        raise RuntimeError(f"Candidate file missing required columns: {missing}. Available columns: {sorted(available)}")


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_generated_answer(str(item)) for item in value if clean_generated_answer(str(item))]
    if value is None:
        return []
    text = clean_generated_answer(str(value))
    return [text] if text else []


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_generated_answer(str(value))
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _matches_any(generated: str, references: list[str]) -> bool:
    return any(entity_match(generated, reference) for reference in references if str(reference).strip())


def _build_c2_prompt(tokenizer, question: str, context: str) -> str:
    return build_user_chat_prompt(
        tokenizer,
        f"Context: {str(context).strip()}\nQuestion: {str(question).strip()}\nAnswer in 1-3 words only.",
    )


def _encode_prompt(tokenizer, prompt: str, max_length: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length))
    return encoded["input_ids"][0].detach().cpu().long()


def _extract_subject(question: str, prop: str) -> str:
    del prop
    q = str(question).strip().rstrip("?")
    prefixes = [
        "Who directed ",
        "Who wrote ",
        "Who composed ",
        "Who produced ",
        "Who was the director of ",
        "Who was the author of ",
        "Who was the screenwriter for ",
        "Who was the producer of ",
        "Who was the composer of ",
        "Who is the director of ",
        "Who is the author of ",
        "Who is the screenwriter of ",
        "Who is the producer of ",
        "Who is the composer of ",
        "Who is the father of ",
        "Who is the mother of ",
        "Who was the father of ",
        "Who was the mother of ",
    ]
    for prefix in prefixes:
        if q.startswith(prefix):
            return q[len(prefix) :].strip()

    if q.startswith("What is the capital of "):
        return q[len("What is the capital of ") :].strip()
    if q.startswith("What is the genre of "):
        return q[len("What is the genre of ") :].strip()
    if q.startswith("What is the religion of "):
        return q[len("What is the religion of ") :].strip()
    if q.startswith("What country is "):
        return q[len("What country is ") :].split(" in")[0].strip()
    if q.startswith("Where was ") and " born" in q:
        return q[len("Where was ") :].replace(" born", "").strip()
    if q.startswith("What sport does ") and q.endswith(" play"):
        return q[len("What sport does ") : -len(" play")].strip()

    match = re.match(r"What is (.+?)'s .+", q)
    if match:
        return match.group(1).strip()

    for prefix in ["Who ", "What ", "Where ", "Which "]:
        if q.startswith(prefix):
            return q[len(prefix) :].strip()
    return q


def _fallback_context(question: str, correct_answer: str) -> str:
    q = str(question).strip().rstrip("?")
    a = str(correct_answer).strip()
    return f"In reference materials, {q.lower()} is associated with {a}."


def _build_context(question: str, correct_answer: str, prop: str) -> str:
    q = str(question).strip().rstrip("?")
    a = str(correct_answer).strip()
    relation = str(prop).strip().lower()
    subject = _extract_subject(q, relation)
    templates = {
        "director": f"{subject} was directed by {a}.",
        "author": f"{subject} was written by {a}.",
        "screenwriter": f"The screenplay for {subject} was written by {a}.",
        "producer": f"{subject} was produced by {a}.",
        "composer": f"The music for {subject} was composed by {a}.",
        "genre": f"{subject} is a work in the {a} genre.",
        "country": f"{subject} is located in {a}.",
        "capital": f"The capital city of {subject} is {a}.",
        "capital of": f"{a} serves as the capital of {subject}.",
        "occupation": f"{subject} works as a {a}.",
        "place of birth": f"{subject} was born in {a}.",
        "religion": f"{subject} is affiliated with the {a} religion.",
        "sport": f"{subject} is known for playing {a}.",
        "father": f"The father of {subject} is {a}.",
        "mother": f"The mother of {subject} is {a}.",
        "color": f"{subject} is associated with the color {a}.",
    }
    return templates.get(relation, _fallback_context(q, a))


def _clean_context(row: dict[str, Any], correct_answer: str) -> str:
    return _build_context(str(row["question"]), correct_answer, str(row.get("prop", "")))


def _with_rebuild_fields(
    row: dict[str, Any],
    *,
    clean_context: str,
    c2_generated: str,
    c2_classification: str,
    c2_parametric: bool,
    c2_context_faithful: bool,
    c2_other: bool,
    likely_label_error: bool,
) -> dict[str, Any]:
    out = dict(row)
    out["context"] = str(clean_context)
    out["clean_context"] = str(clean_context)
    out["c2_generated"] = str(c2_generated)
    out["c2_classification_rebuild"] = str(c2_classification)
    out["c2_parametric"] = bool(c2_parametric)
    out["c2_context_faithful"] = bool(c2_context_faithful)
    out["c2_other"] = bool(c2_other)
    out["likely_label_error"] = bool(likely_label_error)
    out.setdefault("explicit_injection_context", f"According to reference sources, the answer is {out['correct_answer']}.")
    return out


def _stratified_sample(rows: list[dict[str, Any]], target_items: int, seed: int, group_key: str) -> list[dict[str, Any]]:
    if int(target_items) <= 0 or len(rows) <= int(target_items):
        return list(rows)

    rng = random.Random(int(seed))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, ""))].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    total = float(len(rows))
    base_quota: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, group_rows in groups.items():
        exact = float(target_items) * float(len(group_rows)) / total
        quota = min(len(group_rows), int(math.floor(exact)))
        base_quota[key] = quota
        remainders.append((exact - quota, key))

    assigned = sum(base_quota.values())
    for _, key in sorted(remainders, key=lambda pair: (-float(pair[0]), pair[1])):
        if assigned >= int(target_items):
            break
        if base_quota[key] < len(groups[key]):
            base_quota[key] += 1
            assigned += 1

    if assigned < int(target_items):
        spill: list[tuple[str, dict[str, Any]]] = []
        for key, group_rows in groups.items():
            spill.extend((key, row) for row in group_rows[base_quota[key] :])
        rng.shuffle(spill)
        while assigned < int(target_items) and spill:
            key, _ = spill.pop()
            base_quota[key] += 1
            assigned += 1

    sampled: list[dict[str, Any]] = []
    for key, quota in base_quota.items():
        sampled.extend(groups[key][:quota])
    rng.shuffle(sampled)
    return sampled[: int(target_items)]


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    t0 = time.perf_counter()

    candidates_path = _resolve_project_path(args.candidates_path)
    out_dir = _resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_jsonl(candidates_path)
    required = ["item_id", "question", "correct_answer", "parametric_answer", "context", "llm_answer_prob", "prop", "s_pop"]
    _require_columns(candidates, required, candidates_path)

    stage1_rows = [
        row
        for row in candidates
        if math.isfinite(float(row.get("llm_answer_prob", float("nan"))))
        and float(row.get("llm_answer_prob", float("nan"))) >= float(args.min_first_token_prob)
    ]
    stage1_rows = sorted(stage1_rows, key=lambda row: (-float(row["llm_answer_prob"]), str(row.get("item_id", ""))))
    if int(args.max_stage1_items) > 0:
        stage1_rows = stage1_rows[: int(args.max_stage1_items)]
    if not stage1_rows:
        raise RuntimeError(
            f"No candidates survived llm_answer_prob >= {float(args.min_first_token_prob)} from {candidates_path}"
        )

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)

    genuine_conflicts: list[dict[str, Any]] = []
    likely_label_errors: list[dict[str, Any]] = []
    resolved_by_context: list[dict[str, Any]] = []
    c2_other_rows: list[dict[str, Any]] = []
    stats = Counter()

    chunk_size = max(1, int(args.batch_size))
    chunks = [stage1_rows[idx : idx + chunk_size] for idx in range(0, len(stage1_rows), chunk_size)]
    for chunk in tqdm(chunks, desc="PopQA C2-resistant rebuild"):
        for row in chunk:
            correct_answer = clean_generated_answer(str(row["correct_answer"]))
            parametric_answer = clean_generated_answer(str(row["parametric_answer"]))
            correct_aliases = _dedupe_keep_order([correct_answer] + _as_string_list(row.get("correct_aliases", [])))
            clean_context = _clean_context(row, correct_answer)

            parametric_matches_correct = _matches_any(parametric_answer, correct_aliases)
            if parametric_matches_correct:
                likely_label_errors.append(
                    _with_rebuild_fields(
                        row,
                        clean_context=clean_context,
                        c2_generated=parametric_answer,
                        c2_classification="ambiguous_parametric_matches_correct_alias",
                        c2_parametric=True,
                        c2_context_faithful=True,
                        c2_other=False,
                        likely_label_error=True,
                    )
                )
                stats["likely_label_error_parametric_matches_correct"] += 1
                continue

            prompt = _build_c2_prompt(tokenizer, str(row["question"]), clean_context)
            prompt_ids = _encode_prompt(tokenizer, prompt, int(args.max_length))
            generated, gen_ids = generate_short_answer(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                logits_processor=None,
            )
            generated = clean_generated_answer(generated)
            matches_correct = _matches_any(generated, correct_aliases)
            matches_parametric = entity_match(generated, parametric_answer)

            base_kwargs = {
                "clean_context": clean_context,
                "c2_generated": generated,
            }
            if matches_correct and matches_parametric:
                likely_label_errors.append(
                    _with_rebuild_fields(
                        row,
                        **base_kwargs,
                        c2_classification="ambiguous_correct_and_parametric",
                        c2_parametric=True,
                        c2_context_faithful=True,
                        c2_other=False,
                        likely_label_error=True,
                    )
                )
                stats["likely_label_error_ambiguous_c2"] += 1
            elif matches_parametric:
                genuine_conflicts.append(
                    _with_rebuild_fields(
                        row,
                        **base_kwargs,
                        c2_classification="parametric",
                        c2_parametric=True,
                        c2_context_faithful=False,
                        c2_other=False,
                        likely_label_error=False,
                    )
                )
                stats["genuine_conflict"] += 1
            elif matches_correct:
                resolved_by_context.append(
                    _with_rebuild_fields(
                        row,
                        **base_kwargs,
                        c2_classification="context_faithful",
                        c2_parametric=False,
                        c2_context_faithful=True,
                        c2_other=False,
                        likely_label_error=False,
                    )
                )
                stats["resolved_by_context"] += 1
            else:
                c2_other_rows.append(
                    _with_rebuild_fields(
                        row,
                        **base_kwargs,
                        c2_classification="other",
                        c2_parametric=False,
                        c2_context_faithful=False,
                        c2_other=True,
                        likely_label_error=False,
                    )
                )
                stats["c2_other"] += 1
            stats["generated_tokens_total"] += int(len(gen_ids))

    sampled_genuine = _stratified_sample(genuine_conflicts, int(args.target_items), int(args.seed), "prop")
    prop_counts_stage1 = Counter(str(row.get("prop", "")) for row in stage1_rows)
    prop_counts_genuine = Counter(str(row.get("prop", "")) for row in genuine_conflicts)
    prop_counts_written = Counter(str(row.get("prop", "")) for row in sampled_genuine)
    stage1_probs = [float(row["llm_answer_prob"]) for row in stage1_rows]
    genuine_probs = [float(row["llm_answer_prob"]) for row in genuine_conflicts]

    genuine_path = out_dir / "popqa_genuine_conflicts.jsonl"
    likely_label_errors_path = out_dir / "popqa_likely_label_errors.jsonl"
    resolved_path = out_dir / "popqa_resolved_by_context.jsonl"
    other_path = out_dir / "popqa_c2_other.jsonl"
    summary_path = out_dir / "rebuild_summary.json"

    write_jsonl(genuine_path, sampled_genuine)
    write_jsonl(likely_label_errors_path, likely_label_errors)
    write_jsonl(resolved_path, resolved_by_context)
    write_jsonl(other_path, c2_other_rows)

    n_stage2 = int(len(stage1_rows))
    summary = {
        "model_id": str(args.model_id),
        "candidates_path": str(candidates_path),
        "n_candidates_loaded": int(len(candidates)),
        "n_stage1_prob_filtered": int(len(stage1_rows)),
        "n_stage2_scored": int(n_stage2),
        "n_genuine_conflicts": int(len(genuine_conflicts)),
        "n_written_genuine_conflicts": int(len(sampled_genuine)),
        "n_resolved_by_context": int(len(resolved_by_context)),
        "n_likely_label_errors": int(len(likely_label_errors)),
        "n_c2_other": int(len(c2_other_rows)),
        "conflict_rate_stage2": float(len(genuine_conflicts) / max(n_stage2, 1)),
        "resolved_rate_stage2": float(len(resolved_by_context) / max(n_stage2, 1)),
        "likely_label_error_rate_stage2": float(len(likely_label_errors) / max(n_stage2, 1)),
        "other_rate_stage2": float(len(c2_other_rows) / max(n_stage2, 1)),
        "min_first_token_prob": float(args.min_first_token_prob),
        "target_items": int(args.target_items),
        "max_stage1_items": int(args.max_stage1_items),
        "max_new_tokens": int(args.max_new_tokens),
        "mean_stage1_first_token_prob": _mean(stage1_probs),
        "mean_genuine_first_token_prob": _mean(genuine_probs),
        "prop_counts_stage1": dict(sorted(prop_counts_stage1.items())),
        "prop_counts_genuine": dict(sorted(prop_counts_genuine.items())),
        "prop_counts_written": dict(sorted(prop_counts_written.items())),
        "output_files": {
            "popqa_genuine_conflicts": str(genuine_path.resolve()),
            "popqa_likely_label_errors": str(likely_label_errors_path.resolve()),
            "popqa_resolved_by_context": str(resolved_path.resolve()),
            "popqa_c2_other": str(other_path.resolve()),
            "rebuild_summary": str(summary_path.resolve()),
        },
        "stats": dict(stats),
        "metadata": build_metadata(
            args=args,
            input_files=[candidates_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=REPO_ROOT,
        ),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"genuine_conflicts={genuine_path}")
    print(f"likely_label_errors={likely_label_errors_path}")
    print(f"resolved_by_context={resolved_path}")
    print(f"c2_other={other_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
