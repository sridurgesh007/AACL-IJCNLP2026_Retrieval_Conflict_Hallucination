#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
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
    prefill_logits,
    read_table,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct a PopQA genuine-conflict set for Llama-3.1-8B.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--popqa_path", default="data/raw/popQA.tsv")
    parser.add_argument("--download_url", default="")
    parser.add_argument("--out_path", default="data/popqa_conflict_500.jsonl")
    parser.add_argument("--candidates_path", default="data/popqa_conflict_candidates.jsonl")
    parser.add_argument("--results_dir", default="results/popqa")
    parser.add_argument("--target_items", type=int, default=500)
    parser.add_argument("--max_source_items", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--min_first_token_prob", type=float, default=0.3)
    parser.add_argument("--max_entity_words", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _build_no_context_prompt(tokenizer, question: str) -> str:
    return build_user_chat_prompt(
        tokenizer,
        f"Question: {str(question).strip()}\nAnswer in 1-3 words only.",
    )


def _build_injection_context(correct_answer: str) -> str:
    return f"According to reference sources, the answer is {str(correct_answer).strip()}."


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


def _encode_prompt(tokenizer, prompt: str, max_length: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length))
    return encoded["input_ids"][0].detach().cpu().long()


def _sha1_item_id(question: str, correct_answer: str, parametric_answer: str, prop: str) -> str:
    payload = "||".join([str(question), str(correct_answer), str(parametric_answer), str(prop)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _parse_alias_list(cell: Any) -> list[str]:
    text = str(cell or "").strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            payload = parser(text)
            if isinstance(payload, list):
                return [clean_generated_answer(str(item)) for item in payload if clean_generated_answer(str(item))]
        except Exception:
            continue
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        parts = [part.strip(" \"'`") for part in inner.split(",")]
        return [clean_generated_answer(part) for part in parts if clean_generated_answer(part)]
    cleaned = clean_generated_answer(text)
    return [cleaned] if cleaned else []


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _entity_match_any(generated: str, references: list[str]) -> bool:
    return any(entity_match(generated, reference) for reference in references if str(reference).strip())


def _require_columns(rows: list[dict[str, Any]], required: list[str]) -> None:
    if not rows:
        raise RuntimeError("PopQA table is empty.")
    available = set(rows[0].keys())
    missing = [column for column in required if column not in available]
    if missing:
        raise RuntimeError(f"PopQA file missing required columns: {missing}. Available columns: {sorted(available)}")


def _first_token_prob(logits: torch.Tensor, token_id: int) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    vocab = int(probs.numel())
    if not (0 <= int(token_id) < vocab):
        return float("nan")
    return float(probs[int(token_id)].item())


def _stratified_sample(rows: list[dict[str, Any]], target_items: int, seed: int, group_key: str) -> list[dict[str, Any]]:
    if target_items <= 0:
        raise ValueError("target_items must be positive.")
    if len(rows) < target_items:
        raise RuntimeError(f"Only {len(rows)} genuine conflict candidates available; need {target_items}.")

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
        if assigned >= target_items:
            break
        if base_quota[key] < len(groups[key]):
            base_quota[key] += 1
            assigned += 1

    if assigned < target_items:
        spill = []
        for key, group_rows in groups.items():
            spill.extend((key, row) for row in group_rows[base_quota[key] :])
        rng.shuffle(spill)
        while assigned < target_items and spill:
            key, row = spill.pop()
            groups[key].append(row)
            base_quota[key] += 1
            assigned += 1

    sampled: list[dict[str, Any]] = []
    for key, quota in base_quota.items():
        sampled.extend(groups[key][:quota])
    rng.shuffle(sampled)
    return sampled[:target_items]


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    t0 = time.perf_counter()

    popqa_path = (PROJECT_ROOT / args.popqa_path).resolve()
    if not popqa_path.exists():
        download_url = str(args.download_url).strip()
        if download_url:
            popqa_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(download_url, popqa_path)
        else:
            raise FileNotFoundError(
                f"PopQA TSV not found at {popqa_path}. Provide --popqa_path or pass --download_url explicitly."
            )

    rows = read_table(popqa_path)
    _require_columns(rows, ["question", "obj", "possible_answers", "prop", "s_pop"])
    if int(args.max_source_items) > 0:
        rows = rows[: int(args.max_source_items)]
    if not rows:
        raise RuntimeError(f"No PopQA rows available in {popqa_path}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)

    candidates: list[dict[str, Any]] = []
    stats = Counter()
    for idx, row in enumerate(tqdm(rows, desc="PopQA no-context screening")):
        question = str(row["question"]).strip()
        correct_answer = clean_generated_answer(str(row["obj"]))
        correct_aliases = _dedupe_keep_order([correct_answer] + _parse_alias_list(row.get("possible_answers", "")))
        prop = str(row["prop"]).strip()
        s_pop = float(row["s_pop"])
        prompt = _build_no_context_prompt(tokenizer, question)
        prompt_ids = _encode_prompt(tokenizer, prompt, int(args.max_length))
        logits = prefill_logits(model=model, prompt_ids=prompt_ids, device=device)
        llm_answer, gen_ids = generate_short_answer(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            logits_processor=None,
        )
        llm_answer = clean_generated_answer(llm_answer)
        llm_correct = bool(_entity_match_any(llm_answer, correct_aliases))
        first_token_prob = _first_token_prob(logits, gen_ids[0]) if gen_ids else float("nan")
        answer_word_count = len([part for part in correct_answer.split() if part])
        parametric_word_count = len([part for part in llm_answer.split() if part])

        keep = (
            not llm_correct
            and math.isfinite(float(first_token_prob))
            and float(first_token_prob) > float(args.min_first_token_prob)
            and answer_word_count <= int(args.max_entity_words)
            and parametric_word_count <= int(args.max_entity_words)
            and not _entity_match_any(llm_answer, correct_aliases)
        )
        stats["total"] += 1
        stats["correct"] += int(llm_correct)
        stats["wrong"] += int(not llm_correct)
        stats["kept"] += int(keep)
        if not keep:
            continue

        candidate = {
            "item_id": _sha1_item_id(question, correct_answer, llm_answer, prop),
            "question": question,
            "correct_answer": correct_answer,
            "correct_aliases": correct_aliases,
            "parametric_answer": llm_answer,
            "context": _build_context(question, correct_answer, prop),
            "explicit_injection_context": _build_injection_context(correct_answer),
            "llm_answer_prob": float(first_token_prob),
            "llm_correct": bool(llm_correct),
            "prop": prop,
            "s_pop": float(s_pop),
            "source_dataset": "PopQA",
            "source_index": int(idx),
            "correct_answer_word_count": int(answer_word_count),
            "parametric_answer_word_count": int(parametric_word_count),
            "prompt": prompt,
        }
        candidates.append(candidate)

    sampled = _stratified_sample(candidates, int(args.target_items), int(args.seed), "prop")

    out_path = (PROJECT_ROOT / args.out_path).resolve()
    candidates_path = (PROJECT_ROOT / args.candidates_path).resolve()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(candidates_path, candidates)
    write_jsonl(out_path, sampled)

    prop_counts_candidates = Counter(str(row["prop"]) for row in candidates)
    prop_counts_sampled = Counter(str(row["prop"]) for row in sampled)
    summary = {
        "n_source_rows": int(len(rows)),
        "n_candidates": int(len(candidates)),
        "n_sampled": int(len(sampled)),
        "target_items": int(args.target_items),
        "min_first_token_prob": float(args.min_first_token_prob),
        "max_entity_words": int(args.max_entity_words),
        "mean_candidate_first_token_prob": float(sum(float(row["llm_answer_prob"]) for row in candidates) / max(len(candidates), 1)) if candidates else 0.0,
        "prop_counts_candidates": dict(sorted(prop_counts_candidates.items())),
        "prop_counts_sampled": dict(sorted(prop_counts_sampled.items())),
        "stats": dict(stats),
        "metadata": build_metadata(
            args=args,
            input_files=[popqa_path],
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - t0),
            project_root=REPO_ROOT,
        ),
    }
    write_json(results_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"candidates={candidates_path}")
    print(f"sampled={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
