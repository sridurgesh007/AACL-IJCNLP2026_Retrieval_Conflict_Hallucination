#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.modeling import load_model_and_tokenizer

NUMBER_PATTERN = re.compile(r"\d+\.?\d*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClashEval dosage-emission per-step margin diagnostic.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conflict_path", default="data/medical_conflict_500.jsonl")
    parser.add_argument("--out_dir", default="results/clasheval_step_diagnostic")
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash(start_path: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(start_path),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _metadata(args: argparse.Namespace, input_file: Path, wall_clock_seconds: float) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "git_commit_hash": _git_commit_hash(REPO_ROOT),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_file": str(input_file.resolve()),
        "input_file_sha256": _sha256_file(input_file) if input_file.exists() else "",
    }


def _load_conflicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _context_text(ex: dict[str, Any]) -> str:
    if ex.get("context"):
        return str(ex["context"]).strip()
    if ex.get("guideline_text"):
        return str(ex["guideline_text"]).strip()
    if ex.get("retrieved_passages"):
        passages = [str(x).strip() for x in (ex.get("retrieved_passages") or []) if str(x).strip()]
        if passages:
            return "\n\n".join(passages)
    return ""


def _truncate_context_to_dosage_sentence(context_text: str, context_answer: str) -> tuple[str, bool]:
    context = str(context_text or "").strip()
    dosage = _first_number(context_answer)
    if not context or not dosage:
        return context, False
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", context) if part.strip()]
    for part in parts:
        if dosage in part:
            return part, True
    hit = context.find(dosage)
    if hit < 0:
        return context, False
    left = max(context.rfind(".", 0, hit), context.rfind("!", 0, hit), context.rfind("?", 0, hit), context.rfind("\n", 0, hit))
    right_candidates = [pos for pos in (context.find(".", hit), context.find("!", hit), context.find("?", hit), context.find("\n", hit)) if pos >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(context)
    return context[left + 1 : right].strip(), True


def _build_grounded_prompt(question: str, context_text: str, context_answer: str) -> tuple[str, str, bool]:
    prompt_context, found = _truncate_context_to_dosage_sentence(context_text, context_answer)
    prompt_context = prompt_context or str(context_text or "").strip()
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Question: {question}\n"
        f"Clinical Context Sentence:\n{prompt_context}\n\n"
        "Complete the answer using ONLY the dosage number and unit.\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Dosage:"
    ), prompt_context, found


def _first_number(text: str) -> str:
    values = NUMBER_PATTERN.findall(str(text or ""))
    return str(values[0]).strip() if values else ""


def _first_number_value(text: str) -> float | None:
    token = _first_number(text)
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _number_token_variants(tokenizer, text: str) -> set[int]:
    number = _first_number(text)
    if not number:
        return set()
    variants = {
        number,
        f" {number}",
        f"\n{number}",
        f": {number}",
        f"{number}.",
        f"{number},",
    }
    out: set[int] = set()
    special = {int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])}
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if not ids:
            continue
        # Use the first number token for each surface form. Including punctuation
        # tokens from variants like "70." would inflate P_E/P_P with non-dosage mass.
        tok = int(ids[0])
        if tok not in special:
            out.add(tok)
    return out


def _resolve_pad_token_id(model, tokenizer) -> int:
    candidates = [
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    for cand in candidates:
        if isinstance(cand, (list, tuple)):
            cand = cand[0] if cand else None
        if cand is not None:
            return int(cand)
    return 0


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    pos = sum(1 for y, _ in pairs if y == 1)
    neg = sum(1 for y, _ in pairs if y == 0)
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][1])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][1] == ordered[i][1][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(ranks[idx] for idx, (y, _) in enumerate(pairs) if y == 1)
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg))


def _prob_mass_and_margin(logits: torch.Tensor, e_tokens: set[int], p_tokens: set[int]) -> tuple[float, float, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    vocab = int(probs.numel())
    e_valid = sorted({int(tok) for tok in e_tokens if 0 <= int(tok) < vocab})
    p_valid = sorted({int(tok) for tok in p_tokens if 0 <= int(tok) < vocab})
    p_e = float(probs[e_valid].sum().item()) if e_valid else 0.0
    p_p = float(probs[p_valid].sum().item()) if p_valid else 0.0
    margin = float(math.log(max(p_e, 1e-10)) - math.log(max(p_p, 1e-10)))
    return p_e, p_p, float(p_e + p_p), margin


def _contains_dosage_substring(text: str, dosage_text: str) -> bool:
    dosage = _first_number(dosage_text)
    if not dosage:
        return False
    return dosage in str(text or "")


def _classify_generated_window(window_text: str, ctx_text: str, par_text: str) -> str:
    if _contains_dosage_substring(window_text, ctx_text):
        return "faithful"
    if _contains_dosage_substring(window_text, par_text):
        return "hallucinated"
    if _first_number_value(window_text) is None:
        return "other"
    return "other"


def _find_dosage_step(
    *,
    tokenizer,
    generated_ids: list[int],
    generated_text: str,
    e_tokens: set[int],
    p_tokens: set[int],
) -> int | None:
    first_num = _first_number(generated_text)
    target_tokens = set(e_tokens) | set(p_tokens)
    for idx, tok in enumerate(generated_ids):
        if int(tok) in target_tokens:
            return int(idx)
        token_text = tokenizer.decode([int(tok)], skip_special_tokens=True)
        if NUMBER_PATTERN.search(token_text):
            return int(idx)
        if first_num:
            prefix = tokenizer.decode(generated_ids[: idx + 1], skip_special_tokens=True)
            if first_num in NUMBER_PATTERN.findall(prefix):
                return int(idx)
    return None


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _summarize_point(rows: list[dict[str, Any]], *, margin_key: str, mass_key: str | None = None) -> dict[str, Any]:
    faithful = [row for row in rows if row["label"] == "faithful" and row.get(margin_key) is not None]
    hall = [row for row in rows if row["label"] == "hallucinated" and row.get(margin_key) is not None]
    labels = [1 for _ in faithful] + [0 for _ in hall]
    margins = [float(row[margin_key]) for row in faithful] + [float(row[margin_key]) for row in hall]
    summary: dict[str, Any] = {
        "n_faithful": int(len(faithful)),
        "n_hallucinated": int(len(hall)),
        "margin_auc": _binary_auc(labels, margins),
        "mean_margin_faithful": _mean([float(row[margin_key]) for row in faithful]),
        "mean_margin_hallucinated": _mean([float(row[margin_key]) for row in hall]),
    }
    if mass_key is not None:
        summary.update(
            {
                "mean_mass_faithful": _mean([float(row[mass_key]) for row in faithful if row.get(mass_key) is not None]),
                "mean_mass_hallucinated": _mean([float(row[mass_key]) for row in hall if row.get(mass_key) is not None]),
                "mean_mass_all": _mean([float(row[mass_key]) for row in rows if row.get(mass_key) is not None]),
            }
        )
    return summary


def _decision(at_dosage_step: dict[str, Any]) -> str:
    auc = at_dosage_step.get("margin_auc")
    mean_mass = at_dosage_step.get("mean_mass_all")
    if auc is None or mean_mass is None:
        return "UNSOLVABLE"
    auc = float(auc)
    mean_mass = float(mean_mass)
    if auc >= 0.75 and mean_mass >= 0.05:
        return "SOLVABLE"
    if auc < 0.65 or mean_mass < 0.01:
        return "UNSOLVABLE"
    return "BORDERLINE"


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    t0 = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = (PROJECT_ROOT / args.conflict_path).resolve()
    all_rows = _load_conflicts(conflict_path)
    rows = all_rows[: max(0, int(args.n_examples))]
    if not rows:
        raise RuntimeError(f"No rows available in {conflict_path}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    per_items: list[dict[str, Any]] = []
    skip_stats = {"missing_question_or_context": 0, "missing_numeric_answers": 0, "empty_token_sets": 0}

    for idx, ex in enumerate(tqdm(rows, desc="ClashEval step diagnostic")):
        question = str(ex.get("question", "")).strip()
        context = _context_text(ex)
        context_answer = str(ex.get("context_answer", "")).strip()
        parametric_answer = str(ex.get("parametric_answer", "")).strip()
        if not question or not context:
            skip_stats["missing_question_or_context"] += 1
            continue
        if _first_number_value(context_answer) is None or _first_number_value(parametric_answer) is None:
            skip_stats["missing_numeric_answers"] += 1
            continue
        e_tokens = _number_token_variants(tokenizer, context_answer)
        p_tokens = _number_token_variants(tokenizer, parametric_answer)
        overlap = e_tokens & p_tokens
        e_tokens = e_tokens - overlap
        p_tokens = p_tokens - overlap
        if not e_tokens or not p_tokens:
            skip_stats["empty_token_sets"] += 1
            continue

        prompt, prompt_context, context_sentence_found = _build_grounded_prompt(question, context, context_answer)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length)).to(device)
        prompt_ids = encoded["input_ids"][0]
        attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(device)

        with torch.inference_mode():
            prefill_out = model(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            prefill_logits = prefill_out.logits[0, -1, :].detach().float().cpu()
            p_e_prefill, p_p_prefill, mass_prefill, prefill_margin = _prob_mass_and_margin(
                prefill_logits,
                e_tokens,
                p_tokens,
            )
            generated = model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=attention_mask,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                pad_token_id=_resolve_pad_token_id(model, tokenizer),
                output_scores=True,
                return_dict_in_generate=True,
            )
        new_ids = [int(x) for x in generated.sequences[0, prompt_ids.shape[0] :].detach().cpu().tolist()]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        first_10_text = tokenizer.decode(new_ids[:10], skip_special_tokens=True).strip()
        label = _classify_generated_window(first_10_text, context_answer, parametric_answer)
        dosage_step = _find_dosage_step(
            tokenizer=tokenizer,
            generated_ids=new_ids,
            generated_text=generated_text,
            e_tokens=e_tokens,
            p_tokens=p_tokens,
        )
        per_step: list[dict[str, Any]] = []
        for step, logits in enumerate(generated.scores):
            logits_cpu = logits[0].detach().float().cpu()
            p_e, p_p, mass, margin = _prob_mass_and_margin(logits_cpu, e_tokens, p_tokens)
            token_id = new_ids[step] if step < len(new_ids) else None
            per_step.append(
                {
                    "step": int(step),
                    "token": "" if token_id is None else tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "token_id": token_id,
                    "P_E": float(p_e),
                    "P_P": float(p_p),
                    "mass": float(mass),
                    "margin": float(margin),
                }
            )
        dosage_mass = None
        dosage_margin = None
        if dosage_step is not None and 0 <= int(dosage_step) < len(per_step):
            dosage_mass = float(per_step[int(dosage_step)]["mass"])
            dosage_margin = float(per_step[int(dosage_step)]["margin"])
        per_items.append(
            {
                "item_id": int(idx),
                "source": str(ex.get("source", "")),
                "label": label,
                "context_dosage": _first_number(context_answer),
                "parametric_dosage": _first_number(parametric_answer),
                "context_answer": context_answer,
                "parametric_answer": parametric_answer,
                "prompt_context": prompt_context,
                "context_sentence_found": bool(context_sentence_found),
                "dosage_step": dosage_step,
                "generated_text": generated_text,
                "first_10_generated_tokens_text": first_10_text,
                "prefill": {
                    "P_E": float(p_e_prefill),
                    "P_P": float(p_p_prefill),
                    "mass": float(mass_prefill),
                    "margin": float(prefill_margin),
                },
                "dosage_step_mass": dosage_mass,
                "dosage_step_margin": dosage_margin,
                "e_tokens": sorted(int(x) for x in e_tokens),
                "p_tokens": sorted(int(x) for x in p_tokens),
                "token_overlap_removed": int(len(overlap)),
                "per_step": per_step,
            }
        )

    at_dosage_rows = [row for row in per_items if row.get("dosage_step_margin") is not None]
    at_dosage_step = _summarize_point(at_dosage_rows, margin_key="dosage_step_margin", mass_key="dosage_step_mass")
    prefill_proxy_rows = [
        {**row, "prefill_margin_proxy": float(row["prefill"]["margin"])}
        for row in per_items
    ]
    at_prefill = _summarize_point(prefill_proxy_rows, margin_key="prefill_margin_proxy", mass_key=None)

    all_steps = [step for row in per_items for step in row["per_step"]]
    steps_above_001 = sum(1 for step in all_steps if float(step["mass"]) >= 0.001)
    steps_above_01 = sum(1 for step in all_steps if float(step["mass"]) >= 0.01)
    steps_above_05 = sum(1 for step in all_steps if float(step["mass"]) >= 0.05)
    mean_steps_per_item_above_01 = float(
        np.mean([sum(1 for step in row["per_step"] if float(step["mass"]) >= 0.01) for row in per_items])
    ) if per_items else 0.0
    decision = _decision(at_dosage_step)
    counts = {
        "faithful": sum(1 for row in per_items if row["label"] == "faithful"),
        "hallucinated": sum(1 for row in per_items if row["label"] == "hallucinated"),
        "other": sum(1 for row in per_items if row["label"] == "other"),
    }
    summary = {
        "n_items_requested": int(args.n_examples),
        "n_items": int(len(per_items)),
        "n_skipped": int(sum(skip_stats.values())),
        "skip_stats": skip_stats,
        "n_faithful": int(counts["faithful"]),
        "n_hallucinated": int(counts["hallucinated"]),
        "n_other": int(counts["other"]),
        "at_dosage_step": at_dosage_step,
        "at_prefill": at_prefill,
        "counterfact_reference_auc": 0.932,
        "prompt_variant": "question_first_context_dosage_sentence_assistant_dosage_slot",
        "classification_window_tokens": 10,
        "mass_floor_analysis": {
            "steps_above_0.001": int(steps_above_001),
            "steps_above_0.01": int(steps_above_01),
            "steps_above_0.05": int(steps_above_05),
            "mean_steps_per_item_above_0.01": float(mean_steps_per_item_above_01),
        },
        "decision": decision,
        "metadata": _metadata(args, conflict_path, float(time.perf_counter() - t0)),
    }
    _write_jsonl(out_dir / "per_item.jsonl", per_items)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "decision.txt").write_text(f"{decision}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"per_item={out_dir / 'per_item.jsonl'}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"decision={out_dir / 'decision.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
