#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVIDENCE_AWARE_ROOT = REPO_ROOT / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.phase2.common import build_prompt
from evidence_aware.phase2.data import load_examples
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.counterfact_cross_model import build_counterfact_evidence_text, build_metadata, project_path
from medrag.medqa_v3 import set_global_seed
from medrag.utils.io import write_json, write_jsonl


ARTICLE_RE = re.compile(r"^\s*(the|a|an)\s+", flags=re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"^\s*\d+[\.\)]\s+")
SPAN_STOP_RE = re.compile(r"\n|,\s*|\.\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CounterFact generated-P premise check v2.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="../Evidence_aware/data/counterfact_solvent.json")
    parser.add_argument(
        "--evidence_mode",
        default="relation_or_question_answer",
        choices=["cache", "question_answer", "relation_or_question_answer"],
    )
    parser.add_argument("--out_dir", default="results/rebuttal/premise_check_v2")
    parser.add_argument("--max_items", type=int, default=200)
    parser.add_argument(
        "--include_ids_path",
        default="",
        help=(
            "Optional JSON/JSONL/text file of example IDs to evaluate. "
            "When set, examples are selected by id and ordered by this file; max_items limits this ordered list."
        ),
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--c1_max_new_tokens", type=int, default=16)
    parser.add_argument("--prompt_diff_items", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _id_from_payload(row: Any) -> int | None:
    if isinstance(row, int):
        return int(row)
    if isinstance(row, str):
        stripped = row.strip()
        if stripped and re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        return None
    if isinstance(row, dict):
        for key in ("item_id", "example_id", "id", "source_index", "idx"):
            if key in row:
                try:
                    return int(row[key])
                except (TypeError, ValueError):
                    return None
    return None


def _load_include_ids(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing include_ids_path: {path}")
    ids: list[int] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"include_ids_path is empty: {path}")

    parsed_json = False
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        parsed_json = True
        if isinstance(payload, dict):
            for key in ("ids", "example_ids", "item_ids"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise RuntimeError(f"JSON include_ids_path must contain a list or ids/example_ids field: {path}")
        for item in payload:
            value = _id_from_payload(item)
            if value is not None:
                ids.append(int(value))

    if not parsed_json:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value: int | None = None
                if line.startswith("{"):
                    try:
                        value = _id_from_payload(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"Invalid JSONL include-id row at {path}:{line_no}: {exc}") from exc
                else:
                    value = _id_from_payload(line)
                if value is not None:
                    ids.append(int(value))

    deduped: list[int] = []
    seen: set[int] = set()
    for value in ids:
        if int(value) not in seen:
            deduped.append(int(value))
            seen.add(int(value))
    if not deduped:
        raise RuntimeError(f"No usable IDs found in include_ids_path: {path}")
    return deduped


def _select_examples_by_ids(examples: list[Any], include_ids: list[int], max_items: int) -> tuple[list[Any], dict[str, Any]]:
    by_id = {int(getattr(ex, "id")): ex for ex in examples}
    ordered_ids = list(include_ids)
    if int(max_items) > 0:
        ordered_ids = ordered_ids[: int(max_items)]
    selected: list[Any] = []
    missing: list[int] = []
    for item_id in ordered_ids:
        ex = by_id.get(int(item_id))
        if ex is None:
            missing.append(int(item_id))
            continue
        selected.append(ex)
    manifest = {
        "requested_ids": len(ordered_ids),
        "selected_examples": len(selected),
        "missing_ids": len(missing),
        "missing_id_sample": missing[:20],
        "first_selected_ids": [int(getattr(ex, "id")) for ex in selected[:20]],
    }
    if missing:
        raise RuntimeError(
            "include_ids_path contains IDs not present in cache. "
            f"missing={len(missing)} sample={missing[:20]}"
        )
    return selected, manifest


def _normalize_text(text: str) -> str:
    value = str(text or "").casefold()
    value = value.translate(PUNCT_TABLE)
    value = re.sub(r"\s+", "", value)
    return value.strip()


def _clean_token_text(text: str) -> str:
    token = str(text or "")
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.strip()
    token = token.strip(string.punctuation)
    return token.strip()


def token_set_from_text(tokenizer, text: str) -> set[int]:
    cleaned = _clean_token_text(text)
    forms = [cleaned, cleaned.lower(), cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned]
    out: set[int] = set()
    for form in forms:
        form = str(form).strip()
        if not form:
            continue
        for variant in (f" {form}", form):
            ids = tokenizer(variant, add_special_tokens=False).get("input_ids", [])
            if ids:
                out.add(int(ids[0]))
    return out


def _first_token_id(tokenizer, text: str, *, leading_space: bool) -> int | None:
    value = str(text or "").strip()
    if not value:
        return None
    token_text = f" {value}" if leading_space else value
    ids = tokenizer(token_text, add_special_tokens=False).get("input_ids", [])
    return None if not ids else int(ids[0])


def _answer_token_stats(tokenizer, logits: torch.Tensor, answer: str) -> dict[str, Any]:
    probs = torch.softmax(logits.float(), dim=-1)
    candidates: list[dict[str, Any]] = []
    for name, leading in (("leading_space", True), ("no_space", False)):
        token_id = _first_token_id(tokenizer, answer, leading_space=leading)
        if token_id is None or token_id < 0 or token_id >= int(logits.numel()):
            candidates.append(
                {
                    "variant": name,
                    "token_id": None,
                    "token_text": "",
                    "logit": None,
                    "prob": 0.0,
                    "rank": None,
                }
            )
            continue
        logit = float(logits[int(token_id)].item())
        rank = int(torch.sum(logits.float() > logits[int(token_id)].float()).item()) + 1
        candidates.append(
            {
                "variant": name,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": logit,
                "prob": float(probs[int(token_id)].item()),
                "rank": rank,
            }
        )
    valid = [row for row in candidates if row["token_id"] is not None]
    best = max(valid, key=lambda row: float(row["prob"])) if valid else candidates[0]
    return {"selected": best, "variants": candidates}


def _extract_answer_span(generated_text: str) -> tuple[str, bool]:
    text = str(generated_text or "").strip()
    text = LIST_MARKER_RE.sub("", text, count=1).strip()
    text = ARTICLE_RE.sub("", text, count=1).strip()
    hit = SPAN_STOP_RE.search(text)
    if hit is not None:
        text = text[: hit.start()].strip()
    text = text.strip()
    return text, bool(not text)


def _c2_user_text(evidence_text: str, question: str) -> str:
    return build_prompt(str(evidence_text), str(question))


def _c1_user_text(question: str) -> str:
    return f"Question: {str(question)}\nAnswer: "


def _chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": str(user_text)}]
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise RuntimeError("Tokenizer has no chat_template; premise check v2 requires apply_chat_template.")
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def _encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length)).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _generate_greedy(
    *,
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[int], torch.Tensor, int, str]:
    input_ids, attention_mask = _encode_prompt(tokenizer, prompt, max_length, device)
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = prefill.logits[0, -1, :].detach().float().cpu()
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = [int(x) for x in out[0, input_ids.shape[1] :].detach().cpu().tolist()]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    argmax_id = int(torch.argmax(logits).item())
    argmax_token = tokenizer.decode([argmax_id], skip_special_tokens=False)
    return generated_text, new_ids, logits, argmax_id, argmax_token


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if int(rank) == 1:
        return "rank_1"
    if 2 <= int(rank) <= 5:
        return "rank_2_5"
    if 6 <= int(rank) <= 20:
        return "rank_6_20"
    if 21 <= int(rank) <= 100:
        return "rank_21_100"
    return "rank_over_100"


def _frac(count: int, n: int) -> float:
    return float(int(count) / max(int(n), 1))


def _build_prompt_diff(rows: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for idx, row in enumerate(rows, start=1):
        chunks.append("=" * 88)
        chunks.append(f"ITEM {idx} | item_id={row['item_id']}")
        chunks.append("C1 PROMPT (NO EVIDENCE):")
        chunks.append(str(row["c1_prompt"]))
        chunks.append("-" * 88)
        chunks.append("C2 PROMPT (WITH EVIDENCE):")
        chunks.append(str(row["c2_prompt"]))
        chunks.append("-" * 88)
        chunks.append("VERIFICATION NOTE: C1 and C2 use the same chat template; C2 inserts the Evidence block.")
    return "\n".join(chunks) + "\n"


def main() -> int:
    args = parse_args()
    set_global_seed(int(args.seed))
    started = time.perf_counter()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at_utc": _now_iso(),
        "purpose": "CounterFact generated-P premise check v2; diagnosis only, no intervention.",
        "supersedes": "GENERATED_P_PILOT.md and generated_p_pilot invalid run",
        "model_id": str(args.model_id),
        "cache_path": str(args.cache_path),
        "include_ids_path": str(args.include_ids_path),
        "evidence_mode": str(args.evidence_mode),
        "out_dir": str(args.out_dir),
        "max_items": int(args.max_items),
        "max_length": int(args.max_length),
        "c1_max_new_tokens": int(args.c1_max_new_tokens),
        "prompt_diff_items": int(args.prompt_diff_items),
        "seed": int(args.seed),
        "device": str(args.device),
        "torch_dtype": str(args.torch_dtype),
        "frozen_before_model_load": True,
        "do_sample": False,
    }
    write_json(out_dir / "config.json", config)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    cache_path = project_path(PROJECT_ROOT, args.cache_path)
    examples = load_examples(cache_path)
    include_manifest: dict[str, Any] | None = None
    include_ids_path = str(args.include_ids_path).strip()
    if include_ids_path:
        resolved_include_ids_path = project_path(PROJECT_ROOT, include_ids_path)
        include_ids = _load_include_ids(resolved_include_ids_path)
        examples, include_manifest = _select_examples_by_ids(examples, include_ids, int(args.max_items))
        write_json(out_dir / "include_ids_manifest.json", include_manifest | {"include_ids_path": str(resolved_include_ids_path)})
    elif int(args.max_items) > 0:
        examples = examples[: int(args.max_items)]

    per_item: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    prompt_diff_rows: list[dict[str, Any]] = []

    for ex in examples:
        item_id = str(getattr(ex, "id", len(per_item)))
        question = str(getattr(ex, "question", "")).strip()
        target_true = str(getattr(ex, "answer_solid", "")).strip()
        target_new = str(getattr(ex, "answer_liquid", "")).strip()
        if not question:
            dropped.append({"item_id": item_id, "reason": "missing_question"})
            continue
        if not target_true:
            dropped.append({"item_id": item_id, "reason": "missing_target_true"})
            continue
        if not target_new:
            dropped.append({"item_id": item_id, "reason": "missing_target_new"})
            continue

        evidence_text = build_counterfact_evidence_text(ex, str(args.evidence_mode))
        c1_user = _c1_user_text(question)
        c2_user = _c2_user_text(evidence_text, question)
        c1_prompt = _chat_prompt(tokenizer, c1_user)
        c2_prompt = _chat_prompt(tokenizer, c2_user)
        if len(prompt_diff_rows) < int(args.prompt_diff_items):
            prompt_diff_rows.append({"item_id": item_id, "c1_prompt": c1_prompt, "c2_prompt": c2_prompt})

        c1_generation, c1_ids, logits, argmax_id, argmax_token = _generate_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=c1_prompt,
            max_length=int(args.max_length),
            max_new_tokens=int(args.c1_max_new_tokens),
            device=device,
        )
        extracted, extraction_failed = _extract_answer_span(c1_generation)

        true_stats = _answer_token_stats(tokenizer, logits, target_true)
        new_stats = _answer_token_stats(tokenizer, logits, target_new)
        true_selected = true_stats["selected"]
        new_selected = new_stats["selected"]
        logit_true = true_selected["logit"]
        logit_new = new_selected["logit"]
        logit_margin = None
        if logit_true is not None and logit_new is not None:
            logit_margin = float(logit_true) - float(logit_new)

        gold_first_token_text = str(true_selected.get("token_text") or "")
        extracted_first_token_id = _first_token_id(tokenizer, extracted, leading_space=True)
        extracted_first_token_text = (
            "" if extracted_first_token_id is None else tokenizer.decode([int(extracted_first_token_id)], skip_special_tokens=False)
        )
        agree_first_token = _normalize_text(gold_first_token_text) == _normalize_text(extracted_first_token_text)
        agree_token_set = bool(token_set_from_text(tokenizer, target_true) & token_set_from_text(tokenizer, extracted))
        agree_string = _normalize_text(extracted) == _normalize_text(target_true)
        norm_true = _normalize_text(target_true)
        norm_generation = _normalize_text(c1_generation)
        agree_substring = bool(norm_true and norm_true in norm_generation)

        per_item.append(
            {
                "item_id": item_id,
                "query": question,
                "target_true": target_true,
                "target_new": target_new,
                "c1_prompt": c1_prompt,
                "c1_generation": c1_generation,
                "c1_generated_token_ids": c1_ids,
                "extracted_answer": extracted,
                "extraction_failed": bool(extraction_failed),
                "agree_first_token": bool(agree_first_token),
                "agree_token_set": bool(agree_token_set),
                "agree_string": bool(agree_string),
                "agree_substring": bool(agree_substring),
                "p_target_true": float(true_selected["prob"]),
                "p_target_new": float(new_selected["prob"]),
                "rank_target_true": true_selected["rank"],
                "rank_target_new": new_selected["rank"],
                "logit_margin_c1": logit_margin,
                "argmax_token": str(argmax_token),
                "argmax_token_id": int(argmax_id),
                "target_true_token": true_stats,
                "target_new_token": new_stats,
                "rank_bucket": _rank_bucket(true_selected["rank"]),
            }
        )

    (out_dir / "prompt_diff_check.txt").write_text(_build_prompt_diff(prompt_diff_rows), encoding="utf-8")
    write_jsonl(out_dir / "per_item.jsonl", per_item)
    write_jsonl(out_dir / "dropped_items.jsonl", dropped)

    disagreements = [
        {
            "item_id": row["item_id"],
            "query": row["query"],
            "target_true": row["target_true"],
            "target_new": row["target_new"],
            "c1_generation": row["c1_generation"],
            "extracted_answer": row["extracted_answer"],
            "argmax_token": row["argmax_token"],
            "rank_target_true": row["rank_target_true"],
            "p_target_true": row["p_target_true"],
        }
        for row in per_item
        if not bool(row["agree_substring"])
    ]
    write_jsonl(out_dir / "disagreements.jsonl", disagreements)

    taxonomy_template = {
        "status": "requires_human_labeling",
        "instruction": "Label each item into exactly one bucket: same_answer_different_surface, granularity_mismatch, different_belief, degenerate_output, extraction_artifact.",
        "counts": {
            "same_answer_different_surface": None,
            "granularity_mismatch": None,
            "different_belief": None,
            "degenerate_output": None,
            "extraction_artifact": None,
        },
        "items": [{**row, "human_label": None} for row in disagreements[:30]],
    }
    write_json(out_dir / "disagreement_taxonomy.json", taxonomy_template)

    n = len(per_item)
    ranks = [int(row["rank_target_true"]) for row in per_item if row["rank_target_true"] is not None]
    p_true = [float(row["p_target_true"]) for row in per_item]
    margin_values = [float(row["logit_margin_c1"]) for row in per_item if row["logit_margin_c1"] is not None]
    bucket_counts = {
        "rank_1": sum(1 for row in per_item if row["rank_bucket"] == "rank_1"),
        "rank_2_5": sum(1 for row in per_item if row["rank_bucket"] == "rank_2_5"),
        "rank_6_20": sum(1 for row in per_item if row["rank_bucket"] == "rank_6_20"),
        "rank_21_100": sum(1 for row in per_item if row["rank_bucket"] == "rank_21_100"),
        "rank_over_100": sum(1 for row in per_item if row["rank_bucket"] == "rank_over_100"),
    }
    n_failed = sum(1 for row in per_item if bool(row["extraction_failed"]))
    n_rank_over_20 = bucket_counts["rank_21_100"] + bucket_counts["rank_over_100"]
    summary = {
        "n": int(n),
        "n_dropped": int(len(dropped)),
        "n_extraction_failed": int(n_failed),
        "agreement": {
            "agree_first_token": _frac(sum(1 for row in per_item if bool(row["agree_first_token"])), n),
            "agree_token_set": _frac(sum(1 for row in per_item if bool(row["agree_token_set"])), n),
            "agree_string": _frac(sum(1 for row in per_item if bool(row["agree_string"])), n),
            "agree_substring": _frac(sum(1 for row in per_item if bool(row["agree_substring"])), n),
        },
        "belief": {
            "frac_rank_1": _frac(bucket_counts["rank_1"], n),
            "frac_rank_2_5": _frac(bucket_counts["rank_2_5"], n),
            "frac_rank_6_20": _frac(bucket_counts["rank_6_20"], n),
            "frac_rank_21_100": _frac(bucket_counts["rank_21_100"], n),
            "frac_rank_over_100": _frac(bucket_counts["rank_over_100"], n),
            "n_rank_missing": int(sum(1 for row in per_item if row["rank_bucket"] == "missing")),
            "median_p_target_true": float(median(p_true)) if p_true else None,
            "median_rank_target_true": int(median(ranks)) if ranks else None,
            "frac_logit_margin_c1_positive": _frac(sum(1 for value in margin_values if float(value) > 0.0), len(margin_values)),
            "rank_bucket_counts": bucket_counts,
        },
        "subset_audit": {
            "n_model_holds_belief_rank1": int(bucket_counts["rank_1"]),
            "n_model_does_not_hold_belief_rank_over_20": int(n_rank_over_20),
            "frac_subset_potentially_non_conflict": _frac(n_rank_over_20, n),
        },
        "prior_invalid_run": {
            "agreement_rate": 0.39,
            "invalidated_because": "raw completion prompt on instruct model; first-BPE-token equality",
        },
        "metadata": build_metadata(
            args=args,
            input_files=[cache_path] + ([project_path(PROJECT_ROOT, include_ids_path)] if include_ids_path else []),
            seed=int(args.seed),
            wall_clock_seconds=float(time.perf_counter() - started),
            project_root=REPO_ROOT,
        ),
    }
    if include_manifest is not None:
        summary["include_ids_manifest"] = include_manifest
    write_json(out_dir / "summary.json", summary)
    readme = (
        "# CounterFact Premise Check V2\n\n"
        "This reruns the generated-P premise check with the Llama chat template, full-span agreement metrics, "
        "and next-token belief ranks instead of the invalid raw-completion first-BPE-token test. "
        f"On {n} retained items, agree_substring={summary['agreement']['agree_substring']:.4f} and "
        f"frac_rank_1={summary['belief']['frac_rank_1']:.4f}. "
        f"The subset audit marks {summary['subset_audit']['frac_subset_potentially_non_conflict']:.4f} "
        "of retained items as potentially non-conflicts using rank_target_true > 20.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary={out_dir / 'summary.json'}")
    print(f"per_item={out_dir / 'per_item.jsonl'}")
    print(f"disagreements={out_dir / 'disagreements.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
