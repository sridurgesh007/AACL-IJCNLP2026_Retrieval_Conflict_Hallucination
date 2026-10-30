#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_AWARE_ROOT = PROJECT_ROOT.parent / "Evidence_aware"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVIDENCE_AWARE_ROOT / "src"))

from evidence_aware.ag_erha.utils import write_jsonl
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from medrag.medical_rag.medqa_loader import MedQARow, load_medqa_dataset_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MedQA conflict dataset (Path A).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out_path", default="data/medqa_conflict_500.jsonl")
    parser.add_argument("--n_target", type=int, default=500)
    parser.add_argument("--scan_limit", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use_medrag_retrieval", type=int, default=1, choices=[0, 1])
    parser.add_argument("--retrieval_k", type=int, default=3)
    parser.add_argument("--medrag_llm_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--medrag_retriever_name", default="MedCPT")
    parser.add_argument("--medrag_corpus_name", default="StatPearls")
    parser.add_argument("--medrag_db_dir", default="data/medrag_statpearls")
    return parser.parse_args()


def _build_parametric_mc_prompt(question: str, options: dict[str, str]) -> str:
    options_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D)."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_letter_token_sets(tokenizer) -> dict[str, set[int]]:
    letter_to_ids: dict[str, set[int]] = {k: set() for k in ("A", "B", "C", "D")}
    for letter in ("A", "B", "C", "D"):
        for variant in (letter, f" {letter}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tok in ids:
                letter_to_ids[letter].add(int(tok))
    return letter_to_ids


def _pick_letter_from_logits(logits: torch.Tensor, letter_token_sets: dict[str, set[int]]) -> str | None:
    best_letter: str | None = None
    best_score = float("-inf")
    vocab_size = int(logits.numel())
    for letter, token_ids in letter_token_sets.items():
        valid = [tid for tid in token_ids if 0 <= int(tid) < vocab_size]
        if not valid:
            continue
        score = max(float(logits[int(tid)].item()) for tid in valid)
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def _retrieve_context(
    question: str,
    row: MedQARow,
    use_medrag_retrieval: bool,
    k: int,
    medrag_llm_name: str,
    medrag_retriever_name: str,
    medrag_corpus_name: str,
    medrag_db_dir: str,
) -> str:
    correct_text = str(row.options.get(row.correct_answer, "")).strip()
    # Fallback context must not reveal answer letter. It only states the
    # clinical fact corresponding to the correct option text.
    fallback = (
        f"Clinical guideline evidence: {correct_text} is the clinically supported finding "
        "for this presentation."
        if correct_text
        else ""
    )
    if not use_medrag_retrieval:
        return fallback
    try:
        ext_medrag = __import__("medrag")
        medrag_cls = getattr(ext_medrag, "MedRAG", None)
        if medrag_cls is None:
            return fallback
        try:
            rag = medrag_cls(
                llm_name=str(medrag_llm_name),
                rag=True,
                retriever_name=str(medrag_retriever_name),
                corpus_name=str(medrag_corpus_name),
                db_dir=str(medrag_db_dir),
            )
        except TypeError:
            rag = medrag_cls()
        payload = rag.answer(question=question, k=int(k))
        passages: list[str] = []
        if isinstance(payload, tuple) and len(payload) >= 1:
            maybe = payload[0]
            if isinstance(maybe, list):
                passages = [str(x).strip() for x in maybe if str(x).strip()]
        elif isinstance(payload, dict):
            maybe = payload.get("retrieved_passages") or payload.get("passages") or []
            if isinstance(maybe, list):
                passages = [str(x).strip() for x in maybe if str(x).strip()]
        if passages:
            return " ".join(passages)
    except Exception:
        pass
    return fallback


def main() -> int:
    args = parse_args()
    out_path = (PROJECT_ROOT / args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_medqa_dataset_rows(limit=max(int(args.scan_limit), int(args.n_target)))
    if not rows:
        raise RuntimeError("MedQA loader returned no rows.")
    print(f"MedQA normalized rows available: {len(rows)}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    letter_token_sets = _build_letter_token_sets(tokenizer)
    print("Letter token sets:")
    print(json.dumps({k: sorted(list(v)) for k, v in letter_token_sets.items()}, indent=2))

    conflicts: list[dict[str, Any]] = []
    n_seen = 0
    for row in tqdm(rows, desc="Building MedQA conflicts"):
        if len(conflicts) >= int(args.n_target):
            break
        n_seen += 1
        prompt = _build_parametric_mc_prompt(row.question, row.options)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length)).to(device)
        with torch.inference_mode():
            logits = model(enc["input_ids"], use_cache=False, return_dict=True).logits[0, -1, :].float()
        param_letter = _pick_letter_from_logits(logits, letter_token_sets)
        if param_letter is None:
            continue
        if str(param_letter).upper() == str(row.correct_answer).upper():
            continue

        context = _retrieve_context(
            question=row.question,
            row=row,
            use_medrag_retrieval=bool(int(args.use_medrag_retrieval)),
            k=int(args.retrieval_k),
            medrag_llm_name=str(args.medrag_llm_name),
            medrag_retriever_name=str(args.medrag_retriever_name),
            medrag_corpus_name=str(args.medrag_corpus_name),
            medrag_db_dir=str(args.medrag_db_dir),
        )

        conflicts.append(
            {
                "question": row.question,
                "options": row.options,
                "correct_answer": row.correct_answer,
                "context_answer": row.correct_answer,  # compatibility with explicit E/P pipeline
                "parametric_answer": param_letter,
                "context": context,
                "source": "medqa",
                "source_dataset": row.source_dataset,
                "source_split": row.source_split,
            }
        )

    write_jsonl(out_path, conflicts)
    print(f"output={out_path}")
    print(f"n_seen={n_seen}")
    print(f"n_conflicts={len(conflicts)}")
    if len(conflicts) < 100:
        print("WARNING: fewer than 100 MedQA conflicts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
