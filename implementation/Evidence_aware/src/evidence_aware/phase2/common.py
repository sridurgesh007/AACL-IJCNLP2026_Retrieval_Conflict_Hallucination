from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class Phase2Config:
    model_id: str
    cache_path: Path
    phase1_traces_path: Path
    phase1_aggregate_path: Path
    out_dir: Path
    validation_size: int
    eval_size: int | None
    min_eval_examples: int
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    beta_values: list[float]
    include_beta_one_sanity: bool
    target_heads: dict[int, list[int]]
    run_dola: bool
    mature_layer: int
    premature_layers: list[int]
    run_ablation: bool
    smoke_examples: int
    allow_unstable_validation: bool
    random_control_heads_count: int


@dataclass(frozen=True)
class Example:
    id: int
    question: str
    evidence: str
    answer_liquid: str
    answer_solid: str
    subject: str = ""
    relation_id: str = ""
    prompt_template: str = ""


@dataclass(frozen=True)
class PreparedExample:
    id: int
    question: str
    evidence: str
    answer_liquid: str
    answer_solid: str
    prompt: str
    input_ids: list[int]
    evidence_token_id: int
    parametric_token_id: int
    subject: str = ""
    relation_id: str = ""
    prompt_template: str = ""


@dataclass
class EvalRun:
    mode: str
    beta: float | None
    labels_by_id: dict[int, str]
    first_token_by_id: dict[int, int | None]
    elapsed_sec_total: float
    generated_tokens_total: int
    per_example_rows: list[dict[str, Any]]


def parse_target_heads(spec: str) -> dict[int, list[int]]:
    """
    Format: "24:3,4;27:20"
    """
    out: dict[int, list[int]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid target_heads segment: {chunk}")
        layer_str, heads_str = chunk.split(":", maxsplit=1)
        layer_idx = int(layer_str.strip())
        heads = [int(x.strip()) for x in heads_str.split(",") if x.strip()]
        if not heads:
            raise ValueError(f"No heads specified for layer {layer_idx}.")
        out[layer_idx] = sorted(set(heads))
    if not out:
        raise ValueError("No target heads parsed.")
    return out


def parse_phase2_config(args: Any) -> Phase2Config:
    beta_values = [float(x.strip()) for x in str(args.beta_values).split(",") if x.strip()]
    if not beta_values:
        raise ValueError("beta_values cannot be empty.")
    return Phase2Config(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase1_traces_path=Path(args.phase1_traces_path),
        phase1_aggregate_path=Path(args.phase1_aggregate_path),
        out_dir=Path(args.out_dir),
        validation_size=int(args.validation_size),
        eval_size=(None if args.eval_size is None or int(args.eval_size) <= 0 else int(args.eval_size)),
        min_eval_examples=int(args.min_eval_examples),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        beta_values=beta_values,
        include_beta_one_sanity=bool(int(args.include_beta_one_sanity)),
        target_heads=parse_target_heads(str(args.target_heads)),
        run_dola=bool(int(args.run_dola)),
        mature_layer=int(args.mature_layer),
        premature_layers=[int(x.strip()) for x in str(args.premature_layers).split(",") if x.strip()],
        run_ablation=bool(int(args.run_ablation)),
        smoke_examples=int(args.smoke_examples),
        allow_unstable_validation=bool(int(args.allow_unstable_validation)),
        random_control_heads_count=int(args.random_control_heads_count),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(out_dir: Path) -> None:
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)
    (out_dir / "main_eval").mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(evidence: str, question: str) -> str:
    return f"Evidence: {evidence}\n\nQuestion: {question}\nAnswer: "


def first_answer_token_id_in_prompt(tokenizer, answer_text: str) -> int | None:
    answer = str(answer_text).strip()
    if not answer:
        return None
    ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return int(ids[0])


def first_non_whitespace_from_sequence(generated_ids: list[int], tokenizer) -> int | None:
    for token_id in generated_ids:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if text.strip() != "":
            return int(token_id)
    return None


def argmax_non_whitespace_token_id(logits: torch.Tensor, tokenizer, topk_scan: int = 1024) -> int | None:
    k = min(int(topk_scan), int(logits.numel()))
    if k <= 0:
        return None
    top_ids = torch.topk(logits, k=k, dim=-1).indices.tolist()
    for token_id in top_ids:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if text.strip() != "":
            return int(token_id)
    return None


def classify_answer_token(first_token_id: int | None, evidence_token_id: int, parametric_token_id: int) -> str:
    if first_token_id is None:
        return "other"
    if int(first_token_id) == int(evidence_token_id):
        return "faithful"
    if int(first_token_id) == int(parametric_token_id):
        return "hallucinated"
    return "other"
