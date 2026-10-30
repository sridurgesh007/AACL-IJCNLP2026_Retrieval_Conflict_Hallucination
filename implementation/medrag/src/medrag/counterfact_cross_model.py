from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import LogitsProcessorList

from evidence_aware.dataset.counterfact import build_relation_evidence, complete_question_with_answer
from evidence_aware.phase2.common import (
    PreparedExample,
    build_prompt,
    classify_answer_token,
    first_answer_token_id_in_prompt,
    first_non_whitespace_from_sequence,
)
from evidence_aware.phase2.data import load_examples

from medrag.utils.io import read_json, sha256_file, write_json, write_jsonl
from medrag.utils.stats import mcnemar_from_paired, wilson_ci


@dataclass(frozen=True)
class CounterFactPrepared:
    example: PreparedExample
    prompt_ids: torch.Tensor
    prompt_length: int


class NStepEntityBiasProcessor:
    def __init__(
        self,
        *,
        promote_ids: set[int],
        suppress_ids: set[int],
        lambda_pos: float,
        lambda_neg: float,
        n_steps: int,
    ) -> None:
        self.promote_ids = sorted({int(token_id) for token_id in promote_ids})
        self.suppress_ids = sorted({int(token_id) for token_id in suppress_ids})
        self.lambda_pos = float(lambda_pos)
        self.lambda_neg = float(lambda_neg)
        self.n_steps = int(n_steps)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self._step < self.n_steps:
            vocab = int(scores.shape[-1])
            if self.lambda_pos > 0.0:
                for token_id in self.promote_ids:
                    if 0 <= int(token_id) < vocab:
                        scores[..., int(token_id)] += self.lambda_pos
            if self.lambda_neg > 0.0:
                for token_id in self.suppress_ids:
                    if 0 <= int(token_id) < vocab:
                        scores[..., int(token_id)] -= self.lambda_neg
        self._step += 1
        return scores


def project_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected non-empty float grid, got {text!r}")
    return values


def git_commit_hash(project_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(project_root).resolve()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_metadata(
    *,
    args: Any,
    input_files: Sequence[str | Path],
    seed: int,
    wall_clock_seconds: float,
    project_root: str | Path,
) -> dict[str, Any]:
    resolved = [Path(path).expanduser().resolve() for path in input_files]
    return {
        "seed": int(seed),
        "git_commit_hash": git_commit_hash(project_root),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path) for path in resolved],
        "input_file_sha256": {str(path): sha256_file(path) for path in resolved if path.exists()},
    }


def _looks_like_broken_counterfact_evidence(evidence: str, answer: str) -> bool:
    text = str(evidence or "").strip()
    answer_text = str(answer or "").strip()
    return bool(text.startswith("Fact: ") and f" is in {answer_text}" in text)


def build_counterfact_evidence_text(example: Any, evidence_mode: str) -> str:
    mode = str(evidence_mode or "cache").strip().lower()
    if mode == "cache":
        return str(example.evidence)

    if mode not in {"question_answer", "relation_or_question_answer"}:
        raise ValueError(
            "evidence_mode must be one of: cache, question_answer, relation_or_question_answer"
        )

    if mode == "relation_or_question_answer":
        evidence = build_relation_evidence(
            subject=str(getattr(example, "subject", "")),
            answer=str(example.answer_liquid),
            relation_id=str(getattr(example, "relation_id", "")),
            prompt_template=str(getattr(example, "prompt_template", "")),
            question=str(example.question),
        )
    else:
        evidence = complete_question_with_answer(str(example.question), str(example.answer_liquid))

    if not evidence:
        evidence = str(example.evidence)
    if _looks_like_broken_counterfact_evidence(evidence, str(example.answer_liquid)):
        evidence = complete_question_with_answer(str(example.question), str(example.answer_liquid))
    return str(evidence)


def load_counterfact_prepared(
    *,
    project_root: Path,
    cache_path: str | Path,
    tokenizer,
    max_length: int,
    max_items: int,
    evidence_mode: str = "cache",
) -> tuple[list[CounterFactPrepared], Path]:
    resolved_cache = project_path(project_root, cache_path)
    examples = load_examples(resolved_cache)
    prepared_examples: list[PreparedExample] = []
    for row in examples:
        evidence_text = build_counterfact_evidence_text(row, str(evidence_mode))
        evidence_token_id = first_answer_token_id_in_prompt(tokenizer, row.answer_liquid)
        parametric_token_id = first_answer_token_id_in_prompt(tokenizer, row.answer_solid)
        if evidence_token_id is None or parametric_token_id is None:
            continue
        if evidence_token_id == parametric_token_id:
            continue
        prompt = build_prompt(evidence_text, row.question)
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=int(max_length))
        ids = encoded.get("input_ids", [])
        if not ids:
            continue
        prepared_examples.append(
            PreparedExample(
                id=row.id,
                question=row.question,
                evidence=evidence_text,
                answer_liquid=row.answer_liquid,
                answer_solid=row.answer_solid,
                prompt=prompt,
                input_ids=[int(x) for x in ids],
                evidence_token_id=int(evidence_token_id),
                parametric_token_id=int(parametric_token_id),
                subject=getattr(row, "subject", ""),
                relation_id=getattr(row, "relation_id", ""),
                prompt_template=getattr(row, "prompt_template", ""),
            )
        )
    prepared_examples = sorted(prepared_examples, key=lambda ex: int(ex.id))
    if int(max_items) > 0:
        prepared_examples = prepared_examples[: int(max_items)]
    records: list[CounterFactPrepared] = []
    for ex in prepared_examples:
        prompt_ids = torch.tensor(ex.input_ids, dtype=torch.long)
        records.append(CounterFactPrepared(example=ex, prompt_ids=prompt_ids, prompt_length=int(prompt_ids.numel())))
    if not records:
        raise RuntimeError(f"No prepared CounterFact records from {resolved_cache}")
    return records, resolved_cache


def prefill_logits(model, prompt_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        out = model(
            prompt_ids.unsqueeze(0).to(device),
            attention_mask=torch.ones((1, int(prompt_ids.numel())), dtype=torch.long, device=device),
            use_cache=False,
            return_dict=True,
        )
    return out.logits[0, -1, :].detach().float().cpu()


def margin_from_logits(logits: torch.Tensor, evidence_ids: set[int], parametric_ids: set[int]) -> float:
    vocab = int(logits.numel())
    e_valid = sorted({int(token_id) for token_id in evidence_ids if 0 <= int(token_id) < vocab})
    p_valid = sorted({int(token_id) for token_id in parametric_ids if 0 <= int(token_id) < vocab})
    if not e_valid or not p_valid:
        return float("nan")
    e_val = torch.logsumexp(logits[e_valid].float(), dim=0)
    p_val = torch.logsumexp(logits[p_valid].float(), dim=0)
    return float((e_val - p_val).item())


def generate_label(
    *,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    device: torch.device,
    evidence_token_id: int,
    parametric_token_id: int,
    max_new_tokens: int,
    processor: NStepEntityBiasProcessor | None = None,
) -> tuple[str, int | None, str, list[int]]:
    input_ids = prompt_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    processors = LogitsProcessorList([processor]) if processor is not None else None
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = [int(x) for x in out[0, input_ids.shape[1] :].detach().cpu().tolist()]
    first_id = first_non_whitespace_from_sequence(new_ids, tokenizer)
    label = classify_answer_token(first_id, int(evidence_token_id), int(parametric_token_id))
    generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return str(label), first_id, generated, new_ids


def auc_rank(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if math.isfinite(float(score))]
    pos = [score for score, label in pairs if int(label) == 1]
    neg = [score for score, label in pairs if int(label) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for p_score in pos:
        for n_score in neg:
            total += 1.0
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return float(wins / max(total, 1.0))


def select_tau(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    faithful = [float(row["prefill_margin"]) for row in rows if str(row["base_label"]) == "faithful" and math.isfinite(float(row["prefill_margin"]))]
    hallucinated = [float(row["prefill_margin"]) for row in rows if str(row["base_label"]) == "hallucinated" and math.isfinite(float(row["prefill_margin"]))]
    auc = auc_rank(faithful + hallucinated, [1] * len(faithful) + [0] * len(hallucinated))
    values = sorted(set(faithful + hallucinated))
    candidates: list[float] = []
    if values:
        candidates.append(float(values[0]) - 1e-6)
        for left, right in zip(values[:-1], values[1:]):
            candidates.append((float(left) + float(right)) / 2.0)
        candidates.append(float(values[-1]) + 1e-6)

    tau_rows: list[dict[str, float]] = []
    if faithful and hallucinated:
        for tau in candidates:
            tpr = sum(1 for margin in hallucinated if margin < tau) / max(len(hallucinated), 1)
            fpr = sum(1 for margin in faithful if margin < tau) / max(len(faithful), 1)
            tau_rows.append(
                {
                    "tau": float(tau),
                    "tpr_hallucinated": float(tpr),
                    "fpr_faithful": float(fpr),
                    "youden_j": float(tpr - fpr),
                }
            )
    selected = None
    if tau_rows:
        selected = sorted(tau_rows, key=lambda row: (-float(row["youden_j"]), -float(row["tpr_hallucinated"]), float(row["fpr_faithful"]), float(row["tau"])))[0]
    summary = {
        "margin_auc": None if auc is None else float(auc),
        "n_faithful": int(len(faithful)),
        "n_hallucinated": int(len(hallucinated)),
        "mean_margin_faithful": None if not faithful else float(sum(faithful) / len(faithful)),
        "mean_margin_hallucinated": None if not hallucinated else float(sum(hallucinated) / len(hallucinated)),
        "selected_tau": None if selected is None else float(selected["tau"]),
        "selected_tau_metrics": selected,
        "tau_candidates": tau_rows,
        "faithful_margins": faithful,
        "hallucinated_margins": hallucinated,
    }
    return summary, tau_rows


def prepare_baseline_rows(
    *,
    records: list[CounterFactPrepared],
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        ex = rec.example
        logits = prefill_logits(model, rec.prompt_ids, device)
        e_set = {int(ex.evidence_token_id)}
        p_set = {int(ex.parametric_token_id)}
        margin = margin_from_logits(logits, e_set, p_set)
        label, first_id, generated, new_ids = generate_label(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=rec.prompt_ids,
            device=device,
            evidence_token_id=int(ex.evidence_token_id),
            parametric_token_id=int(ex.parametric_token_id),
            max_new_tokens=int(max_new_tokens),
            processor=None,
        )
        rows.append(
            {
                "item_id": str(ex.id),
                "question": str(ex.question),
                "evidence": str(ex.evidence),
                "correct_answer": str(ex.answer_liquid),
                "parametric_answer": str(ex.answer_solid),
                "evidence_token_id": int(ex.evidence_token_id),
                "parametric_token_id": int(ex.parametric_token_id),
                "prompt": str(ex.prompt),
                "prompt_token_ids": [int(x) for x in rec.prompt_ids.tolist()],
                "prompt_length": int(rec.prompt_length),
                "prefill_margin": float(margin),
                "base_label": str(label),
                "base_first_token_id": None if first_id is None else int(first_id),
                "base_generated_answer": str(generated),
                "base_generated_token_ids": new_ids,
            }
        )
    return rows


def evaluate_intervention(
    *,
    baseline_rows: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    tau: float,
    lambda_pos: float,
    lambda_neg: float,
    n_bias_steps: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        prompt_ids = torch.tensor([int(x) for x in row["prompt_token_ids"]], dtype=torch.long)
        margin = float(row["prefill_margin"])
        gate_fired = bool(math.isfinite(margin) and margin < float(tau))
        processor = None
        if gate_fired:
            processor = NStepEntityBiasProcessor(
                promote_ids={int(row["evidence_token_id"])},
                suppress_ids={int(row["parametric_token_id"])},
                lambda_pos=float(lambda_pos),
                lambda_neg=float(lambda_neg),
                n_steps=int(n_bias_steps),
            )
        label, first_id, generated, new_ids = generate_label(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            evidence_token_id=int(row["evidence_token_id"]),
            parametric_token_id=int(row["parametric_token_id"]),
            max_new_tokens=int(max_new_tokens),
            processor=processor,
        )
        base_label = str(row["base_label"])
        out = dict(row)
        out.update(
            {
                "ft_label": str(label),
                "ft_first_token_id": None if first_id is None else int(first_id),
                "ft_generated_answer": str(generated),
                "ft_generated_token_ids": new_ids,
                "ft_faithful": bool(label == "faithful"),
                "ft_hallucinated": bool(label == "hallucinated"),
                "restored": bool(base_label == "hallucinated" and label == "faithful"),
                "harmed": bool(base_label == "faithful" and label != "faithful"),
                "gate_fired": bool(gate_fired),
                "tau": float(tau),
                "lambda_pos": float(lambda_pos),
                "lambda_neg": float(lambda_neg),
                "n_bias_steps": int(n_bias_steps),
            }
        )
        out_rows.append(out)
    return out_rows, condition_summary(out_rows)


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(len(rows))
    base_hall = [row for row in rows if str(row["base_label"]) == "hallucinated"]
    base_faith = [row for row in rows if str(row["base_label"]) == "faithful"]
    restored = sum(1 for row in base_hall if bool(row.get("restored", False)))
    harmed = sum(1 for row in base_faith if bool(row.get("harmed", False)))
    ft_faith = sum(1 for row in rows if str(row.get("ft_label", "")) == "faithful")
    ft_hall = sum(1 for row in rows if str(row.get("ft_label", "")) == "hallucinated")
    ft_other = n - ft_faith - ft_hall
    base_correct = [str(row["base_label"]) == "faithful" for row in rows]
    ft_correct = [str(row.get("ft_label", "")) == "faithful" for row in rows]
    gate_fired = sum(1 for row in rows if bool(row.get("gate_fired", False)))
    return {
        "n_items": int(n),
        "n_base_hallucinated": int(len(base_hall)),
        "n_base_faithful": int(len(base_faith)),
        "n_restored": int(restored),
        "restoration_rate": float(restored / max(len(base_hall), 1)),
        "restoration_rate_ci95_wilson": list(wilson_ci(restored, len(base_hall))),
        "n_harmed": int(harmed),
        "harm_rate": float(harmed / max(len(base_faith), 1)),
        "harm_rate_ci95_wilson": list(wilson_ci(harmed, len(base_faith))),
        "ft_faithful_rate": float(ft_faith / max(n, 1)),
        "ft_hallucinated_rate": float(ft_hall / max(n, 1)),
        "ft_other_rate": float(ft_other / max(n, 1)),
        "gate_fire_rate": float(gate_fired / max(n, 1)),
        "mcnemar_base_vs_ft": mcnemar_from_paired(base_correct, ft_correct),
    }


def choose_best_lambda(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No lambda rows to select from.")
    return sorted(
        rows,
        key=lambda row: (
            -float(row["youden_j"]),
            -float(row["restoration_rate"]),
            float(row["harm_rate"]),
            float(row["lambda_pos"]) + float(row["lambda_neg"]),
            float(row["lambda_pos"]),
            float(row["lambda_neg"]),
        ),
    )[0]


def load_target_site(
    *,
    localization_path: str | Path,
    override_target_layer: int | None,
    override_target_head: int | None,
) -> dict[str, Any]:
    if override_target_layer is not None and override_target_head is not None:
        return {
            "target_layer": int(override_target_layer),
            "target_head": int(override_target_head),
            "source": "override",
            "site_description": f"L{int(override_target_layer)}/H{int(override_target_head)}",
        }
    payload = read_json(localization_path)
    candidate = payload.get("recommended_head") or {}
    if not candidate:
        candidates = list(payload.get("candidate_heads") or payload.get("top_5_heads") or [])
        if not candidates:
            raise RuntimeError(f"No recommended/candidate heads in {localization_path}")
        candidate = dict(candidates[0])
    return {
        "target_layer": int(candidate["layer"]),
        "target_head": int(candidate["head"]),
        "source": "localization",
        "site_description": f"L{int(candidate['layer'])}/H{int(candidate['head'])}",
        "site_payload": candidate,
    }


def write_counterfact_bundle(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "results.jsonl", rows)
