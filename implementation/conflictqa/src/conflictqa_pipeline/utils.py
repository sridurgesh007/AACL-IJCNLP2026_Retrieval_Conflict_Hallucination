from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import torch

ARTICLES = {"a", "an", "the"}
ENTITY_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {p}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash(start_path: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(start_path).expanduser().resolve()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_metadata(
    *,
    args: Any,
    input_files: list[str | Path],
    seed: int,
    wall_clock_seconds: float,
    project_root: str | Path,
) -> dict[str, Any]:
    files = [Path(path).expanduser().resolve() for path in input_files]
    return {
        "seed": int(seed),
        "git_commit_hash": git_commit_hash(project_root),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_files": [str(path) for path in files],
        "input_file_sha256": {str(path): sha256_file(path) for path in files if path.exists()},
    }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
    n = int(n)
    k = int(k)
    if n <= 0:
        return [0.0, 0.0]
    z = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    radius = (
        z
        * math.sqrt((phat * (1.0 - phat) / float(n)) + (z * z) / (4.0 * n * n))
        / denom
    )
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def mcnemar_from_paired(baseline_correct: list[bool], current_correct: list[bool]) -> dict[str, Any]:
    if len(baseline_correct) != len(current_correct):
        raise ValueError("Paired vectors must have the same length.")
    b = 0
    c = 0
    for base_ok, cur_ok in zip(baseline_correct, current_correct):
        if bool(base_ok) and not bool(cur_ok):
            b += 1
        elif (not bool(base_ok)) and bool(cur_ok):
            c += 1
    total = b + c
    if total <= 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "method": "degenerate"}
    tail = 0.0
    for i in range(0, min(b, c) + 1):
        tail += math.comb(total, i) * (0.5 ** total)
    return {
        "b": int(b),
        "c": int(c),
        "n_discordant": int(total),
        "p_value": float(min(1.0, 2.0 * tail)),
        "method": "exact_binomial",
    }


def auc_rank(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s))]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos <= 0 or n_neg <= 0:
        return None
    ordered = sorted(enumerate(pairs), key=lambda x: x[1][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][0] == ordered[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(ranks[idx] for idx, (_, y) in enumerate(pairs) if y == 1)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def normalize_entity(text: str) -> str:
    tokens = [tok.lower() for tok in ENTITY_TOKEN.findall(str(text or ""))]
    tokens = [tok for tok in tokens if tok not in ARTICLES]
    return " ".join(tokens)


def entity_match(generated: str, reference: str) -> bool:
    gen = normalize_entity(generated)
    ref = normalize_entity(reference)
    if not gen or not ref:
        return False
    if ref in gen or gen in ref:
        return True
    gen_tokens = set(gen.split())
    ref_tokens = ref.split()
    if not ref_tokens:
        return False
    overlap = sum(1 for tok in ref_tokens if tok in gen_tokens)
    return float(overlap) / float(len(ref_tokens)) >= 0.5


def partial_entity_overlap(generated: str, reference: str) -> bool:
    gen_tokens = set(normalize_entity(generated).split())
    ref_tokens = set(normalize_entity(reference).split())
    return bool(gen_tokens and ref_tokens and (gen_tokens & ref_tokens))


def classify_answer(generated: str, context_answer: str, parametric_answer: str) -> str:
    if entity_match(generated, context_answer):
        return "context_faithful"
    if entity_match(generated, parametric_answer):
        return "parametric"
    return "other"


def build_user_chat_prompt(tokenizer, content: str) -> str:
    user_content = str(content).strip()
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_conflictqa_prompt(question: str, context: str, *, authoritative: bool = False) -> str:
    instruction = "Answer in 1-3 words only. Use only the provided context."
    if authoritative:
        instruction = (
            "The context above is authoritative. Answer based on the context, not your prior knowledge. "
            "Answer in 1-3 words only."
        )
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Context: {str(context).strip()}\n\n"
        f"Question: {str(question).strip()}\n"
        f"{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def clean_generated_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[\"'`\s]+|[\"'`\s]+$", "", text)
    text = re.split(r"[\n\r.;]", text, maxsplit=1)[0].strip()
    prefixes = [
        "the answer is",
        "answer:",
        "it is",
        "it's",
    ]
    low = text.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            text = text[len(prefix) :].strip(" :,-")
            break
    return text.strip()


def _keep_token(tokenizer, token_id: int) -> bool:
    special = {int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])}
    if int(token_id) in special:
        return False
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    cleaned = str(decoded).replace("\u2581", " ").replace("\u0120", " ").strip()
    return bool(cleaned and any(ch.isalnum() for ch in cleaned))


def first_token_set(tokenizer, entity: str) -> set[int]:
    entity = str(entity or "").strip()
    out: set[int] = set()
    if not entity:
        return out
    stripped = entity.strip("\"'`")
    variants = {
        entity,
        f" {entity}",
        stripped,
        f" {stripped}",
    }
    first_word = entity.split()[0] if entity.split() else entity
    variants.add(first_word)
    variants.add(f" {first_word}")
    for variant in sorted(x for x in variants if x):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if not ids:
            continue
        # Some entity strings begin with punctuation-like prefix tokens
        # (for example quoted aliases or leading apostrophes). For routing
        # analysis we want the first content-bearing lexical token, not the
        # leading punctuation token that the tokenizer may emit first.
        for raw_tok in ids:
            tok = int(raw_tok)
            if _keep_token(tokenizer, tok):
                out.add(tok)
                break
    return out


def margin_from_logits(logits: torch.Tensor, e_tokens: set[int], p_tokens: set[int]) -> float:
    vocab = int(logits.numel())
    e_valid = sorted({int(tok) for tok in e_tokens if 0 <= int(tok) < vocab})
    p_valid = sorted({int(tok) for tok in p_tokens if 0 <= int(tok) < vocab})
    if not e_valid or not p_valid:
        return float("nan")
    e_val = torch.logsumexp(logits[e_valid].float(), dim=0)
    p_val = torch.logsumexp(logits[p_valid].float(), dim=0)
    return float((e_val - p_val).item())


class FirstTokenSuppressProcessor:
    def __init__(self, token_ids: set[int], bias: float) -> None:
        self.token_ids = sorted(int(tok) for tok in token_ids)
        self.bias = float(bias)
        self.step = 0

    def __call__(self, input_ids, scores):
        del input_ids
        if self.step == 0 and self.token_ids and self.bias > 0.0:
            idx = torch.as_tensor(self.token_ids, dtype=torch.long, device=scores.device)
            scores[:, idx] = scores[:, idx] - float(self.bias)
        self.step += 1
        return scores


def resolve_pad_token_id(model, tokenizer) -> int:
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


def generate_short_answer(
    *,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    logits_processor: Any | None = None,
) -> tuple[str, list[int]]:
    from transformers import LogitsProcessorList

    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    kwargs: dict[str, Any] = {}
    if logits_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
    with torch.inference_mode():
        out = model.generate(
            prompt_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=resolve_pad_token_id(model, tokenizer),
            **kwargs,
        )
    gen_ids = [int(x) for x in out[0, prompt_ids.shape[0] :].detach().cpu().tolist()]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return clean_generated_answer(text), gen_ids


def prefill_logits(
    *,
    model,
    prompt_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=prompt_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[0, -1, :].detach().float().cpu()


def summarize_counts(rows: list[dict[str, Any]], *, classification_key: str = "classification") -> dict[str, Any]:
    counts = Counter(str(row.get(classification_key, "other")) for row in rows)
    n = int(len(rows))
    out: dict[str, Any] = {"n": n}
    for key in ("context_faithful", "parametric", "other"):
        k = int(counts.get(key, 0))
        rate_name = {
            "context_faithful": "faithfulness_rate",
            "parametric": "parametric_rate",
            "other": "other_rate",
        }[key]
        out[f"n_{key}"] = k
        out[rate_name] = float(k / max(n, 1))
        out[f"{rate_name}_ci95_wilson"] = wilson_ci(k, n)
    return out


def read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            # Some ConflictQA files use a .json suffix while storing one JSON
            # object per line. Treat them as JSONL instead of failing the build.
            return read_jsonl(path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "examples", "rows", "items"):
                if isinstance(payload.get(key), list):
                    return [row for row in payload[key] if isinstance(row, dict)]
            nested: list[dict[str, Any]] = []
            for key, value in payload.items():
                if isinstance(value, list):
                    for row in value:
                        if not isinstance(row, dict):
                            continue
                        copied = dict(row)
                        copied.setdefault("_source_subset_hint", str(key))
                        nested.append(copied)
            if nested:
                return nested
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    return []
