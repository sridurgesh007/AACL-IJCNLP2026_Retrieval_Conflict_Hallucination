from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
from datasets import load_dataset


DEFAULT_HF_CANDIDATES: tuple[str, ...] = (
    "NeelNanda/counterfact-tracing",
    "counterfact",
)
DEFAULT_FALLBACK_URL = "https://rome.baulab.info/data/dsets/counterfact.json"


@dataclass(frozen=True)
class CounterFactItem:
    id: int
    question: str
    evidence: str
    answer_liquid: str
    answer_solid: str
    subject: str = ""
    relation_id: str = ""
    prompt_template: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "evidence": self.evidence,
            "answer_liquid": self.answer_liquid,
            "answer_solid": self.answer_solid,
            "subject": self.subject,
            "relation_id": self.relation_id,
            "prompt_template": self.prompt_template,
        }


RELATION_TEMPLATES: dict[str, str] = {
    "P17": "{subject} is a country in {answer}.",
    "P19": "{subject} was born in {answer}.",
    "P20": "{subject} died in {answer}.",
    "P27": "{subject} is a citizen of {answer}.",
    "P36": "The capital of {subject} is {answer}.",
    "P103": "{subject}'s native language is {answer}.",
    "P106": "{subject}'s occupation is {answer}.",
    "P1303": "{subject} plays {answer}.",
    "P131": "{subject} is located in {answer}.",
    "P136": "{subject} is a work of {answer}.",
    "P140": "{subject}'s religion is {answer}.",
    "P159": "{subject} is headquartered in {answer}.",
    "P170": "{subject} was created by {answer}.",
    "P176": "{subject} was manufactured by {answer}.",
    "P178": "{subject} was developed by {answer}.",
    "P190": "The twin city of {subject} is {answer}.",
    "P364": "{subject} was originally written in {answer}.",
    "P407": "{subject} was originally written in {answer}.",
    "P495": "{subject} was created in {answer}.",
}


def complete_question_with_answer(question: str, answer: str) -> str:
    stem = str(question).strip()
    answer_text = str(answer).strip()
    if not stem or not answer_text:
        return ""
    if stem.endswith(("?", ".", "!", ":")):
        return f"{stem} {answer_text}."
    return f"{stem} {answer_text}."


def build_relation_evidence(
    *,
    subject: str,
    answer: str,
    relation_id: str = "",
    prompt_template: str = "",
    question: str = "",
) -> str:
    answer_text = str(answer).strip()
    if not answer_text:
        return ""
    rel = str(relation_id or "").strip()
    subj = str(subject or "").strip()
    if rel in RELATION_TEMPLATES and subj:
        return RELATION_TEMPLATES[rel].format(subject=subj, answer=answer_text)
    completed = complete_question_with_answer(question, answer_text)
    if completed:
        return completed
    prompt = str(prompt_template or "").strip()
    if prompt and subj:
        return complete_question_with_answer(CounterFactLoader._format_prompt_static(subj, prompt), answer_text)
    if subj:
        return f"{subj} is associated with {answer_text}."
    return f"The answer is {answer_text}."


class CounterFactLoader:
    """Load CounterFact, normalize schema, and cache locally."""

    def __init__(
        self,
        limit: int | None = None,
        cache_path: str | Path = "data/counterfact_solvent.json",
        hf_candidates: Sequence[str] = DEFAULT_HF_CANDIDATES,
        fallback_url: str = DEFAULT_FALLBACK_URL,
        timeout_seconds: int = 60,
    ) -> None:
        self.limit = limit
        self.cache_path = Path(cache_path)
        self.hf_candidates = tuple(hf_candidates)
        self.fallback_url = fallback_url
        self.timeout_seconds = timeout_seconds
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        if not force_refresh and self.cache_path.exists():
            cached = self._load_cache()
            return self._clip(cached)

        rows, source = self._load_raw_rows()
        items = self._normalize_rows(rows)
        if not items:
            raise RuntimeError(
                "CounterFact normalization produced 0 rows. "
                f"Source={source}. Check schema handling."
            )
        self._write_cache(items)
        return self._clip(items)

    def _load_cache(self) -> list[dict[str, Any]]:
        with self.cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise RuntimeError(f"Cache file must contain a list: {self.cache_path}")
        return data

    def _write_cache(self, rows: list[dict[str, Any]]) -> None:
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)

    def _clip(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.limit is None:
            return rows
        return rows[: self.limit]

    def _load_raw_rows(self) -> tuple[list[Mapping[str, Any]], str]:
        hf_errors: list[str] = []
        for candidate in self.hf_candidates:
            try:
                ds = self._load_hf_dataset(candidate)
                return [dict(row) for row in ds], f"huggingface:{candidate}"
            except Exception as exc:
                hf_errors.append(f"{candidate}: {exc}")

        try:
            response = requests.get(self.fallback_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "data" in payload:
                payload = payload["data"]
            if not isinstance(payload, list):
                raise RuntimeError("Fallback URL did not return a list payload.")
            return [dict(row) for row in payload if isinstance(row, Mapping)], "fallback_url"
        except Exception as exc:
            error_lines = "\n".join(hf_errors) if hf_errors else "(none)"
            raise RuntimeError(
                "Failed to download CounterFact from Hugging Face and fallback URL.\n"
                f"HF errors:\n{error_lines}\n"
                f"Fallback error:\n{exc}"
            ) from exc

    def _load_hf_dataset(self, name: str) -> Iterable[Mapping[str, Any]]:
        try:
            return load_dataset(name, split="train")
        except Exception:
            ds = load_dataset(name)
            if not ds:
                raise RuntimeError(f"Dataset has no splits: {name}")
            first_split = next(iter(ds.keys()))
            return ds[first_split]

    def _normalize_rows(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            parsed = self._parse_row(row)
            if parsed is None:
                continue

            subject = parsed["subject"].strip()
            question = self._build_question(subject, parsed["prompt_template"])
            answer_liquid = parsed["target_new"].strip()
            answer_solid = parsed["target_true"].strip()
            relation_id = parsed.get("relation_id", "").strip()
            if not question or not answer_liquid or not answer_solid:
                continue

            item = CounterFactItem(
                id=idx,
                question=question,
                evidence=build_relation_evidence(
                    subject=subject,
                    answer=answer_liquid,
                    relation_id=relation_id,
                    prompt_template=parsed["prompt_template"],
                    question=question,
                ),
                answer_liquid=answer_liquid,
                answer_solid=answer_solid,
                subject=subject,
                relation_id=relation_id,
                prompt_template=parsed["prompt_template"],
            )
            normalized.append(item.to_dict())
        return normalized

    def _build_question(self, subject: str, prompt_template: str) -> str:
        return self._format_prompt_static(subject, prompt_template)

    @staticmethod
    def _format_prompt_static(subject: str, prompt_template: str) -> str:
        template = (prompt_template or "").strip()
        if "{}" in template:
            return template.format(subject).strip()
        if "{subject}" in template:
            return template.format(subject=subject).strip()
        if subject.lower() in template.lower():
            return template
        return f"{subject} {template}".strip()

    def _parse_row(self, row: Mapping[str, Any]) -> dict[str, str] | None:
        if "requested_rewrite" in row:
            rewrite = row.get("requested_rewrite", {})
            if not isinstance(rewrite, Mapping):
                return None
            subject = str(rewrite.get("subject", "")).strip()
            prompt = str(rewrite.get("prompt", "")).strip()
            target_new = self._extract_text_value(rewrite.get("target_new"))
            target_true = self._extract_text_value(rewrite.get("target_true"))
            relation_id = self._extract_relation_id(rewrite)
            if not all([subject, prompt, target_new, target_true]):
                return None
            return {
                "subject": subject,
                "prompt_template": prompt,
                "target_new": target_new,
                "target_true": target_true,
                "relation_id": relation_id,
            }

        if "subject" in row and "prompt" in row:
            target_new_raw = row.get("target_new", row.get("target_false"))
            target_true_raw = row.get("target_true")
            target_new = self._extract_text_value(target_new_raw)
            target_true = self._extract_text_value(target_true_raw)
            subject = str(row.get("subject", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not all([subject, prompt, target_new, target_true]):
                return None
            return {
                "subject": subject,
                "prompt_template": prompt,
                "target_new": target_new,
                "target_true": target_true,
                "relation_id": self._extract_relation_id(row),
            }

        return None

    def _extract_relation_id(self, row: Mapping[str, Any]) -> str:
        for key in ("relation_id", "property_id", "predicate_id", "relation", "property"):
            value = row.get(key)
            if value:
                return str(value).strip()
        return ""

    def _extract_text_value(self, value: Any) -> str:
        if isinstance(value, Mapping):
            for key in ("str", "text", "value"):
                if key in value and value[key]:
                    return str(value[key]).strip()
            return ""
        if value is None:
            return ""
        return str(value).strip()
