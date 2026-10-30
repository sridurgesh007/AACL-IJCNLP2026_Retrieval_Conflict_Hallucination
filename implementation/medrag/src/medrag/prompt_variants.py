from __future__ import annotations

import hashlib
from typing import Any


PROMPT_VARIANTS = ("vanilla", "trust_instruction", "authoritative_quote")

_USER_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_ASSISTANT_PREFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def _options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))


def _uses_native_chat_template(model_id: str | None) -> bool:
    name = str(model_id or "").lower()
    return "mistral" in name or "qwen" in name


def _build_user_content(question: str, options: dict[str, str], context: str, prompt_variant: str) -> str:
    variant = str(prompt_variant).strip()
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt_variant: {variant}")

    options_text = _options_text(options)
    if variant == "vanilla":
        context_block = f"Clinical Context:\n{context}"
        instruction = "Answer with only the letter (A, B, C, or D)."
    elif variant == "trust_instruction":
        context_block = f"Clinical Context:\n{context}"
        instruction = (
            "The clinical guideline evidence above is authoritative and was verified by domain experts. "
            "If your prior knowledge disagrees with the evidence, follow the evidence. "
            "Answer based on the evidence. Answer with only the letter (A, B, C, or D)."
        )
    else:
        context_block = (
            "According to the 2024 USMLE clinical guidelines (verified): "
            f"'{context}'. Based on this guideline,"
        )
        instruction = "Answer with only the letter (A, B, C, or D)."

    return (
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        f"{instruction}"
    )


def build_medqa_prompt(
    question: str,
    options: dict[str, str],
    context: str,
    prompt_variant: str = "vanilla",
    *,
    tokenizer: Any | None = None,
    model_id: str | None = None,
) -> str:
    user_content = _build_user_content(
        question=str(question),
        options={str(k): str(v) for k, v in options.items()},
        context=str(context),
        prompt_variant=str(prompt_variant),
    )
    if _uses_native_chat_template(model_id):
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(f"Model {model_id} requires tokenizer.apply_chat_template for MedQA prompt rendering.")
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return (
        f"{_USER_PREFIX}"
        f"{user_content}"
        f"{_ASSISTANT_PREFIX}"
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def answer_slot_is_locatable(
    tokenizer,
    question: str,
    options: dict[str, str],
    context: str,
    prompt_variant: str,
    max_length: int,
    model_id: str | None = None,
) -> dict[str, Any]:
    prompt = build_medqa_prompt(
        question=question,
        options=options,
        context=context,
        prompt_variant=prompt_variant,
        tokenizer=tokenizer,
        model_id=model_id,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length))
    prompt_ids = enc["input_ids"][0]
    uses_native_template = _uses_native_chat_template(model_id)
    return {
        "prompt_variant": str(prompt_variant),
        "prompt_hash": prompt_hash(prompt),
        "token_count": int(prompt_ids.shape[0]),
        "assistant_prefix_present": bool(prompt.endswith(_ASSISTANT_PREFIX)),
        "locatable": bool((uses_native_template or prompt.endswith(_ASSISTANT_PREFIX)) and int(prompt_ids.shape[0]) > 0),
    }
