from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from evidence_aware.phase2.common import PreparedExample, classify_answer_token, first_non_whitespace_from_sequence
from evidence_aware.phase2.modeling import to_tensors


@dataclass
class MGGateRunOutput:
    label: str
    first_token_id: int | None
    generated_ids: list[int]
    gate_fired: int
    gate_blocked: int
    gate_fired_frac: float
    mean_margin: float


class MarginGatedHeadAmplifier:
    """
    MG-ERHA gate for a configurable attention head.
    Gate condition (from mgerha.md): fire when M_post_attn[target_layer] < tau_margin.

    Implementation details:
    - Gate decision uses one-step lag (decision from margin at t-1 is applied at t).
    - Amplification is applied at layer27 self_attn.o_proj *input* (same space as ERHA).
    - Current-step margin is computed from the exact h_post_attn at layer 27
      via post_attention_layernorm pre-hook (after ERHA modification).
    """

    def __init__(
        self,
        model,
        evidence_set: set[int],
        parametric_set: set[int],
        tau_margin: float,
        beta: float = 5.0,
        init_prev_margin: float = 0.0,
        first_token_only: bool = False,
        bootstrap_first_token_margin: bool = False,
        target_layer_idx: int = 27,
        target_head_idx: int = 20,
    ) -> None:
        self.model = model
        self.tau_margin = float(tau_margin)
        self.beta = float(beta)
        self.init_prev_margin = float(init_prev_margin)
        self.first_token_only = bool(first_token_only)
        self.bootstrap_first_token_margin = bool(bootstrap_first_token_margin)
        self.target_layer_idx = int(target_layer_idx)
        self.target_head_idx = int(target_head_idx)
        self._prev_margin = float(init_prev_margin)
        self._should_amplify = bool(self._prev_margin < self.tau_margin)
        self._step = 0
        self._bootstrap_mode = False
        self._bootstrap_margin = float("nan")

        num_layers = int(model.config.num_hidden_layers)
        num_heads = int(model.config.num_attention_heads)
        if self.target_layer_idx < 0 or self.target_layer_idx >= num_layers:
            raise ValueError(
                f"Invalid target_layer_idx={self.target_layer_idx}; valid range=[0, {num_layers - 1}]"
            )
        if self.target_head_idx < 0 or self.target_head_idx >= num_heads:
            raise ValueError(
                f"Invalid target_head_idx={self.target_head_idx}; valid range=[0, {num_heads - 1}]"
            )

        self.head_dim = int(model.config.hidden_size) // num_heads
        self.slice_start = self.target_head_idx * self.head_dim
        self.slice_end = (self.target_head_idx + 1) * self.head_dim

        self._lm_head = model.lm_head
        self._norm = model.model.norm
        # Stable margin path: keep one fp32 copy for projection in margin computation.
        self._lm_weight_fp32 = self._lm_head.weight.detach().float()
        lm_bias = getattr(self._lm_head, "bias", None)
        self._lm_bias_fp32 = lm_bias.detach().float() if lm_bias is not None else None
        self.hooks: list[Any] = []

        self._evidence_ids: list[int] = []
        self._parametric_ids: list[int] = []
        self.update_sets(evidence_set, parametric_set)

        self.gate_log: list[bool] = []
        self.margin_log: list[float] = []
        self.step_log: list[dict[str, float | int | bool]] = []
        self.margin_error_count: int = 0
        self.margin_error_samples: list[str] = []

    def update_sets(self, evidence_set: set[int], parametric_set: set[int]) -> None:
        self._evidence_ids = sorted(int(x) for x in evidence_set)
        self._parametric_ids = sorted(int(x) for x in parametric_set)

    def reset_log(self) -> None:
        self.gate_log = []
        self.margin_log = []
        self.step_log = []
        self.margin_error_count = 0
        self.margin_error_samples = []
        self._step = 0
        self._prev_margin = float(self.init_prev_margin)
        self._should_amplify = bool(self._prev_margin < self.tau_margin)

    def gate_summary(self) -> dict[str, float | int]:
        if not self.gate_log:
            return {
                "fired": 0,
                "blocked": 0,
                "fired_frac": 0.0,
                "mean_margin": float("nan"),
                "margin_steps_collected": 0,
                "margin_errors": int(self.margin_error_count),
                "first_margin_error": (self.margin_error_samples[0] if self.margin_error_samples else ""),
            }
        fired = int(sum(self.gate_log))
        total = int(len(self.gate_log))
        valid = [m for m in self.margin_log if math.isfinite(m)]
        return {
            "fired": fired,
            "blocked": int(total - fired),
            "fired_frac": float(fired / max(total, 1)),
            "mean_margin": float(sum(valid) / len(valid)) if valid else float("nan"),
            "margin_steps_collected": int(len(self.margin_log)),
            "margin_errors": int(self.margin_error_count),
            "first_margin_error": (self.margin_error_samples[0] if self.margin_error_samples else ""),
        }

    def _current_step_fire(self) -> bool:
        return bool(self._prev_margin < self.tau_margin)

    def _compute_margin_from_h_post_attn(self, h_post_attn_last: torch.Tensor) -> float:
        if not self._evidence_ids or not self._parametric_ids:
            return float("inf")

        with torch.no_grad():
            normed = self._norm(h_post_attn_last)
            # Force projection in fp32 to avoid bf16 underflow/numerical edge cases.
            logits = F.linear(
                normed.float(),
                self._lm_weight_fp32,
                self._lm_bias_fp32,
            )[0, 0, :]
            vocab_size = int(logits.numel())
            evidence_ids = [tid for tid in self._evidence_ids if 0 <= int(tid) < vocab_size]
            parametric_ids = [tid for tid in self._parametric_ids if 0 <= int(tid) < vocab_size]
            if not evidence_ids or not parametric_ids:
                return float("inf")
            e_idx = torch.as_tensor(evidence_ids, device=logits.device, dtype=torch.long)
            p_idx = torch.as_tensor(parametric_ids, device=logits.device, dtype=torch.long)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_p_e = torch.logsumexp(log_probs.index_select(0, e_idx), dim=0)
            log_p_p = torch.logsumexp(log_probs.index_select(0, p_idx), dim=0)
            margin = log_p_e - log_p_p
            return float(margin.item())

    def _make_h_post_attn_hook(self):
        def hook_fn(module, args):
            fired = bool(self._should_amplify)
            prev_margin = float(self._prev_margin)

            margin = float("nan")
            try:
                if args and args[0] is not None and args[0].ndim == 3:
                    # pre-hook on post_attention_layernorm receives h_post_attn directly.
                    h_last = args[0][:, -1:, :]
                    margin = self._compute_margin_from_h_post_attn(h_last)
                    if math.isfinite(margin):
                        self._prev_margin = margin
                self._should_amplify = self._current_step_fire()
            except Exception as err:
                margin = float("nan")
                self._should_amplify = False
                self.margin_error_count += 1
                if len(self.margin_error_samples) < 5:
                    self.margin_error_samples.append(str(err))

            if self._bootstrap_mode:
                self._bootstrap_margin = float(margin)
            else:
                self.margin_log.append(float(margin))
                self.step_log.append(
                    {
                        "step": int(self._step),
                        "fired": bool(fired),
                        "prev_margin": float(prev_margin),
                        "margin": float(margin),
                        "tau_margin": float(self.tau_margin),
                    }
                )
                self._step += 1

            return args

        return hook_fn

    def _make_o_proj_hook(self):
        beta = self.beta
        s, e = self.slice_start, self.slice_end

        def hook_fn(module, args):
            if self._bootstrap_mode:
                return args
            fired = bool(self._should_amplify)
            if self.first_token_only and self._step > 0:
                fired = False
            self.gate_log.append(fired)
            if not fired:
                return args
            if not args:
                return args
            x = args[0]
            if x is None or x.ndim != 3:
                return args
            x_mod = x.clone()
            x_mod[:, -1:, s:e] = x_mod[:, -1:, s:e] * beta
            return (x_mod,) + tuple(args[1:])

        return hook_fn

    def attach(self) -> None:
        layer = self.model.model.layers[self.target_layer_idx]
        h_margin = layer.post_attention_layernorm.register_forward_pre_hook(self._make_h_post_attn_hook())
        h_proj = layer.self_attn.o_proj.register_forward_pre_hook(self._make_o_proj_hook())
        self.hooks = [h_margin, h_proj]

    def detach(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def bootstrap_margin_from_prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if not self.bootstrap_first_token_margin:
            return
        self._bootstrap_mode = True
        self._bootstrap_margin = float("nan")
        try:
            with torch.inference_mode():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            self._bootstrap_mode = False

        if math.isfinite(self._bootstrap_margin):
            self._prev_margin = float(self._bootstrap_margin)
        else:
            self._prev_margin = float(self.init_prev_margin)
        self._should_amplify = bool(self._prev_margin < self.tau_margin)


def decode_with_margin_gate(
    model,
    tokenizer,
    device: torch.device,
    example: PreparedExample,
    max_new_tokens: int,
    gate: MarginGatedHeadAmplifier,
) -> MGGateRunOutput:
    input_ids, attention_mask = to_tensors(example, device=device)
    if gate.first_token_only and gate.bootstrap_first_token_margin:
        gate.bootstrap_margin_from_prefill(input_ids=input_ids, attention_mask=attention_mask)
    generated = input_ids
    mask = attention_mask
    past_key_values = None
    out_ids: list[int] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            step_input = generated if past_key_values is None else generated[:, -1:]
            outputs = model(
                input_ids=step_input,
                attention_mask=mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            out_ids.append(token_id)
            generated = torch.cat([generated, next_token], dim=1)
            mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)], dim=1)
            past_key_values = outputs.past_key_values
            if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                break

    first_token_id = first_non_whitespace_from_sequence(out_ids, tokenizer)
    label = classify_answer_token(first_token_id, example.evidence_token_id, example.parametric_token_id)
    summary = gate.gate_summary()
    return MGGateRunOutput(
        label=label,
        first_token_id=first_token_id,
        generated_ids=out_ids,
        gate_fired=int(summary["fired"]),
        gate_blocked=int(summary["blocked"]),
        gate_fired_frac=float(summary["fired_frac"]),
        mean_margin=float(summary["mean_margin"]),
    )
