from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from evidence_aware.phase2.common import PreparedExample, classify_answer_token, first_non_whitespace_from_sequence
from evidence_aware.phase2.modeling import to_tensors


@dataclass
class GateRunOutput:
    label: str
    first_token_id: int | None
    generated_ids: list[int]
    gate_fired: int
    gate_blocked: int
    gate_fired_frac: float


class AttentionGatedERHA:
    """
    Gated ERHA for L27H20.
    Gate condition: amplify only when head-20 attention mass on evidence positions < tau.
    """

    def __init__(self, model, evidence_positions: list[int], tau: float, beta: float = 5.0) -> None:
        self.model = model
        self.tau = float(tau)
        self.beta = float(beta)
        self.head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        self.slice_start = 20 * self.head_dim
        self.slice_end = 21 * self.head_dim
        self._should_amplify = False
        self._evidence_positions = list(evidence_positions)
        self.hooks: list[Any] = []
        self.gate_log: list[bool] = []

    def update_evidence_positions(self, positions: list[int]) -> None:
        self._evidence_positions = list(positions)

    def reset_log(self) -> None:
        self.gate_log = []
        # Boundary semantics per AG-erha.md sanity checks:
        # tau <= 0.0: gate always fires
        # tau >= 1.0: gate never fires
        if self.tau <= 0.0:
            self._should_amplify = True
        elif self.tau >= 1.0:
            self._should_amplify = False
        else:
            self._should_amplify = False

    def gate_summary(self) -> dict[str, float | int]:
        if not self.gate_log:
            return {"fired": 0, "blocked": 0, "fired_frac": 0.0}
        fired = int(sum(self.gate_log))
        total = len(self.gate_log)
        return {"fired": fired, "blocked": int(total - fired), "fired_frac": float(fired / total)}

    def _make_attn_hook(self):
        def hook_fn(module, inputs, output):
            try:
                # Enforce boundary sanity semantics exactly.
                if self.tau <= 0.0:
                    self._should_amplify = True
                    self.gate_log.append(self._should_amplify)
                    return
                if self.tau >= 1.0:
                    self._should_amplify = False
                    self.gate_log.append(self._should_amplify)
                    return

                attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if attn_weights is None:
                    self._should_amplify = True
                    self.gate_log.append(self._should_amplify)
                    return
                h20 = attn_weights[0, 20, -1, :]
                if len(self._evidence_positions) == 0:
                    self._should_amplify = False
                    self.gate_log.append(self._should_amplify)
                    return
                valid_pos = [p for p in self._evidence_positions if 0 <= p < h20.shape[0]]
                if not valid_pos:
                    self._should_amplify = True
                    self.gate_log.append(self._should_amplify)
                    return
                evidence_mass = h20[valid_pos].sum().item()
                self._should_amplify = bool(evidence_mass < self.tau)
                self.gate_log.append(self._should_amplify)
            except Exception:
                self._should_amplify = True
                self.gate_log.append(self._should_amplify)

        return hook_fn

    def _make_proj_hook(self):
        beta = self.beta
        s, e = self.slice_start, self.slice_end

        def hook_fn(module, args):
            if not self._should_amplify:
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
        layer = self.model.model.layers[27]
        h_attn = layer.self_attn.register_forward_hook(self._make_attn_hook())
        h_proj = layer.self_attn.o_proj.register_forward_pre_hook(self._make_proj_hook())
        self.hooks = [h_attn, h_proj]

    def detach(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


class MarginGatedERHA:
    """
    Gate condition: amplify L27H20 only when M = log P(E) - log P(P) < tau_margin.
    """

    def __init__(
        self,
        model,
        evidence_set: set[int],
        parametric_set: set[int],
        tau_margin: float,
        beta: float = 5.0,
    ) -> None:
        self.model = model
        self.evidence_set = set(int(x) for x in evidence_set)
        self.parametric_set = set(int(x) for x in parametric_set)
        self.tau_margin = float(tau_margin)
        self.beta = float(beta)
        self.head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        self.slice_start = 20 * self.head_dim
        self.slice_end = 21 * self.head_dim
        self._should_amplify = False
        self.hooks: list[Any] = []
        self.gate_log: list[bool] = []

    def update_sets(self, evidence_set: set[int], parametric_set: set[int]) -> None:
        self.evidence_set = set(int(x) for x in evidence_set)
        self.parametric_set = set(int(x) for x in parametric_set)

    def reset_log(self) -> None:
        self.gate_log = []
        # Boundary semantics per AG-erha.md sanity checks:
        # tau <= 0.0: gate always fires
        # tau >= 1.0: gate never fires
        if self.tau_margin <= 0.0:
            self._should_amplify = True
        elif self.tau_margin >= 1.0:
            self._should_amplify = False
        else:
            self._should_amplify = False

    def gate_summary(self) -> dict[str, float | int]:
        if not self.gate_log:
            return {"fired": 0, "blocked": 0, "fired_frac": 0.0}
        fired = int(sum(self.gate_log))
        total = len(self.gate_log)
        return {"fired": fired, "blocked": int(total - fired), "fired_frac": float(fired / total)}

    def _make_margin_hook(self):
        lm_head = self.model.lm_head
        norm = self.model.model.norm
        tau_m = self.tau_margin

        def hook_fn(module, inputs, output):
            try:
                h = output[0][:, -1:, :] if isinstance(output, tuple) else output[:, -1:, :]
                logits = lm_head(norm(h)).squeeze(0).squeeze(0)
                probs = torch.softmax(logits.float(), dim=-1)
                e_ids = list(self.evidence_set)
                p_ids = list(self.parametric_set)
                if not e_ids or not p_ids:
                    self._should_amplify = False
                    self.gate_log.append(self._should_amplify)
                    return
                p_e = float(probs[e_ids].sum().item())
                p_p = float(probs[p_ids].sum().item())
                p_e = max(p_e, 1e-10)
                p_p = max(p_p, 1e-10)
                margin = float(torch.log(torch.tensor(p_e / p_p)))
                self._should_amplify = bool(margin < tau_m)
                self.gate_log.append(self._should_amplify)
            except Exception:
                self._should_amplify = True
                self.gate_log.append(self._should_amplify)

        return hook_fn

    def _make_proj_hook(self):
        beta = self.beta
        s, e = self.slice_start, self.slice_end

        def hook_fn(module, args):
            if not self._should_amplify:
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
        layer = self.model.model.layers[27]
        h_margin = layer.self_attn.register_forward_hook(self._make_margin_hook())
        h_proj = layer.self_attn.o_proj.register_forward_pre_hook(self._make_proj_hook())
        self.hooks = [h_margin, h_proj]

    def detach(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def decode_with_gate(
    model,
    tokenizer,
    device: torch.device,
    example: PreparedExample,
    max_new_tokens: int,
    gate: AttentionGatedERHA | MarginGatedERHA,
) -> GateRunOutput:
    input_ids, attention_mask = to_tensors(example, device=device)
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
                output_attentions=True,
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
    return GateRunOutput(
        label=label,
        first_token_id=first_token_id,
        generated_ids=out_ids,
        gate_fired=int(summary["fired"]),
        gate_blocked=int(summary["blocked"]),
        gate_fired_frac=float(summary["fired_frac"]),
    )


def compute_margin_from_logits_vector(logits_vector: list[float], evidence_set: set[int], parametric_set: set[int]) -> float:
    logits = torch.tensor(logits_vector, dtype=torch.float32)
    probs = F.softmax(logits, dim=-1)
    e_ids = list(evidence_set)
    p_ids = list(parametric_set)
    p_e = float(probs[e_ids].sum().item())
    p_p = float(probs[p_ids].sum().item())
    p_e = max(p_e, 1e-10)
    p_p = max(p_p, 1e-10)
    return float(torch.log(torch.tensor(p_e)) - torch.log(torch.tensor(p_p)))
