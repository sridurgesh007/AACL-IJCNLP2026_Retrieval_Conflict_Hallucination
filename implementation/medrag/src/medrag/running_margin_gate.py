"""
Running margin gate for the isolated medrag workflow.

Supports:
1) Standard running gate (existing behavior).
2) Evidence-position bootstrap (Approach 2).
3) Criticality-aware running gate for multi-token medical answers.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

NUMBER_PATTERN = re.compile(r"\d+\.?\d*")


class RunningMarginGate:
    def __init__(
        self,
        model,
        evidence_set: set[int],
        parametric_set: set[int],
        tokenizer=None,
        target_layer: int = 27,
        target_head: int = 20,
        tau_margin: float = 0.0,
        alpha: float = 1.0,
        lambda_max: float = 10.0,
        first_token_only: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = int(target_layer)
        self.target_head = int(target_head)
        self.tau_margin = float(tau_margin)
        self.alpha = float(alpha)
        self.lambda_max = float(lambda_max)
        self.first_token_only = bool(first_token_only)

        self.num_layers = int(model.config.num_hidden_layers)
        self.num_heads = int(model.config.num_attention_heads)
        self.hidden_size = int(model.config.hidden_size)
        if self.hidden_size % self.num_heads != 0:
            raise RuntimeError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads
        if not (0 <= self.target_layer < self.num_layers):
            raise ValueError(f"Invalid target layer {self.target_layer}")
        if not (0 <= self.target_head < self.num_heads):
            raise ValueError(f"Invalid target head {self.target_head}")

        self._lm_weight_fp32 = model.lm_head.weight.detach().float()
        lm_bias = getattr(model.lm_head, "bias", None)
        self._lm_bias_fp32 = lm_bias.detach().float() if lm_bias is not None else None
        self._norm = model.model.norm
        self._target_device = model.lm_head.weight.device
        self._target_dtype = model.lm_head.weight.dtype

        w_o = model.model.layers[self.target_layer].self_attn.o_proj.weight.detach().float()
        s = int(self.target_head) * self.head_dim
        e = (int(self.target_head) + 1) * self.head_dim
        self._W_O_h = w_o[:, s:e]
        self._s = int(s)
        self._e = int(e)

        self._hooks: list[Any] = []
        self._prefill_mode = False
        self._decode_step = 0
        self._current_margin = 0.0
        self._margin_at_evidence_pos = 0.0
        self._lambda = 0.0
        self._lambda_clipped = False
        self._evidence_position = -1
        self._position_found = False
        self._v_star: torch.Tensor | None = None
        self._gain = 1.0
        self._margin_errors = 0
        self._margin_error_samples: list[str] = []
        self.step_log: list[dict[str, Any]] = []

        self.update_sets(evidence_set, parametric_set)

    def update_sets(self, evidence_set: set[int], parametric_set: set[int]) -> None:
        self.E_ids = sorted(int(x) for x in evidence_set)
        e_set = set(self.E_ids)
        self.P_ids = sorted(int(x) for x in parametric_set if int(x) not in e_set)

    def setup_example(
        self,
        evidence_token_id: int,
        parametric_token_id: int,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = self._target_device
        d_ep = (
            self._lm_weight_fp32[int(evidence_token_id)].to(device=device)
            - self._lm_weight_fp32[int(parametric_token_id)].to(device=device)
        )
        w_o_h = self._W_O_h.to(device=device)
        self._v_star = w_o_h.transpose(0, 1).matmul(d_ep)
        self._gain = max(float(torch.dot(self._v_star, self._v_star).item()), 1e-8)
        self.step_log = []
        self._decode_step = 0
        self._lambda = 0.0
        self._lambda_clipped = False

    @staticmethod
    def _extract_first_number(text: str) -> str:
        items = NUMBER_PATTERN.findall(str(text or ""))
        return str(items[0]).strip() if items else ""

    def _find_subsequence(self, prompt_list: list[int], sub_ids: list[int], prefer_last: bool = False) -> int | None:
        if not sub_ids:
            return None
        max_start = len(prompt_list) - len(sub_ids)
        if max_start < 0:
            return None
        if bool(prefer_last):
            iterator = range(max_start, -1, -1)
        else:
            iterator = range(max_start + 1)
        for start in iterator:
            if prompt_list[start : start + len(sub_ids)] == sub_ids:
                return int(start)
        return None

    def _find_evidence_position(self, prompt_ids: torch.Tensor, context_answer: str) -> int | None:
        if self.tokenizer is None:
            return None
        prompt_list = [int(x) for x in prompt_ids.tolist()]
        answer = str(context_answer or "").strip()
        candidates: list[list[int]] = []

        num = self._extract_first_number(answer)
        if num:
            for txt in (" " + num, num):
                ids = [int(x) for x in self.tokenizer.encode(txt, add_special_tokens=False)]
                if ids:
                    candidates.append(ids)

        if answer:
            for txt in (" " + answer, answer):
                ids = [int(x) for x in self.tokenizer.encode(txt, add_special_tokens=False)]
                if ids:
                    candidates.append(ids)

        seen: set[tuple[int, ...]] = set()
        uniq_candidates: list[list[int]] = []
        for ids in candidates:
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            uniq_candidates.append(ids)

        for ids in uniq_candidates:
            pos = self._find_subsequence(prompt_list, ids, prefer_last=True)
            if pos is not None:
                return int(pos)
        return None

    def _compute_margin_and_logits(self, h_last: torch.Tensor) -> tuple[float, torch.Tensor]:
        h = h_last
        if h.ndim == 2:
            h = h.unsqueeze(1)
        if h.ndim != 3:
            raise ValueError(f"Expected hidden state [batch, seq, hidden], got {tuple(h.shape)}")

        h = h.to(device=self._target_device, dtype=self._target_dtype)
        with torch.no_grad():
            normed = self._norm(h)
            logits = F.linear(normed.float(), self._lm_weight_fp32, self._lm_bias_fp32)[0, 0, :]
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            if not self.E_ids or not self.P_ids:
                return float("nan"), logits

            vocab_size = int(logits.numel())
            e_ids = [tid for tid in self.E_ids if 0 <= int(tid) < vocab_size]
            p_ids = [tid for tid in self.P_ids if 0 <= int(tid) < vocab_size]
            if not e_ids or not p_ids:
                return float("nan"), logits
            e_idx = torch.as_tensor(e_ids, device=logits.device, dtype=torch.long)
            p_idx = torch.as_tensor(p_ids, device=logits.device, dtype=torch.long)
            log_p = torch.log_softmax(logits, dim=-1)
            log_pE = torch.logsumexp(log_p.index_select(0, e_idx), dim=0)
            log_pP = torch.logsumexp(log_p.index_select(0, p_idx), dim=0)
            return float((log_pE - log_pP).item()), logits

    def _compute_margin(self, h_last: torch.Tensor) -> float:
        margin, _ = self._compute_margin_and_logits(h_last)
        return float(margin)

    def _compute_prefill_lambda(self, margin: float) -> tuple[float, bool]:
        if self._v_star is None or not np.isfinite(margin):
            return 0.0, False
        deficit = float(self.tau_margin - float(margin))
        if deficit <= 0.0:
            return 0.0, False
        raw_lambda = float(self.alpha * deficit / max(self._gain, 1e-8))
        lam = float(min(raw_lambda, self.lambda_max))
        clipped = bool(lam < raw_lambda)
        return lam, clipped

    def bootstrap_prefill(
        self,
        prompt_ids: torch.Tensor,
        device: torch.device,
        context_answer: str | None = None,
        parametric_answer: str | None = None,
        model=None,
    ) -> None:
        del parametric_answer
        if prompt_ids.ndim != 1:
            raise ValueError("prompt_ids must be 1D token ids")

        model_for_prefill = model if model is not None else self.model
        fallback_pos = int(len(prompt_ids) - 1)
        ev_pos = self._find_evidence_position(prompt_ids, context_answer or "") if context_answer else None
        if ev_pos is None:
            ev_pos = fallback_pos
            position_found = False
        else:
            position_found = True

        captured: dict[str, Any] = {}

        def _hook(module, args):
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            seq_len = int(hidden.shape[1])
            pos = int(ev_pos) if 0 <= int(ev_pos) < seq_len else seq_len - 1
            margin = self._compute_margin(hidden[:, pos : pos + 1, :])
            captured["margin"] = float(margin)
            captured["position"] = int(pos)
            return args

        handle = (
            model_for_prefill.model.layers[self.target_layer]
            .post_attention_layernorm.register_forward_pre_hook(_hook)
        )
        self._prefill_mode = True
        try:
            with torch.inference_mode():
                _ = model_for_prefill(prompt_ids.unsqueeze(0).to(device), use_cache=False, return_dict=True)
        finally:
            self._prefill_mode = False
            handle.remove()

        margin = float(captured.get("margin", 0.0))
        self._current_margin = margin
        self._margin_at_evidence_pos = margin
        self._evidence_position = int(captured.get("position", ev_pos))
        self._position_found = bool(position_found and self._evidence_position != fallback_pos)
        self._lambda, self._lambda_clipped = self._compute_prefill_lambda(margin)

    def _make_live_margin_hook(self):
        def hook_fn(module, args):
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            try:
                self._current_margin = self._compute_margin(hidden[:, -1:, :])
            except Exception as err:
                self._margin_errors += 1
                if len(self._margin_error_samples) < 5:
                    self._margin_error_samples.append(str(err))
            return args

        return hook_fn

    def _make_steer_hook(self, device: torch.device):
        del device
        s = self._s
        e = self._e

        def hook_fn(module, args):
            if self._prefill_mode or not args:
                return args
            x = args[0]
            if x is None or x.ndim != 3:
                return args
            if self.first_token_only and self._decode_step > 0:
                self.step_log.append(
                    {"steered": False, "reason": "first_token_only", "margin": float(self._current_margin), "lambda": 0.0}
                )
                return args
            if self._v_star is None:
                self.step_log.append({"steered": False, "reason": "no_v_star", "margin": float(self._current_margin), "lambda": 0.0})
                return args

            margin = float(self._current_margin)
            deficit = float(self.tau_margin - margin) if np.isfinite(margin) else 0.0
            if deficit <= 0.0:
                self.step_log.append(
                    {"steered": False, "reason": "margin_ok", "margin": float(margin), "lambda": 0.0}
                )
                return args

            raw_lambda = float(self.alpha * deficit / max(self._gain, 1e-8))
            lam = float(min(raw_lambda, self.lambda_max))
            clipped = bool(lam < raw_lambda)
            x_mod = x.clone()
            v = self._v_star.to(device=x_mod.device, dtype=torch.float32)
            updated = x_mod[:, -1:, s:e].float() + lam * v.view(1, 1, -1)
            x_mod[:, -1:, s:e] = updated.to(dtype=x_mod.dtype)
            self.step_log.append(
                {
                    "steered": True,
                    "reason": "steered",
                    "margin": float(margin),
                    "lambda": float(lam),
                    "clipped": clipped,
                    "E_present": None,
                    "P_present": None,
                }
            )
            return (x_mod,) + tuple(args[1:])

        return hook_fn

    def _make_steer_hook_with_criticality(self, device: torch.device, top_k: int = 50):
        del device
        s = self._s
        e = self._e
        top_k = max(1, int(top_k))

        def hook_fn(module, args):
            if self._prefill_mode or not args:
                return args
            x = args[0]
            if x is None or x.ndim != 3:
                return args
            if self.first_token_only and self._decode_step > 0:
                self.step_log.append(
                    {"steered": False, "reason": "first_token_only", "margin": float(self._current_margin), "lambda": 0.0}
                )
                return args
            if self._v_star is None:
                self.step_log.append({"steered": False, "reason": "no_v_star", "margin": float(self._current_margin), "lambda": 0.0})
                return args
            if abs(float(self._lambda)) < 1e-10:
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "no_lambda",
                        "margin": float(self._current_margin),
                        "lambda": 0.0,
                        "E_present": False,
                        "P_present": False,
                    }
                )
                return args

            try:
                margin, logits = self._compute_margin_and_logits(x[:, -1:, :])
                self._current_margin = float(margin)
            except Exception as err:
                self._margin_errors += 1
                if len(self._margin_error_samples) < 5:
                    self._margin_error_samples.append(str(err))
                self.step_log.append(
                    {"steered": False, "reason": "margin_error", "margin": 0.0, "lambda": 0.0}
                )
                return args

            vocab_size = int(logits.numel())
            k_eff = min(top_k, vocab_size)
            top_k_ids = set(int(x) for x in logits.topk(k_eff).indices.tolist())
            e_present = bool(set(self.E_ids) & top_k_ids)
            p_present = bool(set(self.P_ids) & top_k_ids)
            if not (e_present or p_present):
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "not_commitment_token",
                        "margin": float(margin) if np.isfinite(margin) else 0.0,
                        "lambda": 0.0,
                        "E_present": False,
                        "P_present": False,
                    }
                )
                return args

            if not np.isfinite(margin) or float(margin) >= float(self.tau_margin):
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "margin_ok",
                        "margin": float(margin) if np.isfinite(margin) else 0.0,
                        "lambda": 0.0,
                        "E_present": e_present,
                        "P_present": p_present,
                    }
                )
                return args

            lam = float(self._lambda)
            x_mod = x.clone()
            v = self._v_star.to(device=x_mod.device, dtype=torch.float32)
            updated = x_mod[:, -1:, s:e].float() + lam * v.view(1, 1, -1)
            x_mod[:, -1:, s:e] = updated.to(dtype=x_mod.dtype)
            self.step_log.append(
                {
                    "steered": True,
                    "reason": "steered",
                    "margin": float(margin),
                    "lambda": float(lam),
                    "clipped": bool(self._lambda_clipped),
                    "E_present": e_present,
                    "P_present": p_present,
                }
            )
            return (x_mod,) + tuple(args[1:])

        return hook_fn

    def _make_step_counter(self):
        def hook_fn(module, args, output):
            if not self._prefill_mode:
                self._decode_step += 1
            return output

        return hook_fn

    def attach(self, device: torch.device) -> None:
        self.detach()
        layer = self.model.model.layers[self.target_layer]
        h1 = layer.post_attention_layernorm.register_forward_pre_hook(self._make_live_margin_hook())
        h2 = layer.self_attn.o_proj.register_forward_pre_hook(self._make_steer_hook(device))
        h3 = layer.self_attn.register_forward_hook(self._make_step_counter())
        self._hooks = [h1, h2, h3]

    def attach_with_criticality(self, device: torch.device, top_k: int = 50) -> None:
        self.detach()
        layer = self.model.model.layers[self.target_layer]
        h1 = layer.post_attention_layernorm.register_forward_pre_hook(self._make_live_margin_hook())
        h2 = layer.self_attn.o_proj.register_forward_pre_hook(
            self._make_steer_hook_with_criticality(device=device, top_k=int(top_k))
        )
        h3 = layer.self_attn.register_forward_hook(self._make_step_counter())
        self._hooks = [h1, h2, h3]

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self) -> None:
        self._current_margin = 0.0
        self._margin_at_evidence_pos = 0.0
        self._lambda = 0.0
        self._lambda_clipped = False
        self._evidence_position = -1
        self._position_found = False
        self.step_log = []
        self._v_star = None
        self._gain = 1.0
        self._decode_step = 0
        self._margin_errors = 0
        self._margin_error_samples = []

    def summary(self) -> dict[str, Any]:
        steered = [row for row in self.step_log if bool(row.get("steered", False))]
        lambdas = [float(row.get("lambda", 0.0)) for row in steered]
        margins = [float(row.get("margin", 0.0)) for row in self.step_log]
        clipped_n = sum(1 for row in steered if bool(row.get("clipped", False)))
        return {
            "n_steps_total": int(len(self.step_log)),
            "n_steps_steered": int(len(steered)),
            "frac_steered": float(len(steered) / max(len(self.step_log), 1)),
            "mean_lambda": float(np.mean(lambdas)) if lambdas else 0.0,
            "clip_fraction": float(clipped_n / max(len(steered), 1)),
            "margins_by_step": margins,
            "margin_errors": int(self._margin_errors),
            "first_margin_error": (self._margin_error_samples[0] if self._margin_error_samples else ""),
            "prefill_lambda": float(self._lambda),
            "prefill_margin": float(self._margin_at_evidence_pos),
            "evidence_position": int(self._evidence_position),
            "evidence_position_found": bool(self._position_found),
        }


class DynamicLambdaRunningGate(RunningMarginGate):
    """
    Running gate variant for integration-failure regime.

    Static gate:
      lambda is derived once from prefill margin at evidence position.
    Dynamic gate:
      lambda is recomputed at each commitment token based on live margin deficit.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dynamic_mode = True
        self._prefill_lambda_saved = 0.0
        self._prefill_margin_saved = 0.0

    def bootstrap_prefill(
        self,
        prompt_ids: torch.Tensor,
        device: torch.device,
        context_answer: str | None = None,
        parametric_answer: str | None = None,
        model=None,
    ) -> None:
        super().bootstrap_prefill(
            prompt_ids=prompt_ids,
            device=device,
            context_answer=context_answer,
            parametric_answer=parametric_answer,
            model=model,
        )
        self._prefill_margin_saved = float(self._margin_at_evidence_pos)
        self._prefill_lambda_saved = float(self._lambda)
        # Force dynamic mode: do not use static prefill lambda in decode.
        self._lambda = 0.0
        self._lambda_clipped = False

    def _make_steer_hook_with_criticality(self, device: torch.device, top_k: int = 50):
        del device
        s = self._s
        e = self._e
        top_k = max(1, int(top_k))

        def hook_fn(module, args):
            if self._prefill_mode or not args:
                return args
            x = args[0]
            if x is None or x.ndim != 3:
                return args
            if self.first_token_only and self._decode_step > 0:
                self.step_log.append(
                    {"steered": False, "reason": "first_token_only", "margin": float(self._current_margin), "lambda": 0.0}
                )
                return args
            if self._v_star is None:
                self.step_log.append({"steered": False, "reason": "no_v_star", "margin": float(self._current_margin), "lambda": 0.0})
                return args

            try:
                margin, logits = self._compute_margin_and_logits(x[:, -1:, :])
                self._current_margin = float(margin)
            except Exception as err:
                self._margin_errors += 1
                if len(self._margin_error_samples) < 5:
                    self._margin_error_samples.append(str(err))
                self.step_log.append(
                    {"steered": False, "reason": "margin_error", "margin": 0.0, "lambda": 0.0}
                )
                return args

            vocab_size = int(logits.numel())
            k_eff = min(top_k, vocab_size)
            top_k_ids = set(int(x) for x in logits.topk(k_eff).indices.tolist())
            e_present = bool(set(self.E_ids) & top_k_ids)
            p_present = bool(set(self.P_ids) & top_k_ids)
            if not (e_present or p_present):
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "not_commitment_token",
                        "margin": float(margin) if np.isfinite(margin) else 0.0,
                        "lambda": 0.0,
                        "E_present": False,
                        "P_present": False,
                        "dynamic": True,
                    }
                )
                return args

            if not np.isfinite(margin):
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "margin_nan",
                        "margin": 0.0,
                        "lambda": 0.0,
                        "E_present": e_present,
                        "P_present": p_present,
                        "dynamic": True,
                    }
                )
                return args

            deficit = float(self.tau_margin - float(margin))
            if deficit <= 0.0:
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "margin_ok",
                        "margin": float(margin),
                        "lambda": 0.0,
                        "E_present": e_present,
                        "P_present": p_present,
                        "dynamic": True,
                    }
                )
                return args

            raw_lambda = float(self.alpha * deficit / max(self._gain, 1e-8))
            lam = float(min(raw_lambda, self.lambda_max))
            clipped = bool(lam < raw_lambda)
            if lam <= 0.0:
                self.step_log.append(
                    {
                        "steered": False,
                        "reason": "no_lambda",
                        "margin": float(margin),
                        "lambda": 0.0,
                        "E_present": e_present,
                        "P_present": p_present,
                        "dynamic": True,
                    }
                )
                return args

            x_mod = x.clone()
            v = self._v_star.to(device=x_mod.device, dtype=torch.float32)
            updated = x_mod[:, -1:, s:e].float() + lam * v.view(1, 1, -1)
            x_mod[:, -1:, s:e] = updated.to(dtype=x_mod.dtype)
            self.step_log.append(
                {
                    "steered": True,
                    "reason": "steered",
                    "margin": float(margin),
                    "margin_live": float(margin),
                    "lambda": float(lam),
                    "clipped": clipped,
                    "E_present": e_present,
                    "P_present": p_present,
                    "dynamic": True,
                }
            )
            return (x_mod,) + tuple(args[1:])

        return hook_fn

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        payload["dynamic_mode"] = True
        payload["prefill_margin"] = float(self._prefill_margin_saved)
        payload["prefill_lambda"] = float(self._prefill_lambda_saved)
        return payload
