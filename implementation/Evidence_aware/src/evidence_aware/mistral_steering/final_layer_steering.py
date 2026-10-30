from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class MistralFinalLayerSteering:
    """
    Final-layer steering for routed architectures.

    The intervention is applied after `model.model.norm` and before `lm_head`.
    Steering is applied to the last token position only.
    """

    def __init__(
        self,
        model,
        evidence_set: set[int],
        parametric_set: set[int],
        tau_margin: float = 0.0,
        alpha: float = 1.0,
        lambda_max: float = 5.0,
        first_token_only: bool = True,
    ) -> None:
        self._model = model
        self._lm_head = model.lm_head
        self._norm = model.model.norm

        self.tau_margin = float(tau_margin)
        self.alpha = float(alpha)
        self.lambda_max = float(lambda_max)
        self.first_token_only = bool(first_token_only)

        lm_bias = getattr(self._lm_head, "bias", None)
        self._lm_bias_fp32 = lm_bias.detach().float() if lm_bias is not None else None

        self._hooks: list[Any] = []
        self._step = 0
        self._lambda = 0.0
        self._margin = float("nan")
        self._d_ep_gain_one: torch.Tensor | None = None
        self._last_ev_id: int | None = None
        self._last_par_id: int | None = None
        self.log: list[dict[str, Any]] = []

        self.E_ids: list[int] = []
        self.P_ids: list[int] = []
        self.update_sets(evidence_set=evidence_set, parametric_set=parametric_set)

    def update_sets(self, evidence_set: set[int], parametric_set: set[int]) -> None:
        self.E_ids = sorted({int(x) for x in evidence_set})
        self.P_ids = sorted({int(x) for x in parametric_set})

    def reset(self) -> None:
        self._step = 0
        self._lambda = 0.0
        self._margin = float("nan")
        self._d_ep_gain_one = None
        self._last_ev_id = None
        self._last_par_id = None
        self.log = []

    def setup_example(
        self,
        evidence_token_id: int,
        parametric_token_id: int,
        model,
        device,
    ) -> None:
        weight = model.lm_head.weight.detach().float()
        ev = int(evidence_token_id)
        par = int(parametric_token_id)
        if ev < 0 or par < 0 or ev >= weight.shape[0] or par >= weight.shape[0]:
            self._d_ep_gain_one = None
            self._lambda = 0.0
            return
        d_ep = weight[ev] - weight[par]
        denom = float(torch.dot(d_ep, d_ep).item())
        if denom <= 1e-12:
            self._d_ep_gain_one = None
            self._lambda = 0.0
            return

        # Gain-one direction: adding lam * d causes exact LD increase of lam.
        self._d_ep_gain_one = (d_ep / denom).to(device=device, dtype=torch.float32)
        self._last_ev_id = ev
        self._last_par_id = par

    def _compute_margin_from_hidden(self, h_post_attn_last: torch.Tensor) -> float:
        if not self.E_ids or not self.P_ids:
            return float("nan")
        lm_weight = self._lm_head.weight
        h = h_post_attn_last.to(device=lm_weight.device, dtype=lm_weight.dtype)
        with torch.no_grad():
            normed = self._norm(h)
            logits = self._lm_head(normed)[0, 0, :].float()
            vocab = int(logits.numel())
            e_ids = [tid for tid in self.E_ids if 0 <= int(tid) < vocab]
            p_ids = [tid for tid in self.P_ids if 0 <= int(tid) < vocab]
            if not e_ids or not p_ids:
                return float("nan")
            e_idx = torch.as_tensor(e_ids, device=logits.device, dtype=torch.long)
            p_idx = torch.as_tensor(p_ids, device=logits.device, dtype=torch.long)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_p_e = torch.logsumexp(log_probs.index_select(0, e_idx), dim=0)
            log_p_p = torch.logsumexp(log_probs.index_select(0, p_idx), dim=0)
            return float((log_p_e - log_p_p).item())

    def bootstrap_prefill(
        self,
        model,
        prompt_ids: torch.Tensor,
        device,
    ) -> None:
        if self._d_ep_gain_one is None:
            self._margin = float("nan")
            self._lambda = 0.0
            return

        captured: dict[str, float] = {"margin": float("nan")}

        def hook_fn(module, args):
            if not args:
                return args
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            captured["margin"] = self._compute_margin_from_hidden(hidden[:, -1:, :])
            return args

        handle = model.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(hook_fn)
        try:
            with torch.inference_mode():
                if prompt_ids.ndim == 1:
                    input_ids = prompt_ids.unsqueeze(0).to(device)
                else:
                    input_ids = prompt_ids.to(device)
                attn = torch.ones_like(input_ids)
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()

        self._margin = float(captured.get("margin", float("nan")))
        if np.isfinite(self._margin):
            deficit = float(self.tau_margin - self._margin)
            self._lambda = float(min(self.alpha * deficit, self.lambda_max)) if deficit > 0.0 else 0.0
        else:
            self._lambda = 0.0

    def _make_norm_output_hook(self):
        def hook_fn(module, args, output):
            if not isinstance(output, torch.Tensor):
                return output

            should_consider = not (self.first_token_only and self._step > 0)
            steered = False
            clipped = False
            lam = 0.0

            if should_consider and self._d_ep_gain_one is not None and abs(self._lambda) > 1e-12:
                lam = float(self._lambda)
                steer_vec = self._d_ep_gain_one.to(device=output.device, dtype=output.dtype)
                out = output.clone()
                out[:, -1, :] = out[:, -1, :] + (lam * steer_vec)
                output = out
                steered = True
                clipped = bool(lam >= (self.lambda_max - 1e-12))

            self.log.append(
                {
                    "step": int(self._step),
                    "steered": bool(steered),
                    "lambda": float(lam),
                    "clipped": bool(clipped),
                    "margin": float(self._margin),
                }
            )
            self._step += 1
            return output

        return hook_fn

    def attach(self, model) -> None:
        self.detach()
        h = model.model.norm.register_forward_hook(self._make_norm_output_hook())
        self._hooks = [h]

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def summary(self) -> dict[str, Any]:
        steered_rows = [row for row in self.log if bool(row.get("steered", False))]
        n_total = int(len(self.log))
        n_steered = int(len(steered_rows))
        return {
            "n_steps_total": n_total,
            "n_steps_steered": n_steered,
            "frac_steered": float(n_steered / max(n_total, 1)),
            "mean_lambda": float(np.mean([float(r["lambda"]) for r in steered_rows])) if steered_rows else 0.0,
            "clip_fraction": (
                float(sum(1 for r in steered_rows if bool(r.get("clipped", False))) / max(n_steered, 1))
                if steered_rows
                else 0.0
            ),
            "bootstrap_margin": float(self._margin),
            "evidence_token_id": self._last_ev_id,
            "parametric_token_id": self._last_par_id,
        }


def test_final_layer_steering_exactness(
    model,
    evidence_token_id: int,
    parametric_token_id: int,
    device,
    lambda_value: float = 2.5,
    tolerance: float = 1e-2,
) -> dict[str, Any]:
    """
    Checks LD exactness under gain-one direction steering:
      h' = h + lambda * (d_ep / ||d_ep||^2)
      LD(h') - LD(h) = lambda
    """
    weight = model.lm_head.weight.detach().float()
    ev = int(evidence_token_id)
    par = int(parametric_token_id)
    if ev < 0 or par < 0 or ev >= weight.shape[0] or par >= weight.shape[0]:
        raise RuntimeError(f"Invalid token ids for exactness test: ev={ev}, par={par}")

    d_ep = weight[ev] - weight[par]
    denom = float(torch.dot(d_ep, d_ep).item())
    if denom <= 1e-12:
        raise RuntimeError("Exactness test failed: degenerate d_ep direction (norm^2 too small).")
    d_gain_one = (d_ep / denom).to(device=device, dtype=torch.float32)

    hidden_size = int(weight.shape[1])
    h = torch.randn(hidden_size, device=device, dtype=torch.float32)
    lm_weight = model.lm_head.weight

    def _project(vec: torch.Tensor) -> torch.Tensor:
        v = vec.to(device=lm_weight.device, dtype=lm_weight.dtype).unsqueeze(0)
        return model.lm_head(v).float().squeeze(0)

    logits_before = _project(h)
    ld_before = float((logits_before[ev] - logits_before[par]).item())

    lam = float(lambda_value)
    logits_after = _project(h + lam * d_gain_one)
    ld_after = float((logits_after[ev] - logits_after[par]).item())

    actual_delta = float(ld_after - ld_before)
    expected_delta = float(lam)
    error = float(abs(actual_delta - expected_delta))
    passed = bool(error < float(tolerance))
    payload = {
        "passed": passed,
        "expected_delta": expected_delta,
        "actual_delta": actual_delta,
        "abs_error": error,
        "tolerance": float(tolerance),
        "evidence_token_id": ev,
        "parametric_token_id": par,
    }
    if not passed:
        raise RuntimeError(
            "Final-layer steering exactness test failed: "
            f"expected_delta={expected_delta:.6f}, actual_delta={actual_delta:.6f}, abs_error={error:.6f}, "
            f"tolerance={float(tolerance):.6f}"
        )
    return payload
