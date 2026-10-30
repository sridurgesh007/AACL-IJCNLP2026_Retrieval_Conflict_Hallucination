from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from medrag.running_margin_gate import RunningMarginGate


@dataclass
class DecomposedGateState:
    prefill_margin: float
    gate_lambda: float
    lambda_clipped: bool
    conflict_detected: bool
    lambda_pos_applied: float
    lambda_neg_applied: float


class DecomposedMarginGate:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        evidence_set: set[int],
        parametric_set: set[int],
        target_layer: int = 27,
        target_head: int = 20,
        tau_margin: float = 0.0,
        alpha: float = 1.0,
        lambda_max: float = 10.0,
    ) -> None:
        self._gate = RunningMarginGate(
            model=model,
            tokenizer=tokenizer,
            evidence_set=evidence_set,
            parametric_set=parametric_set,
            target_layer=int(target_layer),
            target_head=int(target_head),
            tau_margin=float(tau_margin),
            alpha=float(alpha),
            lambda_max=float(lambda_max),
            first_token_only=True,
        )

    @property
    def gate(self) -> RunningMarginGate:
        return self._gate

    def setup_example(self, evidence_token_id: int, parametric_token_id: int, device: torch.device) -> None:
        self._gate.setup_example(int(evidence_token_id), int(parametric_token_id), device=device)

    def bootstrap_from_hidden(self, hidden_pre_norm: torch.Tensor) -> tuple[float, float, bool]:
        hidden = hidden_pre_norm
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0).unsqueeze(0)
        elif hidden.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected hidden_pre_norm shape: {tuple(hidden.shape)}")
        margin = float(self._gate._compute_margin(hidden))
        lam, clipped = self._gate._compute_prefill_lambda(float(margin))
        self._gate._current_margin = float(margin)
        self._gate._margin_at_evidence_pos = float(margin)
        self._gate._evidence_position = -1
        self._gate._position_found = False
        self._gate._lambda = float(lam)
        self._gate._lambda_clipped = bool(clipped)
        return float(margin), float(lam), bool(clipped)

    def decide(self, *, hidden_pre_norm: torch.Tensor, lambda_pos: float, lambda_neg: float) -> DecomposedGateState:
        margin, gate_lambda, clipped = self.bootstrap_from_hidden(hidden_pre_norm)
        conflict = bool(np.isfinite(margin) and margin < float(self._gate.tau_margin))
        return DecomposedGateState(
            prefill_margin=float(margin),
            gate_lambda=float(gate_lambda),
            lambda_clipped=bool(clipped),
            conflict_detected=bool(conflict),
            lambda_pos_applied=float(lambda_pos) if conflict else 0.0,
            lambda_neg_applied=float(lambda_neg) if conflict else 0.0,
        )


def _self_test() -> None:
    pure_suppression = DecomposedGateState(
        prefill_margin=-1.0,
        gate_lambda=2.0,
        lambda_clipped=False,
        conflict_detected=True,
        lambda_pos_applied=0.0,
        lambda_neg_applied=5.0,
    )
    assert pure_suppression.lambda_pos_applied == 0.0
    assert pure_suppression.lambda_neg_applied > 0.0


_self_test()
