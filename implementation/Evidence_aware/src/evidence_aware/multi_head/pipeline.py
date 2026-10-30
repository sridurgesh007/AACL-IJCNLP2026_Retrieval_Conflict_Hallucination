from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from evidence_aware.ag_erha.track_b import build_evidence_set, build_parametric_set
from evidence_aware.ag_erha.utils import (
    load_cache_rows,
    normalize_example_rows,
    prepare_examples,
    read_json,
    read_jsonl,
    seed_everything,
    write_json,
    write_jsonl,
)
from evidence_aware.phase2.common import PreparedExample, classify_answer_token, first_non_whitespace_from_sequence
from evidence_aware.phase2.modeling import load_model_and_tokenizer, to_tensors


@dataclass(frozen=True)
class MultiHeadConfig:
    model_id: str
    cache_path: Path
    phase1_ld_path: Path
    phase2_main_eval_dir: Path
    ag_track_a_dir: Path
    out_dir: Path
    target_layers: list[int]
    late_layers: list[int]
    multi_betas: list[float]
    tau_values: list[float]
    single_beta: float
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    sanity_tolerance: float
    first_token_only: bool
    bootstrap_first_token_margin: bool


def _parse_float_list(value: str) -> list[float]:
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty float list, got: {value}")
    return vals


def _parse_int_list(value: str) -> list[int]:
    vals = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty int list, got: {value}")
    return vals


def parse_multi_head_config(args: Any) -> MultiHeadConfig:
    return MultiHeadConfig(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase1_ld_path=Path(args.phase1_ld_path),
        phase2_main_eval_dir=Path(args.phase2_main_eval_dir),
        ag_track_a_dir=Path(args.ag_track_a_dir),
        out_dir=Path(args.out_dir),
        target_layers=_parse_int_list(args.target_layers),
        late_layers=_parse_int_list(args.late_layers),
        multi_betas=_parse_float_list(getattr(args, "multi_betas", "2.0,3.0,5.0")),
        tau_values=_parse_float_list(args.tau_values),
        single_beta=float(args.single_beta),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        sanity_tolerance=float(args.sanity_tolerance),
        first_token_only=bool(int(args.first_token_only)),
        bootstrap_first_token_margin=bool(int(args.bootstrap_first_token_margin)),
    )


def _metric_as_rate(payload: dict[str, Any], key: str) -> float:
    if key in payload:
        val = float(payload[key])
    elif key == "stability_rate" and "stability" in payload:
        val = float(payload["stability"])
    elif key == "accuracy" and "n_current_faithful" in payload and "n_total" in payload:
        val = float(payload["n_current_faithful"]) / max(float(payload["n_total"]), 1.0)
    else:
        raise KeyError(f"Missing metric '{key}' in payload keys={sorted(payload.keys())}")
    return val / 100.0 if val > 1.0 else val


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _load_baseline_map(main_eval_dir: Path) -> dict[int, dict[str, Any]]:
    baseline_examples_path = main_eval_dir / "baseline_examples.jsonl"
    if not baseline_examples_path.exists():
        raise FileNotFoundError(f"Missing baseline examples: {baseline_examples_path}")
    rows = read_jsonl(baseline_examples_path)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(row["id"])] = {"label": str(row.get("label", "other"))}
    if not out:
        raise RuntimeError("baseline_examples.jsonl is empty.")
    return out


def _load_eval_examples(
    cache_path: Path,
    tokenizer,
    max_length: int,
    baseline_map: dict[int, dict[str, Any]],
) -> list[PreparedExample]:
    rows = normalize_example_rows(load_cache_rows(cache_path))
    eval_ids = set(int(x) for x in baseline_map.keys())
    eval_rows = [r for r in rows if int(r.id) in eval_ids]
    prepared = prepare_examples(eval_rows, tokenizer, max_length=max_length)
    prepared_ids = {x.id for x in prepared}
    missing = sorted(eval_ids - prepared_ids)
    if missing:
        raise RuntimeError(
            "Prepared eval examples do not match baseline id set. "
            f"missing_count={len(missing)} sample={missing[:10]}"
        )
    order = {idx: pos for pos, idx in enumerate(baseline_map.keys())}
    prepared.sort(key=lambda x: order.get(x.id, 10**18))
    return prepared


def _build_sets_for_eval_examples(
    model,
    tokenizer,
    device: torch.device,
    eval_examples: list[PreparedExample],
) -> dict[int, dict[str, set[int]]]:
    out: dict[int, dict[str, set[int]]] = {}
    for ex in eval_examples:
        evidence_set = set(build_evidence_set(tokenizer, ex.answer_liquid))
        if not evidence_set:
            evidence_set = {int(ex.evidence_token_id)}
        input_ids, attention_mask = to_tensors(ex, device=device)
        with torch.inference_mode():
            base_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        base_logits = base_out.logits[0, -1, :].float()
        baseline_top = torch.topk(base_logits, k=min(20, int(base_logits.numel()))).indices.tolist()
        parametric_set = set(build_parametric_set([int(t) for t in baseline_top], evidence_set, k=5))
        parametric_set = {int(x) for x in parametric_set if int(x) not in evidence_set}
        if not parametric_set:
            parametric_set = {int(ex.parametric_token_id)}
        out[ex.id] = {
            "evidence_set": {int(x) for x in evidence_set},
            "parametric_set": {int(x) for x in parametric_set},
        }
    return out


def _load_top_heads_per_layer(
    phase1_ld_path: Path,
    target_layers: list[int],
    out_path: Path,
) -> tuple[dict[int, int], dict[int, float]]:
    if not phase1_ld_path.exists():
        raise FileNotFoundError(f"Missing phase1 LD file: {phase1_ld_path}")
    rows = read_jsonl(phase1_ld_path)
    faithful: list[dict[str, Any]] = []
    for row in rows:
        if "label" in row:
            if str(row.get("label", "")).lower() == "faithful":
                faithful.append(row)
            continue
        if "is_hallucinated" in row and not bool(row.get("is_hallucinated", True)):
            faithful.append(row)
    if not faithful:
        raise RuntimeError("No faithful rows found in phase1 LD file.")

    top_heads: dict[int, int] = {}
    top_values: dict[int, float] = {}
    out_json: dict[str, dict[str, float | int | str]] = {}

    # Accept mixed schemas across rows:
    # 1) dense matrix fields: head_ld_matrix or head_ld
    # 2) sparse dict field: head_delta_attn_evidence_by_layer
    per_layer_vectors: dict[int, list[np.ndarray]] = {}
    per_layer_sources: dict[int, set[str]] = {}
    expected_heads: int | None = None
    used_rows = 0

    target_layer_set = {int(x) for x in target_layers}
    for row in faithful:
        row_used = False
        matrix_payload = None
        source_name = ""

        if "head_ld_matrix" in row:
            matrix_payload = row.get("head_ld_matrix")
            source_name = "head_ld_matrix"
        elif "head_ld" in row:
            matrix_payload = row.get("head_ld")
            source_name = "head_ld"

        if matrix_payload is not None:
            mat = np.array(matrix_payload, dtype=np.float64)
            if mat.ndim != 2:
                raise RuntimeError(
                    f"{source_name} must be 2D per row. example_id={row.get('example_id', row.get('id', 'unknown'))}"
                )
            num_layers, num_heads = int(mat.shape[0]), int(mat.shape[1])
            if expected_heads is None:
                expected_heads = num_heads
            elif int(num_heads) != int(expected_heads):
                raise RuntimeError(
                    "Inconsistent head dimension across phase1 rows. "
                    f"expected={expected_heads} got={num_heads} source={source_name}"
                )
            for layer_idx in target_layers:
                if 0 <= int(layer_idx) < num_layers:
                    vec = mat[int(layer_idx)]
                    per_layer_vectors.setdefault(int(layer_idx), []).append(vec)
                    per_layer_sources.setdefault(int(layer_idx), set()).add(source_name)
                    row_used = True

        layer_map = row.get("head_delta_attn_evidence_by_layer", None)
        if isinstance(layer_map, dict):
            for layer_key, vec in layer_map.items():
                layer_idx = int(layer_key)
                if int(layer_idx) not in target_layer_set:
                    continue
                arr = np.array(vec, dtype=np.float64)
                if arr.ndim != 1:
                    raise RuntimeError(
                        "head_delta_attn_evidence_by_layer values must be 1D per layer. "
                        f"example_id={row.get('example_id', 'unknown')} layer={layer_idx}"
                    )
                if expected_heads is None:
                    expected_heads = int(arr.shape[0])
                elif int(arr.shape[0]) != int(expected_heads):
                    raise RuntimeError(
                        "Inconsistent head dimension in head_delta_attn_evidence_by_layer. "
                        f"expected={expected_heads} got={arr.shape[0]} at layer={layer_idx}"
                    )
                per_layer_vectors.setdefault(int(layer_idx), []).append(arr)
                per_layer_sources.setdefault(int(layer_idx), set()).add("head_delta_attn_evidence_by_layer")
                row_used = True

        if row_used:
            used_rows += 1

    if used_rows == 0:
        first = faithful[0]
        raise RuntimeError(
            "Unsupported phase1 file schema for multi-head extraction. "
            "Expected one of: 'head_ld_matrix', 'head_ld', 'head_delta_attn_evidence_by_layer'. "
            f"First faithful row keys={sorted(first.keys())}"
        )

    missing_layers = [int(l) for l in target_layers if int(l) not in per_layer_vectors]
    if missing_layers:
        available_layers = sorted(int(k) for k in per_layer_vectors.keys())
        raise RuntimeError(
            "Requested target layers are missing in phase1 LD traces. "
            f"missing={missing_layers}; available={available_layers}. "
            "Set target layers to available layers or regenerate phase1 LD outputs."
        )

    for layer_idx in target_layers:
        mat = np.stack(per_layer_vectors[int(layer_idx)], axis=0)
        mean_vec = np.mean(mat, axis=0)
        head = int(np.argmax(mean_vec))
        val = float(mean_vec[head])
        top_heads[int(layer_idx)] = int(head)
        top_values[int(layer_idx)] = float(val)
        out_json[str(int(layer_idx))] = {
            "head": int(head),
            "mean_ld_faithful": float(val),
            "n_rows_used": int(mat.shape[0]),
            "source": ",".join(sorted(per_layer_sources.get(int(layer_idx), {"unknown"}))),
        }

    write_json(out_path, out_json)
    return top_heads, top_values

class MultiLayerMarginGatedHeadAmplifier:
    def __init__(
        self,
        model,
        layer_to_head: dict[int, int],
        evidence_set: set[int],
        parametric_set: set[int],
        tau_margin: float,
        beta: float,
        first_token_only: bool,
        bootstrap_first_token_margin: bool,
    ) -> None:
        self.model = model
        self.layer_to_head = {int(k): int(v) for k, v in layer_to_head.items()}
        self.layers = sorted(self.layer_to_head.keys())
        self.tau_margin = float(tau_margin)
        self.beta = float(beta)
        self.first_token_only = bool(first_token_only)
        self.bootstrap_first_token_margin = bool(bootstrap_first_token_margin)

        self.num_layers = int(model.config.num_hidden_layers)
        self.num_heads = int(model.config.num_attention_heads)
        self.hidden_size = int(model.config.hidden_size)
        if self.hidden_size % self.num_heads != 0:
            raise RuntimeError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads
        for layer_idx, head_idx in self.layer_to_head.items():
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(f"Invalid target layer {layer_idx}; valid [0,{self.num_layers - 1}]")
            if head_idx < 0 or head_idx >= self.num_heads:
                raise ValueError(f"Invalid target head {head_idx}; valid [0,{self.num_heads - 1}]")

        self._lm_weight_fp32 = model.lm_head.weight.detach().float()
        lm_bias = getattr(model.lm_head, "bias", None)
        self._lm_bias_fp32 = lm_bias.detach().float() if lm_bias is not None else None
        self._norm = model.model.norm

        self._evidence_ids: list[int] = []
        self._parametric_ids: list[int] = []
        self.hooks: list[Any] = []

        self._bootstrap_mode = False
        self._step = 0
        self._layer_margins: dict[int, float] = {}
        self._gate_decisions: dict[int, bool] = {}
        self._fired_count: dict[int, int] = {}
        self._blocked_count: dict[int, int] = {}
        self._margin_errors = 0
        self._margin_error_samples: list[str] = []
        self.update_sets(evidence_set, parametric_set)

    def update_sets(self, evidence_set: set[int], parametric_set: set[int]) -> None:
        self._evidence_ids = sorted(int(x) for x in evidence_set)
        self._parametric_ids = sorted(int(x) for x in parametric_set)

    def reset_log(self) -> None:
        self._bootstrap_mode = False
        self._step = 0
        self._layer_margins = {}
        self._gate_decisions = {}
        self._fired_count = {int(l): 0 for l in self.layers}
        self._blocked_count = {int(l): 0 for l in self.layers}
        self._margin_errors = 0
        self._margin_error_samples = []

    def _compute_margin(self, h_post_attn_last: torch.Tensor) -> float:
        if not self._evidence_ids or not self._parametric_ids:
            return float("nan")
        with torch.no_grad():
            normed = self._norm(h_post_attn_last)
            logits = F.linear(normed.float(), self._lm_weight_fp32, self._lm_bias_fp32)[0, 0, :]
            vocab_size = int(logits.numel())
            evidence_ids = [tid for tid in self._evidence_ids if 0 <= int(tid) < vocab_size]
            parametric_ids = [tid for tid in self._parametric_ids if 0 <= int(tid) < vocab_size]
            if not evidence_ids or not parametric_ids:
                return float("nan")
            e_idx = torch.as_tensor(evidence_ids, device=logits.device, dtype=torch.long)
            p_idx = torch.as_tensor(parametric_ids, device=logits.device, dtype=torch.long)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_p_e = torch.logsumexp(log_probs.index_select(0, e_idx), dim=0)
            log_p_p = torch.logsumexp(log_probs.index_select(0, p_idx), dim=0)
            return float((log_p_e - log_p_p).item())

    def _make_margin_hook(self, layer_idx: int):
        def hook_fn(module, args):
            if not args:
                return args
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            try:
                margin = self._compute_margin(hidden[:, -1:, :])
                self._layer_margins[int(layer_idx)] = float(margin)
                self._gate_decisions[int(layer_idx)] = bool(np.isfinite(margin) and (float(margin) < self.tau_margin))
            except Exception as err:
                self._layer_margins[int(layer_idx)] = float("nan")
                self._gate_decisions[int(layer_idx)] = False
                self._margin_errors += 1
                if len(self._margin_error_samples) < 5:
                    self._margin_error_samples.append(str(err))
            return args

        return hook_fn

    def _make_o_proj_hook(self, layer_idx: int):
        head_idx = int(self.layer_to_head[layer_idx])
        start = head_idx * self.head_dim
        end = (head_idx + 1) * self.head_dim
        last_layer = int(self.layers[-1])

        def hook_fn(module, args):
            if self._bootstrap_mode:
                return args
            if not args:
                return args
            x = args[0]
            if x is None or x.ndim != 3:
                return args

            should_eval = (self._step == 0) if self.first_token_only else True
            fired = False
            if should_eval:
                fired = bool(self._gate_decisions.get(int(layer_idx), False))
                if fired:
                    self._fired_count[int(layer_idx)] += 1
                else:
                    self._blocked_count[int(layer_idx)] += 1
            if fired:
                x_mod = x.clone()
                x_mod[:, -1:, start:end] = x_mod[:, -1:, start:end] * self.beta
                out_args = (x_mod,) + tuple(args[1:])
            else:
                out_args = args

            if int(layer_idx) == last_layer:
                self._step += 1
            return out_args

        return hook_fn

    def attach(self) -> None:
        self.hooks = []
        for layer_idx in self.layers:
            layer = self.model.model.layers[int(layer_idx)]
            h_margin = layer.post_attention_layernorm.register_forward_pre_hook(self._make_margin_hook(int(layer_idx)))
            h_proj = layer.self_attn.o_proj.register_forward_pre_hook(self._make_o_proj_hook(int(layer_idx)))
            self.hooks.extend([h_margin, h_proj])

    def detach(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def bootstrap_prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if not self.bootstrap_first_token_margin:
            for layer_idx in self.layers:
                self._layer_margins[int(layer_idx)] = float("nan")
                self._gate_decisions[int(layer_idx)] = False
            return
        self._bootstrap_mode = True
        self._step = 0
        with torch.inference_mode():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        self._bootstrap_mode = False

    def gate_summary(self) -> dict[str, Any]:
        fired_layers = [int(l) for l in self.layers if int(self._fired_count.get(int(l), 0)) > 0]
        blocked_layers = [int(l) for l in self.layers if int(self._blocked_count.get(int(l), 0)) > 0]
        valid_margins = [float(self._layer_margins.get(int(l), float("nan"))) for l in self.layers]
        valid_margins = [x for x in valid_margins if np.isfinite(x)]
        per_layer = {}
        for layer_idx in self.layers:
            fired = int(self._fired_count.get(int(layer_idx), 0))
            blocked = int(self._blocked_count.get(int(layer_idx), 0))
            total = fired + blocked
            per_layer[str(int(layer_idx))] = {
                "margin": float(self._layer_margins.get(int(layer_idx), float("nan"))),
                "decision": bool(self._gate_decisions.get(int(layer_idx), False)),
                "fired": fired,
                "blocked": blocked,
                "fired_frac": float(fired / total) if total > 0 else 0.0,
            }
        return {
            "fired_layers": fired_layers,
            "blocked_layers": blocked_layers,
            "n_fired_layers": int(len(fired_layers)),
            "n_blocked_layers": int(len(blocked_layers)),
            "mean_margin_target_layers": (float(np.mean(valid_margins)) if valid_margins else float("nan")),
            "margin_errors": int(self._margin_errors),
            "first_margin_error": (self._margin_error_samples[0] if self._margin_error_samples else ""),
            "per_layer": per_layer,
        }


def _decode_with_multi_head_gate(
    model,
    tokenizer,
    device: torch.device,
    example: PreparedExample,
    max_new_tokens: int,
    gate: MultiLayerMarginGatedHeadAmplifier,
) -> dict[str, Any]:
    input_ids, attention_mask = to_tensors(example, device=device)
    gate.bootstrap_prefill(input_ids=input_ids, attention_mask=attention_mask)

    generated = input_ids
    mask = attention_mask
    past_key_values = None
    out_ids: list[int] = []

    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
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
    return {
        "label": label,
        "first_token_id": first_token_id,
        "generated_ids": out_ids,
        "gate_summary": gate.gate_summary(),
    }


def _compute_rates(
    metrics: dict[str, Any],
    baseline_map: dict[int, dict[str, Any]],
    eval_ids: list[int],
) -> None:
    baseline_labels = [baseline_map.get(i, {"label": "other"})["label"] for i in eval_ids]
    n_baseline_hall = sum(1 for x in baseline_labels if x == "hallucinated")
    n_baseline_faith = sum(1 for x in baseline_labels if x == "faithful")
    metrics["accuracy"] = float(metrics["n_faithful"] / max(metrics["n_total"], 1))
    metrics["restoration_rate"] = float(metrics["n_restored"] / max(n_baseline_hall, 1))
    metrics["harm_rate"] = float(metrics["n_harmed"] / max(n_baseline_faith, 1))
    metrics["stability_rate"] = float(metrics["n_stable"] / max(n_baseline_faith, 1))
    metrics["restoration_improvement"] = float(metrics["restoration_rate"] - metrics["harm_rate"])


def _build_per_layer_template(layers: list[int]) -> dict[int, dict[str, dict[str, int]]]:
    out: dict[int, dict[str, dict[str, int]]] = {}
    for layer_idx in layers:
        out[int(layer_idx)] = {
            "hallucinated": {"fired": 0, "total": 0},
            "faithful": {"fired": 0, "total": 0},
            "all": {"fired": 0, "total": 0},
        }
    return out


def _run_single_config_tau(
    *,
    model,
    tokenizer,
    device: torch.device,
    eval_examples: list[PreparedExample],
    baseline_map: dict[int, dict[str, Any]],
    sets_by_id: dict[int, dict[str, set[int]]],
    layer_to_head: dict[int, int],
    config_name: str,
    tau_margin: float,
    beta: float,
    first_token_only: bool,
    bootstrap_first_token_margin: bool,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, dict[str, int]]]]:
    gate = MultiLayerMarginGatedHeadAmplifier(
        model=model,
        layer_to_head=layer_to_head,
        evidence_set=set(),
        parametric_set=set(),
        tau_margin=float(tau_margin),
        beta=float(beta),
        first_token_only=bool(first_token_only),
        bootstrap_first_token_margin=bool(bootstrap_first_token_margin),
    )
    gate.attach()

    per_layer_counts = _build_per_layer_template(sorted(layer_to_head.keys()))
    metrics: dict[str, Any] = {
        "config_name": str(config_name),
        "tau_margin": float(tau_margin),
        "beta": float(beta),
        "layers": [int(x) for x in sorted(layer_to_head.keys())],
        "heads_by_layer": {str(int(k)): int(v) for k, v in sorted(layer_to_head.items())},
        "n_total": 0,
        "n_faithful": 0,
        "n_hallucinated": 0,
        "n_other": 0,
        "n_restored": 0,
        "n_harmed": 0,
        "n_stable": 0,
        "n_gate_fired_layers_total": 0,
        "n_gate_events_total": 0,
        "mean_fired_frac_hallucinated": float("nan"),
        "mean_fired_frac_faithful": float("nan"),
    }
    gate_rows: list[dict[str, Any]] = []
    fired_frac_by_label = {"faithful": [], "hallucinated": []}

    try:
        for ex in eval_examples:
            sets = sets_by_id.get(ex.id)
            if sets is None:
                raise RuntimeError(f"Missing token sets for example id={ex.id}")
            evidence_set = set(int(x) for x in sets["evidence_set"])
            parametric_set = set(int(x) for x in sets["parametric_set"])
            if not evidence_set or not parametric_set:
                raise RuntimeError(f"Invalid empty token sets for example id={ex.id}")

            gate.update_sets(evidence_set, parametric_set)
            gate.reset_log()
            out = _decode_with_multi_head_gate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                example=ex,
                max_new_tokens=max_new_tokens,
                gate=gate,
            )
            baseline_label = str(baseline_map.get(ex.id, {"label": "other"})["label"])
            label = str(out["label"])
            summary = out["gate_summary"]

            metrics["n_total"] += 1
            metrics[f"n_{label}"] += 1
            metrics["n_gate_fired_layers_total"] += int(summary["n_fired_layers"])
            metrics["n_gate_events_total"] += int(len(layer_to_head))

            if baseline_label == "hallucinated" and label == "faithful":
                metrics["n_restored"] += 1
            if baseline_label == "faithful" and label == "hallucinated":
                metrics["n_harmed"] += 1
            if baseline_label == "faithful" and label == "faithful":
                metrics["n_stable"] += 1

            for layer_idx in sorted(layer_to_head.keys()):
                layer_key = str(int(layer_idx))
                layer_payload = summary["per_layer"][layer_key]
                fired = int(layer_payload["fired"] > 0)
                per_layer_counts[int(layer_idx)]["all"]["fired"] += fired
                per_layer_counts[int(layer_idx)]["all"]["total"] += 1
                if baseline_label == "hallucinated":
                    per_layer_counts[int(layer_idx)]["hallucinated"]["fired"] += fired
                    per_layer_counts[int(layer_idx)]["hallucinated"]["total"] += 1
                if baseline_label == "faithful":
                    per_layer_counts[int(layer_idx)]["faithful"]["fired"] += fired
                    per_layer_counts[int(layer_idx)]["faithful"]["total"] += 1

            if baseline_label in ("faithful", "hallucinated"):
                fired_frac_by_label[baseline_label].append(float(summary["n_fired_layers"]) / max(len(layer_to_head), 1))

            gate_rows.append(
                {
                    "example_id": int(ex.id),
                    "config_name": str(config_name),
                    "tau_margin": float(tau_margin),
                    "beta": float(beta),
                    "baseline_label": baseline_label,
                    "label": label,
                    "first_token_id": out["first_token_id"],
                    "generated_ids": out["generated_ids"],
                    "fired_layers": [int(x) for x in summary["fired_layers"]],
                    "blocked_layers": [int(x) for x in summary["blocked_layers"]],
                    "n_fired_layers": int(summary["n_fired_layers"]),
                    "mean_margin_target_layers": float(summary["mean_margin_target_layers"]),
                    "margin_errors": int(summary["margin_errors"]),
                    "first_margin_error": str(summary["first_margin_error"]),
                    "per_layer": summary["per_layer"],
                }
            )
    finally:
        gate.detach()

    _compute_rates(metrics, baseline_map=baseline_map, eval_ids=[ex.id for ex in eval_examples])
    if fired_frac_by_label["faithful"]:
        metrics["mean_fired_frac_faithful"] = float(np.mean(fired_frac_by_label["faithful"]))
    if fired_frac_by_label["hallucinated"]:
        metrics["mean_fired_frac_hallucinated"] = float(np.mean(fired_frac_by_label["hallucinated"]))
    return metrics, gate_rows, per_layer_counts


def _find_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Cannot select best row from empty list.")
    return max(
        rows,
        key=lambda r: (
            float(r["restoration_improvement"]),
            float(r["stability_rate"]),
            -float(r["harm_rate"]),
            float(r["restoration_rate"]),
        ),
    )


def _compute_per_layer_fire_rates(
    per_layer_counts: dict[int, dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for layer_idx in sorted(per_layer_counts.keys()):
        out[str(int(layer_idx))] = {}
        for label in ("hallucinated", "faithful", "all"):
            fired = int(per_layer_counts[layer_idx][label]["fired"])
            total = int(per_layer_counts[layer_idx][label]["total"])
            out[str(int(layer_idx))][label] = {
                "fired": fired,
                "total": total,
                "fired_frac": float(fired / total) if total > 0 else 0.0,
            }
    return out


def _load_ag_best_metrics(ag_track_a_dir: Path) -> dict[str, Any] | None:
    summary_path = ag_track_a_dir / "ag_erha_sweep_summary.json"
    if not summary_path.exists():
        return None
    rows = read_json(summary_path)
    if not isinstance(rows, list) or not rows:
        return None
    candidates = [r for r in rows if float(r.get("tau", 999.0)) not in (0.0, 1.0)]
    if not candidates:
        candidates = rows
    return max(candidates, key=lambda r: (float(r.get("restoration_improvement", 0.0)), float(r.get("stability_rate", 0.0))))


def _plot_restoration_improvement_by_config(out_dir: Path, best_by_config: list[dict[str, Any]]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r["config_name"]) for r in best_by_config]
    vals = [float(r["restoration_improvement"]) for r in best_by_config]
    taus = [float(r["tau_margin"]) for r in best_by_config]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(x, vals)
    for i, tau in enumerate(taus):
        ax.text(i, vals[i], f"tau={tau:g}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("restoration_improvement")
    ax.set_title("Restoration Improvement by Config (best tau per config)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "restoration_improvement_by_config.png", dpi=240)
    plt.close(fig)


def _plot_stability_by_config(out_dir: Path, best_by_config: list[dict[str, Any]]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r["config_name"]) for r in best_by_config]
    vals = [float(r["stability_rate"]) for r in best_by_config]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(x, vals)
    ax.axhline(0.92, color="orange", linestyle="--", linewidth=1.2, label="target 0.92")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("stability_rate")
    ax.set_title("Stability by Config (best tau per config)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stability_by_config.png", dpi=240)
    plt.close(fig)


def _plot_per_layer_gate_fire_rate(
    out_dir: Path,
    per_layer_rates: dict[str, dict[str, dict[str, float]]],
    title_suffix: str,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(int(x) for x in per_layer_rates.keys())
    hall = [float(per_layer_rates[str(l)]["hallucinated"]["fired_frac"]) for l in layers]
    faith = [float(per_layer_rates[str(l)]["faithful"]["fired_frac"]) for l in layers]
    all_vals = [float(per_layer_rates[str(l)]["all"]["fired_frac"]) for l in layers]

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.plot(layers, hall, marker="o", linewidth=2.0, label="hallucinated")
    ax.plot(layers, faith, marker="s", linewidth=2.0, label="faithful")
    ax.plot(layers, all_vals, marker="^", linewidth=1.8, label="all")
    ax.set_xlabel("layer")
    ax.set_ylabel("gate fired fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Per-layer Gate Fire Rate ({title_suffix})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "per_layer_gate_fire_rate.png", dpi=240)
    plt.close(fig)


def _plot_restoration_vs_tau_multi_vs_single(
    out_dir: Path,
    rows_single: list[dict[str, Any]],
    rows_multi: list[dict[str, Any]],
    multi_name: str,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows_single = sorted(rows_single, key=lambda x: float(x["tau_margin"]))
    rows_multi = sorted(rows_multi, key=lambda x: float(x["tau_margin"]))
    xs_single = [float(r["tau_margin"]) for r in rows_single]
    ys_single = [float(r["restoration_improvement"]) for r in rows_single]
    xs_multi = [float(r["tau_margin"]) for r in rows_multi]
    ys_multi = [float(r["restoration_improvement"]) for r in rows_multi]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(xs_single, ys_single, marker="o", linewidth=2.0, label="single_L27H20")
    ax.plot(xs_multi, ys_multi, marker="s", linewidth=2.0, label=multi_name)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("tau_margin")
    ax.set_ylabel("restoration_improvement")
    ax.set_title("Restoration Improvement vs tau (multi-head vs single-head)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "restoration_vs_tau_multi_vs_single.png", dpi=240)
    plt.close(fig)


def _plot_comparison_bar_all_methods(
    out_dir: Path,
    method_metrics: list[tuple[str, dict[str, Any]]],
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [name for name, _ in method_metrics]
    restoration = [float(m.get("restoration_rate", 0.0)) for _, m in method_metrics]
    stability = [float(m.get("stability_rate", 0.0)) for _, m in method_metrics]
    improvement = [float(m.get("restoration_improvement", 0.0)) for _, m in method_metrics]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.bar(x - width, restoration, width, label="restoration_rate")
    ax.bar(x, stability, width, label="stability_rate")
    ax.bar(x + width, improvement, width, label="restoration_improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Comparison Across Methods")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_bar_all_methods.png", dpi=240)
    plt.close(fig)


def run_multi_head_pipeline(config: MultiHeadConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "gate_stats").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(
        model_id=config.model_id,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )

    baseline_map = _load_baseline_map(config.phase2_main_eval_dir)
    eval_examples = _load_eval_examples(
        cache_path=config.cache_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        baseline_map=baseline_map,
    )
    sets_by_id = _build_sets_for_eval_examples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_examples,
    )

    required_layers = sorted(set(int(x) for x in config.target_layers) | set(int(x) for x in config.late_layers))
    top_heads_json_path = config.out_dir / "top_heads_per_layer.json"
    top_heads, top_values = _load_top_heads_per_layer(
        phase1_ld_path=config.phase1_ld_path,
        target_layers=required_layers,
        out_path=top_heads_json_path,
    )
    missing_for_configs = [int(l) for l in required_layers if int(l) not in top_heads]
    if missing_for_configs:
        raise RuntimeError(
            "Top-head extraction did not produce all required layers for configs. "
            f"missing={missing_for_configs}; available={sorted(top_heads.keys())}"
        )

    method_configs: list[dict[str, Any]] = []
    for beta in config.multi_betas:
        beta_name = str(beta).replace(".", "p")
        method_configs.append(
            {
                "name": f"multi_all_layers_b{beta_name}",
                "layer_to_head": {int(l): int(top_heads[int(l)]) for l in config.target_layers},
                "beta": float(beta),
            }
        )
        method_configs.append(
            {
                "name": f"multi_late_layers_b{beta_name}",
                "layer_to_head": {int(l): int(top_heads[int(l)]) for l in config.late_layers},
                "beta": float(beta),
            }
        )

    method_configs.append(
        {
            "name": "single_L27H20",
            "layer_to_head": {27: 20},
            "beta": float(config.single_beta),
        }
    )

    run_rows: list[dict[str, Any]] = []
    per_layer_counts_by_config_tau: dict[tuple[str, float], dict[int, dict[str, dict[str, int]]]] = {}

    for item in method_configs:
        for tau in config.tau_values:
            metrics, gate_rows, per_layer_counts = _run_single_config_tau(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eval_examples=eval_examples,
                baseline_map=baseline_map,
                sets_by_id=sets_by_id,
                layer_to_head=item["layer_to_head"],
                config_name=str(item["name"]),
                tau_margin=float(tau),
                beta=float(item["beta"]),
                first_token_only=bool(config.first_token_only),
                bootstrap_first_token_margin=bool(config.bootstrap_first_token_margin),
                max_new_tokens=config.max_new_tokens,
            )
            run_rows.append(metrics)
            per_layer_counts_by_config_tau[(str(item["name"]), float(tau))] = per_layer_counts
            out_name = f"gate_stats_config_{item['name']}_tau_{tau:g}.jsonl"
            write_jsonl(config.out_dir / "gate_stats" / out_name, gate_rows)

    # Sanity checks from multi_head.md implementation order.
    sanity_row_beta1, _, _ = _run_single_config_tau(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_examples,
        baseline_map=baseline_map,
        sets_by_id=sets_by_id,
        layer_to_head={int(l): int(top_heads[int(l)]) for l in config.target_layers},
        config_name="sanity_beta1_all_layers",
        tau_margin=0.0,
        beta=1.0,
        first_token_only=bool(config.first_token_only),
        bootstrap_first_token_margin=bool(config.bootstrap_first_token_margin),
        max_new_tokens=config.max_new_tokens,
    )

    sanity_row_active, _, _ = _run_single_config_tau(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_examples[:200],
        baseline_map=baseline_map,
        sets_by_id=sets_by_id,
        layer_to_head={int(l): int(top_heads[int(l)]) for l in config.target_layers},
        config_name="sanity_always_fire_probe",
        tau_margin=999.0,
        beta=2.0,
        first_token_only=bool(config.first_token_only),
        bootstrap_first_token_margin=bool(config.bootstrap_first_token_margin),
        max_new_tokens=config.max_new_tokens,
    )

    baseline_metrics = read_json(config.phase2_main_eval_dir / "baseline_results.json")
    baseline_accuracy = _metric_as_rate(baseline_metrics, "accuracy")
    baseline_stability = _metric_as_rate(baseline_metrics, "stability_rate")
    beta1_accuracy = float(sanity_row_beta1["accuracy"])
    beta1_stability = float(sanity_row_beta1["stability_rate"])
    sanity_checks = {
        "beta_one_matches_baseline_accuracy": {
            "got": beta1_accuracy,
            "ref": baseline_accuracy,
            "abs_diff": abs(beta1_accuracy - baseline_accuracy),
            "passed": abs(beta1_accuracy - baseline_accuracy) <= float(config.sanity_tolerance),
        },
        "beta_one_matches_baseline_stability": {
            "got": beta1_stability,
            "ref": baseline_stability,
            "abs_diff": abs(beta1_stability - baseline_stability),
            "passed": abs(beta1_stability - baseline_stability) <= float(config.sanity_tolerance),
        },
        "activation_probe_has_firing": {
            "n_gate_fired_layers_total": int(sanity_row_active["n_gate_fired_layers_total"]),
            "n_gate_events_total": int(sanity_row_active["n_gate_events_total"]),
            "passed": int(sanity_row_active["n_gate_fired_layers_total"]) > 0,
        },
    }
    sanity_checks["all_passed"] = bool(
        sanity_checks["beta_one_matches_baseline_accuracy"]["passed"]
        and sanity_checks["beta_one_matches_baseline_stability"]["passed"]
        and sanity_checks["activation_probe_has_firing"]["passed"]
    )
    write_json(config.out_dir / "sanity_checks.json", sanity_checks)

    run_rows_sorted = sorted(run_rows, key=lambda r: (str(r["config_name"]), float(r["tau_margin"])))
    write_json(config.out_dir / "multi_head_sweep_summary.json", run_rows_sorted)

    best_by_config: list[dict[str, Any]] = []
    for item in method_configs:
        rows = [r for r in run_rows_sorted if str(r["config_name"]) == str(item["name"])]
        if rows:
            best_by_config.append(_find_best_row(rows))

    multi_rows = [r for r in run_rows_sorted if str(r["config_name"]).startswith("multi_")]
    single_rows = [r for r in run_rows_sorted if str(r["config_name"]) == "single_L27H20"]
    best_multi = _find_best_row(multi_rows)
    best_single = next((r for r in single_rows if abs(float(r["tau_margin"]) - 0.0) < 1e-12), _find_best_row(single_rows))

    best_multi_key = (str(best_multi["config_name"]), float(best_multi["tau_margin"]))
    per_layer_rates = _compute_per_layer_fire_rates(per_layer_counts_by_config_tau[best_multi_key])
    write_json(config.out_dir / "gate_stats" / "per_layer_gate_fire_rate.json", per_layer_rates)

    _plot_restoration_improvement_by_config(config.out_dir, best_by_config)
    _plot_stability_by_config(config.out_dir, best_by_config)
    _plot_per_layer_gate_fire_rate(
        config.out_dir,
        per_layer_rates=per_layer_rates,
        title_suffix=f"{best_multi['config_name']} @ tau={best_multi['tau_margin']:g}",
    )
    _plot_restoration_vs_tau_multi_vs_single(
        config.out_dir,
        rows_single=single_rows,
        rows_multi=[r for r in run_rows_sorted if str(r["config_name"]) == str(best_multi["config_name"])],
        multi_name=str(best_multi["config_name"]),
    )

    erha_uncond = _load_optional_json(config.phase2_main_eval_dir / "erha_results_beta_5.0.json")
    ag_best = _load_ag_best_metrics(config.ag_track_a_dir)
    dola_metrics = _load_optional_json(config.phase2_main_eval_dir / "dola_results.json")
    method_metrics: list[tuple[str, dict[str, Any]]] = [
        ("baseline", baseline_metrics),
        ("erha_unconditional_beta5", (erha_uncond or {})),
        ("ag_erha_best", (ag_best or {})),
        ("mg_erha_single_head", best_single),
        ("mg_erha_multi_head_best", best_multi),
        ("dola", (dola_metrics or {})),
    ]
    _plot_comparison_bar_all_methods(config.out_dir, method_metrics=method_metrics)

    delta = float(best_multi["restoration_improvement"]) - float(best_single["restoration_improvement"])
    if delta > 0.01:
        verdict = "MULTI_BETTER"
    elif abs(delta) <= 0.01:
        verdict = "COMPARABLE"
    else:
        verdict = "SINGLE_BETTER"

    verdict_lines = [
        f"MULTI_HEAD_BEST_CONFIG: {best_multi['config_name']}",
        f"MULTI_HEAD_BEST_TAU: {float(best_multi['tau_margin'])}",
        f"MULTI_HEAD_RESTORATION_IMPROVEMENT: {float(best_multi['restoration_improvement'])}",
        f"SINGLE_HEAD_RESTORATION_IMPROVEMENT: {float(best_single['restoration_improvement'])}",
        f"IMPROVEMENT_DELTA: {delta}",
        f"MULTI_HEAD_STABILITY: {float(best_multi['stability_rate'])}",
        f"MULTI_HEAD_HARM_RATE: {float(best_multi['harm_rate'])}",
        f"VERDICT: {verdict}",
    ]
    (config.out_dir / "multi_head_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    final_summary = {
        "target_layers": [int(x) for x in config.target_layers],
        "late_layers": [int(x) for x in config.late_layers],
        "multi_betas": [float(x) for x in config.multi_betas],
        "top_heads_per_layer": {str(k): int(v) for k, v in sorted(top_heads.items())},
        "top_head_mean_ld_faithful": {str(k): float(v) for k, v in sorted(top_values.items())},
        "run_rows": run_rows_sorted,
        "best_by_config": best_by_config,
        "best_multi": best_multi,
        "best_single_reference": best_single,
        "sanity_checks": sanity_checks,
        "per_layer_gate_fire_rate": per_layer_rates,
        "verdict": verdict,
        "outputs": {
            "top_heads_per_layer": str(config.out_dir / "top_heads_per_layer.json"),
            "sweep_summary": str(config.out_dir / "multi_head_sweep_summary.json"),
            "verdict_txt": str(config.out_dir / "multi_head_verdict.txt"),
            "gate_stats_dir": str(config.out_dir / "gate_stats"),
            "plots_dir": str(config.out_dir / "plots"),
        },
    }
    write_json(config.out_dir / "final_summary.json", final_summary)
    return final_summary
