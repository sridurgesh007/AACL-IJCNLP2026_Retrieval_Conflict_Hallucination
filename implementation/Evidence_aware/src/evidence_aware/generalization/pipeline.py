from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from evidence_aware.ag_erha.track_b import build_evidence_set, build_parametric_set, run_track_b
from evidence_aware.ag_erha.utils import load_cache_rows, normalize_example_rows, seed_everything, write_json, write_jsonl
from evidence_aware.mg_erha import MGERHAConfig, run_mg_erha_pipeline
from evidence_aware.phase2.common import (
    Example,
    PreparedExample,
    argmax_non_whitespace_token_id,
    classify_answer_token,
)
from evidence_aware.phase2.evaluation import (
    choose_random_head_control,
    evaluate_mode,
    summarize_mode_metrics,
)
from evidence_aware.phase2.modeling import load_model_and_tokenizer, logit_lens, to_tensors


@dataclass(frozen=True)
class GeneralizationConfig:
    model_id: str
    model_shortname: str
    cache_path: Path
    out_root: Path
    llama_results_dir: Path
    phase1_start_idx: int
    phase1_end_idx: int
    phase2_val_start_idx: int
    phase2_val_end_idx: int
    phase2_eval_start_idx: int
    phase2_eval_end_idx: int
    max_length: int
    max_new_tokens: int
    label_max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    beta: float
    force_prompt_style: str
    baseline_probe_examples: int
    mg_tau_margin_grid: list[float]
    mg_sanity_taus: list[float]
    mg_first_token_only: bool
    mg_bootstrap_first_token_margin: bool
    mg_sanity_tolerance: float


def parse_generalization_config(args: Any) -> GeneralizationConfig:
    tau_grid = [float(x.strip()) for x in str(args.mg_tau_margin_grid).split(",") if x.strip()]
    if not tau_grid:
        raise ValueError("mg_tau_margin_grid cannot be empty.")
    sanity_taus = [float(x.strip()) for x in str(args.mg_sanity_taus).split(",") if x.strip()]
    if len(sanity_taus) < 2:
        raise ValueError("mg_sanity_taus must contain at least two values.")
    force_prompt_style = str(args.force_prompt_style).strip().lower()
    if force_prompt_style not in {"auto", "chat", "raw"}:
        raise ValueError("force_prompt_style must be one of: auto, chat, raw")

    return GeneralizationConfig(
        model_id=str(args.model_id),
        model_shortname=str(args.model_shortname),
        cache_path=Path(args.cache_path),
        out_root=Path(args.out_root),
        llama_results_dir=Path(args.llama_results_dir),
        phase1_start_idx=int(args.phase1_start_idx),
        phase1_end_idx=int(args.phase1_end_idx),
        phase2_val_start_idx=int(args.phase2_val_start_idx),
        phase2_val_end_idx=int(args.phase2_val_end_idx),
        phase2_eval_start_idx=int(args.phase2_eval_start_idx),
        phase2_eval_end_idx=int(args.phase2_eval_end_idx),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        label_max_new_tokens=int(args.label_max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        beta=float(args.beta),
        force_prompt_style=force_prompt_style,
        baseline_probe_examples=int(args.baseline_probe_examples),
        mg_tau_margin_grid=tau_grid,
        mg_sanity_taus=sanity_taus,
        mg_first_token_only=bool(int(args.mg_first_token_only)),
        mg_bootstrap_first_token_margin=bool(int(args.mg_bootstrap_first_token_margin)),
        mg_sanity_tolerance=float(args.mg_sanity_tolerance),
    )


def _build_prompt_raw(evidence_text: str, question: str) -> str:
    return f"Evidence: {evidence_text}\n\nQuestion: {question}\nAnswer: "


def _build_prompt_chat(evidence_text: str, question: str) -> str:
    instruction = f"Evidence: {evidence_text}\n\nQuestion: {question}\nAnswer:"
    return f"<s>[INST] {instruction} [/INST]"


def _strict_single_token_id(tokenizer, answer: str) -> int | None:
    answer = str(answer).strip()
    if not answer:
        return None
    for text in [" " + answer, answer]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def _validate_hook_locations(model) -> dict[str, bool]:
    layer = model.model.layers[0]
    checks = {
        "has_self_attn": hasattr(layer, "self_attn"),
        "has_o_proj": hasattr(layer.self_attn, "o_proj"),
        "has_input_layernorm": hasattr(layer, "input_layernorm"),
        "has_post_attention_layernorm": hasattr(layer, "post_attention_layernorm"),
        "has_mlp": hasattr(layer, "mlp"),
    }
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Mistral hook-location validation failed. Missing: {missing}")
    return checks


def _rows_in_range(rows: list[Example], start_idx: int, end_idx: int) -> list[Example]:
    lo = min(int(start_idx), int(end_idx))
    hi = max(int(start_idx), int(end_idx))
    return [row for row in rows if lo <= int(row.id) <= hi]


def _prepare_examples_with_prompt(
    rows: list[Example],
    tokenizer,
    max_length: int,
    prompt_builder: Callable[[str, str], str],
) -> tuple[list[PreparedExample], dict[str, int]]:
    prepared: list[PreparedExample] = []
    dropped = {
        "not_single_token_answers": 0,
        "same_token_answers": 0,
        "empty_prompt_ids": 0,
    }
    for row in rows:
        ev_id = _strict_single_token_id(tokenizer, row.answer_liquid)
        pa_id = _strict_single_token_id(tokenizer, row.answer_solid)
        if ev_id is None or pa_id is None:
            dropped["not_single_token_answers"] += 1
            continue
        if int(ev_id) == int(pa_id):
            dropped["same_token_answers"] += 1
            continue
        prompt = prompt_builder(row.evidence, row.question)
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=int(max_length))
        input_ids = encoded.get("input_ids", [])
        if not input_ids:
            dropped["empty_prompt_ids"] += 1
            continue
        prepared.append(
            PreparedExample(
                id=int(row.id),
                question=str(row.question),
                evidence=str(row.evidence),
                answer_liquid=str(row.answer_liquid),
                answer_solid=str(row.answer_solid),
                prompt=prompt,
                input_ids=[int(x) for x in input_ids],
                evidence_token_id=int(ev_id),
                parametric_token_id=int(pa_id),
            )
        )
    return prepared, dropped


def _classify_with_span_fallback(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    first_token_id: int | None,
    evidence_token_id: int,
    parametric_token_id: int,
    max_new_tokens: int,
) -> str:
    if first_token_id is not None:
        label = classify_answer_token(first_token_id, evidence_token_id, parametric_token_id)
        if label in {"faithful", "hallucinated"}:
            return label

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    continuation = out[0, input_ids.shape[1] :].tolist()
    ev_pos = next((i for i, tid in enumerate(continuation) if int(tid) == int(evidence_token_id)), None)
    pa_pos = next((i for i, tid in enumerate(continuation) if int(tid) == int(parametric_token_id)), None)
    if ev_pos is None and pa_pos is None:
        return "other"
    if ev_pos is not None and (pa_pos is None or ev_pos < pa_pos):
        return "faithful"
    if pa_pos is not None and (ev_pos is None or pa_pos < ev_pos):
        return "hallucinated"
    return "other"


def _baseline_prompt_probe(
    model,
    tokenizer,
    device: torch.device,
    rows: list[Example],
    prompt_builder: Callable[[str, str], str],
    max_length: int,
    sample_size: int,
    label_max_new_tokens: int,
) -> dict[str, Any]:
    prepared, dropped = _prepare_examples_with_prompt(
        rows=rows,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_builder=prompt_builder,
    )
    subset = prepared[: max(1, int(sample_size))]
    if not subset:
        return {
            "n_total": 0,
            "n_faithful": 0,
            "n_hallucinated": 0,
            "n_other": 0,
            "accuracy": 0.0,
            "dropped": dropped,
        }

    counts = {"n_total": 0, "n_faithful": 0, "n_hallucinated": 0, "n_other": 0}
    for ex in subset:
        input_ids, attention_mask = to_tensors(ex, device=device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits[0, -1, :].float()
        first_token = argmax_non_whitespace_token_id(logits, tokenizer)
        label = _classify_with_span_fallback(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            first_token_id=first_token,
            evidence_token_id=ex.evidence_token_id,
            parametric_token_id=ex.parametric_token_id,
            max_new_tokens=label_max_new_tokens,
        )
        counts["n_total"] += 1
        counts[f"n_{label}"] += 1

    accuracy = float(counts["n_faithful"] / max(counts["n_total"], 1))
    return {
        **counts,
        "accuracy": accuracy,
        "dropped": dropped,
    }


class LDHooks:
    def __init__(self, model) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.clear()

    def clear(self) -> None:
        self.residual_in: dict[int, torch.Tensor] = {}
        self.attn_output: dict[int, torch.Tensor] = {}
        self.post_mlp: dict[int, torch.Tensor] = {}
        self.attention_pre_o: dict[int, torch.Tensor] = {}

    def register(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._mk_layer_pre_hook(layer_idx)))
            self.handles.append(layer.self_attn.register_forward_hook(self._mk_attn_hook(layer_idx)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._mk_o_proj_pre_hook(layer_idx)))
            self.handles.append(layer.register_forward_hook(self._mk_layer_post_hook(layer_idx)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _mk_layer_pre_hook(self, layer_idx: int):
        def hook(module, inputs):
            hidden = inputs[0]
            self.residual_in[layer_idx] = hidden[:, -1, :].detach().float().squeeze(0)

        return hook

    def _mk_attn_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            attn_out = outputs[0] if isinstance(outputs, tuple) else outputs
            self.attn_output[layer_idx] = attn_out[:, -1, :].detach().float().squeeze(0)

        return hook

    def _mk_o_proj_pre_hook(self, layer_idx: int):
        def hook(module, inputs):
            if not inputs:
                return
            pre_o = inputs[0]
            if pre_o is None or pre_o.ndim != 3:
                return
            last = pre_o[:, -1, :].detach().float().squeeze(0)
            num_heads = int(getattr(self.model.config, "num_attention_heads"))
            if num_heads <= 0 or last.numel() % num_heads != 0:
                return
            head_dim = last.numel() // num_heads
            self.attention_pre_o[layer_idx] = last.view(num_heads, head_dim)

        return hook

    def _mk_layer_post_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            self.post_mlp[layer_idx] = hidden[:, -1, :].detach().float().squeeze(0)

        return hook

    def has_full_capture(self, num_layers: int) -> bool:
        for layer_idx in range(num_layers):
            if layer_idx not in self.residual_in:
                return False
            if layer_idx not in self.attn_output:
                return False
            if layer_idx not in self.post_mlp:
                return False
            if layer_idx not in self.attention_pre_o:
                return False
        return True


def _save_heatmap(matrix: np.ndarray, out_path: Path, title: str, center_zero: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    if center_zero:
        vmax = float(np.max(np.abs(matrix))) if matrix.size > 0 else 1e-8
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1e-8
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    else:
        im = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _topk_heads(matrix: np.ndarray, k: int) -> list[dict[str, Any]]:
    ranked: list[tuple[float, int, int]] = []
    for layer_idx in range(matrix.shape[0]):
        for head_idx in range(matrix.shape[1]):
            ranked.append((float(matrix[layer_idx, head_idx]), int(layer_idx), int(head_idx)))
    ranked.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for value, layer_idx, head_idx in ranked[: max(0, int(k))]:
        out.append({"layer": layer_idx, "head": head_idx, "value": value})
    return out


def _group_heads_by_layer(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer_idx, head_idx in pairs:
        out.setdefault(int(layer_idx), [])
        out[layer_idx].append(int(head_idx))
    for layer_idx in list(out.keys()):
        out[layer_idx] = sorted(set(out[layer_idx]))
    return out


def _compute_ld_matrix_per_example(
    model,
    evidence_token_id: int,
    parametric_token_id: int,
    attention_pre_o: dict[int, torch.Tensor],
) -> np.ndarray:
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // num_heads

    lm_weight = model.lm_head.weight.detach().float()
    d_ep = (lm_weight[int(evidence_token_id)] - lm_weight[int(parametric_token_id)]).to(dtype=torch.float32)

    ld_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
    for layer_idx in range(num_layers):
        pre_heads = attention_pre_o[layer_idx].to(dtype=torch.float32, device=lm_weight.device)
        w_o = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().float()
        w_o_view = w_o.view(hidden_size, num_heads, head_dim)
        delta_by_head = torch.einsum("dnh,nh->nd", w_o_view, pre_heads)
        ld_vec = torch.einsum("nd,d->n", delta_by_head, d_ep).float()
        ld_matrix[layer_idx, :] = ld_vec.detach().cpu().numpy().astype(np.float64)
    return ld_matrix


def _compute_phase1_outputs(
    model,
    tokenizer,
    device: torch.device,
    prepared_examples: list[PreparedExample],
    out_dir: Path,
    split_name: str,
    label_max_new_tokens: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hooks = LDHooks(model)
    hooks.register()

    num_layers = int(model.config.num_hidden_layers)
    per_example_rows: list[dict[str, Any]] = []
    dropped = {
        "missing_hook_capture": 0,
        "label_other": 0,
    }

    try:
        for ex in prepared_examples:
            hooks.clear()
            input_ids, attention_mask = to_tensors(ex, device=device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

            if not hooks.has_full_capture(num_layers=num_layers):
                dropped["missing_hook_capture"] += 1
                continue

            final_logits = outputs.logits[0, -1, :].float()
            first_token = argmax_non_whitespace_token_id(final_logits, tokenizer)
            label = _classify_with_span_fallback(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_token_id=first_token,
                evidence_token_id=ex.evidence_token_id,
                parametric_token_id=ex.parametric_token_id,
                max_new_tokens=label_max_new_tokens,
            )
            if label not in {"faithful", "hallucinated"}:
                dropped["label_other"] += 1
                continue

            ld_matrix = _compute_ld_matrix_per_example(
                model=model,
                evidence_token_id=ex.evidence_token_id,
                parametric_token_id=ex.parametric_token_id,
                attention_pre_o=hooks.attention_pre_o,
            )

            baseline_top = torch.topk(final_logits, k=min(20, int(final_logits.numel()))).indices.tolist()
            evidence_set = set(build_evidence_set(tokenizer, ex.answer_liquid))
            if not evidence_set:
                evidence_set = {int(ex.evidence_token_id)}
            parametric_set = set(build_parametric_set([int(t) for t in baseline_top], evidence_set, k=5))
            parametric_set = {int(t) for t in parametric_set if int(t) not in evidence_set}
            if not parametric_set:
                parametric_set = {int(ex.parametric_token_id)}
            margin_token_ids = sorted(set(int(t) for t in evidence_set | parametric_set))

            log_probs_pre: list[list[float]] = []
            log_probs_attn: list[list[float]] = []
            log_probs_mlp: list[list[float]] = []
            p_evidence_post_attn: list[float] = []
            m_pre: list[float] = []
            m_post_attn: list[float] = []
            m_post_mlp: list[float] = []
            delta_attn: list[float] = []
            delta_mlp: list[float] = []

            for layer_idx in range(num_layers):
                residual_pre = hooks.residual_in[layer_idx]
                post_attn = hooks.residual_in[layer_idx] + hooks.attn_output[layer_idx]
                post_mlp = hooks.post_mlp[layer_idx]

                logits_pre = logit_lens(model, residual_pre)
                logits_attn = logit_lens(model, post_attn)
                logits_mlp = logit_lens(model, post_mlp)

                lp_pre = F.log_softmax(logits_pre, dim=-1)
                lp_attn = F.log_softmax(logits_attn, dim=-1)
                lp_mlp = F.log_softmax(logits_mlp, dim=-1)
                log_probs_pre.append([float(lp_pre[i].item()) for i in margin_token_ids])
                log_probs_attn.append([float(lp_attn[i].item()) for i in margin_token_ids])
                log_probs_mlp.append([float(lp_mlp[i].item()) for i in margin_token_ids])
                p_evidence_post_attn.append(float(torch.softmax(logits_attn, dim=-1)[ex.evidence_token_id].item()))

                ev_idx = torch.as_tensor(sorted(int(x) for x in evidence_set), dtype=torch.long, device=lp_pre.device)
                pa_idx = torch.as_tensor(sorted(int(x) for x in parametric_set), dtype=torch.long, device=lp_pre.device)
                margin_pre = float((torch.logsumexp(lp_pre.index_select(0, ev_idx), dim=0) - torch.logsumexp(lp_pre.index_select(0, pa_idx), dim=0)).item())
                margin_attn = float((torch.logsumexp(lp_attn.index_select(0, ev_idx), dim=0) - torch.logsumexp(lp_attn.index_select(0, pa_idx), dim=0)).item())
                margin_mlp = float((torch.logsumexp(lp_mlp.index_select(0, ev_idx), dim=0) - torch.logsumexp(lp_mlp.index_select(0, pa_idx), dim=0)).item())
                m_pre.append(margin_pre)
                m_post_attn.append(margin_attn)
                m_post_mlp.append(margin_mlp)
                delta_attn.append(margin_attn - margin_pre)
                delta_mlp.append(margin_mlp - margin_attn)

            per_example_rows.append(
                {
                    "example_id": int(ex.id),
                    "label": label,
                    "is_hallucinated": bool(label == "hallucinated"),
                    "evidence_token_id": int(ex.evidence_token_id),
                    "parametric_token_id": int(ex.parametric_token_id),
                    "baseline_top_tokens": [int(x) for x in baseline_top],
                    "evidence_set_token_ids": sorted(int(x) for x in evidence_set),
                    "parametric_set_token_ids": sorted(int(x) for x in parametric_set),
                    "margin_token_ids": [int(x) for x in margin_token_ids],
                    "log_probs_pre_attn_selected": log_probs_pre,
                    "log_probs_post_attn_selected": log_probs_attn,
                    "log_probs_post_mlp_selected": log_probs_mlp,
                    "p_evidence_post_attn": p_evidence_post_attn,
                    "M_pre_attn": m_pre,
                    "M_post_attn": m_post_attn,
                    "M_post_mlp": m_post_mlp,
                    "delta_attn": delta_attn,
                    "delta_mlp": delta_mlp,
                    "head_ld_matrix": ld_matrix.tolist(),
                }
            )
    finally:
        hooks.close()

    faithful = [row for row in per_example_rows if row["label"] == "faithful"]
    hallucinated = [row for row in per_example_rows if row["label"] == "hallucinated"]
    if not faithful or not hallucinated:
        raise RuntimeError(
            f"{split_name} produced empty class after filtering. "
            f"faithful={len(faithful)}, hallucinated={len(hallucinated)}, dropped={dropped}"
        )

    mats_f = np.array([r["head_ld_matrix"] for r in faithful], dtype=np.float64)
    mats_h = np.array([r["head_ld_matrix"] for r in hallucinated], dtype=np.float64)
    mean_faithful = mats_f.mean(axis=0)
    mean_hallucinated = mats_h.mean(axis=0)
    delta = mean_faithful - mean_hallucinated

    top5_faithful = _topk_heads(mean_faithful, k=5)
    top5_hallucinated = _topk_heads(mean_hallucinated, k=5)
    top3_delta = _topk_heads(delta, k=3)
    top_head = top3_delta[0]
    top_layer = int(top_head["layer"])
    top_head_idx = int(top_head["head"])
    faithful_val = float(mean_faithful[top_layer, top_head_idx])
    hall_val = float(mean_hallucinated[top_layer, top_head_idx])
    ratio = float(abs(faithful_val) / max(abs(hall_val), 1e-12))
    same_head = bool(top_layer == 27 and top_head_idx == 20)
    analogous = bool(ratio > 3.0)

    _save_heatmap(mean_faithful, out_dir / "head_ld_heatmap_faithful.png", f"{split_name}: mean LD faithful")
    _save_heatmap(mean_hallucinated, out_dir / "head_ld_heatmap_hallucinated.png", f"{split_name}: mean LD hallucinated")
    _save_heatmap(delta, out_dir / "head_ld_heatmap_delta.png", f"{split_name}: delta LD (faithful - hallucinated)")

    write_jsonl(out_dir / "per_example_ld.jsonl", per_example_rows)
    write_json(out_dir / "top5_heads_faithful.json", top5_faithful)
    write_json(out_dir / "top5_heads_hallucinated.json", top5_hallucinated)

    verdict_lines = [
        f"MISTRAL_TOP_HEAD: layer={top_layer}, head={top_head_idx}",
        f"MISTRAL_TOP_HEAD_FAITHFUL_LD: {faithful_val}",
        f"MISTRAL_TOP_HEAD_HALLUCINATED_LD: {hall_val}",
        f"MISTRAL_LD_RATIO: {ratio}",
        f"TOP3_HEADS: {[(int(x['layer']), int(x['head'])) for x in top3_delta]}",
        "LLAMA_TOP_HEAD: layer=27, head=20",
        f"SAME_HEAD: {'yes' if same_head else 'no'}",
        f"ANALOGOUS_BEHAVIOR: {'yes' if analogous else 'no'}",
    ]
    (out_dir / "phase1_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    summary = {
        "split_name": split_name,
        "n_examples": int(len(per_example_rows)),
        "n_faithful": int(len(faithful)),
        "n_hallucinated": int(len(hallucinated)),
        "dropped": dropped,
        "top_head": {"layer": top_layer, "head": top_head_idx},
        "top3_heads_delta": [{"layer": int(x["layer"]), "head": int(x["head"]), "value": float(x["value"])} for x in top3_delta],
        "top5_heads_faithful": top5_faithful,
        "top5_heads_hallucinated": top5_hallucinated,
        "top_head_faithful_ld": faithful_val,
        "top_head_hallucinated_ld": hall_val,
        "ld_ratio": ratio,
        "same_head_as_llama_l27h20": same_head,
        "analogous_behavior": analogous,
    }
    write_json(out_dir / "phase1_summary.json", summary)
    return summary


def _phase1_validation_summary(
    phase1_summary: dict[str, Any],
    validation_summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    phase1_top3 = [(int(x["layer"]), int(x["head"])) for x in phase1_summary["top3_heads_delta"]]
    val_top3 = [(int(x["layer"]), int(x["head"])) for x in validation_summary["top3_heads_delta"]]
    overlap = sorted(set(phase1_top3) & set(val_top3))
    stable = bool(len(overlap) >= 2)

    payload = {
        "phase1_top3": phase1_top3,
        "validation_top3": val_top3,
        "top3_overlap": overlap,
        "top3_overlap_count": len(overlap),
        "stable": stable,
    }
    write_json(out_dir / "validation_summary.json", payload)
    lines = [
        f"VALIDATION_STABLE: {'yes' if stable else 'no'}",
        f"TOP3_OVERLAP_COUNT: {len(overlap)}",
        f"TOP3_OVERLAP: {overlap}",
    ]
    (out_dir / "validation_verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _write_phase2_artifacts(
    phase2_dir: Path,
    baseline_run,
    baseline_metrics: dict[str, Any],
    top_head_metrics: dict[str, Any],
    top3_metrics: dict[str, Any],
    random_metrics: dict[str, Any],
    dola_metrics: dict[str, Any],
) -> None:
    main_eval_dir = phase2_dir / "main_eval"
    main_eval_dir.mkdir(parents=True, exist_ok=True)

    write_json(phase2_dir / "baseline_results.json", baseline_metrics)
    write_json(phase2_dir / "erha_mistral_top_head.json", top_head_metrics)
    write_json(phase2_dir / "erha_top3_combined.json", top3_metrics)
    write_json(phase2_dir / "random_head_control.json", random_metrics)
    write_json(phase2_dir / "dola_results.json", dola_metrics)
    write_jsonl(phase2_dir / "baseline_examples.jsonl", baseline_run.per_example_rows)

    write_json(main_eval_dir / "baseline_results.json", baseline_metrics)
    write_json(main_eval_dir / "erha_results_beta_5.0.json", top_head_metrics)
    write_json(main_eval_dir / "dola_results.json", dola_metrics)
    write_jsonl(main_eval_dir / "baseline_examples.jsonl", baseline_run.per_example_rows)


def _plot_phase2_ablation(phase2_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(r["mode"]) for r in rows]
    restoration = [float(r["restoration_rate"]) for r in rows]
    stability = [float(r["stability_rate"]) for r in rows]
    improvement = [float(r["restoration_improvement"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width, restoration, width, label="restoration_rate")
    ax.bar(x, stability, width, label="stability_rate")
    ax.bar(x + width, improvement, width, label="restoration_improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Mistral Phase-2 Ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(phase2_dir / "ablation_bar_chart.png", dpi=220)
    plt.close(fig)


def _run_phase2_ablation(
    model,
    tokenizer,
    device: torch.device,
    eval_examples: list[PreparedExample],
    phase2_dir: Path,
    top_head: tuple[int, int],
    top3_heads: list[tuple[int, int]],
    max_new_tokens: int,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    if not eval_examples:
        raise RuntimeError("No eval examples available for Phase-2 ablation.")

    top_layer, top_head_idx = int(top_head[0]), int(top_head[1])
    top_head_dict = {top_layer: [top_head_idx]}
    top3_dict = _group_heads_by_layer(top3_heads)

    baseline_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples,
        max_new_tokens=max_new_tokens,
        mode_name="baseline",
        decode_mode="baseline",
        target_heads=None,
        beta=None,
        mature_layer=31,
        premature_layers=[14, 16, 18, 20],
    )
    baseline_metrics = summarize_mode_metrics(run=baseline_run, baseline_run=baseline_run, mode_name="baseline")

    top_head_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples,
        max_new_tokens=max_new_tokens,
        mode_name="erha_mistral_top_head",
        decode_mode="erha",
        target_heads=top_head_dict,
        beta=float(beta),
        mature_layer=31,
        premature_layers=[14, 16, 18, 20],
    )
    top_head_metrics = summarize_mode_metrics(run=top_head_run, baseline_run=baseline_run, mode_name="erha_mistral_top_head")

    top3_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples,
        max_new_tokens=max_new_tokens,
        mode_name="erha_top3_combined",
        decode_mode="erha",
        target_heads=top3_dict,
        beta=float(beta),
        mature_layer=31,
        premature_layers=[14, 16, 18, 20],
    )
    top3_metrics = summarize_mode_metrics(run=top3_run, baseline_run=baseline_run, mode_name="erha_top3_combined")

    random_heads = choose_random_head_control(
        model=model,
        excluded_heads=top3_dict,
        count=3,
        seed=int(seed),
    )
    random_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples,
        max_new_tokens=max_new_tokens,
        mode_name="random_head_control",
        decode_mode="erha",
        target_heads=random_heads,
        beta=float(beta),
        mature_layer=31,
        premature_layers=[14, 16, 18, 20],
    )
    random_metrics = summarize_mode_metrics(run=random_run, baseline_run=baseline_run, mode_name="random_head_control")
    random_metrics["random_heads"] = random_heads

    dola_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples,
        max_new_tokens=max_new_tokens,
        mode_name="dola",
        decode_mode="dola",
        target_heads=None,
        beta=None,
        mature_layer=31,
        premature_layers=[14, 16, 18, 20],
    )
    dola_metrics = summarize_mode_metrics(run=dola_run, baseline_run=baseline_run, mode_name="dola")

    _write_phase2_artifacts(
        phase2_dir=phase2_dir,
        baseline_run=baseline_run,
        baseline_metrics=baseline_metrics,
        top_head_metrics=top_head_metrics,
        top3_metrics=top3_metrics,
        random_metrics=random_metrics,
        dola_metrics=dola_metrics,
    )
    _plot_phase2_ablation(
        phase2_dir=phase2_dir,
        rows=[baseline_metrics, top_head_metrics, top3_metrics, random_metrics, dola_metrics],
    )

    summary = {
        "top_head": {"layer": top_layer, "head": top_head_idx},
        "top3_heads": [{"layer": int(l), "head": int(h)} for l, h in top3_heads],
        "random_heads": random_heads,
        "baseline": baseline_metrics,
        "erha_mistral_top_head": top_head_metrics,
        "erha_top3_combined": top3_metrics,
        "random_head_control": random_metrics,
        "dola": dola_metrics,
        "num_eval_examples": len(eval_examples),
    }
    write_json(phase2_dir / "phase2_summary.json", summary)
    return summary


def _find_divergence_start_layer(track_b_margin_path: Path) -> int:
    if not track_b_margin_path.exists():
        return -1
    rows = []
    with track_b_margin_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    faithful = [r for r in rows if not bool(r.get("is_hallucinated", False))]
    hall = [r for r in rows if bool(r.get("is_hallucinated", False))]
    if not faithful or not hall:
        return -1
    f_mean = np.array([r["M_post_mlp"] for r in faithful], dtype=np.float64).mean(axis=0)
    h_mean = np.array([r["M_post_mlp"] for r in hall], dtype=np.float64).mean(axis=0)
    diff = f_mean - h_mean
    for layer_idx in range(max(0, len(diff) - 3)):
        window = diff[layer_idx : layer_idx + 4]
        if np.sum(window > 0.0) >= 3:
            return int(layer_idx)
    return int(np.argmax(diff))


def _load_llama_reference(llama_results_dir: Path) -> dict[str, Any]:
    ref = {
        "top_head": {"layer": 27, "head": 20},
        "ld_ratio": 6.1,
        "auc_margin_post_attn": 0.932,
        "cohens_d_margin": 2.041,
        "mlp_erodes_margin": False,
        "divergence_start_layer": 20,
        "mg_stability": 0.987,
        "mg_restoration_improvement": 0.057,
        "mg_harm_rate": 0.0,
        "source": "generalization.md defaults",
    }
    mg_final = llama_results_dir / "mg_erha_first_token" / "final_summary.json"
    if mg_final.exists():
        payload = json.loads(mg_final.read_text(encoding="utf-8"))
        best = payload.get("best_row", {})
        ref["mg_stability"] = float(best.get("stability_rate", ref["mg_stability"]))
        ref["mg_restoration_improvement"] = float(best.get("restoration_improvement", ref["mg_restoration_improvement"]))
        ref["mg_harm_rate"] = float(best.get("harm_rate", ref["mg_harm_rate"]))
        ref["source"] = "llama artifacts + generalization.md defaults"
    return ref


def _plot_cross_model_table(out_path: Path, table_rows: list[list[str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title("Cross-Model Generalization Comparison", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _generalization_verdict(
    mistral: dict[str, Any],
    llama: dict[str, Any],
) -> str:
    top_layer = int(mistral["top_head"]["layer"])
    top_head = int(mistral["top_head"]["head"])
    same_head = bool(top_layer == int(llama["top_head"]["layer"]) and top_head == int(llama["top_head"]["head"]))
    ld_ratio = float(mistral["ld_ratio"])
    auc_margin = float(mistral["auc_margin_post_attn"])
    mg_stability = float(mistral["mg_stability"])
    mg_resto = float(mistral["mg_restoration_improvement"])

    if (not same_head) and (auc_margin > 0.85) and (mg_stability > 0.92) and (mg_resto > 0.02):
        return "STRONG_GENERALIZATION"
    if (not same_head) and (auc_margin > 0.75) and (mg_stability > 0.88):
        return "MODERATE_GENERALIZATION"
    if same_head:
        return "SAME_HEAD_FOUND"
    if ld_ratio < 2.0:
        return "ARCHITECTURE_DEPENDENT"
    return "ARCHITECTURE_DEPENDENT"


def run_generalization_pipeline(config: GeneralizationConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    rows = normalize_example_rows(load_cache_rows(config.cache_path))
    if not rows:
        raise RuntimeError(f"No valid dataset rows loaded from {config.cache_path}")

    model, tokenizer, device = load_model_and_tokenizer(
        model_id=config.model_id,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )
    hook_checks = _validate_hook_locations(model)

    phase1_rows = _rows_in_range(rows, config.phase1_start_idx, config.phase1_end_idx)
    val_rows = _rows_in_range(rows, config.phase2_val_start_idx, config.phase2_val_end_idx)
    eval_rows = _rows_in_range(rows, config.phase2_eval_start_idx, config.phase2_eval_end_idx)
    if not phase1_rows or not val_rows or not eval_rows:
        raise RuntimeError(
            "Empty split detected. "
            f"phase1={len(phase1_rows)}, val={len(val_rows)}, eval={len(eval_rows)}"
        )

    phase1_dir = config.out_root / config.model_shortname / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    if config.force_prompt_style == "auto":
        probe_chat = _baseline_prompt_probe(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=eval_rows,
            prompt_builder=_build_prompt_chat,
            max_length=config.max_length,
            sample_size=config.baseline_probe_examples,
            label_max_new_tokens=config.label_max_new_tokens,
        )
        probe_raw = _baseline_prompt_probe(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=eval_rows,
            prompt_builder=_build_prompt_raw,
            max_length=config.max_length,
            sample_size=config.baseline_probe_examples,
            label_max_new_tokens=config.label_max_new_tokens,
        )
        prompt_style = "chat" if probe_chat["accuracy"] >= probe_raw["accuracy"] else "raw"
        prompt_probe = {"chat": probe_chat, "raw": probe_raw, "selected": prompt_style}
    else:
        prompt_style = config.force_prompt_style
        prompt_probe = {"selected": prompt_style}
    prompt_builder = _build_prompt_chat if prompt_style == "chat" else _build_prompt_raw
    write_json(phase1_dir / "prompt_style_probe.json", prompt_probe)

    phase1_prepared, phase1_dropped = _prepare_examples_with_prompt(
        rows=phase1_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
        prompt_builder=prompt_builder,
    )
    val_prepared, val_dropped = _prepare_examples_with_prompt(
        rows=val_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
        prompt_builder=prompt_builder,
    )
    eval_prepared, eval_dropped = _prepare_examples_with_prompt(
        rows=eval_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
        prompt_builder=prompt_builder,
    )
    write_json(
        phase1_dir / "single_token_filter_stats.json",
        {
            "phase1_prepared": len(phase1_prepared),
            "phase1_total": len(phase1_rows),
            "phase1_dropped": phase1_dropped,
            "val_prepared": len(val_prepared),
            "val_total": len(val_rows),
            "val_dropped": val_dropped,
            "eval_prepared": len(eval_prepared),
            "eval_total": len(eval_rows),
            "eval_dropped": eval_dropped,
        },
    )
    if not phase1_prepared or not val_prepared or not eval_prepared:
        raise RuntimeError(
            "Single-token filtering produced empty split. "
            f"phase1={len(phase1_prepared)}, val={len(val_prepared)}, eval={len(eval_prepared)}"
        )

    mistral_root = config.out_root / config.model_shortname
    phase2_dir = mistral_root / "phase2"
    track_b_dir = mistral_root / "track_b"
    mg_dir = mistral_root / "mg_erha"
    generalization_dir = config.out_root / "generalization"
    for path in [phase2_dir, track_b_dir, mg_dir, generalization_dir]:
        path.mkdir(parents=True, exist_ok=True)

    phase1_summary = _compute_phase1_outputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prepared_examples=phase1_prepared,
        out_dir=phase1_dir,
        split_name="mistral_phase1",
        label_max_new_tokens=config.label_max_new_tokens,
    )
    val_phase1_summary = _compute_phase1_outputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prepared_examples=val_prepared,
        out_dir=phase1_dir / "validation",
        split_name="mistral_phase1_validation",
        label_max_new_tokens=config.label_max_new_tokens,
    )
    validation_summary = _phase1_validation_summary(
        phase1_summary=phase1_summary,
        validation_summary=val_phase1_summary,
        out_dir=phase1_dir,
    )

    top3_heads = [(int(x["layer"]), int(x["head"])) for x in phase1_summary["top3_heads_delta"]]
    top_head = (int(phase1_summary["top_head"]["layer"]), int(phase1_summary["top_head"]["head"]))
    phase2_summary = _run_phase2_ablation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_prepared,
        phase2_dir=phase2_dir,
        top_head=top_head,
        top3_heads=top3_heads,
        max_new_tokens=config.max_new_tokens,
        beta=config.beta,
        seed=config.seed,
    )

    examples_by_id = {int(row.id): row for row in rows}
    track_b_summary = run_track_b(
        tokenizer=tokenizer,
        examples_by_id=examples_by_id,
        traces_path=phase1_dir / "per_example_ld.jsonl",
        out_dir=track_b_dir,
        focus_layer=int(top_head[0]),
    )

    mg_summary = run_mg_erha_pipeline(
        MGERHAConfig(
            model_id=config.model_id,
            cache_path=config.cache_path,
            phase2_main_eval_dir=phase2_dir / "main_eval",
            track_b_margin_path=track_b_dir / "margin_trajectories.jsonl",
            ag_track_a_dir=config.out_root / "ag_erha",
            out_dir=mg_dir,
            tau_margin_grid=config.mg_tau_margin_grid,
            sanity_taus=config.mg_sanity_taus,
            beta=config.beta,
            max_length=config.max_length,
            max_new_tokens=config.max_new_tokens,
            device=config.device,
            torch_dtype=config.torch_dtype,
            seed=config.seed,
            sanity_tolerance=config.mg_sanity_tolerance,
            init_prev_margin=0.0,
            precompute_missing_sets=True,
            strict_track_b_sets=False,
            first_token_only=config.mg_first_token_only,
            bootstrap_first_token_margin=config.mg_bootstrap_first_token_margin,
            target_layer_idx=int(top_head[0]),
            target_head_idx=int(top_head[1]),
        )
    )

    divergence_start = _find_divergence_start_layer(track_b_dir / "margin_trajectories.jsonl")
    mistral_summary = {
        "top_head": {"layer": int(top_head[0]), "head": int(top_head[1])},
        "top3_heads": [{"layer": int(l), "head": int(h)} for l, h in top3_heads],
        "ld_ratio": float(phase1_summary["ld_ratio"]),
        "auc_margin_post_attn": float(track_b_summary.get("auc_margin_post_attn", float("nan"))),
        "auc_margin_post_mlp": float(track_b_summary.get("auc_margin_post_mlp", float("nan"))),
        "cohens_d_margin": float(track_b_summary.get(f"cohens_d_margin_L{int(top_head[0])}", float("nan"))),
        "mlp_erodes_margin": bool(track_b_summary.get("mlp_erodes_margin_in_hallucinated", False)),
        "divergence_start_layer": int(divergence_start),
        "mg_stability": float(mg_summary["best_row"]["stability_rate"]),
        "mg_restoration_improvement": float(mg_summary["best_row"]["restoration_improvement"]),
        "mg_harm_rate": float(mg_summary["best_row"]["harm_rate"]),
    }
    llama_summary = _load_llama_reference(config.llama_results_dir)
    verdict = _generalization_verdict(mistral_summary, llama_summary)

    comparison_payload = {"llama": llama_summary, "mistral": mistral_summary, "verdict": verdict}
    write_json(generalization_dir / "cross_model_comparison.json", comparison_payload)

    table_rows = [
        ["Metric", "Llama 3.1 8B", "Mistral 7B"],
        ["Evidence-routing head", f"L{llama_summary['top_head']['layer']}H{llama_summary['top_head']['head']}", f"L{top_head[0]}H{top_head[1]}"],
        ["Head LD ratio (f/h)", str(llama_summary["ld_ratio"]), f"{mistral_summary['ld_ratio']:.4f}"],
        ["Margin AUC at top layer", f"{llama_summary['auc_margin_post_attn']:.4f}", f"{mistral_summary['auc_margin_post_attn']:.4f}"],
        ["Cohen's d", f"{llama_summary['cohens_d_margin']:.4f}", f"{mistral_summary['cohens_d_margin']:.4f}"],
        ["MLP erodes margin", "Yes" if llama_summary["mlp_erodes_margin"] else "No", "Yes" if mistral_summary["mlp_erodes_margin"] else "No"],
        ["Divergence start layer", str(llama_summary["divergence_start_layer"]), str(mistral_summary["divergence_start_layer"])],
        ["MG-ERHA stability", f"{llama_summary['mg_stability']:.4f}", f"{mistral_summary['mg_stability']:.4f}"],
        ["MG-ERHA restoration impr.", f"{llama_summary['mg_restoration_improvement']:.4f}", f"{mistral_summary['mg_restoration_improvement']:.4f}"],
        ["MG-ERHA harm rate", f"{llama_summary['mg_harm_rate']:.4f}", f"{mistral_summary['mg_harm_rate']:.4f}"],
        [
            "Same head as Llama",
            "-",
            "Yes" if (top_head[0] == int(llama_summary["top_head"]["layer"]) and top_head[1] == int(llama_summary["top_head"]["head"])) else "No",
        ],
    ]
    _plot_cross_model_table(generalization_dir / "cross_model_comparison_table.png", table_rows)
    (generalization_dir / "generalization_verdict.txt").write_text(f"{verdict}\n", encoding="utf-8")

    final_summary = {
        "model_id": config.model_id,
        "model_shortname": config.model_shortname,
        "hook_checks": hook_checks,
        "prompt_probe": prompt_probe,
        "phase1_summary": phase1_summary,
        "phase1_validation_summary": validation_summary,
        "phase2_summary": phase2_summary,
        "track_b_summary": track_b_summary,
        "mg_summary": mg_summary,
        "cross_model": comparison_payload,
        "outputs": {
            "mistral_root": str(mistral_root),
            "phase1": str(phase1_dir),
            "phase2": str(phase2_dir),
            "track_b": str(track_b_dir),
            "mg_erha": str(mg_dir),
            "generalization": str(generalization_dir),
        },
    }
    write_json(mistral_root / "generalization_final_summary.json", final_summary)
    return final_summary
