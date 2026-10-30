from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TraceBundle:
    residual_in: dict[int, torch.Tensor]
    attn_output: dict[int, torch.Tensor]
    post_attn: dict[int, torch.Tensor]
    mlp_output: dict[int, torch.Tensor]
    post_mlp: dict[int, torch.Tensor]
    attention_last: dict[int, torch.Tensor]
    attention_pre_o: dict[int, torch.Tensor]


class LayerDecompositionHooks:
    def __init__(self, model) -> None:
        self.model = model
        self.handles = []
        self.clear()

    def clear(self) -> None:
        self.residual_in: dict[int, torch.Tensor] = {}
        self.attn_output: dict[int, torch.Tensor] = {}
        self.post_attn: dict[int, torch.Tensor] = {}
        self.mlp_output: dict[int, torch.Tensor] = {}
        self.post_mlp: dict[int, torch.Tensor] = {}
        self.attention_last: dict[int, torch.Tensor] = {}
        self.attention_pre_o: dict[int, torch.Tensor] = {}

    def register(self) -> None:
        layers = self.model.model.layers
        for layer_idx, layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(self._mk_layer_pre_hook(layer_idx)))
            self.handles.append(layer.self_attn.register_forward_hook(self._mk_attn_hook(layer_idx)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._mk_o_proj_pre_hook(layer_idx)))
            self.handles.append(layer.mlp.register_forward_hook(self._mk_mlp_hook(layer_idx)))
            self.handles.append(layer.register_forward_hook(self._mk_layer_post_hook(layer_idx)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def bundle(self, num_layers: int) -> TraceBundle:
        missing = []
        for l in range(num_layers):
            if l not in self.residual_in:
                missing.append(f"residual_in[{l}]")
            if l not in self.attn_output:
                missing.append(f"attn_output[{l}]")
            if l not in self.post_attn:
                missing.append(f"post_attn[{l}]")
            if l not in self.mlp_output:
                missing.append(f"mlp_output[{l}]")
            if l not in self.post_mlp:
                missing.append(f"post_mlp[{l}]")
            if l not in self.attention_last:
                missing.append(f"attention_last[{l}]")
            if l not in self.attention_pre_o:
                missing.append(f"attention_pre_o[{l}]")
        if missing:
            raise RuntimeError(f"Missing hook captures: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        return TraceBundle(
            residual_in=self.residual_in,
            attn_output=self.attn_output,
            post_attn=self.post_attn,
            mlp_output=self.mlp_output,
            post_mlp=self.post_mlp,
            attention_last=self.attention_last,
            attention_pre_o=self.attention_pre_o,
        )

    def _mk_layer_pre_hook(self, layer_idx: int):
        def hook(module, inputs):
            hidden = inputs[0]
            self.residual_in[layer_idx] = hidden[:, -1, :].detach().float().squeeze(0)

        return hook

    def _mk_attn_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                attn_out = outputs[0]
                attn_w = outputs[1] if len(outputs) > 1 else None
            else:
                attn_out = outputs
                attn_w = None

            attn_last = attn_out[:, -1, :].detach().float().squeeze(0)
            self.attn_output[layer_idx] = attn_last
            if layer_idx in self.residual_in:
                self.post_attn[layer_idx] = self.residual_in[layer_idx] + attn_last
            else:
                self.post_attn[layer_idx] = attn_last

            if attn_w is not None and attn_w.ndim == 4:
                self.attention_last[layer_idx] = attn_w[0, :, -1, :].detach().float()  # [heads, seq_len]

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

    def _mk_mlp_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                mlp_out = outputs[0]
            else:
                mlp_out = outputs
            self.mlp_output[layer_idx] = mlp_out[:, -1, :].detach().float().squeeze(0)

        return hook

    def _mk_layer_post_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                hidden = outputs[0]
            else:
                hidden = outputs
            self.post_mlp[layer_idx] = hidden[:, -1, :].detach().float().squeeze(0)

        return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 diagnostics for Evidence Routing Decomposition (ERD).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--target_per_class", type=int, default=250)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out_dir", default="results/phase1_diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_max_new_tokens", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(resolve_cache_path(Path(args.cache_path)))
    model, tokenizer, device = load_model_and_tokenizer(args.model_id, args.device, args.torch_dtype)
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    head_analysis_layers = [l for l in range(22, 28) if l < num_layers]

    hooks = LayerDecompositionHooks(model)
    hooks.register()

    per_example: list[dict[str, Any]] = []
    faithful_count = 0
    hallucinated_count = 0
    drop_stats: Counter[str] = Counter()

    for row in tqdm(rows, desc="Phase1 diagnostics"):
        if len(per_example) >= int(args.max_examples):
            break
        if faithful_count >= int(args.target_per_class) and hallucinated_count >= int(args.target_per_class):
            break

        sample = normalize_row(row)
        if sample is None:
            drop_stats["invalid_schema"] += 1
            continue
        evidence_token_id = first_answer_token_id_in_prompt(tokenizer, sample["answer_liquid"])
        parametric_token_id = first_answer_token_id_in_prompt(tokenizer, sample["answer_solid"])
        if evidence_token_id is None or parametric_token_id is None:
            drop_stats["empty_answer_first_token"] += 1
            continue
        if int(evidence_token_id) == int(parametric_token_id):
            drop_stats["same_first_answer_token"] += 1
            continue

        prompt = build_prompt(sample["evidence"], sample["question"])
        encoded_with_offsets = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        full_ids = encoded_with_offsets["input_ids"]
        offsets = encoded_with_offsets["offset_mapping"]
        evidence_positions = evidence_positions_from_offsets(prompt, sample["evidence"], offsets)
        if not full_ids:
            drop_stats["empty_prompt_or_evidence_tokens"] += 1
            continue
        if not evidence_positions:
            drop_stats["evidence_span_not_found"] += 1
            continue

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        hooks.clear()
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

        if not ensure_attention_last(hooks, outputs, num_layers):
            drop_stats["missing_output_attentions"] += 1
            continue

        bundle = hooks.bundle(num_layers=num_layers)
        final_logits = outputs.logits[0, -1, :].float()
        generated_token_id = argmax_non_whitespace_token_id(final_logits, tokenizer)
        if generated_token_id is None:
            drop_stats["no_non_whitespace_token"] += 1
            continue
        evidence_token_id = int(evidence_token_id)
        parametric_token_id = int(parametric_token_id)

        if generated_token_id == evidence_token_id:
            label = "faithful"
        elif generated_token_id == parametric_token_id:
            label = "hallucinated"
        else:
            label = classify_from_generation_span(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                evidence_token_id=evidence_token_id,
                parametric_token_id=parametric_token_id,
                max_new_tokens=int(args.label_max_new_tokens),
            )
            if label != "other":
                drop_stats["label_recovered_by_span"] += 1

        if label == "faithful" and faithful_count >= int(args.target_per_class):
            drop_stats["faithful_over_target"] += 1
            continue
        if label == "hallucinated" and hallucinated_count >= int(args.target_per_class):
            drop_stats["hallucinated_over_target"] += 1
            continue
        if label == "other":
            drop_stats["label_other"] += 1
            continue

        trace = compute_trace(
            model=model,
            tokenizer=tokenizer,
            bundle=bundle,
            num_layers=num_layers,
            head_analysis_layers=head_analysis_layers,
            evidence_token_id=evidence_token_id,
            parametric_token_id=parametric_token_id,
            evidence_positions=evidence_positions,
            generated_token_id=generated_token_id,
            sample=sample,
            label=label,
            final_logits=final_logits,
        )
        per_example.append(trace)
        if label == "faithful":
            faithful_count += 1
        elif label == "hallucinated":
            hallucinated_count += 1

    hooks.close()

    write_jsonl(out_dir / "per_example_traces.jsonl", per_example)
    agg = aggregate_statistics(per_example, num_layers=num_layers)
    with (out_dir / "aggregate_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    generate_plots(per_example, agg, plots_dir, num_layers=num_layers)
    verdict_text = build_hypothesis_verdict(per_example, agg)
    (out_dir / "hypothesis_verdict.txt").write_text(verdict_text, encoding="utf-8")

    print(json.dumps({"n_examples": len(per_example), "n_faithful": faithful_count, "n_hallucinated": hallucinated_count}, indent=2))
    print(f"traces={out_dir / 'per_example_traces.jsonl'}")
    print(f"aggregate={out_dir / 'aggregate_statistics.json'}")
    print(f"plots={plots_dir}")
    print(f"verdict={out_dir / 'hypothesis_verdict.txt'}")
    print(f"drop_stats={json.dumps(dict(drop_stats), indent=2, sort_keys=True)}")


def load_rows(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"CounterFact cache not found: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Cache file must contain a JSON list.")
    return rows


def resolve_cache_path(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    raise FileNotFoundError(f"Explicit CounterFact cache does not exist: {cache_path}; prepare or import it first")




def load_model_and_tokenizer(model_id: str, device: str, torch_dtype: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = {"torch_dtype": dtype_map[torch_dtype]}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer, torch.device(device)


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    needed = ["id", "question", "evidence", "answer_liquid", "answer_solid"]
    if not all(k in row for k in needed):
        return None
    return {
        "id": int(row["id"]),
        "question": str(row["question"]),
        "evidence": str(row["evidence"]),
        "answer_liquid": str(row["answer_liquid"]),
        "answer_solid": str(row["answer_solid"]),
    }


def build_prompt(evidence: str, question: str) -> str:
    return f"Evidence: {evidence}\n\nQuestion: {question}\nAnswer: "


def find_subsequence(full_ids: list[int], sub_ids: list[int]) -> int | None:
    if not sub_ids or len(sub_ids) > len(full_ids):
        return None
    m = len(sub_ids)
    for i in range(len(full_ids) - m + 1):
        if full_ids[i : i + m] == sub_ids:
            return i
    return None




def answer_token_ids_in_prompt(tokenizer, answer_text: str) -> list[int]:
    prefix = "Answer: "
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    combined_ids = tokenizer(prefix + answer_text, add_special_tokens=False)["input_ids"]
    if len(combined_ids) <= len(prefix_ids):
        return []
    return combined_ids[len(prefix_ids) :]


def first_answer_token_id_in_prompt(tokenizer, answer_text: str) -> int | None:
    answer = str(answer_text).strip()
    if not answer:
        return None
    ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return int(ids[0])


def build_single_token_evidence_variants(tokenizer, answer_text: str) -> list[int]:
    answer = str(answer_text).strip()
    if not answer:
        return []
    variants = [
        answer,
        " " + answer,
        answer.capitalize(),
        " " + answer.capitalize(),
        answer.lower(),
        " " + answer.lower(),
    ]
    token_ids: set[int] = set()
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            token_ids.add(int(ids[0]))
    return sorted(token_ids)


def evidence_token_ids_in_prompt(tokenizer, evidence_text: str) -> list[int]:
    prefix = "Evidence: "
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    combined_ids = tokenizer(prefix + evidence_text, add_special_tokens=False)["input_ids"]
    if len(combined_ids) <= len(prefix_ids):
        return []
    return combined_ids[len(prefix_ids) :]


def evidence_positions_from_offsets(prompt: str, evidence_text: str, offsets: list[tuple[int, int]]) -> list[int]:
    start = prompt.find(evidence_text)
    if start < 0:
        return []
    end = start + len(evidence_text)
    positions: list[int] = []
    for idx, (s, e) in enumerate(offsets):
        if s < end and e > start:
            positions.append(idx)
    return positions


def argmax_non_whitespace_token_id(logits: torch.Tensor, tokenizer, topk_scan: int = 512) -> int | None:
    k = min(int(topk_scan), int(logits.numel()))
    top_ids = torch.topk(logits, k=k, dim=-1).indices.tolist()
    for tid in top_ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        if piece.strip() != "":
            return int(tid)
    return None


def classify_from_generation_span(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    evidence_token_id: int,
    parametric_token_id: int,
    max_new_tokens: int,
) -> str:
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
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


def ensure_attention_last(hooks: LayerDecompositionHooks, outputs, num_layers: int) -> bool:
    attentions = getattr(outputs, "attentions", None)
    if attentions is not None:
        for layer_idx in range(min(num_layers, len(attentions))):
            attn_w = attentions[layer_idx]
            if attn_w is None:
                continue
            if attn_w.ndim != 4:
                continue
            hooks.attention_last[layer_idx] = attn_w[0, :, -1, :].detach().float()
    return all(layer_idx in hooks.attention_last for layer_idx in range(num_layers))


def logit_lens(model, hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"logit_lens expects 1D hidden vector, got shape={tuple(hidden.shape)}")
    # Keep LM path in model compute dtype/device, then upcast logits for stable metrics.
    target_dtype = model.lm_head.weight.dtype
    target_device = model.lm_head.weight.device
    h = hidden.view(1, 1, -1).to(device=target_device, dtype=target_dtype)
    normed = model.model.norm(h)
    logits = model.lm_head(normed)[0, 0, :].float()
    return logits


def logit_lens_batch(model, hidden_batch: torch.Tensor) -> torch.Tensor:
    if hidden_batch.ndim != 2:
        raise ValueError(f"logit_lens_batch expects 2D tensor, got shape={tuple(hidden_batch.shape)}")
    target_dtype = model.lm_head.weight.dtype
    target_device = model.lm_head.weight.device
    h = hidden_batch.to(device=target_device, dtype=target_dtype).unsqueeze(1)
    normed = model.model.norm(h)
    logits = model.lm_head(normed)[:, 0, :].float()
    return logits


def compute_head_delta_attn_evidence_for_layer(
    model,
    layer_idx: int,
    residual_in: torch.Tensor,
    head_pre_o: torch.Tensor,
    evidence_token_id: int,
    p_evidence_pre: float,
) -> list[float]:
    if head_pre_o.ndim != 2:
        raise ValueError(f"head_pre_o must be [num_heads, head_dim], got {tuple(head_pre_o.shape)}")

    layer = model.model.layers[layer_idx]
    o_proj = layer.self_attn.o_proj
    num_heads, head_dim = int(head_pre_o.shape[0]), int(head_pre_o.shape[1])
    hidden_size = num_heads * head_dim

    target_dtype = o_proj.weight.dtype
    target_device = o_proj.weight.device
    head_pre = head_pre_o.to(device=target_device, dtype=target_dtype)

    head_only = torch.zeros((num_heads, hidden_size), device=target_device, dtype=target_dtype)
    for head_idx in range(num_heads):
        start = head_idx * head_dim
        end = start + head_dim
        head_only[head_idx, start:end] = head_pre[head_idx]

    contrib = o_proj(head_only).float()
    residual = residual_in.to(device=target_device, dtype=torch.float32).unsqueeze(0).expand(num_heads, -1)
    combined = residual + contrib

    logits_batch = logit_lens_batch(model, combined)
    probs = F.softmax(logits_batch, dim=-1)[:, evidence_token_id]
    deltas = probs - float(p_evidence_pre)
    return [float(x.item()) for x in deltas]


def compute_trace(
    model,
    tokenizer,
    bundle: TraceBundle,
    num_layers: int,
    head_analysis_layers: list[int],
    evidence_token_id: int,
    parametric_token_id: int,
    evidence_positions: list[int],
    generated_token_id: int,
    sample: dict[str, Any],
    label: str,
    final_logits: torch.Tensor,
) -> dict[str, Any]:
    p_ev_pre: list[float] = []
    p_ev_post_attn: list[float] = []
    p_ev_post_mlp: list[float] = []
    p_pa_pre: list[float] = []
    p_pa_post_attn: list[float] = []
    p_pa_post_mlp: list[float] = []
    d_attn_ev: list[float] = []
    d_mlp_ev: list[float] = []
    d_attn_pa: list[float] = []
    d_mlp_pa: list[float] = []
    evidence_mass: list[float] = []
    kl_attn_to_mlp: list[float] = []
    top10_ev_removed_after_mlp: list[int] = []
    head_delta_attn_evidence_by_layer: dict[str, list[float]] = {}
    head_layer_set = set(head_analysis_layers)

    baseline_top_tokens = torch.topk(final_logits, k=min(20, int(final_logits.numel()))).indices.tolist()
    evidence_set_token_ids = build_single_token_evidence_variants(tokenizer, sample["answer_liquid"])
    if not evidence_set_token_ids:
        evidence_set_token_ids = [int(evidence_token_id)]
    parametric_set_token_ids: list[int] = []
    for tid in baseline_top_tokens:
        tok = int(tid)
        if tok in evidence_set_token_ids:
            continue
        parametric_set_token_ids.append(tok)
        if len(parametric_set_token_ids) >= 5:
            break
    if not parametric_set_token_ids:
        parametric_set_token_ids = [int(parametric_token_id)]
    margin_token_ids = sorted(set(evidence_set_token_ids + parametric_set_token_ids))

    log_probs_pre_selected: list[list[float]] = []
    log_probs_post_attn_selected: list[list[float]] = []
    log_probs_post_mlp_selected: list[list[float]] = []

    for l in range(num_layers):
        logits_pre = logit_lens(model, bundle.residual_in[l])
        logits_attn = logit_lens(model, bundle.post_attn[l])
        logits_mlp = logit_lens(model, bundle.post_mlp[l])

        prob_pre = F.softmax(logits_pre, dim=-1)
        prob_attn = F.softmax(logits_attn, dim=-1)
        prob_mlp = F.softmax(logits_mlp, dim=-1)

        logprob_pre = F.log_softmax(logits_pre, dim=-1)
        logprob_attn = F.log_softmax(logits_attn, dim=-1)
        logprob_mlp = F.log_softmax(logits_mlp, dim=-1)
        log_probs_pre_selected.append([float(logprob_pre[i].item()) for i in margin_token_ids])
        log_probs_post_attn_selected.append([float(logprob_attn[i].item()) for i in margin_token_ids])
        log_probs_post_mlp_selected.append([float(logprob_mlp[i].item()) for i in margin_token_ids])

        pev_pre = float(prob_pre[evidence_token_id].item())
        pev_attn = float(prob_attn[evidence_token_id].item())
        pev_mlp = float(prob_mlp[evidence_token_id].item())
        ppa_pre = float(prob_pre[parametric_token_id].item())
        ppa_attn = float(prob_attn[parametric_token_id].item())
        ppa_mlp = float(prob_mlp[parametric_token_id].item())

        p_ev_pre.append(pev_pre)
        p_ev_post_attn.append(pev_attn)
        p_ev_post_mlp.append(pev_mlp)
        p_pa_pre.append(ppa_pre)
        p_pa_post_attn.append(ppa_attn)
        p_pa_post_mlp.append(ppa_mlp)

        d_attn_ev.append(pev_attn - pev_pre)
        d_mlp_ev.append(pev_mlp - pev_attn)
        d_attn_pa.append(ppa_attn - ppa_pre)
        d_mlp_pa.append(ppa_mlp - ppa_attn)

        kl = torch.sum(prob_attn * (torch.log(prob_attn + 1e-12) - torch.log(prob_mlp + 1e-12)))
        kl_attn_to_mlp.append(float(kl.item()))

        top10_attn = torch.topk(logits_attn, k=10).indices.tolist()
        top10_mlp = torch.topk(logits_mlp, k=10).indices.tolist()
        removed = int((evidence_token_id in top10_attn) and (evidence_token_id not in top10_mlp))
        top10_ev_removed_after_mlp.append(removed)

        attn = bundle.attention_last[l]  # [heads, seq_len]
        den = float(attn.sum().item()) + 1e-12
        mass = float(attn[:, evidence_positions].sum().item() / den)
        evidence_mass.append(mass)

        if l in head_layer_set:
            head_delta_attn_evidence_by_layer[str(l)] = compute_head_delta_attn_evidence_for_layer(
                model=model,
                layer_idx=l,
                residual_in=bundle.residual_in[l],
                head_pre_o=bundle.attention_pre_o[l],
                evidence_token_id=evidence_token_id,
                p_evidence_pre=pev_pre,
            )

    conflict_layer = int(np.argmin(np.array(d_mlp_ev)))
    return {
        "example_id": int(sample["id"]),
        "is_hallucinated": bool(label == "hallucinated"),
        "label": label,
        "evidence_token_id": int(evidence_token_id),
        "parametric_token_id": int(parametric_token_id),
        "generated_token_id": int(generated_token_id),
        "baseline_top_tokens": [int(x) for x in baseline_top_tokens],
        "evidence_set_token_ids": [int(x) for x in evidence_set_token_ids],
        "parametric_set_token_ids": [int(x) for x in parametric_set_token_ids],
        "margin_token_ids": [int(x) for x in margin_token_ids],
        "log_probs_pre_attn_selected": log_probs_pre_selected,
        "log_probs_post_attn_selected": log_probs_post_attn_selected,
        "log_probs_post_mlp_selected": log_probs_post_mlp_selected,
        "p_evidence_pre_attn": p_ev_pre,
        "p_evidence_post_attn": p_ev_post_attn,
        "p_evidence_post_mlp": p_ev_post_mlp,
        "p_parametric_pre_attn": p_pa_pre,
        "p_parametric_post_attn": p_pa_post_attn,
        "p_parametric_post_mlp": p_pa_post_mlp,
        "delta_attn_evidence": d_attn_ev,
        "delta_mlp_evidence": d_mlp_ev,
        "delta_attn_parametric": d_attn_pa,
        "delta_mlp_parametric": d_mlp_pa,
        "evidence_attn_mass": evidence_mass,
        "kl_post_attn_to_post_mlp": kl_attn_to_mlp,
        "evidence_top10_removed_after_mlp": top10_ev_removed_after_mlp,
        "head_delta_attn_evidence_by_layer": head_delta_attn_evidence_by_layer,
        "conflict_layer": conflict_layer,
    }


def aggregate_statistics(traces: list[dict[str, Any]], num_layers: int) -> dict[str, Any]:
    faithful = [t for t in traces if not t["is_hallucinated"]]
    hall = [t for t in traces if t["is_hallucinated"]]

    def mean_vec(rows: list[dict[str, Any]], key: str) -> list[float]:
        if not rows:
            return [0.0] * num_layers
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        return arr.mean(axis=0).tolist()

    all_head_layers = sorted(
        {
            int(layer_str)
            for row in traces
            for layer_str in row.get("head_delta_attn_evidence_by_layer", {}).keys()
        }
    )

    out = {
        "n_faithful": int(len(faithful)),
        "n_hallucinated": int(len(hall)),
        "mean_delta_mlp_evidence_faithful": mean_vec(faithful, "delta_mlp_evidence"),
        "mean_delta_mlp_evidence_hallucinated": mean_vec(hall, "delta_mlp_evidence"),
        "mean_delta_attn_evidence_faithful": mean_vec(faithful, "delta_attn_evidence"),
        "mean_delta_attn_evidence_hallucinated": mean_vec(hall, "delta_attn_evidence"),
        "mean_conflict_layer_faithful": float(np.mean([r["conflict_layer"] for r in faithful])) if faithful else -1.0,
        "mean_conflict_layer_hallucinated": float(np.mean([r["conflict_layer"] for r in hall])) if hall else -1.0,
        "mean_evidence_attn_mass_faithful": mean_vec(faithful, "evidence_attn_mass"),
        "mean_evidence_attn_mass_hallucinated": mean_vec(hall, "evidence_attn_mass"),
        "head_analysis_layers": all_head_layers,
        "mean_head_delta_attn_evidence_faithful": aggregate_head_delta_matrix(faithful, all_head_layers),
        "mean_head_delta_attn_evidence_hallucinated": aggregate_head_delta_matrix(hall, all_head_layers),
    }
    return out


def aggregate_head_delta_matrix(rows: list[dict[str, Any]], layers: list[int]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for layer_idx in layers:
        key = str(layer_idx)
        vectors = [
            np.array(row["head_delta_attn_evidence_by_layer"][key], dtype=np.float64)
            for row in rows
            if key in row.get("head_delta_attn_evidence_by_layer", {})
        ]
        if not vectors:
            continue
        out[key] = np.mean(np.stack(vectors, axis=0), axis=0).tolist()
    return out


def plot_head_delta_heatmaps(agg: dict[str, Any], plots_dir: Path) -> None:
    layers = [int(x) for x in agg.get("head_analysis_layers", [])]
    if not layers:
        return

    faithful_map = agg.get("mean_head_delta_attn_evidence_faithful", {})
    hall_map = agg.get("mean_head_delta_attn_evidence_hallucinated", {})

    first_vec = None
    for layer in layers:
        key = str(layer)
        if key in faithful_map and faithful_map[key]:
            first_vec = faithful_map[key]
            break
        if key in hall_map and hall_map[key]:
            first_vec = hall_map[key]
            break
    if first_vec is None:
        return

    num_heads = int(len(first_vec))
    mat_f = np.full((num_heads, len(layers)), np.nan, dtype=np.float64)
    mat_h = np.full((num_heads, len(layers)), np.nan, dtype=np.float64)

    for col, layer in enumerate(layers):
        key = str(layer)
        if key in faithful_map and len(faithful_map[key]) == num_heads:
            mat_f[:, col] = np.array(faithful_map[key], dtype=np.float64)
        if key in hall_map and len(hall_map[key]) == num_heads:
            mat_h[:, col] = np.array(hall_map[key], dtype=np.float64)

    merged = np.concatenate([np.nan_to_num(mat_f, nan=0.0).ravel(), np.nan_to_num(mat_h, nan=0.0).ravel()])
    vabs = float(np.max(np.abs(merged))) if merged.size > 0 else 0.0
    if not np.isfinite(vabs) or vabs <= 0.0:
        vabs = 1e-8
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    def draw_one(matrix: np.ndarray, label: str, file_name: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto", origin="lower")
        ax.set_title(f"Head-wise Attention Effect on Evidence Probability ({label})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers)
        step = max(1, num_heads // 8)
        ax.set_yticks(np.arange(0, num_heads, step))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean delta p(evidence) from isolated head contribution")
        fig.tight_layout()
        fig.savefig(plots_dir / file_name, dpi=240)
        plt.close(fig)

    draw_one(mat_f, "faithful", "head_delta_heatmap_faithful.png")
    draw_one(mat_h, "hallucinated", "head_delta_heatmap_hallucinated.png")


def generate_plots(traces: list[dict[str, Any]], agg: dict[str, Any], plots_dir: Path, num_layers: int) -> None:
    layers = np.arange(num_layers)
    faithful = [t for t in traces if not t["is_hallucinated"]]
    hall = [t for t in traces if t["is_hallucinated"]]

    def mean_vec(rows: list[dict[str, Any]], key: str) -> np.ndarray:
        if not rows:
            return np.zeros(num_layers)
        return np.array([r[key] for r in rows], dtype=np.float64).mean(axis=0)

    ev_attn_f = mean_vec(faithful, "p_evidence_post_attn")
    ev_attn_h = mean_vec(hall, "p_evidence_post_attn")
    ev_mlp_f = mean_vec(faithful, "p_evidence_post_mlp")
    ev_mlp_h = mean_vec(hall, "p_evidence_post_mlp")

    pa_attn_f = mean_vec(faithful, "p_parametric_post_attn")
    pa_attn_h = mean_vec(hall, "p_parametric_post_attn")
    pa_mlp_f = mean_vec(faithful, "p_parametric_post_mlp")
    pa_mlp_h = mean_vec(hall, "p_parametric_post_mlp")

    dmlp_f = mean_vec(faithful, "delta_mlp_evidence")
    dmlp_h = mean_vec(hall, "delta_mlp_evidence")
    datt_f = mean_vec(faithful, "delta_attn_evidence")
    datt_h = mean_vec(hall, "delta_attn_evidence")
    attn_mass_f = mean_vec(faithful, "evidence_attn_mass")
    attn_mass_h = mean_vec(hall, "evidence_attn_mass")

    # Plot 1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, ev_attn_f, color="green", linestyle="-", label="faithful post-attn")
    ax.plot(layers, ev_mlp_f, color="green", linestyle="--", label="faithful post-mlp")
    ax.plot(layers, ev_attn_h, color="red", linestyle="-", label="hallucinated post-attn")
    ax.plot(layers, ev_mlp_h, color="red", linestyle="--", label="hallucinated post-mlp")
    ax.set_title("Evidence token probability across layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean p(evidence token)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "evidence_prob_trajectory.png", dpi=220)
    plt.close(fig)

    # Plot 2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, pa_attn_f, color="green", linestyle="-", label="faithful post-attn")
    ax.plot(layers, pa_mlp_f, color="green", linestyle="--", label="faithful post-mlp")
    ax.plot(layers, pa_attn_h, color="red", linestyle="-", label="hallucinated post-attn")
    ax.plot(layers, pa_mlp_h, color="red", linestyle="--", label="hallucinated post-mlp")
    ax.set_title("Parametric token probability across layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean p(parametric token)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "parametric_prob_trajectory.png", dpi=220)
    plt.close(fig)

    # Plot 3
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, dmlp_f, color="green", linewidth=2.0, label="faithful")
    ax.plot(layers, dmlp_h, color="red", linewidth=2.0, label="hallucinated")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title("MLP effect on evidence probability per layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean delta_mlp_evidence")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_mlp_evidence_by_layer.png", dpi=220)
    plt.close(fig)

    # Plot 4
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, datt_f, color="green", linewidth=2.0, label="faithful")
    ax.plot(layers, datt_h, color="red", linewidth=2.0, label="hallucinated")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title("Attention effect on evidence probability per layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean delta_attn_evidence")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_attn_evidence_by_layer.png", dpi=220)
    plt.close(fig)

    # Plot 5
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(-0.5, num_layers + 0.5, 1.0)
    ax.hist([r["conflict_layer"] for r in faithful], bins=bins, alpha=0.6, color="green", label="faithful")
    ax.hist([r["conflict_layer"] for r in hall], bins=bins, alpha=0.6, color="red", label="hallucinated")
    ax.set_title("Conflict layer histogram")
    ax.set_xlabel("Conflict layer")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "conflict_layer_histogram.png", dpi=220)
    plt.close(fig)

    # Plot 6
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, attn_mass_f, color="green", linewidth=2.0, label="faithful")
    ax.plot(layers, attn_mass_h, color="red", linewidth=2.0, label="hallucinated")
    ax.set_title("Evidence attention mass by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean evidence attention mass")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "evidence_attn_mass_by_layer.png", dpi=220)
    plt.close(fig)

    plot_head_delta_heatmaps(agg, plots_dir)


def build_hypothesis_verdict(traces: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    faithful = [t for t in traces if not t["is_hallucinated"]]
    hall = [t for t in traces if t["is_hallucinated"]]

    def conflict_signature(row: dict[str, Any]) -> bool:
        attn = np.array(row["delta_attn_evidence"], dtype=np.float64)
        mlp = np.array(row["delta_mlp_evidence"], dtype=np.float64)
        return bool(np.any((attn > 0.0) & (mlp < 0.0)))

    sig_f = np.mean([conflict_signature(r) for r in faithful]) if faithful else 0.0
    sig_h = np.mean([conflict_signature(r) for r in hall]) if hall else 0.0

    min_mlp_f = float(np.min(np.array(agg["mean_delta_mlp_evidence_faithful"], dtype=np.float64)))
    min_mlp_h = float(np.min(np.array(agg["mean_delta_mlp_evidence_hallucinated"], dtype=np.float64)))

    hall_layers = np.array([r["conflict_layer"] for r in hall], dtype=np.int64) if hall else np.array([], dtype=np.int64)
    faithful_layers = (
        np.array([r["conflict_layer"] for r in faithful], dtype=np.int64) if faithful else np.array([], dtype=np.int64)
    )
    if hall_layers.size > 0:
        h_counts = np.bincount(hall_layers, minlength=32) / max(1, hall_layers.size)
        f_counts = np.bincount(faithful_layers, minlength=32) / max(1, faithful_layers.size) if faithful_layers.size > 0 else np.zeros(32)
        peak_layer = int(np.argmax(h_counts))
        concentration_advantage = float(h_counts[peak_layer] - f_counts[peak_layer])
    else:
        peak_layer = -1
        concentration_advantage = 0.0

    c1 = sig_h > sig_f
    c2 = min_mlp_h < min_mlp_f
    c3 = concentration_advantage > 0.0

    verdict = "CONFIRMED" if (c1 and c2 and c3) else "FALSIFIED"
    lines = [
        f"HYPOTHESIS_VERDICT={verdict}",
        "",
        "Criteria:",
        f"1) conflict signature rate hallucinated > faithful: {sig_h:.4f} vs {sig_f:.4f} -> {c1}",
        f"2) min mean delta_mlp_evidence hallucinated < faithful: {min_mlp_h:.6f} vs {min_mlp_f:.6f} -> {c2}",
        f"3) conflict layer concentration advantage > 0 at peak layer {peak_layer}: {concentration_advantage:.4f} -> {c3}",
    ]
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
