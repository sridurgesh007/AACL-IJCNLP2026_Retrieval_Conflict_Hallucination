from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from evidence_aware.phase2.common import PreparedExample


def load_model_and_tokenizer(model_id: str, device: str, torch_dtype: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_attempts = (
        {"dtype": dtype_map[torch_dtype], "attn_implementation": "eager"},
        {"torch_dtype": dtype_map[torch_dtype], "attn_implementation": "eager"},
        {"dtype": dtype_map[torch_dtype]},
        {"torch_dtype": dtype_map[torch_dtype]},
    )
    last_error: Exception | None = None
    model = None
    for kwargs in load_attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            break
        except TypeError as exc:
            last_error = exc
            continue
    if model is None:
        assert last_error is not None
        raise last_error
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    try:
        gen_cfg = model.generation_config
        gen_cfg.do_sample = False
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None
    except Exception:
        pass
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer, torch.device(device)


def to_tensors(example: PreparedExample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hidden_to_logits(model, hidden_last: torch.Tensor) -> torch.Tensor:
    if hidden_last.ndim != 2:
        raise ValueError(f"hidden_to_logits expects [batch, hidden], got {tuple(hidden_last.shape)}")
    target_dtype = model.lm_head.weight.dtype
    target_device = model.lm_head.weight.device
    h = hidden_last.to(device=target_device, dtype=target_dtype).unsqueeze(1)
    normed = model.model.norm(h)
    logits = model.lm_head(normed)[:, 0, :].float()
    return logits


def logit_lens(model, hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"logit_lens expects 1D vector, got {tuple(hidden.shape)}")
    target_dtype = model.lm_head.weight.dtype
    target_device = model.lm_head.weight.device
    h = hidden.view(1, 1, -1).to(device=target_device, dtype=target_dtype)
    normed = model.model.norm(h)
    logits = model.lm_head(normed)[0, 0, :].float()
    return logits


def logit_lens_batch(model, hidden_batch: torch.Tensor) -> torch.Tensor:
    if hidden_batch.ndim != 2:
        raise ValueError(f"logit_lens_batch expects [batch, hidden], got {tuple(hidden_batch.shape)}")
    target_dtype = model.lm_head.weight.dtype
    target_device = model.lm_head.weight.device
    h = hidden_batch.to(device=target_device, dtype=target_dtype).unsqueeze(1)
    normed = model.model.norm(h)
    logits = model.lm_head(normed)[:, 0, :].float()
    return logits


class EvidenceHeadAmplifier:
    """
    Amplify selected attention head slices at the input of o_proj.
    """

    def __init__(self, model, target_heads: dict[int, list[int]], beta: float) -> None:
        self.model = model
        self.target_heads = {int(k): sorted({int(h) for h in v}) for k, v in target_heads.items()}
        self.beta = float(beta)
        self.hooks: list[Any] = []
        self.hidden_size = int(model.config.hidden_size)
        self.num_heads = int(model.config.num_attention_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.head_dim = self.hidden_size // self.num_heads
        self._validate()

    def _validate(self) -> None:
        max_layer = int(getattr(self.model.config, "num_hidden_layers")) - 1
        for layer_idx, heads in self.target_heads.items():
            if layer_idx < 0 or layer_idx > max_layer:
                raise ValueError(f"Invalid layer index {layer_idx}; max={max_layer}")
            for head_idx in heads:
                if head_idx < 0 or head_idx >= self.num_heads:
                    raise ValueError(f"Invalid head {head_idx}; valid range [0, {self.num_heads - 1}]")

    def _make_hook(self, layer_idx: int):
        heads = self.target_heads[layer_idx]
        head_dim = self.head_dim
        beta = self.beta

        def hook_fn(module, args):
            if not args:
                return args
            x = args[0]
            if x is None or x.ndim != 3:
                return args
            x_mod = x.clone()
            for head_idx in heads:
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                x_mod[:, -1:, start:end] = x_mod[:, -1:, start:end] * beta
            return (x_mod,) + tuple(args[1:])

        return hook_fn

    def attach(self) -> None:
        for layer_idx in sorted(self.target_heads.keys()):
            layer = self.model.model.layers[layer_idx]
            handle = layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(layer_idx))
            self.hooks.append(handle)

    def detach(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def decode_greedy(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], float, int]:
    generated = input_ids
    mask = attention_mask
    past_key_values = None
    out_ids: list[int] = []
    synchronize_if_cuda(generated.device)
    start = time.perf_counter()
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
    synchronize_if_cuda(generated.device)
    elapsed = time.perf_counter() - start
    return out_ids, elapsed, len(out_ids)


def js_divergence_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + 1e-12) - torch.log(m + 1e-12)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q + 1e-12) - torch.log(m + 1e-12)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def decode_dola(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    mature_layer: int,
    premature_layers: list[int],
) -> tuple[list[int], float, int]:
    generated = input_ids
    mask = attention_mask
    past_key_values = None
    out_ids: list[int] = []
    synchronize_if_cuda(generated.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            step_input = generated if past_key_values is None else generated[:, -1:]
            outputs = model(
                input_ids=step_input,
                attention_mask=mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("DoLa requires output_hidden_states=True, but hidden_states is None.")

            num_layers = int(getattr(model.config, "num_hidden_layers"))
            mature_idx = max(0, min(mature_layer, num_layers - 1))
            mature_logits = hidden_to_logits(model, hidden_states[mature_idx + 1][:, -1, :])

            valid_premature = [l for l in premature_layers if 0 <= l < num_layers and l != mature_idx]
            if valid_premature:
                candidates: list[tuple[float, torch.Tensor]] = []
                for layer_idx in valid_premature:
                    pre_logits = hidden_to_logits(model, hidden_states[layer_idx + 1][:, -1, :])
                    jsd = float(js_divergence_from_logits(mature_logits, pre_logits).item())
                    candidates.append((jsd, pre_logits))
                candidates.sort(key=lambda x: x[0], reverse=True)
                pre_logits = candidates[0][1]
                corrected = F.log_softmax(mature_logits, dim=-1) - F.log_softmax(pre_logits, dim=-1)
            else:
                corrected = mature_logits

            next_token = torch.argmax(corrected, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            out_ids.append(token_id)
            generated = torch.cat([generated, next_token], dim=1)
            mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)], dim=1)
            past_key_values = outputs.past_key_values
            if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                break
    synchronize_if_cuda(generated.device)
    elapsed = time.perf_counter() - start
    return out_ids, elapsed, len(out_ids)


@dataclass
class ValidationHooks:
    model: Any

    def __post_init__(self) -> None:
        self.handles: list[Any] = []
        self.clear()

    def clear(self) -> None:
        self.residual_in: dict[int, torch.Tensor] = {}
        self.attention_pre_o: dict[int, torch.Tensor] = {}

    def register(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._mk_layer_pre_hook(layer_idx)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._mk_o_proj_pre_hook(layer_idx)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _mk_layer_pre_hook(self, layer_idx: int):
        def hook(module, inputs):
            hidden = inputs[0]
            self.residual_in[layer_idx] = hidden[:, -1, :].detach().float().squeeze(0)

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
