from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PatchSpec:
    layer: int
    head: int
    position: int


def _normalize_patch_tensor(
    *,
    patch_value: torch.Tensor,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    patch = torch.as_tensor(patch_value, dtype=torch.float32, device=device)
    if patch.ndim == 1:
        patch = patch.unsqueeze(0)
    if tuple(patch.shape) != (1, int(head_dim)):
        raise ValueError(
            f"patched_value must have shape {(1, head_dim)} or {(head_dim,)}, got {tuple(patch.shape)}."
        )
    return patch


def _target_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def _resolve_position(seq_len: int, position: int) -> int:
    pos = int(position)
    if pos < 0:
        pos = int(seq_len) + pos
    if not (0 <= pos < int(seq_len)):
        raise IndexError(f"Position {position} is out of bounds for sequence length {seq_len}.")
    return int(pos)


def head_geometry(model, layer: int, head: int) -> tuple[int, int, int]:
    del layer
    num_heads = int(getattr(model.config, "num_attention_heads"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    if num_heads <= 0 or hidden_size % num_heads != 0:
        raise RuntimeError("Model head geometry is invalid.")
    if not (0 <= int(head) < num_heads):
        raise ValueError(f"Invalid head index {head}; valid range is [0, {num_heads - 1}].")
    head_dim = hidden_size // num_heads
    start = int(head) * int(head_dim)
    end = int(start + head_dim)
    return int(start), int(end), int(head_dim)


def run_without_patch(
    model,
    prompt_ids: torch.Tensor,
    *,
    use_cache: bool = False,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be a 1D tensor of token ids.")
    device = _target_device(model)
    with torch.inference_mode():
        outputs = model(
            prompt_ids.unsqueeze(0).to(device),
            use_cache=bool(use_cache),
            return_dict=True,
        )
    return outputs.logits[0, -1, :].detach().float().cpu()


def cache_head_output(
    model,
    prompt_ids: torch.Tensor,
    layer: int,
    head: int,
    position: int,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be a 1D tensor of token ids.")
    start, end, _ = head_geometry(model, layer=layer, head=head)
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, args):
        del module
        if not args:
            return args
        hidden = args[0]
        if hidden is None or hidden.ndim != 3:
            return args
        pos = _resolve_position(int(hidden.shape[1]), int(position))
        captured["value"] = hidden[:, pos, start:end].detach().float().clone().cpu()
        return args

    handle = model.model.layers[int(layer)].self_attn.o_proj.register_forward_pre_hook(_hook)
    try:
        _ = run_without_patch(model, prompt_ids, use_cache=False)
    finally:
        handle.remove()
    if "value" not in captured:
        raise RuntimeError(
            f"Failed to capture head output for layer={layer} head={head} position={position}."
        )
    return captured["value"].squeeze(0)


def cache_multihead_outputs(
    model,
    prompt_ids: torch.Tensor,
    patch_specs: list[PatchSpec],
) -> dict[PatchSpec, torch.Tensor]:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be a 1D tensor of token ids.")
    if not patch_specs:
        raise ValueError("patch_specs must be non-empty.")

    grouped: dict[int, list[tuple[PatchSpec, int, int]]] = {}
    for spec in patch_specs:
        start, end, _ = head_geometry(model, layer=int(spec.layer), head=int(spec.head))
        grouped.setdefault(int(spec.layer), []).append((spec, int(start), int(end)))

    captured: dict[PatchSpec, torch.Tensor] = {}
    handles = []

    def _make_hook(entries: list[tuple[PatchSpec, int, int]]):
        def _hook(module, args):
            del module
            if not args:
                return args
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            seq_len = int(hidden.shape[1])
            for spec, start, end in entries:
                pos = _resolve_position(seq_len, int(spec.position))
                captured[spec] = hidden[:, pos, start:end].detach().float().clone().cpu()
            return args

        return _hook

    for layer, entries in grouped.items():
        handle = model.model.layers[int(layer)].self_attn.o_proj.register_forward_pre_hook(_make_hook(entries))
        handles.append(handle)
    try:
        _ = run_without_patch(model, prompt_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [spec for spec in patch_specs if spec not in captured]
    if missing:
        first = missing[0]
        raise RuntimeError(
            f"Failed to capture multi-head output for layer={first.layer} head={first.head} position={first.position}."
        )
    return {spec: captured[spec].squeeze(0) for spec in patch_specs}


def run_with_patch(
    model,
    prompt_ids: torch.Tensor,
    layer: int,
    head: int,
    position: int,
    patched_value: torch.Tensor,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be a 1D tensor of token ids.")
    start, end, head_dim = head_geometry(model, layer=layer, head=head)
    device = _target_device(model)
    patch = _normalize_patch_tensor(patch_value=patched_value, head_dim=int(head_dim), device=device)

    def _hook(module, args):
        del module
        if not args:
            return args
        hidden = args[0]
        if hidden is None or hidden.ndim != 3:
            return args
        pos = _resolve_position(int(hidden.shape[1]), int(position))
        hidden_mod = hidden.clone()
        hidden_mod[:, pos, start:end] = patch.to(device=hidden_mod.device, dtype=hidden_mod.dtype)
        return (hidden_mod,) + tuple(args[1:])

    handle = model.model.layers[int(layer)].self_attn.o_proj.register_forward_pre_hook(_hook)
    try:
        return run_without_patch(model, prompt_ids, use_cache=False)
    finally:
        handle.remove()


def run_with_multihead_patch(
    model,
    prompt_ids: torch.Tensor,
    patched_values: dict[PatchSpec, torch.Tensor],
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be a 1D tensor of token ids.")
    if not patched_values:
        raise ValueError("patched_values must be non-empty.")

    device = _target_device(model)
    grouped: dict[int, list[tuple[PatchSpec, int, int, torch.Tensor]]] = {}
    for spec, patch_value in patched_values.items():
        start, end, head_dim = head_geometry(model, layer=int(spec.layer), head=int(spec.head))
        patch = _normalize_patch_tensor(patch_value=patch_value, head_dim=int(head_dim), device=device)
        grouped.setdefault(int(spec.layer), []).append((spec, int(start), int(end), patch))

    handles = []

    def _make_hook(entries: list[tuple[PatchSpec, int, int, torch.Tensor]]):
        def _hook(module, args):
            del module
            if not args:
                return args
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            hidden_mod = hidden.clone()
            seq_len = int(hidden.shape[1])
            for spec, start, end, patch in entries:
                pos = _resolve_position(seq_len, int(spec.position))
                hidden_mod[:, pos, start:end] = patch.to(device=hidden_mod.device, dtype=hidden_mod.dtype)
            return (hidden_mod,) + tuple(args[1:])

        return _hook

    for layer, entries in grouped.items():
        handle = model.model.layers[int(layer)].self_attn.o_proj.register_forward_pre_hook(_make_hook(entries))
        handles.append(handle)
    try:
        return run_without_patch(model, prompt_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()


def run_with_zero_ablation(
    model,
    prompt_ids: torch.Tensor,
    layer: int,
    head: int,
    position: int,
) -> torch.Tensor:
    _, _, head_dim = head_geometry(model, layer=layer, head=head)
    zeros = torch.zeros(int(head_dim), dtype=torch.float32)
    return run_with_patch(
        model=model,
        prompt_ids=prompt_ids,
        layer=int(layer),
        head=int(head),
        position=int(position),
        patched_value=zeros,
    )


def null_patch_test(
    model,
    prompt_ids: torch.Tensor,
    layer: int,
    head: int,
    position: int,
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    baseline = run_without_patch(model, prompt_ids, use_cache=False)
    cached = cache_head_output(model, prompt_ids, layer=layer, head=head, position=position)
    patched = run_with_patch(
        model=model,
        prompt_ids=prompt_ids,
        layer=int(layer),
        head=int(head),
        position=int(position),
        patched_value=cached,
    )
    diff = (patched - baseline).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    passed = bool(max_abs_diff <= float(atol))
    return {
        "layer": int(layer),
        "head": int(head),
        "position": int(position),
        "atol": float(atol),
        "max_abs_diff": float(max_abs_diff),
        "passed": bool(passed),
    }
