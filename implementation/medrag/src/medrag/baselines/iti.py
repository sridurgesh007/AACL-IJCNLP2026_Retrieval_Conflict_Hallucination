from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class HeadProbe:
    layer: int
    head: int
    truthful_mean: torch.Tensor
    hallucinated_mean: torch.Tensor
    direction: torch.Tensor
    threshold: float
    accuracy: float
    separation: float
    n_truthful: int
    n_hallucinated: int


@dataclass(frozen=True)
class SelectedHead:
    layer: int
    head: int
    accuracy: float
    separation: float
    direction: torch.Tensor
    threshold: float


def _coerce_prompt_ids(example: dict[str, Any]) -> torch.Tensor:
    prompt_ids = example.get("prompt_ids")
    if isinstance(prompt_ids, torch.Tensor):
        if prompt_ids.ndim != 1:
            raise ValueError("prompt_ids must be a 1D tensor.")
        return prompt_ids.detach().clone().long().cpu()
    raise ValueError("Each probe-training example must contain a 1D torch.Tensor prompt_ids field.")


def _coerce_label(example: dict[str, Any]) -> int:
    label = example.get("label")
    if label is None:
        raise ValueError("Each probe-training example must contain a binary label field.")
    return 1 if bool(label) else 0


class _LastTokenHeadActivationTracer:
    def __init__(self, model, target_layers: list[int]) -> None:
        self.model = model
        self.target_layers = list(int(x) for x in target_layers)
        self._target_set = set(self.target_layers)
        self._num_heads = int(getattr(self.model.config, "num_attention_heads"))
        self._handles: list[Any] = []
        self._captured: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "_LastTokenHeadActivationTracer":
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[int(layer_idx)]
            handle = layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(int(layer_idx)))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear(self) -> None:
        self._captured = {}

    def pop(self) -> torch.Tensor:
        missing = [layer for layer in self.target_layers if int(layer) not in self._captured]
        if missing:
            raise RuntimeError(f"Missing attention_pre_o captures for layers: {missing[:8]}")
        return torch.stack([self._captured[int(layer)] for layer in self.target_layers], dim=0)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs):
            del module
            if layer_idx not in self._target_set or not inputs:
                return None
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return None
            last = hidden[:, -1, :].detach().float().cpu().squeeze(0)
            if last.ndim != 1 or last.numel() % self._num_heads != 0:
                return None
            head_dim = int(last.numel() // self._num_heads)
            self._captured[int(layer_idx)] = last.reshape(self._num_heads, head_dim)
            return None

        return hook


def collect_last_token_head_activations(
    *,
    model,
    dataset: list[dict[str, Any]],
    target_layers: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not dataset:
        raise RuntimeError("Cannot collect ITI activations from an empty dataset.")
    labels = torch.as_tensor([_coerce_label(example) for example in dataset], dtype=torch.long)
    captured: list[torch.Tensor] = []
    with _LastTokenHeadActivationTracer(model=model, target_layers=target_layers) as tracer:
        for example in dataset:
            prompt_ids = _coerce_prompt_ids(example).to(device)
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            tracer.clear()
            with torch.inference_mode():
                model(
                    input_ids=prompt_ids.unsqueeze(0),
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            captured.append(tracer.pop())
    return torch.stack(captured, dim=0), labels


def train_probes(
    model,
    tokenizer,
    dataset: list[dict[str, Any]],
    target_layers: list[int],
    device: torch.device,
) -> dict[tuple[int, int], HeadProbe]:
    del tokenizer
    activations, labels = collect_last_token_head_activations(
        model=model,
        dataset=dataset,
        target_layers=target_layers,
        device=device,
    )
    if activations.ndim != 4:
        raise RuntimeError(f"Expected activations of shape [n, layers, heads, dim], got {tuple(activations.shape)}")
    n_truthful = int((labels == 1).sum().item())
    n_hallucinated = int((labels == 0).sum().item())
    if n_truthful <= 0 or n_hallucinated <= 0:
        raise RuntimeError(
            f"ITI probe training requires both classes. Got truthful={n_truthful}, hallucinated={n_hallucinated}."
        )

    results: dict[tuple[int, int], HeadProbe] = {}
    labels_bool = labels.bool()
    target_layers = [int(x) for x in target_layers]
    for layer_offset, layer_idx in enumerate(target_layers):
        layer_acts = activations[:, layer_offset, :, :]
        num_heads = int(layer_acts.shape[1])
        for head_idx in range(num_heads):
            head_acts = layer_acts[:, head_idx, :]
            truthful = head_acts[labels_bool]
            hallucinated = head_acts[~labels_bool]
            truthful_mean = truthful.mean(dim=0)
            hallucinated_mean = hallucinated.mean(dim=0)
            raw_direction = truthful_mean - hallucinated_mean
            direction_norm = float(torch.linalg.norm(raw_direction).item())
            if direction_norm <= 0.0 or (not math.isfinite(direction_norm)):
                direction = torch.zeros_like(raw_direction)
                threshold = 0.0
                accuracy = 0.5
                separation = 0.0
            else:
                direction = raw_direction / float(direction_norm)
                truthful_proj = truthful @ direction
                hallucinated_proj = hallucinated @ direction
                truthful_center = float(truthful_proj.mean().item())
                hallucinated_center = float(hallucinated_proj.mean().item())
                threshold = 0.5 * (truthful_center + hallucinated_center)
                all_proj = head_acts @ direction
                preds = all_proj >= float(threshold)
                accuracy = float((preds == labels_bool).float().mean().item())
                separation = float(truthful_center - hallucinated_center)
            results[(int(layer_idx), int(head_idx))] = HeadProbe(
                layer=int(layer_idx),
                head=int(head_idx),
                truthful_mean=truthful_mean.detach().cpu(),
                hallucinated_mean=hallucinated_mean.detach().cpu(),
                direction=direction.detach().cpu(),
                threshold=float(threshold),
                accuracy=float(accuracy),
                separation=float(separation),
                n_truthful=int(n_truthful),
                n_hallucinated=int(n_hallucinated),
            )
    return results


def select_heads(
    probe_results: dict[tuple[int, int], HeadProbe],
    k: int,
) -> list[SelectedHead]:
    ranked = sorted(
        probe_results.values(),
        key=lambda probe: (
            float(probe.accuracy),
            abs(float(probe.separation)),
            -int(probe.layer),
            -int(probe.head),
        ),
        reverse=True,
    )
    out: list[SelectedHead] = []
    for probe in ranked[: max(int(k), 0)]:
        out.append(
            SelectedHead(
                layer=int(probe.layer),
                head=int(probe.head),
                accuracy=float(probe.accuracy),
                separation=float(probe.separation),
                direction=probe.direction.detach().clone(),
                threshold=float(probe.threshold),
            )
        )
    return out


class ITIController:
    def __init__(
        self,
        *,
        model,
        selected_heads: list[SelectedHead],
        alpha: float,
        position_mode: str,
    ) -> None:
        self.model = model
        self.selected_heads = list(selected_heads)
        self.alpha = float(alpha)
        self.position_mode = str(position_mode)
        if self.position_mode not in {"every_position", "last_position_only"}:
            raise ValueError(f"Unsupported ITI position_mode: {self.position_mode}")
        self._handles: list[Any] = []
        self._num_heads = int(getattr(self.model.config, "num_attention_heads"))
        self._grouped: dict[int, list[SelectedHead]] = {}
        self._direction_cache: dict[tuple[int, int], torch.Tensor] = {}
        for head in self.selected_heads:
            self._grouped.setdefault(int(head.layer), []).append(head)

    def register(self) -> list[Any]:
        if self._handles:
            return self._handles
        ref_param = next(self.model.parameters())
        for selections in self._grouped.values():
            for selection in selections:
                key = (int(selection.layer), int(selection.head))
                self._direction_cache[key] = selection.direction.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype,
                )
        for layer_idx, selections in sorted(self._grouped.items()):
            layer = self.model.model.layers[int(layer_idx)]
            handle = layer.self_attn.o_proj.register_forward_pre_hook(
                self._make_hook(int(layer_idx), list(selections))
            )
            self._handles.append(handle)
        return self._handles

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "ITIController":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.remove()

    def _make_hook(self, layer_idx: int, selections: list[SelectedHead]):
        def hook(module, inputs):
            del module
            if not inputs:
                return None
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return None
            batch, seq_len, hidden_size = hidden.shape
            if batch <= 0 or seq_len <= 0 or hidden_size % self._num_heads != 0:
                return None
            head_dim = int(hidden_size // self._num_heads)
            hidden_mod = hidden.clone()
            view = hidden_mod.reshape(batch, seq_len, self._num_heads, head_dim)
            for selection in selections:
                direction = self._direction_cache[(int(layer_idx), int(selection.head))]
                scaled = float(self.alpha) * direction.view(1, 1, 1, head_dim)
                if self.position_mode == "every_position":
                    view[:, :, int(selection.head), :] = view[:, :, int(selection.head), :] + scaled.squeeze(2)
                else:
                    if seq_len > 1:
                        view[:, -1, int(selection.head), :] = view[:, -1, int(selection.head), :] + scaled.squeeze(1).squeeze(1)
            args = list(inputs)
            args[0] = hidden_mod
            return tuple(args)

        return hook


def apply_iti(
    model,
    selected_heads: list[SelectedHead],
    directions: Any = None,
    alpha: float = 0.0,
    position_mode: str = "every_position",
) -> ITIController:
    del directions
    controller = ITIController(
        model=model,
        selected_heads=selected_heads,
        alpha=float(alpha),
        position_mode=str(position_mode),
    )
    controller.register()
    return controller


def remove_iti(handles: ITIController | list[Any]) -> None:
    if isinstance(handles, ITIController):
        handles.remove()
        return
    for handle in handles:
        handle.remove()


def _serialize_probe(probe: HeadProbe) -> dict[str, Any]:
    return {
        "layer": int(probe.layer),
        "head": int(probe.head),
        "truthful_mean": probe.truthful_mean.tolist(),
        "hallucinated_mean": probe.hallucinated_mean.tolist(),
        "direction": probe.direction.tolist(),
        "threshold": float(probe.threshold),
        "accuracy": float(probe.accuracy),
        "separation": float(probe.separation),
        "n_truthful": int(probe.n_truthful),
        "n_hallucinated": int(probe.n_hallucinated),
    }


def _deserialize_probe(payload: dict[str, Any]) -> HeadProbe:
    return HeadProbe(
        layer=int(payload["layer"]),
        head=int(payload["head"]),
        truthful_mean=torch.as_tensor(payload["truthful_mean"], dtype=torch.float32),
        hallucinated_mean=torch.as_tensor(payload["hallucinated_mean"], dtype=torch.float32),
        direction=torch.as_tensor(payload["direction"], dtype=torch.float32),
        threshold=float(payload["threshold"]),
        accuracy=float(payload["accuracy"]),
        separation=float(payload["separation"]),
        n_truthful=int(payload["n_truthful"]),
        n_hallucinated=int(payload["n_hallucinated"]),
    )


def save_probe_bundle(
    path: str | Path,
    *,
    probe_results: dict[tuple[int, int], HeadProbe],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "metadata": metadata,
        "probes": [_serialize_probe(probe) for _, probe in sorted(probe_results.items())],
    }
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_probe_bundle(path: str | Path) -> tuple[dict[str, Any], dict[tuple[int, int], HeadProbe]]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    probes_raw = payload.get("probes", [])
    probe_results: dict[tuple[int, int], HeadProbe] = {}
    for row in probes_raw:
        probe = _deserialize_probe(dict(row))
        probe_results[(int(probe.layer), int(probe.head))] = probe
    return metadata, probe_results
