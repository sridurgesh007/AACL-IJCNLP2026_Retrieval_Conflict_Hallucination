from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import torch


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return clean.strip("_") or "default"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


class ActivationCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, model_id: str, item_id: str, prompt: str) -> Path:
        model_slug = _slug(model_id)
        item_slug = _slug(item_id)
        prompt_sig = _prompt_hash(prompt)
        return self.root / model_slug / item_slug / f"{prompt_sig}.pt"

    def load(self, model_id: str, item_id: str, prompt: str) -> dict[str, Any] | None:
        path = self.path_for(model_id, item_id, prompt)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return None
        payload["cache_hit"] = True
        payload["cache_path"] = str(path)
        return payload

    def store(self, model_id: str, item_id: str, prompt: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(model_id, item_id, prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def get_or_compute(
        self,
        *,
        model,
        tokenizer,
        device: torch.device,
        model_id: str,
        item_id: str,
        prompt: str,
        target_layer: int,
        max_length: int,
    ) -> dict[str, Any]:
        cached = self.load(model_id=model_id, item_id=item_id, prompt=prompt)
        if cached is not None:
            return cached

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length)).to(device)
        prompt_ids = enc["input_ids"][0].detach().cpu()
        captured: dict[str, Any] = {}

        def _hook(module, args):
            del module
            hidden = args[0]
            if hidden is None or hidden.ndim != 3:
                return args
            captured["hidden_pre_norm"] = hidden[:, -1:, :].detach().float().cpu()
            return args

        handle = model.model.layers[int(target_layer)].post_attention_layernorm.register_forward_pre_hook(_hook)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                logits = model(enc["input_ids"], use_cache=False, return_dict=True).logits[0, -1, :].detach().float().cpu()
        finally:
            handle.remove()
        elapsed = float(time.perf_counter() - t0)

        payload = {
            "model_id": str(model_id),
            "item_id": str(item_id),
            "prompt_hash": _prompt_hash(prompt),
            "prompt_ids": prompt_ids,
            "cached_logits": logits,
            "hidden_pre_norm": captured.get("hidden_pre_norm"),
            "wall_clock_seconds": elapsed,
            "token_count": int(prompt_ids.shape[0]),
            "cache_hit": False,
        }
        path = self.store(model_id=model_id, item_id=item_id, prompt=prompt, payload=payload)
        payload["cache_path"] = str(path)
        return payload

    def describe_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        hidden = entry.get("hidden_pre_norm")
        return {
            "cache_hit": bool(entry.get("cache_hit", False)),
            "cache_path": str(entry.get("cache_path", "")),
            "token_count": int(entry.get("token_count", 0)),
            "has_hidden_pre_norm": bool(isinstance(hidden, torch.Tensor)),
        }

    def write_manifest(self, path: str | Path, entries: list[dict[str, Any]]) -> None:
        out_path = Path(path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "item_id": str(entry.get("item_id", "")),
                "prompt_hash": str(entry.get("prompt_hash", "")),
                "cache_path": str(entry.get("cache_path", "")),
                "cache_hit": bool(entry.get("cache_hit", False)),
                "token_count": int(entry.get("token_count", 0)),
                "wall_clock_seconds": float(entry.get("wall_clock_seconds", 0.0)),
            }
            for entry in entries
        ]
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
