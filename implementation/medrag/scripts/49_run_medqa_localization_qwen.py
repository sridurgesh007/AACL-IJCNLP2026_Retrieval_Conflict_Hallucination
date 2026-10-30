#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medrag.final_paper import mean_ci95
from medrag.utils.io import read_json, write_json


BASE_PATH = PROJECT_ROOT / "scripts" / "43_run_medqa_localization.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("medqa_localization_base_qwen", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base localization script: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_defaults() -> tuple[Path, Path]:
    defaults = {
        "--model_id": "Qwen/Qwen2-7B-Instruct",
        "--out_dir": "results/cross_model/qwen_localization",
        "--figures_dir": "results/cross_model/qwen_localization/figures",
        "--reference_layer": "27",
        "--reference_head": "20",
    }
    argv = sys.argv[1:]
    for flag, value in defaults.items():
        if flag in argv or any(str(arg).startswith(f"{flag}=") for arg in argv):
            continue
        sys.argv.extend([flag, value])

    out_dir = PROJECT_ROOT / defaults["--out_dir"]
    figures_dir = PROJECT_ROOT / defaults["--figures_dir"]
    for idx, token in enumerate(sys.argv):
        if token == "--out_dir" and idx + 1 < len(sys.argv):
            out_dir = (PROJECT_ROOT / sys.argv[idx + 1]).resolve()
        if token == "--figures_dir" and idx + 1 < len(sys.argv):
            figures_dir = (PROJECT_ROOT / sys.argv[idx + 1]).resolve()
    return out_dir.resolve(), figures_dir.resolve()


def _layer_payload(head_importance: dict[str, object], per_item_rows: list[dict[str, object]]) -> dict[str, object]:
    top5_heads = list(head_importance.get("top_5_heads", []))
    if not per_item_rows:
        raise RuntimeError("No per-item localization rows found for Qwen aggregation.")

    layer_item_values: dict[int, list[float]] = {}
    first_row = per_item_rows[0]
    layer_keys = sorted(int(k) for k in dict(first_row["per_layer_head_margin_contribution"]).keys())
    for layer in layer_keys:
        layer_item_values[int(layer)] = []

    for row in per_item_rows:
        per_layer = dict(row["per_layer_head_margin_contribution"])
        for layer in layer_keys:
            head_values = [float(x) for x in list(per_layer[str(layer)])]
            layer_item_values[int(layer)].append(float(np.mean(head_values)))

    layer_rows: list[dict[str, object]] = []
    for layer in layer_keys:
        mean, std, ci_low, ci_high = mean_ci95(layer_item_values[int(layer)])
        layer_rows.append(
            {
                "layer": int(layer),
                "mean_contribution": float(mean),
                "std": float(std),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "n_items": int(len(layer_item_values[int(layer)])),
            }
        )
    layer_rows.sort(key=lambda row: (float(row["mean_contribution"]), -int(row["layer"])), reverse=True)
    for rank, row in enumerate(layer_rows, start=1):
        row["rank"] = int(rank)

    top_layer = dict(layer_rows[0])
    ref_layer = next((dict(row) for row in layer_rows if int(row["layer"]) == 27), None)
    return {
        "architecture_note": (
            "Qwen2-7B uses grouped-query attention. Head-local intervention is not treated as the deployment site; "
            "the late residual-stream layer is the architecture-appropriate analog."
        ),
        "top_layer": top_layer,
        "layers": layer_rows,
        "reference_layer": ref_layer,
        "reference_head_rank_from_llama_coordinate": None if not top5_heads else int(next((row["rank"] for row in top5_heads if int(row["layer"]) == 27 and int(row["head"]) == 20), top5_heads[0]["rank"])),
        "metadata": head_importance.get("metadata", {}),
    }


def _plot_layer_heatmap(layer_payload: dict[str, object], out_path: Path) -> None:
    rows = list(layer_payload["layers"])
    values = np.asarray([[float(row["mean_contribution"])] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.5, 8))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_title("Qwen Layer Importance")
    ax.set_xlabel("Residual Site")
    ax.set_ylabel("Layer")
    ax.set_xticks([0])
    ax.set_xticklabels(["late residual"])
    ytick_idx = list(range(0, len(rows), max(1, len(rows) // 8)))
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([str(rows[idx]["layer"]) for idx in ytick_idx])
    top_layer = dict(layer_payload["top_layer"])
    top_idx = next(idx for idx, row in enumerate(rows) if int(row["layer"]) == int(top_layer["layer"]))
    ax.scatter(0, top_idx, marker="*", s=180, color="black", edgecolors="white", linewidths=0.8)
    fig.colorbar(im, ax=ax, label="Mean contribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    started = time.perf_counter()
    out_dir, figures_dir = _ensure_defaults()
    module = _load_base()
    rc = int(module.main())
    if rc != 0:
        return rc

    head_importance = read_json(out_dir / "head_importance.json")
    per_item_rows: list[dict[str, object]] = []
    per_item_path = out_dir / "per_item_contributions.jsonl"
    with per_item_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                per_item_rows.append(json.loads(line))

    layer_payload = _layer_payload(head_importance, per_item_rows)
    layer_payload["metadata"] = dict(layer_payload.get("metadata", {})) | {
        "aggregation_wall_clock_seconds": float(time.perf_counter() - started),
    }
    write_json(out_dir / "layer_importance.json", layer_payload)
    _plot_layer_heatmap(layer_payload, figures_dir / "layer_importance_heatmap.png")

    print(json.dumps({"top_layer": layer_payload["top_layer"]}, indent=2))
    print(f"layer_importance={out_dir / 'layer_importance.json'}")
    print(f"layer_heatmap={figures_dir / 'layer_importance_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
