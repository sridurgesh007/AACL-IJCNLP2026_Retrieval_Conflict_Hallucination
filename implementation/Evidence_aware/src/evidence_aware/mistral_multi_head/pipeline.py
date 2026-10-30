from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evidence_aware.ag_erha.utils import read_json, read_jsonl, seed_everything, write_json, write_jsonl
from evidence_aware.multi_head.pipeline import (
    _build_sets_for_eval_examples,
    _compute_per_layer_fire_rates,
    _find_best_row,
    _load_baseline_map,
    _load_eval_examples,
    _run_single_config_tau,
)
from evidence_aware.phase2.modeling import load_model_and_tokenizer


@dataclass(frozen=True)
class MistralExtractionConfig:
    track_b_path: Path
    phase1_ld_path: Path
    out_dir: Path
    top_k: int
    min_layer: int
    fallback_layers: list[int]


@dataclass(frozen=True)
class MistralMultiHeadConfig:
    model_id: str
    cache_path: Path
    phase2_main_eval_dir: Path
    target_layers_path: Path
    top_heads_path: Path
    out_dir: Path
    tau_values: list[float]
    beta_values: list[float]
    selected_beta: float | None
    calibration_n: int
    calibration_only: bool
    single_head_layer: int
    single_head_index: int
    single_head_beta: float
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    harm_target: float
    harm_reject: float
    generalization_json_path: Path
    generalization_plot_path: Path
    llama_single_restoration: float
    mistral_single_restoration: float
    mistral_single_stability: float
    mistral_single_harm_rate: float


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


def parse_mistral_extraction_config(args: Any) -> MistralExtractionConfig:
    return MistralExtractionConfig(
        track_b_path=Path(args.track_b_path),
        phase1_ld_path=Path(args.phase1_ld_path),
        out_dir=Path(args.out_dir),
        top_k=int(args.top_k),
        min_layer=int(getattr(args, "min_layer", 20)),
        fallback_layers=_parse_int_list(getattr(args, "fallback_layers", "24,25,26,27,28,29,30,31")),
    )


def parse_mistral_multi_head_config(args: Any) -> MistralMultiHeadConfig:
    selected_beta = None
    if str(args.selected_beta).strip().lower() not in {"", "none", "nan"}:
        selected_beta = float(args.selected_beta)
    return MistralMultiHeadConfig(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase2_main_eval_dir=Path(args.phase2_main_eval_dir),
        target_layers_path=Path(args.target_layers_path),
        top_heads_path=Path(args.top_heads_path),
        out_dir=Path(args.out_dir),
        tau_values=_parse_float_list(args.tau_values),
        beta_values=_parse_float_list(args.beta_values),
        selected_beta=selected_beta,
        calibration_n=int(args.calibration_n),
        calibration_only=bool(int(args.calibration_only)),
        single_head_layer=int(args.single_head_layer),
        single_head_index=int(args.single_head_index),
        single_head_beta=float(args.single_head_beta),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        harm_target=float(args.harm_target),
        harm_reject=float(args.harm_reject),
        generalization_json_path=Path(args.generalization_json_path),
        generalization_plot_path=Path(args.generalization_plot_path),
        llama_single_restoration=float(args.llama_single_restoration),
        mistral_single_restoration=float(args.mistral_single_restoration),
        mistral_single_stability=float(args.mistral_single_stability),
        mistral_single_harm_rate=float(args.mistral_single_harm_rate),
    )


def _extract_faithful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if "label" in row:
            if str(row.get("label", "")).strip().lower() == "faithful":
                out.append(row)
            continue
        if "is_hallucinated" in row and not bool(row.get("is_hallucinated", True)):
            out.append(row)
    return out


def _layer_vectors_from_row(row: dict[str, Any]) -> tuple[dict[int, np.ndarray], str] | None:
    if "head_ld_matrix" in row:
        mat = np.array(row["head_ld_matrix"], dtype=np.float64)
        if mat.ndim != 2:
            raise RuntimeError("head_ld_matrix must be 2D.")
        return {int(i): mat[int(i)] for i in range(int(mat.shape[0]))}, "head_ld_matrix"
    if "head_ld" in row:
        mat = np.array(row["head_ld"], dtype=np.float64)
        if mat.ndim != 2:
            raise RuntimeError("head_ld must be 2D.")
        return {int(i): mat[int(i)] for i in range(int(mat.shape[0]))}, "head_ld"
    layer_map = row.get("head_delta_attn_evidence_by_layer")
    if isinstance(layer_map, dict) and layer_map:
        out: dict[int, np.ndarray] = {}
        for key, val in layer_map.items():
            arr = np.array(val, dtype=np.float64)
            if arr.ndim != 1:
                raise RuntimeError("head_delta_attn_evidence_by_layer values must be 1D.")
            out[int(key)] = arr
        return out, "head_delta_attn_evidence_by_layer"
    return None


def _extract_top_heads_per_layer(phase1_ld_path: Path, target_layers: list[int]) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
    if not phase1_ld_path.exists():
        raise FileNotFoundError(f"Missing phase1 LD file: {phase1_ld_path}")
    rows = read_jsonl(phase1_ld_path)
    faithful = _extract_faithful_rows(rows)
    if not faithful:
        raise RuntimeError("No faithful rows found in phase1 LD file.")

    target_set = {int(x) for x in target_layers}
    per_layer_vectors: dict[int, list[np.ndarray]] = {}
    per_layer_sources: dict[int, set[str]] = {}
    expected_heads: int | None = None

    for row in faithful:
        parsed = _layer_vectors_from_row(row)
        if parsed is None:
            continue
        layer_map, source = parsed
        for layer_idx, vec in layer_map.items():
            if int(layer_idx) not in target_set:
                continue
            if expected_heads is None:
                expected_heads = int(vec.shape[0])
            elif int(vec.shape[0]) != int(expected_heads):
                raise RuntimeError("Inconsistent head dimension across phase1 LD rows.")
            per_layer_vectors.setdefault(int(layer_idx), []).append(vec)
            per_layer_sources.setdefault(int(layer_idx), set()).add(source)

    missing = [int(l) for l in target_layers if int(l) not in per_layer_vectors]
    if missing:
        raise RuntimeError(f"Missing target layers in phase1 LD rows: {missing}")

    top_heads: dict[int, int] = {}
    layer_stats: dict[int, dict[str, Any]] = {}
    for layer_idx in target_layers:
        mat = np.stack(per_layer_vectors[int(layer_idx)], axis=0)
        mean_vec = np.mean(mat, axis=0)
        head = int(np.argmax(mean_vec))
        top_heads[int(layer_idx)] = int(head)
        layer_stats[int(layer_idx)] = {
            "head": int(head),
            "mean_ld_faithful": float(mean_vec[head]),
            "n_rows_used": int(mat.shape[0]),
            "sources": sorted(per_layer_sources.get(int(layer_idx), set())),
        }
    return top_heads, layer_stats


def run_mistral_target_extraction(config: MistralExtractionConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if not config.track_b_path.exists():
        raise FileNotFoundError(f"Missing track-b path: {config.track_b_path}")
    rows = read_jsonl(config.track_b_path)
    faithful, hallucinated = [], []
    for row in rows:
        vec = row.get("M_post_mlp")
        if not isinstance(vec, list) or not vec:
            continue
        arr = np.array(vec, dtype=np.float64)
        if arr.ndim != 1:
            continue
        if bool(row.get("is_hallucinated", False)):
            hallucinated.append(arr)
        else:
            faithful.append(arr)
    if not faithful or not hallucinated:
        raise RuntimeError("Track-B margin file must contain both faithful and hallucinated rows.")

    f_mean = np.mean(np.stack(faithful, axis=0), axis=0)
    h_mean = np.mean(np.stack(hallucinated, axis=0), axis=0)
    margin_gap = f_mean - h_mean
    gap_delta = np.diff(margin_gap)
    ranked = np.argsort(gap_delta)[-int(config.top_k) :][::-1].tolist()
    selected = sorted(int(x) for x in ranked if int(x) >= int(config.min_layer))
    if not selected:
        selected = [int(x) for x in config.fallback_layers]

    top_heads, layer_stats = _extract_top_heads_per_layer(config.phase1_ld_path, selected)
    target_payload = {
        "target_layers": [int(x) for x in selected],
        "top_k": int(config.top_k),
        "min_layer": int(config.min_layer),
        "fallback_layers": [int(x) for x in config.fallback_layers],
        "num_faithful_track_b": int(len(faithful)),
        "num_hallucinated_track_b": int(len(hallucinated)),
        "margin_gap": [float(x) for x in margin_gap.tolist()],
        "gap_delta": [float(x) for x in gap_delta.tolist()],
    }
    top_payload = {
        "layer_to_head": {str(k): int(v) for k, v in sorted(top_heads.items())},
        "layer_stats": {str(k): v for k, v in sorted(layer_stats.items())},
    }
    write_json(config.out_dir / "target_layers.json", target_payload)
    write_json(config.out_dir / "top_heads_per_layer.json", top_payload)
    return {
        "target_layers_json": str(config.out_dir / "target_layers.json"),
        "top_heads_json": str(config.out_dir / "top_heads_per_layer.json"),
        "target_layers": target_payload["target_layers"],
        "top_heads": top_payload["layer_to_head"],
    }


def _load_target_layers(path: Path) -> list[int]:
    payload = read_json(path)
    layers = payload.get("target_layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"Invalid target_layers.json: {path}")
    return [int(x) for x in layers]


def _load_top_heads(path: Path) -> dict[int, int]:
    payload = read_json(path)
    mapping = payload.get("layer_to_head", payload)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"Invalid top_heads_per_layer.json: {path}")
    out: dict[int, int] = {}
    for k, v in mapping.items():
        if isinstance(v, dict) and "head" in v:
            out[int(k)] = int(v["head"])
        else:
            out[int(k)] = int(v)
    return out


def _choose_beta(calibration_rows: list[dict[str, Any]], harm_target: float, harm_reject: float) -> tuple[float, dict[str, Any]]:
    multi_rows = [r for r in calibration_rows if str(r.get("config_name", "")).startswith("mistral_multi_b")]
    if not multi_rows:
        raise RuntimeError("Calibration rows missing multi-head candidates.")
    strict = [r for r in multi_rows if float(r.get("harm_rate", 1.0)) <= float(harm_target)]
    if strict:
        best = _find_best_row(strict)
        return float(best["beta"]), {"selection_rule": "harm<=target", "selected": best}
    soft = [r for r in multi_rows if float(r.get("harm_rate", 1.0)) <= float(harm_reject)]
    if soft:
        best = _find_best_row(soft)
        return float(best["beta"]), {"selection_rule": "harm<=reject", "selected": best}
    best = sorted(
        multi_rows,
        key=lambda r: (float(r.get("harm_rate", 1.0)), -float(r.get("restoration_improvement", -1e9))),
    )[0]
    return float(best["beta"]), {"selection_rule": "min_harm_fallback", "selected": best}


def _plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, yline: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    x = np.arange(len(labels))
    ax.bar(x, values)
    if yline is not None:
        ax.axhline(float(yline), color="orange", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_per_layer(path: Path, per_layer_rates: dict[str, dict[str, dict[str, float]]]) -> None:
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
    ax.set_title("Per-layer Gate Fire Rate (Mistral Multi-head)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_comparison_bar(path: Path, baseline: dict[str, Any], single_best: dict[str, Any], multi_best: dict[str, Any]) -> None:
    rows = [("baseline", baseline), ("single_head", single_best), ("multi_head_best", multi_best)]
    labels = [x[0] for x in rows]
    restoration = [float(x[1].get("restoration_rate", 0.0)) for x in rows]
    stability = [float(x[1].get("stability_rate", 0.0)) for x in rows]
    harm = [float(x[1].get("harm_rate", 0.0)) for x in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.bar(x - width, restoration, width, label="restoration_rate")
    ax.bar(x, stability, width, label="stability_rate")
    ax.bar(x + width, harm, width, label="harm_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Mistral Methods: Baseline vs Single-head vs Multi-head")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _update_cross_model(config: MistralMultiHeadConfig, multi_best: dict[str, Any]) -> dict[str, Any]:
    path = config.generalization_json_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(path) if path.exists() else {}
    payload.setdefault(
        "llama",
        {
            "single_head_restoration": float(config.llama_single_restoration),
            "multi_head_restoration": None,
            "single_head_stability": 0.987,
            "multi_head_stability": None,
            "harm_rate_single": 0.0,
            "harm_rate_multi": None,
        },
    )
    payload.setdefault("mistral", {})
    payload["mistral"]["single_head_restoration"] = float(config.mistral_single_restoration)
    payload["mistral"]["single_head_stability"] = float(config.mistral_single_stability)
    payload["mistral"]["harm_rate_single"] = float(config.mistral_single_harm_rate)
    payload["mistral"]["multi_head_restoration"] = float(multi_best.get("restoration_improvement", float("nan")))
    payload["mistral"]["multi_head_stability"] = float(multi_best.get("stability_rate", float("nan")))
    payload["mistral"]["harm_rate_multi"] = float(multi_best.get("harm_rate", float("nan")))
    write_json(path, payload)
    return payload


def _plot_final_cross_model(path: Path, payload: dict[str, Any]) -> None:
    def _to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    models = [("llama", payload.get("llama", {})), ("mistral", payload.get("mistral", {}))]
    defs = [("single", "single_head_restoration", "single_head_stability", "harm_rate_single"),
            ("multi", "multi_head_restoration", "multi_head_stability", "harm_rate_multi")]
    for r_idx, (model_name, mp) in enumerate(models):
        for c_idx, (col_name, rk, sk, hk) in enumerate(defs):
            ax = axes[r_idx][c_idx]
            vals = [_to_float(mp.get(rk, np.nan)), _to_float(mp.get(sk, np.nan)), _to_float(mp.get(hk, np.nan))]
            clean = [0.0 if not np.isfinite(v) else float(v) for v in vals]
            ax.bar(np.arange(3), clean)
            for i, v in enumerate(vals):
                txt = "NA" if not np.isfinite(v) else f"{float(v):.3f}"
                ax.text(i, clean[i], txt, ha="center", va="bottom", fontsize=8)
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(["restoration", "stability", "harm"], rotation=10)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{model_name.capitalize()} - {col_name}")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Final Cross-Model Comparison: Single-head vs Multi-head", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def run_mistral_multi_head_pipeline(config: MistralMultiHeadConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "gate_stats").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    target_layers = _load_target_layers(config.target_layers_path)
    top_heads = _load_top_heads(config.top_heads_path)
    missing_layers = [int(l) for l in target_layers if int(l) not in top_heads]
    if missing_layers:
        raise RuntimeError(
            "top_heads_per_layer mapping does not cover all target layers. "
            f"missing={missing_layers}; available={sorted(top_heads.keys())}"
        )
    layer_to_head_multi = {int(l): int(top_heads[int(l)]) for l in target_layers}

    model, tokenizer, device = load_model_and_tokenizer(config.model_id, config.device, config.torch_dtype)
    baseline_map = _load_baseline_map(config.phase2_main_eval_dir)
    eval_examples = _load_eval_examples(config.cache_path, tokenizer, config.max_length, baseline_map)
    sets_by_id = _build_sets_for_eval_examples(model, tokenizer, device, eval_examples)

    calib_examples = eval_examples[: int(config.calibration_n)]
    calibration_rows: list[dict[str, Any]] = []
    for beta in config.beta_values:
        name = f"mistral_multi_b{str(beta).replace('.', 'p')}_target"
        m, g, _ = _run_single_config_tau(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_examples=calib_examples,
            baseline_map=baseline_map,
            sets_by_id=sets_by_id,
            layer_to_head=layer_to_head_multi,
            config_name=name,
            tau_margin=0.0,
            beta=float(beta),
            first_token_only=True,
            bootstrap_first_token_margin=True,
            max_new_tokens=config.max_new_tokens,
        )
        calibration_rows.append(m)
        write_jsonl(config.out_dir / "gate_stats" / f"calibration_{name}.jsonl", g)

    single_name = f"mistral_single_b{str(config.single_head_beta).replace('.', 'p')}_L{config.single_head_layer}H{config.single_head_index}"
    single_calib, single_gate_rows, _ = _run_single_config_tau(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=calib_examples,
        baseline_map=baseline_map,
        sets_by_id=sets_by_id,
        layer_to_head={int(config.single_head_layer): int(config.single_head_index)},
        config_name=single_name,
        tau_margin=0.0,
        beta=float(config.single_head_beta),
        first_token_only=True,
        bootstrap_first_token_margin=True,
        max_new_tokens=config.max_new_tokens,
    )
    calibration_rows.append(single_calib)
    write_jsonl(config.out_dir / "gate_stats" / f"calibration_{single_name}.jsonl", single_gate_rows)

    chosen_beta, selection = _choose_beta(calibration_rows, config.harm_target, config.harm_reject)
    calibration_payload = {
        "calibration_n": int(config.calibration_n),
        "rows": calibration_rows,
        "chosen_beta": float(chosen_beta),
        "selection": selection,
        "harm_target": float(config.harm_target),
        "harm_reject": float(config.harm_reject),
    }
    write_json(config.out_dir / "calibration_100ex.json", calibration_payload)
    if config.calibration_only:
        return {"mode": "calibration_only", "calibration_json": str(config.out_dir / "calibration_100ex.json")}

    # Full sweep must compare true simultaneous multi-head configs vs single-head baseline.
    # Keep two explicit multi-head betas as in multiheadmistral.md.
    sweep_multi_betas = [2.0, 3.0]
    if config.selected_beta is not None:
        # Optional override for debugging single-beta sweeps.
        sweep_multi_betas = [float(config.selected_beta)]

    min_layer = int(min(target_layers))
    max_layer = int(max(target_layers))
    method_configs = []
    for beta in sweep_multi_betas:
        method_configs.append(
            {
                "name": f"multi_L{min_layer}to{max_layer}_b{str(beta).replace('.', 'p')}",
                "layer_to_head": layer_to_head_multi,
                "beta": float(beta),
            }
        )
    method_configs.append(
        {
            "name": single_name,
            "layer_to_head": {int(config.single_head_layer): int(config.single_head_index)},
            "beta": float(config.single_head_beta),
        }
    )
    run_rows: list[dict[str, Any]] = []
    per_layer_counts: dict[tuple[str, float], dict[int, dict[str, dict[str, int]]]] = {}
    gate_rows_by_cfg_tau: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for cfg in method_configs:
        for tau in config.tau_values:
            m, g, plc = _run_single_config_tau(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eval_examples=eval_examples,
                baseline_map=baseline_map,
                sets_by_id=sets_by_id,
                layer_to_head=cfg["layer_to_head"],
                config_name=str(cfg["name"]),
                tau_margin=float(tau),
                beta=float(cfg["beta"]),
                first_token_only=True,
                bootstrap_first_token_margin=True,
                max_new_tokens=config.max_new_tokens,
            )
            run_rows.append(m)
            key = (str(cfg["name"]), float(tau))
            per_layer_counts[key] = plc
            gate_rows_by_cfg_tau[key] = g
            write_jsonl(config.out_dir / "gate_stats" / f"gate_stats_{cfg['name']}_tau_{tau:g}.jsonl", g)
    run_rows_sorted = sorted(run_rows, key=lambda r: (str(r["config_name"]), float(r["tau_margin"])))
    write_json(config.out_dir / "multi_head_sweep_summary.json", run_rows_sorted)

    multi_rows = [r for r in run_rows_sorted if str(r["config_name"]).startswith("multi_")]
    single_rows = [r for r in run_rows_sorted if str(r["config_name"]) == single_name]
    best_multi = _find_best_row(multi_rows)
    best_single = next((r for r in single_rows if abs(float(r["tau_margin"])) < 1e-12), _find_best_row(single_rows))
    best_key = (str(best_multi["config_name"]), float(best_multi["tau_margin"]))
    write_jsonl(config.out_dir / "gate_stats" / f"gate_stats_tau_{float(best_multi['tau_margin']):g}.jsonl", gate_rows_by_cfg_tau[best_key])
    per_layer_rates = _compute_per_layer_fire_rates(per_layer_counts[best_key])
    write_json(config.out_dir / "gate_stats" / "per_layer_gate_fire_rate.json", per_layer_rates)

    best_by_config = []
    for cfg in method_configs:
        rows_cfg = [r for r in run_rows_sorted if str(r["config_name"]) == str(cfg["name"])]
        if rows_cfg:
            best_by_config.append(_find_best_row(rows_cfg))
    _plot_bar(
        config.out_dir / "plots" / "restoration_improvement_by_config.png",
        [str(r["config_name"]) for r in best_by_config],
        [float(r["restoration_improvement"]) for r in best_by_config],
        "Restoration Improvement by Config (best tau)",
        "restoration_improvement",
    )
    _plot_bar(
        config.out_dir / "plots" / "stability_by_config.png",
        [str(r["config_name"]) for r in best_by_config],
        [float(r["stability_rate"]) for r in best_by_config],
        "Stability by Config (best tau)",
        "stability_rate",
        yline=0.92,
    )
    _plot_per_layer(config.out_dir / "plots" / "per_layer_gate_fire_rate_mistral.png", per_layer_rates)
    baseline_payload = read_json(config.phase2_main_eval_dir / "baseline_results.json")
    baseline_metrics = {
        "restoration_rate": float(baseline_payload.get("restoration_rate", 0.0)),
        "stability_rate": float(baseline_payload.get("stability_rate", baseline_payload.get("stability", 1.0))),
        "harm_rate": 0.0,
    }
    if float(baseline_metrics["stability_rate"]) > 1.0:
        baseline_metrics["stability_rate"] = float(baseline_metrics["stability_rate"]) / 100.0
    _plot_comparison_bar(config.out_dir / "plots" / "comparison_bar_mistral_all_methods.png", baseline_metrics, best_single, best_multi)

    multi_impr = float(best_multi.get("restoration_improvement", float("nan")))
    single_ref = float(config.mistral_single_restoration)
    delta = multi_impr - single_ref
    if multi_impr >= 0.04:
        verdict = "FULL_GENERALIZATION"
    elif multi_impr >= 0.02:
        verdict = "PARTIAL_GENERALIZATION"
    else:
        verdict = "METHOD_LIMITED"
    verdict_lines = [
        f"MISTRAL_MULTI_HEAD_BEST_CONFIG:          {best_multi['config_name']}",
        f"MISTRAL_MULTI_HEAD_BEST_TAU:             {float(best_multi['tau_margin'])}",
        f"MISTRAL_MULTI_HEAD_RESTORATION_IMPR:     {multi_impr}",
        f"MISTRAL_SINGLE_HEAD_RESTORATION_IMPR:    {single_ref}",
        f"MISTRAL_IMPROVEMENT_DELTA:               {delta}",
        f"MISTRAL_MULTI_HEAD_STABILITY:            {float(best_multi['stability_rate'])}",
        f"MISTRAL_MULTI_HEAD_HARM_RATE:            {float(best_multi['harm_rate'])}",
        "",
        "CROSS_MODEL_VERDICT:",
        f"  {verdict}",
    ]
    (config.out_dir / "multi_head_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    cross_payload = _update_cross_model(config, best_multi)
    _plot_final_cross_model(config.generalization_plot_path, cross_payload)
    final_summary = {
        "target_layers": [int(x) for x in target_layers],
        "top_heads_per_layer": {str(k): int(v) for k, v in sorted(top_heads.items())},
        "chosen_beta": float(chosen_beta),
        "sweep_multi_betas": [float(x) for x in sweep_multi_betas],
        "best_multi": best_multi,
        "best_single": best_single,
        "verdict": verdict,
        "outputs": {
            "calibration_json": str(config.out_dir / "calibration_100ex.json"),
            "sweep_summary_json": str(config.out_dir / "multi_head_sweep_summary.json"),
            "verdict_txt": str(config.out_dir / "multi_head_verdict.txt"),
            "gate_stats_dir": str(config.out_dir / "gate_stats"),
            "plots_dir": str(config.out_dir / "plots"),
            "cross_model_json": str(config.generalization_json_path),
            "cross_model_plot": str(config.generalization_plot_path),
        },
    }
    write_json(config.out_dir / "final_summary.json", final_summary)
    return final_summary
