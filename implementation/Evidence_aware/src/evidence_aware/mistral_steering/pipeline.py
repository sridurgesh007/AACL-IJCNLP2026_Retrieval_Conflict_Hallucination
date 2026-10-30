from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evidence_aware.ag_erha.utils import write_json, write_jsonl, seed_everything
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from evidence_aware.true_multi_head.pipeline import (
    _build_gain_diagnostics,
    _build_sets_for_eval_examples,
    _find_best_row,
    _load_baseline_map,
    _load_eval_examples,
    _run_lambda_magnitude_unit_test,
    _run_single_steering_config,
)


@dataclass(frozen=True)
class MistralSteeringConfig:
    model_id: str
    cache_path: Path
    phase2_main_eval_dir: Path
    out_dir: Path
    target_heads: list[tuple[int, int]]
    selected_head: tuple[int, int] | None
    lambda_max_override: float | None
    alpha_grid: list[float]
    tau_grid: list[float]
    pilot_n: int
    pilot_alpha: float
    pilot_tau: float
    pilot_min_improvement: float
    pilot_min_stability: float
    pilot_max_harm: float
    pilot_max_other_to_faithful: int
    force_full_grid: bool
    run_pilot_only: bool
    gain_sample_n: int
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    first_token_only: bool
    bootstrap_first_token_margin: bool
    amplification_ceiling: float


def _parse_float_list(value: str) -> list[float]:
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty float list, got: {value}")
    return vals


def _parse_head_list(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid head spec '{chunk}'. Expected L:H format.")
        l_txt, h_txt = chunk.split(":", 1)
        out.append((int(l_txt.strip()), int(h_txt.strip())))
    if not out:
        raise ValueError(f"Expected non-empty target head list, got: {value}")
    return out


def _parse_optional_head(value: str) -> tuple[int, int] | None:
    txt = str(value).strip()
    if not txt or txt.lower() in {"none", "auto"}:
        return None
    pairs = _parse_head_list(txt)
    if len(pairs) != 1:
        raise ValueError(f"selected_head must contain exactly one L:H entry, got: {value}")
    return pairs[0]


def parse_mistral_steering_config(args: Any) -> MistralSteeringConfig:
    return MistralSteeringConfig(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase2_main_eval_dir=Path(args.phase2_main_eval_dir),
        out_dir=Path(args.out_dir),
        target_heads=_parse_head_list(args.target_heads),
        selected_head=_parse_optional_head(getattr(args, "selected_head", "")),
        lambda_max_override=(
            None
            if str(getattr(args, "lambda_max_override", "")).strip() in {"", "none", "auto"}
            else float(getattr(args, "lambda_max_override"))
        ),
        alpha_grid=_parse_float_list(args.alpha_grid),
        tau_grid=_parse_float_list(args.tau_grid),
        pilot_n=int(args.pilot_n),
        pilot_alpha=float(args.pilot_alpha),
        pilot_tau=float(args.pilot_tau),
        pilot_min_improvement=float(args.pilot_min_improvement),
        pilot_min_stability=float(args.pilot_min_stability),
        pilot_max_harm=float(args.pilot_max_harm),
        pilot_max_other_to_faithful=int(args.pilot_max_other_to_faithful),
        force_full_grid=bool(int(args.force_full_grid)),
        run_pilot_only=bool(int(args.run_pilot_only)),
        gain_sample_n=int(args.gain_sample_n),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        first_token_only=bool(int(args.first_token_only)),
        bootstrap_first_token_margin=bool(int(args.bootstrap_first_token_margin)),
        amplification_ceiling=float(args.amplification_ceiling),
    )


def _safe_name(value: float | str) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _baseline_faithful_count(baseline_map: dict[int, dict[str, Any]], ids: list[int]) -> int:
    return int(sum(1 for i in ids if str((baseline_map.get(int(i), {}) or {}).get("label", "other")) == "faithful"))


def _choose_head_and_lambda(
    config: MistralSteeringConfig,
    gain_diag: dict[str, Any],
) -> tuple[tuple[int, int], float, dict[str, Any]]:
    heads_payload = gain_diag.get("heads", {}) or {}
    if not isinstance(heads_payload, dict) or not heads_payload:
        raise RuntimeError("Gain diagnostic did not produce head statistics.")

    candidates = {(int(l), int(h)) for l, h in config.target_heads}

    if config.selected_head is not None:
        if tuple(config.selected_head) not in candidates:
            raise RuntimeError(
                f"selected_head={config.selected_head} not in configured target_heads={sorted(candidates)}"
            )
        chosen = tuple(config.selected_head)
    else:
        best: tuple[float, tuple[int, int]] | None = None
        for (layer_idx, head_idx) in sorted(candidates):
            key = f"L{int(layer_idx)}H{int(head_idx)}"
            row = (heads_payload.get(key, {}) or {})
            median_gain = float(row.get("median_gain", float("nan")))
            if not np.isfinite(median_gain):
                continue
            rank = (median_gain, (int(layer_idx), int(head_idx)))
            if best is None or rank[0] > best[0]:
                best = (rank[0], rank[1])
        if best is None:
            raise RuntimeError("Could not choose steering head: all candidate gains are invalid.")
        chosen = best[1]

    key = f"L{int(chosen[0])}H{int(chosen[1])}"
    chosen_stats = (heads_payload.get(key, {}) or {})
    chosen_median_gain = float(chosen_stats.get("median_gain", float("nan")))

    if config.lambda_max_override is not None:
        lambda_max = float(config.lambda_max_override)
        lambda_rule = "manual_override"
    else:
        if not np.isfinite(chosen_median_gain):
            raise RuntimeError(f"Chosen head {key} has invalid median_gain={chosen_median_gain}")
        if chosen_median_gain > 1.0:
            lambda_max = 10.0
            lambda_rule = "gain_gt_1.0"
        elif chosen_median_gain >= 0.1:
            lambda_max = 5.0
            lambda_rule = "gain_in_0.1_to_1.0"
        elif chosen_median_gain >= 0.01:
            raise RuntimeError(
                f"Chosen head {key} has weak gain={chosen_median_gain:.6f} (<0.1). "
                "Spec says skip Task 2 or choose higher-gain head."
            )
        else:
            raise RuntimeError(
                f"Chosen head {key} has very low gain={chosen_median_gain:.6f} (<0.01). "
                "Task 2 is not viable with this head."
            )

    decision = {
        "selected_head": {"layer": int(chosen[0]), "head": int(chosen[1]), "key": key},
        "selected_head_stats": chosen_stats,
        "lambda_max": float(lambda_max),
        "lambda_rule": lambda_rule,
    }
    return chosen, float(lambda_max), decision


def _validate_mistral_layer_interface(model, layer_idx: int) -> dict[str, bool]:
    layer = model.model.layers[int(layer_idx)]
    checks = {
        "has_post_attention_layernorm": bool(hasattr(layer, "post_attention_layernorm")),
        "has_self_attn": bool(hasattr(layer, "self_attn")),
        "has_o_proj": bool(hasattr(getattr(layer, "self_attn", object()), "o_proj")),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Target layer interface mismatch at layer={layer_idx}. checks={checks}. "
            "Update hook location for this model implementation."
        )
    return checks


def _plot_lambda_distribution(out_path: Path, gate_rows: list[dict[str, Any]], lambda_max: float) -> None:
    hall = [float(r.get("mean_lambda_when_steered", 0.0)) for r in gate_rows if str(r.get("baseline_label")) == "hallucinated"]
    faith = [float(r.get("mean_lambda_when_steered", 0.0)) for r in gate_rows if str(r.get("baseline_label")) == "faithful"]
    max_x = max(float(lambda_max) * 1.2, 1.0)
    bins = np.linspace(0.0, max_x, 30)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.hist(faith, bins=bins, alpha=0.6, label="faithful", color="#DD8452")
    ax.hist(hall, bins=bins, alpha=0.6, label="hallucinated", color="#4C72B0")
    ax.axvline(float(lambda_max), color="red", linestyle="--", linewidth=1.2, label=f"lambda_max={lambda_max:g}")
    ax.set_xlabel("mean lambda when steered")
    ax.set_ylabel("count")
    ax.set_title("Mistral OV Steering: Lambda Distribution by Baseline Label")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_metric_vs_tau(
    out_path: Path,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    y_label: str,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    by_cfg: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cfg.setdefault(str(row["config_name"]), []).append(row)
    for cfg_name in sorted(by_cfg.keys()):
        vals = sorted(by_cfg[cfg_name], key=lambda x: float(x["tau_margin"]))
        xs = [float(r["tau_margin"]) for r in vals]
        ys = [float(r[metric_key]) for r in vals]
        ax.plot(xs, ys, marker="o", linewidth=2, label=cfg_name)
    ax.set_xlabel("tau_margin")
    ax.set_ylabel(y_label)
    ax.set_ylim(floor, ceil)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_comparison_bar(
    out_path: Path,
    baseline_metrics: dict[str, float],
    steering_best: dict[str, Any],
    amplification_ceiling: float,
) -> None:
    labels = ["Baseline", "Amplification ceiling", "Mistral steering"]
    vals = [
        float(baseline_metrics.get("restoration_improvement", 0.0)),
        float(amplification_ceiling),
        float(steering_best.get("restoration_improvement", 0.0)),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = ["#888888", "#DD8452", "#4C72B0"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("restoration improvement")
    ax.set_title("Mistral: Baseline vs Amplification Ceiling vs OV Steering")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def run_mistral_gain_check(config: MistralSteeringConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)

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
    if not eval_examples:
        raise RuntimeError("No eval examples prepared from cache + baseline map.")

    layer_to_head = {int(l): int(h) for l, h in config.target_heads}
    gain_diag = _build_gain_diagnostics(
        model=model,
        eval_examples=eval_examples,
        layer_to_head=layer_to_head,
        sample_n=int(max(config.gain_sample_n, 1)),
    )

    chosen, lambda_max, decision = _choose_head_and_lambda(config, gain_diag)
    interface_checks = _validate_mistral_layer_interface(model, layer_idx=int(chosen[0]))

    unit_test = _run_lambda_magnitude_unit_test(
        model=model,
        evidence_token_id=int(eval_examples[0].evidence_token_id),
        parametric_token_id=int(eval_examples[0].parametric_token_id),
        layer_idx=int(chosen[0]),
        head_idx=int(chosen[1]),
        alpha=1.0,
        lambda_max=float(lambda_max),
        deficit=3.0,
        max_perturbation=20.0,
    )

    payload = {
        "model_id": config.model_id,
        "phase2_main_eval_dir": str(config.phase2_main_eval_dir),
        "target_heads": [{"layer": int(l), "head": int(h)} for l, h in config.target_heads],
        "gain_sample_n": int(max(config.gain_sample_n, 1)),
        "gain_diagnostic": gain_diag,
        "selection": decision,
        "interface_checks": interface_checks,
        "unit_test": unit_test,
    }
    write_json(config.out_dir / "gain_diagnostic.json", payload)
    return payload


def run_mistral_steering_pipeline(config: MistralSteeringConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir = config.out_dir / "gate_stats"
    plots_dir = config.out_dir / "plots"
    gate_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    if not eval_examples:
        raise RuntimeError("No eval examples prepared from cache + baseline map.")

    sets_by_id = _build_sets_for_eval_examples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_examples,
    )

    gain_diag = _build_gain_diagnostics(
        model=model,
        eval_examples=eval_examples,
        layer_to_head={int(l): int(h) for l, h in config.target_heads},
        sample_n=int(max(config.gain_sample_n, 1)),
    )
    chosen, lambda_max, decision = _choose_head_and_lambda(config, gain_diag)
    interface_checks = _validate_mistral_layer_interface(model, layer_idx=int(chosen[0]))

    unit_test = _run_lambda_magnitude_unit_test(
        model=model,
        evidence_token_id=int(eval_examples[0].evidence_token_id),
        parametric_token_id=int(eval_examples[0].parametric_token_id),
        layer_idx=int(chosen[0]),
        head_idx=int(chosen[1]),
        alpha=1.0,
        lambda_max=float(lambda_max),
        deficit=3.0,
        max_perturbation=20.0,
    )

    gain_payload = {
        "model_id": config.model_id,
        "phase2_main_eval_dir": str(config.phase2_main_eval_dir),
        "target_heads": [{"layer": int(l), "head": int(h)} for l, h in config.target_heads],
        "gain_sample_n": int(max(config.gain_sample_n, 1)),
        "gain_diagnostic": gain_diag,
        "selection": decision,
        "interface_checks": interface_checks,
        "unit_test": unit_test,
    }
    write_json(config.out_dir / "gain_diagnostic.json", gain_payload)

    pilot_examples = eval_examples[: max(int(config.pilot_n), 1)]
    pilot_layer_to_head = {int(chosen[0]): int(chosen[1])}
    pilot_config_name = (
        f"mistral_steering_L{int(chosen[0])}H{int(chosen[1])}_"
        f"clip{_safe_name(lambda_max)}_alpha{_safe_name(config.pilot_alpha)}"
    )
    pilot_metrics, pilot_gate_rows = _run_single_steering_config(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=pilot_examples,
        baseline_map=baseline_map,
        sets_by_id=sets_by_id,
        layer_to_head=pilot_layer_to_head,
        config_name=pilot_config_name,
        alpha=float(config.pilot_alpha),
        tau_margin=float(config.pilot_tau),
        lambda_max=float(lambda_max),
        first_token_only=bool(config.first_token_only),
        bootstrap_first_token_margin=bool(config.bootstrap_first_token_margin),
        max_new_tokens=config.max_new_tokens,
    )

    pilot_ids = [int(ex.id) for ex in pilot_examples]
    pilot_baseline_faithful = _baseline_faithful_count(baseline_map, pilot_ids)
    pilot_n_faithful_increase = int(pilot_metrics["n_faithful"]) - int(pilot_baseline_faithful)
    pilot_n_other_to_faithful = int(pilot_n_faithful_increase) - int(pilot_metrics["n_restored"])

    pilot_checks = {
        "restoration_improvement_gt_threshold": {
            "value": float(pilot_metrics["restoration_improvement"]),
            "threshold": float(config.pilot_min_improvement),
            "passed": bool(float(pilot_metrics["restoration_improvement"]) > float(config.pilot_min_improvement)),
        },
        "stability_gt_threshold": {
            "value": float(pilot_metrics["stability_rate"]),
            "threshold": float(config.pilot_min_stability),
            "passed": bool(float(pilot_metrics["stability_rate"]) > float(config.pilot_min_stability)),
        },
        "harm_lt_threshold": {
            "value": float(pilot_metrics["harm_rate"]),
            "threshold": float(config.pilot_max_harm),
            "passed": bool(float(pilot_metrics["harm_rate"]) < float(config.pilot_max_harm)),
        },
        "mean_lambda_below_lambda_max": {
            "value": float(pilot_metrics["mean_lambda_when_steered"]),
            "threshold": float(lambda_max),
            "passed": bool(float(pilot_metrics["mean_lambda_when_steered"]) < float(lambda_max)),
        },
        "other_contamination_within_limit": {
            "value": int(pilot_n_other_to_faithful),
            "threshold": int(config.pilot_max_other_to_faithful),
            "passed": bool(int(pilot_n_other_to_faithful) <= int(config.pilot_max_other_to_faithful)),
        },
    }
    pilot_passed = bool(all(bool(v["passed"]) for v in pilot_checks.values()))

    pilot_payload = {
        "pilot_config": {
            "name": pilot_config_name,
            "layer_to_head": {str(int(chosen[0])): int(chosen[1])},
            "alpha": float(config.pilot_alpha),
            "tau_margin": float(config.pilot_tau),
            "lambda_max": float(lambda_max),
        },
        "num_examples": int(len(pilot_examples)),
        "metrics": pilot_metrics,
        "baseline_n_faithful": int(pilot_baseline_faithful),
        "n_faithful_increase": int(pilot_n_faithful_increase),
        "n_other_to_faithful": int(pilot_n_other_to_faithful),
        "checks": pilot_checks,
        "pilot_passed": bool(pilot_passed),
    }
    write_json(config.out_dir / "pilot_200ex.json", pilot_payload)
    write_jsonl(
        gate_dir / f"gate_stats_{pilot_config_name}_tau_{_safe_name(config.pilot_tau)}__pilot.jsonl",
        pilot_gate_rows,
    )

    if bool(config.run_pilot_only):
        final_summary = {
            "mode": "pilot_only",
            "gain": gain_payload,
            "pilot": pilot_payload,
            "outputs": {
                "gain_diagnostic": str(config.out_dir / "gain_diagnostic.json"),
                "pilot": str(config.out_dir / "pilot_200ex.json"),
                "gate_stats": str(gate_dir),
            },
        }
        write_json(config.out_dir / "final_summary.json", final_summary)
        return final_summary

    if not pilot_passed and not bool(config.force_full_grid):
        raise RuntimeError(
            "Mistral steering pilot did not pass criteria. "
            "Set --force_full_grid 1 to override."
        )

    layer_to_head = {int(chosen[0]): int(chosen[1])}
    run_rows: list[dict[str, Any]] = []
    gate_rows_map: dict[tuple[str, float], list[dict[str, Any]]] = {}
    eval_ids = [int(ex.id) for ex in eval_examples]
    baseline_n_faithful = _baseline_faithful_count(baseline_map, eval_ids)

    for alpha in config.alpha_grid:
        cfg_name = f"mistral_steering_L{int(chosen[0])}H{int(chosen[1])}_clip{_safe_name(lambda_max)}_alpha{_safe_name(alpha)}"
        for tau in config.tau_grid:
            row, gate_rows = _run_single_steering_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eval_examples=eval_examples,
                baseline_map=baseline_map,
                sets_by_id=sets_by_id,
                layer_to_head=layer_to_head,
                config_name=cfg_name,
                alpha=float(alpha),
                tau_margin=float(tau),
                lambda_max=float(lambda_max),
                first_token_only=bool(config.first_token_only),
                bootstrap_first_token_margin=bool(config.bootstrap_first_token_margin),
                max_new_tokens=config.max_new_tokens,
            )
            row["baseline_n_faithful"] = int(baseline_n_faithful)
            row["n_faithful_increase_over_baseline"] = int(row["n_faithful"]) - int(baseline_n_faithful)
            row["n_other_to_faithful"] = int(row["n_faithful_increase_over_baseline"]) - int(row["n_restored"])
            run_rows.append(row)
            gate_rows_map[(str(cfg_name), float(tau))] = gate_rows
            write_jsonl(
                gate_dir / f"gate_stats_{cfg_name}_tau_{_safe_name(tau)}.jsonl",
                gate_rows,
            )

    run_rows = sorted(run_rows, key=lambda x: (str(x["config_name"]), float(x["tau_margin"])))
    write_json(config.out_dir / "steering_sweep_summary.json", run_rows)

    best_row = _find_best_row(run_rows)
    best_key = (str(best_row["config_name"]), float(best_row["tau_margin"]))
    best_gate_rows = gate_rows_map[best_key]

    _plot_lambda_distribution(
        out_path=plots_dir / "lambda_distribution_mistral.png",
        gate_rows=best_gate_rows,
        lambda_max=float(lambda_max),
    )
    _plot_metric_vs_tau(
        out_path=plots_dir / "restoration_vs_tau_mistral.png",
        rows=run_rows,
        metric_key="restoration_improvement",
        title="Mistral Steering: Restoration Improvement vs Tau",
        y_label="restoration improvement",
        floor=-0.05,
        ceil=0.5,
    )
    _plot_metric_vs_tau(
        out_path=plots_dir / "stability_vs_tau_mistral.png",
        rows=run_rows,
        metric_key="stability_rate",
        title="Mistral Steering: Stability vs Tau",
        y_label="stability rate",
        floor=0.0,
        ceil=1.02,
    )

    baseline_metrics = {
        "accuracy": float(_baseline_faithful_count(baseline_map, eval_ids) / max(len(eval_ids), 1)),
        "restoration_rate": 0.0,
        "stability_rate": 1.0,
        "harm_rate": 0.0,
        "restoration_improvement": 0.0,
    }
    _plot_comparison_bar(
        out_path=plots_dir / "comparison_bar_mistral_with_steering.png",
        baseline_metrics=baseline_metrics,
        steering_best=best_row,
        amplification_ceiling=float(config.amplification_ceiling),
    )

    best_impr = float(best_row["restoration_improvement"])
    if best_impr > 0.03:
        verdict = "BREAKS_CEILING"
    elif best_impr >= 0.015:
        verdict = "MARGINAL"
    else:
        verdict = "METHOD_LIMITED"

    verdict_lines = [
        f"MISTRAL_STEERING_BEST_CONFIG:       {best_row['config_name']}",
        f"MISTRAL_STEERING_BEST_TAU:          {float(best_row['tau_margin'])}",
        f"MISTRAL_STEERING_ALPHA:             {float(best_row['alpha'])}",
        f"MISTRAL_STEERING_LAMBDA_MAX:        {float(best_row['lambda_max'])}",
        f"MISTRAL_STEERING_RESTORATION_IMPR:  {float(best_row['restoration_improvement'])}",
        f"MISTRAL_STEERING_STABILITY:         {float(best_row['stability_rate'])}",
        f"MISTRAL_STEERING_HARM_RATE:         {float(best_row['harm_rate'])}",
        f"MISTRAL_STEERING_MEAN_LAMBDA:       {float(best_row['mean_lambda_when_steered'])}",
        f"MISTRAL_STEERING_CLIP_FRACTION:     {float(best_row.get('clip_fraction', 0.0))}",
        "",
        f"MISTRAL_AMPLIFICATION_CEILING:      {float(config.amplification_ceiling)}",
        f"MISTRAL_STEERING_DELTA:             {float(best_row['restoration_improvement']) - float(config.amplification_ceiling)}",
        "",
        f"VERDICT: {verdict}",
    ]
    (config.out_dir / "steering_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    final_summary = {
        "model_id": config.model_id,
        "phase2_main_eval_dir": str(config.phase2_main_eval_dir),
        "selection": decision,
        "gain": gain_payload,
        "pilot": pilot_payload,
        "sweep_rows": run_rows,
        "best_row": best_row,
        "verdict": verdict,
        "outputs": {
            "gain_diagnostic": str(config.out_dir / "gain_diagnostic.json"),
            "pilot": str(config.out_dir / "pilot_200ex.json"),
            "sweep_summary": str(config.out_dir / "steering_sweep_summary.json"),
            "verdict": str(config.out_dir / "steering_verdict.txt"),
            "gate_stats_dir": str(gate_dir),
            "plots_dir": str(plots_dir),
        },
    }
    write_json(config.out_dir / "final_summary.json", final_summary)
    return final_summary
