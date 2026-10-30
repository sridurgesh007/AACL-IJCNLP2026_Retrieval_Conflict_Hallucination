from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_aware.ag_erha.utils import seed_everything, write_json, write_jsonl
from evidence_aware.mistral_steering.final_layer_steering import (
    MistralFinalLayerSteering,
    test_final_layer_steering_exactness,
)
from evidence_aware.multi_head.pipeline import (
    _build_sets_for_eval_examples,
    _load_baseline_map,
    _load_eval_examples,
)
from evidence_aware.phase2.common import PreparedExample, first_non_whitespace_from_sequence
from evidence_aware.phase2.modeling import load_model_and_tokenizer, to_tensors


@dataclass(frozen=True)
class MistralFinalLayerSweepConfig:
    model_id: str
    cache_path: Path
    phase2_main_eval_dir: Path
    out_dir: Path
    alpha_grid: list[float]
    tau_grid: list[float]
    lambda_max: float
    pilot_n: int
    pilot_alpha: float
    pilot_tau: float
    pilot_min_restoration: float
    pilot_max_harm: float
    pilot_min_stability: float
    run_pilot_only: bool
    force_full_sweep: bool
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    first_token_only: bool


def _parse_float_list(value: str) -> list[float]:
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty float list, got: {value}")
    return vals


def _safe_name(value: float | str) -> str:
    text = str(value)
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def parse_config(args: argparse.Namespace) -> MistralFinalLayerSweepConfig:
    return MistralFinalLayerSweepConfig(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase2_main_eval_dir=Path(args.phase2_main_eval_dir),
        out_dir=Path(args.out_dir),
        alpha_grid=_parse_float_list(args.alpha_grid),
        tau_grid=_parse_float_list(args.tau_grid),
        lambda_max=float(args.lambda_max),
        pilot_n=int(args.pilot_n),
        pilot_alpha=float(args.pilot_alpha),
        pilot_tau=float(args.pilot_tau),
        pilot_min_restoration=float(args.pilot_min_restoration),
        pilot_max_harm=float(args.pilot_max_harm),
        pilot_min_stability=float(args.pilot_min_stability),
        run_pilot_only=bool(int(args.run_pilot_only)),
        force_full_sweep=bool(int(args.force_full_sweep)),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        first_token_only=bool(int(args.first_token_only)),
    )


def _classify(first_token_id: int | None, evidence_set: set[int], parametric_set: set[int]) -> str:
    if first_token_id is None:
        return "other"
    if int(first_token_id) in evidence_set:
        return "faithful"
    if int(first_token_id) in parametric_set:
        return "hallucinated"
    return "other"


def _run_single_config(
    *,
    model,
    tokenizer,
    device: torch.device,
    eval_examples: list[PreparedExample],
    baseline_map: dict[int, dict[str, Any]],
    sets_by_id: dict[int, dict[str, set[int]]],
    alpha: float,
    tau_margin: float,
    lambda_max: float,
    config_name: str,
    max_new_tokens: int,
    first_token_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steerer = MistralFinalLayerSteering(
        model=model,
        evidence_set=set(),
        parametric_set=set(),
        tau_margin=float(tau_margin),
        alpha=float(alpha),
        lambda_max=float(lambda_max),
        first_token_only=bool(first_token_only),
    )

    metrics: dict[str, Any] = {
        "config_name": str(config_name),
        "alpha": float(alpha),
        "tau_margin": float(tau_margin),
        "lambda_max": float(lambda_max),
        "n_total": 0,
        "n_faithful": 0,
        "n_hallucinated": 0,
        "n_other": 0,
        "n_restored": 0,
        "n_harmed": 0,
        "n_stable": 0,
        "n_skipped_missing_sets": 0,
    }
    gate_rows: list[dict[str, Any]] = []

    for ex in tqdm(eval_examples, desc=f"{config_name} a={alpha:g} tau={tau_margin:g}", leave=False):
        sets = sets_by_id.get(int(ex.id))
        if sets is None:
            metrics["n_skipped_missing_sets"] += 1
            continue
        evidence_set = {int(x) for x in sets.get("evidence_set", set())}
        parametric_set = {int(x) for x in sets.get("parametric_set", set())}
        if not evidence_set or not parametric_set:
            metrics["n_skipped_missing_sets"] += 1
            continue

        input_ids, attention_mask = to_tensors(ex, device=device)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[0, -1, :].float()

        ev_id = max(evidence_set, key=lambda t: float(logits[int(t)].item()))
        par_id = max(parametric_set, key=lambda t: float(logits[int(t)].item()))

        steerer.reset()
        steerer.update_sets(evidence_set=evidence_set, parametric_set=parametric_set)
        steerer.setup_example(
            evidence_token_id=int(ev_id),
            parametric_token_id=int(par_id),
            model=model,
            device=device,
        )
        steerer.bootstrap_prefill(model=model, prompt_ids=input_ids.squeeze(0), device=device)
        steerer.attach(model)
        try:
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                )
        finally:
            steerer.detach()

        gen_ids = [int(x) for x in out[0, input_ids.shape[1] :].tolist()]
        first_token_id = first_non_whitespace_from_sequence(gen_ids, tokenizer)
        label = _classify(first_token_id=first_token_id, evidence_set=evidence_set, parametric_set=parametric_set)
        baseline_label = str((baseline_map.get(int(ex.id), {}) or {}).get("label", "other"))

        metrics["n_total"] += 1
        metrics[f"n_{label}"] += 1
        if baseline_label == "hallucinated" and label == "faithful":
            metrics["n_restored"] += 1
        if baseline_label == "faithful" and label == "hallucinated":
            metrics["n_harmed"] += 1
        if baseline_label == "faithful" and label == "faithful":
            metrics["n_stable"] += 1

        summary = steerer.summary()
        gate_rows.append(
            {
                "example_id": int(ex.id),
                "baseline_label": baseline_label,
                "label": str(label),
                "first_token_id": first_token_id,
                "generated_ids": gen_ids,
                "steered": bool(summary["n_steps_steered"] > 0),
                "n_steps_total": int(summary["n_steps_total"]),
                "n_steps_steered": int(summary["n_steps_steered"]),
                "mean_lambda_when_steered": float(summary["mean_lambda"]),
                "clip_fraction": float(summary["clip_fraction"]),
                "bootstrap_margin": float(summary["bootstrap_margin"]),
                "evidence_token_id": summary["evidence_token_id"],
                "parametric_token_id": summary["parametric_token_id"],
            }
        )

    n_baseline_hall = int(sum(1 for row in gate_rows if str(row["baseline_label"]) == "hallucinated"))
    n_baseline_faith = int(sum(1 for row in gate_rows if str(row["baseline_label"]) == "faithful"))
    fired_hall = [row for row in gate_rows if str(row["baseline_label"]) == "hallucinated" and bool(row["steered"])]
    fired_faith = [row for row in gate_rows if str(row["baseline_label"]) == "faithful" and bool(row["steered"])]
    steered_rows = [row for row in gate_rows if bool(row["steered"])]
    faithful_margins = [float(row["bootstrap_margin"]) for row in gate_rows if str(row["baseline_label"]) == "faithful"]
    hall_margins = [float(row["bootstrap_margin"]) for row in gate_rows if str(row["baseline_label"]) == "hallucinated"]

    metrics["accuracy"] = float(metrics["n_faithful"] / max(metrics["n_total"], 1))
    metrics["restoration_rate"] = float(metrics["n_restored"] / max(n_baseline_hall, 1))
    metrics["harm_rate"] = float(metrics["n_harmed"] / max(n_baseline_faith, 1))
    metrics["stability_rate"] = float(metrics["n_stable"] / max(n_baseline_faith, 1))
    metrics["restoration_improvement"] = float(metrics["restoration_rate"] - metrics["harm_rate"])
    metrics["gate_fired_frac_hall"] = float(len(fired_hall) / max(n_baseline_hall, 1))
    metrics["gate_fired_frac_faith"] = float(len(fired_faith) / max(n_baseline_faith, 1))
    metrics["mean_lambda_steered"] = (
        float(np.mean([float(row["mean_lambda_when_steered"]) for row in steered_rows])) if steered_rows else 0.0
    )
    metrics["mean_margin_hallucinated"] = float(np.mean(hall_margins)) if hall_margins else float("nan")
    metrics["mean_margin_faithful"] = float(np.mean(faithful_margins)) if faithful_margins else float("nan")
    metrics["n_baseline_hallucinated"] = int(n_baseline_hall)
    metrics["n_baseline_faithful"] = int(n_baseline_faith)
    return metrics, gate_rows


def _find_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Cannot select best row from empty sweep.")
    return max(
        rows,
        key=lambda r: (
            float(r["restoration_improvement"]),
            float(r["stability_rate"]),
            -float(r["harm_rate"]),
            float(r["restoration_rate"]),
        ),
    )


def run_mistral_final_layer_sweep(config: MistralFinalLayerSweepConfig) -> dict[str, Any]:
    seed_everything(int(config.seed))
    config.out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir = config.out_dir / "gate_stats"
    gate_dir.mkdir(parents=True, exist_ok=True)

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

    first_example = None
    for ex in eval_examples:
        sets = sets_by_id.get(int(ex.id), {})
        evidence_set = set(int(x) for x in sets.get("evidence_set", set()))
        parametric_set = set(int(x) for x in sets.get("parametric_set", set()))
        if evidence_set and parametric_set:
            first_example = (ex, evidence_set, parametric_set)
            break
    if first_example is None:
        raise RuntimeError("No eval example has non-empty E/P sets for final-layer steering.")

    ex0, e0, p0 = first_example
    input_ids0, attention_mask0 = to_tensors(ex0, device=device)
    with torch.inference_mode():
        logits0 = model(
            input_ids=input_ids0,
            attention_mask=attention_mask0,
            use_cache=False,
            return_dict=True,
        ).logits[0, -1, :].float()
    ev0 = max(e0, key=lambda t: float(logits0[int(t)].item()))
    par0 = max(p0, key=lambda t: float(logits0[int(t)].item()))
    unit_test = test_final_layer_steering_exactness(
        model=model,
        evidence_token_id=int(ev0),
        parametric_token_id=int(par0),
        device=device,
    )
    write_json(config.out_dir / "unit_test.json", unit_test)

    pilot_examples = eval_examples[: max(int(config.pilot_n), 1)]
    pilot_metrics, pilot_gate_rows = _run_single_config(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=pilot_examples,
        baseline_map=baseline_map,
        sets_by_id=sets_by_id,
        alpha=float(config.pilot_alpha),
        tau_margin=float(config.pilot_tau),
        lambda_max=float(config.lambda_max),
        config_name="pilot",
        max_new_tokens=int(config.max_new_tokens),
        first_token_only=bool(config.first_token_only),
    )
    write_json(config.out_dir / "pilot.json", pilot_metrics)
    write_jsonl(gate_dir / "pilot_gate.jsonl", pilot_gate_rows)

    pilot_pass = bool(
        float(pilot_metrics["restoration_improvement"]) >= float(config.pilot_min_restoration)
        and float(pilot_metrics["harm_rate"]) <= float(config.pilot_max_harm)
        and float(pilot_metrics["stability_rate"]) >= float(config.pilot_min_stability)
    )
    pilot_check = {
        "pilot_passed": pilot_pass,
        "thresholds": {
            "pilot_min_restoration": float(config.pilot_min_restoration),
            "pilot_max_harm": float(config.pilot_max_harm),
            "pilot_min_stability": float(config.pilot_min_stability),
        },
        "pilot_metrics": pilot_metrics,
    }
    write_json(config.out_dir / "pilot_check.json", pilot_check)

    if config.run_pilot_only:
        final_summary = {
            "mode": "pilot_only",
            "unit_test": unit_test,
            "pilot": pilot_metrics,
            "pilot_check": pilot_check,
            "outputs": {
                "unit_test": str(config.out_dir / "unit_test.json"),
                "pilot": str(config.out_dir / "pilot.json"),
                "pilot_check": str(config.out_dir / "pilot_check.json"),
                "pilot_gate": str(gate_dir / "pilot_gate.jsonl"),
            },
        }
        write_json(config.out_dir / "final_summary.json", final_summary)
        return final_summary

    if not pilot_pass and not config.force_full_sweep:
        raise RuntimeError(
            "Pilot failed thresholds for final-layer steering. "
            "Use --force_full_sweep 1 to continue."
        )

    sweep_rows: list[dict[str, Any]] = []
    for alpha in config.alpha_grid:
        for tau in config.tau_grid:
            cfg_name = f"final_steer_a{_safe_name(alpha)}_t{_safe_name(tau)}"
            metrics, gate_rows = _run_single_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eval_examples=eval_examples,
                baseline_map=baseline_map,
                sets_by_id=sets_by_id,
                alpha=float(alpha),
                tau_margin=float(tau),
                lambda_max=float(config.lambda_max),
                config_name=cfg_name,
                max_new_tokens=int(config.max_new_tokens),
                first_token_only=bool(config.first_token_only),
            )
            sweep_rows.append(metrics)
            write_jsonl(gate_dir / f"{cfg_name}.jsonl", gate_rows)

    write_json(config.out_dir / "sweep_summary.json", sweep_rows)
    best_row = _find_best_row(sweep_rows)
    best_impr = float(best_row["restoration_improvement"])
    if best_impr >= 0.04:
        verdict = "BREAKS_CEILING"
    elif best_impr >= 0.015:
        verdict = "MARGINAL"
    else:
        verdict = "METHOD_LIMITED"

    verdict_lines = [
        f"MISTRAL_FINAL_LAYER_BEST_CONFIG: {best_row['config_name']}",
        f"MISTRAL_FINAL_LAYER_BEST_TAU: {float(best_row['tau_margin'])}",
        f"MISTRAL_FINAL_LAYER_BEST_ALPHA: {float(best_row['alpha'])}",
        f"MISTRAL_FINAL_LAYER_LAMBDA_MAX: {float(config.lambda_max)}",
        f"MISTRAL_FINAL_LAYER_RESTORATION_IMPROVEMENT: {best_impr:.6f}",
        f"MISTRAL_FINAL_LAYER_STABILITY: {float(best_row['stability_rate']):.6f}",
        f"MISTRAL_FINAL_LAYER_HARM_RATE: {float(best_row['harm_rate']):.6f}",
        f"MISTRAL_FINAL_LAYER_MEAN_LAMBDA: {float(best_row['mean_lambda_steered']):.6f}",
        "",
        "MISTRAL_AMPLIFICATION_CEILING: 0.013",
        "",
        f"VERDICT: {verdict}",
    ]
    (config.out_dir / "verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    final_summary = {
        "model_id": config.model_id,
        "unit_test": unit_test,
        "pilot": pilot_metrics,
        "pilot_check": pilot_check,
        "sweep_rows": sweep_rows,
        "best_row": best_row,
        "verdict": verdict,
        "outputs": {
            "pilot": str(config.out_dir / "pilot.json"),
            "sweep_summary": str(config.out_dir / "sweep_summary.json"),
            "verdict": str(config.out_dir / "verdict.txt"),
            "gate_stats_dir": str(gate_dir),
        },
    }
    write_json(config.out_dir / "final_summary.json", final_summary)
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mistral final-layer steering sweep")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--cache_path", default="data/counterfact_solvent.json")
    parser.add_argument("--phase2_main_eval_dir", default="results/mistral/phase2/main_eval")
    parser.add_argument("--out_dir", default="results/mistral/final_layer_steering")
    parser.add_argument("--alpha_grid", default="0.5,1.0,1.5")
    parser.add_argument("--tau_grid", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--lambda_max", type=float, default=5.0)
    parser.add_argument("--pilot_n", type=int, default=200)
    parser.add_argument("--pilot_alpha", type=float, default=1.0)
    parser.add_argument("--pilot_tau", type=float, default=0.0)
    parser.add_argument("--pilot_min_restoration", type=float, default=0.02)
    parser.add_argument("--pilot_max_harm", type=float, default=0.01)
    parser.add_argument("--pilot_min_stability", type=float, default=0.92)
    parser.add_argument("--run_pilot_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--force_full_sweep", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_token_only", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    config = parse_config(parse_args())
    summary = run_mistral_final_layer_sweep(config)
    print(json.dumps(summary, indent=2))
    print(f"pilot={Path(config.out_dir) / 'pilot.json'}")
    print(f"sweep_summary={Path(config.out_dir) / 'sweep_summary.json'}")
    print(f"verdict={Path(config.out_dir) / 'verdict.txt'}")
    print(f"gate_stats={Path(config.out_dir) / 'gate_stats'}")


if __name__ == "__main__":
    main()
