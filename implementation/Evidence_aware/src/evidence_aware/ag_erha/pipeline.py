from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from evidence_aware.ag_erha.gating import MarginGatedERHA, decode_with_gate
from evidence_aware.ag_erha.track_a import load_baseline_label_map, run_track_a
from evidence_aware.ag_erha.track_b import build_evidence_set, build_parametric_set, run_track_b
from evidence_aware.ag_erha.utils import (
    load_cache_rows,
    normalize_example_rows,
    prepare_examples,
    seed_everything,
    split_by_fixed_indices,
    write_json,
)
from evidence_aware.phase2.modeling import load_model_and_tokenizer, to_tensors


@dataclass(frozen=True)
class AGERHAConfig:
    model_id: str
    cache_path: Path
    phase1_traces_path: Path
    phase2_main_eval_dir: Path
    ag_out_dir: Path
    track_b_out_dir: Path
    margin_out_dir: Path
    tau_grid: list[float]
    beta: float
    tau_margin_grid: list[float]
    max_length: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    seed: int
    run_margin_upgrade: bool


def parse_ag_erha_config(args: Any) -> AGERHAConfig:
    tau_grid = [float(x.strip()) for x in str(args.tau_grid).split(",") if x.strip()]
    if not tau_grid:
        raise ValueError("tau_grid cannot be empty.")
    tau_margin_grid = [float(x.strip()) for x in str(args.tau_margin_grid).split(",") if x.strip()]
    if not tau_margin_grid:
        raise ValueError("tau_margin_grid cannot be empty.")
    return AGERHAConfig(
        model_id=str(args.model_id),
        cache_path=Path(args.cache_path),
        phase1_traces_path=Path(args.phase1_traces_path),
        phase2_main_eval_dir=Path(args.phase2_main_eval_dir),
        ag_out_dir=Path(args.ag_out_dir),
        track_b_out_dir=Path(args.track_b_out_dir),
        margin_out_dir=Path(args.margin_out_dir),
        tau_grid=tau_grid,
        beta=float(args.beta),
        tau_margin_grid=tau_margin_grid,
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        run_margin_upgrade=bool(int(args.run_margin_upgrade)),
    )


def _compute_metrics(
    counts: dict[str, Any],
    baseline_map: dict[int, dict[str, Any]],
    eval_ids: list[int],
) -> dict[str, float]:
    baseline_labels = [baseline_map.get(i, {"label": "other"})["label"] for i in eval_ids]
    n_baseline_hall = sum(1 for x in baseline_labels if x == "hallucinated")
    n_baseline_faith = sum(1 for x in baseline_labels if x == "faithful")
    accuracy = counts["n_faithful"] / max(counts["n_total"], 1)
    restoration_rate = counts["n_restored"] / max(n_baseline_hall, 1)
    harm_rate = counts["n_harmed"] / max(n_baseline_faith, 1)
    stability_rate = counts["n_stable"] / max(n_baseline_faith, 1)
    return {
        "accuracy": float(accuracy),
        "restoration_rate": float(restoration_rate),
        "harm_rate": float(harm_rate),
        "stability_rate": float(stability_rate),
        "restoration_improvement": float(restoration_rate - harm_rate),
    }


def _run_margin_gated_upgrade(
    model,
    tokenizer,
    device,
    eval_examples,
    baseline_map: dict[int, dict[str, Any]],
    tau_margin_grid: list[float],
    beta: float,
    max_new_tokens: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for tau_margin in tau_margin_grid:
        gate = MarginGatedERHA(model=model, evidence_set=set(), parametric_set=set(), tau_margin=tau_margin, beta=beta)
        gate.attach()
        counts = {
            "tau_margin": tau_margin,
            "beta": beta,
            "n_total": 0,
            "n_faithful": 0,
            "n_hallucinated": 0,
            "n_other": 0,
            "n_restored": 0,
            "n_harmed": 0,
            "n_stable": 0,
        }
        try:
            for ex in eval_examples:
                evidence_set = build_evidence_set(tokenizer, ex.answer_liquid)
                input_ids, attention_mask = to_tensors(ex, device=device)
                with torch.inference_mode():
                    base_out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                base_logits = base_out.logits[0, -1, :]
                baseline_top = torch.topk(base_logits, k=min(20, int(base_logits.numel()))).indices.tolist()
                parametric_set = build_parametric_set(baseline_top, evidence_set, k=5)
                gate.update_sets(evidence_set, parametric_set)
                gate.reset_log()
                out = decode_with_gate(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    example=ex,
                    max_new_tokens=max_new_tokens,
                    gate=gate,
                )
                base = baseline_map.get(ex.id, {"label": "other"})["label"]
                counts["n_total"] += 1
                counts[f"n_{out.label}"] += 1
                if base == "hallucinated" and out.label == "faithful":
                    counts["n_restored"] += 1
                if base == "faithful" and out.label == "hallucinated":
                    counts["n_harmed"] += 1
                if base == "faithful" and out.label == "faithful":
                    counts["n_stable"] += 1
        finally:
            gate.detach()
        counts.update(_compute_metrics(counts, baseline_map=baseline_map, eval_ids=[e.id for e in eval_examples]))
        results.append(counts)
        write_json(out_dir / f"mg_erha_tau_{tau_margin:.2f}.json", counts)

    write_json(out_dir / "mg_erha_sweep_summary.json", results)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = [float(r["tau_margin"]) for r in results]
    stability = [float(r["stability_rate"]) for r in results]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.plot(xs, stability, marker="o", linewidth=2.0)
    ax.set_xlabel("tau_margin")
    ax.set_ylabel("stability_rate")
    ax.set_title("Stability vs tau_margin (Margin-Gated ERHA)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "stability_vs_tau_margin.png", dpi=220)
    plt.close(fig)
    return {"results": results}


def run_ag_erha_pipeline(config: AGERHAConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.ag_out_dir.mkdir(parents=True, exist_ok=True)
    config.track_b_out_dir.mkdir(parents=True, exist_ok=True)
    config.margin_out_dir.mkdir(parents=True, exist_ok=True)

    rows = normalize_example_rows(load_cache_rows(config.cache_path))
    phase1_rows, _, eval_rows = split_by_fixed_indices(rows)
    model, tokenizer, device = load_model_and_tokenizer(config.model_id, config.device, config.torch_dtype)

    baseline_map_for_eval = load_baseline_label_map(config.phase2_main_eval_dir)
    eval_id_set = set(int(x) for x in baseline_map_for_eval.keys())
    track_a_rows = [r for r in rows if int(r.id) in eval_id_set]
    eval_examples = prepare_examples(track_a_rows, tokenizer, max_length=config.max_length)

    track_a_summary = run_track_a(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_examples=eval_examples,
        tau_grid=config.tau_grid,
        beta=config.beta,
        max_new_tokens=config.max_new_tokens,
        phase2_main_eval_dir=config.phase2_main_eval_dir,
        out_dir=config.ag_out_dir,
    )

    # Track B must match traces by exact example id from Phase 1 output,
    # so map all dataset rows, not only fixed 0..499 split.
    examples_by_id = {x.id: x for x in rows}
    track_b_summary = run_track_b(
        tokenizer=tokenizer,
        examples_by_id=examples_by_id,
        traces_path=config.phase1_traces_path,
        out_dir=config.track_b_out_dir,
    )

    margin_upgrade_summary = None
    if config.run_margin_upgrade and track_b_summary["verdict"] == "MARGIN_BETTER":
        baseline_map = load_baseline_label_map(config.phase2_main_eval_dir)
        margin_upgrade_summary = _run_margin_gated_upgrade(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_examples=eval_examples,
            baseline_map=baseline_map,
            tau_margin_grid=config.tau_margin_grid,
            beta=config.beta,
            max_new_tokens=config.max_new_tokens,
            out_dir=config.margin_out_dir,
        )

    final_summary = {
        "track_a": track_a_summary,
        "track_b": track_b_summary,
        "margin_upgrade": margin_upgrade_summary,
    }
    write_json(config.ag_out_dir / "ag_erha_final_summary.json", final_summary)
    return final_summary
