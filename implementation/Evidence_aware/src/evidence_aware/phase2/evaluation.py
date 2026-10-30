from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from evidence_aware.phase2.common import EvalRun, PreparedExample, classify_answer_token, first_non_whitespace_from_sequence, write_json, write_jsonl
from evidence_aware.phase2.modeling import EvidenceHeadAmplifier, decode_dola, decode_greedy, to_tensors


def evaluate_mode(
    model,
    tokenizer,
    device: torch.device,
    examples: list[PreparedExample],
    max_new_tokens: int,
    mode_name: str,
    decode_mode: str,
    target_heads: dict[int, list[int]] | None,
    beta: float | None,
    mature_layer: int,
    premature_layers: list[int],
) -> EvalRun:
    if decode_mode not in {"baseline", "erha", "dola"}:
        raise ValueError(f"Unknown decode mode: {decode_mode}")

    amplifier: EvidenceHeadAmplifier | None = None
    if decode_mode == "erha":
        if beta is None or target_heads is None:
            raise ValueError("ERHA mode requires beta and target_heads.")
        amplifier = EvidenceHeadAmplifier(model=model, target_heads=target_heads, beta=beta)
        amplifier.attach()

    labels_by_id: dict[int, str] = {}
    first_token_by_id: dict[int, int | None] = {}
    per_example_rows: list[dict[str, Any]] = []
    elapsed_total = 0.0
    generated_total = 0
    try:
        for ex in examples:
            input_ids, attention_mask = to_tensors(ex, device=device)
            if decode_mode in {"baseline", "erha"}:
                gen_ids, elapsed, generated_tokens = decode_greedy(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            else:
                gen_ids, elapsed, generated_tokens = decode_dola(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    mature_layer=mature_layer,
                    premature_layers=premature_layers,
                )
            elapsed_total += float(elapsed)
            generated_total += int(generated_tokens)
            first_token_id = first_non_whitespace_from_sequence(gen_ids, tokenizer)
            label = classify_answer_token(first_token_id, ex.evidence_token_id, ex.parametric_token_id)
            labels_by_id[ex.id] = label
            first_token_by_id[ex.id] = first_token_id
            per_example_rows.append(
                {
                    "id": ex.id,
                    "mode": mode_name,
                    "beta": beta,
                    "label": label,
                    "first_answer_token_id": first_token_id,
                    "generated_ids": gen_ids,
                    "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
                    "elapsed_ms": elapsed * 1000.0,
                    "generated_tokens": generated_tokens,
                }
            )
    finally:
        if amplifier is not None:
            amplifier.detach()

    return EvalRun(
        mode=mode_name,
        beta=beta,
        labels_by_id=labels_by_id,
        first_token_by_id=first_token_by_id,
        elapsed_sec_total=elapsed_total,
        generated_tokens_total=generated_total,
        per_example_rows=per_example_rows,
    )


def summarize_mode_metrics(run: EvalRun, baseline_run: EvalRun, mode_name: str) -> dict[str, Any]:
    ids = sorted(set(baseline_run.labels_by_id.keys()) & set(run.labels_by_id.keys()))
    n_total = len(ids)
    if n_total == 0:
        raise RuntimeError(f"No overlapping ids between baseline and mode {mode_name}.")

    baseline_hall = {idx for idx in ids if baseline_run.labels_by_id[idx] == "hallucinated"}
    baseline_faithful = {idx for idx in ids if baseline_run.labels_by_id[idx] == "faithful"}
    current_faithful = {idx for idx in ids if run.labels_by_id[idx] == "faithful"}
    current_hall = {idx for idx in ids if run.labels_by_id[idx] == "hallucinated"}
    current_other = {idx for idx in ids if run.labels_by_id[idx] == "other"}

    n_baseline_hall = len(baseline_hall)
    n_baseline_faithful = len(baseline_faithful)
    restored = len([idx for idx in baseline_hall if run.labels_by_id[idx] == "faithful"])
    unresolved = len([idx for idx in baseline_hall if run.labels_by_id[idx] == "hallucinated"])
    baseline_hall_to_other = len([idx for idx in baseline_hall if run.labels_by_id[idx] == "other"])
    stable = len([idx for idx in baseline_faithful if run.labels_by_id[idx] == "faithful"])
    harmed = len([idx for idx in baseline_faithful if run.labels_by_id[idx] == "hallucinated"])
    baseline_faithful_to_other = len([idx for idx in baseline_faithful if run.labels_by_id[idx] == "other"])

    if mode_name == "baseline":
        restoration_rate = 0.0
        stability_rate = 1.0
        harm_rate = 0.0
        restoration_improvement = 0.0
    else:
        restoration_rate = float(restored / n_baseline_hall) if n_baseline_hall > 0 else 0.0
        stability_rate = float(stable / n_baseline_faithful) if n_baseline_faithful > 0 else 0.0
        harm_rate = float(harmed / n_baseline_faithful) if n_baseline_faithful > 0 else 0.0
        restoration_improvement = restoration_rate - harm_rate

    accuracy = float(len(current_faithful) / n_total)
    mean_ms_per_token = (
        float(run.elapsed_sec_total * 1000.0 / run.generated_tokens_total) if run.generated_tokens_total > 0 else 0.0
    )
    baseline_ms_per_token = (
        float(baseline_run.elapsed_sec_total * 1000.0 / baseline_run.generated_tokens_total)
        if baseline_run.generated_tokens_total > 0
        else 0.0
    )
    overhead_percent = (
        float((mean_ms_per_token - baseline_ms_per_token) / baseline_ms_per_token * 100.0)
        if baseline_ms_per_token > 0
        else 0.0
    )

    return {
        "mode": mode_name,
        "beta": run.beta,
        "n_total": n_total,
        "accuracy": accuracy,
        "restoration_rate": restoration_rate,
        "stability_rate": stability_rate,
        "harm_rate": harm_rate,
        "restoration_improvement": restoration_improvement,
        "n_restored": restored,
        "n_harmed": harmed,
        "n_stable": stable,
        "n_unresolved": unresolved,
        "n_baseline_hall_to_other": baseline_hall_to_other,
        "n_baseline_faithful_to_other": baseline_faithful_to_other,
        "n_current_faithful": len(current_faithful),
        "n_current_hallucinated": len(current_hall),
        "n_current_other": len(current_other),
        "n_baseline_hallucinated": n_baseline_hall,
        "n_baseline_faithful": n_baseline_faithful,
        "latency": {
            "mean_ms_per_token_baseline": baseline_ms_per_token,
            "mean_ms_per_token_mode": mean_ms_per_token,
            "overhead_percent": overhead_percent,
            "elapsed_sec_total": run.elapsed_sec_total,
            "generated_tokens_total": run.generated_tokens_total,
        },
    }


def write_mode_outputs(main_eval_dir: Path, run: EvalRun, metrics: dict[str, Any]) -> None:
    if run.mode == "baseline":
        file_name = "baseline_results.json"
    elif run.mode.startswith("erha_beta_"):
        beta = run.mode.split("erha_beta_", maxsplit=1)[1]
        file_name = f"erha_results_beta_{beta}.json"
    elif run.mode == "dola":
        file_name = "dola_results.json"
    else:
        file_name = f"{run.mode.replace('/', '_')}_results.json"
    write_json(main_eval_dir / file_name, metrics)
    write_jsonl(main_eval_dir / f"{run.mode}_examples.jsonl", run.per_example_rows)


def build_comparison_table(
    baseline_metrics: dict[str, Any],
    erha_metrics_rows: list[dict[str, Any]],
    dola_metrics: dict[str, Any] | None,
    sanity_checks: dict[str, Any],
    target_heads: dict[int, list[int]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [baseline_metrics]
    rows.extend(sorted(erha_metrics_rows, key=lambda x: (float(x["beta"]) if x.get("beta") is not None else 999.0)))
    if dola_metrics is not None:
        rows.append(dola_metrics)
    return {"target_heads": target_heads, "rows": rows, "sanity_checks": sanity_checks}


def select_best_erha(erha_metrics_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not erha_metrics_rows:
        return None
    candidates = [row for row in erha_metrics_rows if row.get("beta") is not None and float(row["beta"]) >= 1.5]
    if not candidates:
        candidates = erha_metrics_rows
    return max(
        candidates,
        key=lambda r: (
            float(r.get("restoration_improvement", 0.0)),
            float(r.get("accuracy", 0.0)),
            float(r.get("stability_rate", 0.0)),
        ),
    )


def run_sanity_checks(baseline_run: EvalRun, erha_runs: list[EvalRun], target_heads: dict[int, list[int]]) -> dict[str, Any]:
    beta_one_run = next((run for run in erha_runs if run.beta is not None and abs(float(run.beta) - 1.0) < 1e-9), None)
    out = {"has_beta_one_run": beta_one_run is not None, "beta_one_identical_to_baseline": None, "target_heads_non_empty": bool(target_heads)}
    if beta_one_run is not None:
        same_labels = all(baseline_run.labels_by_id.get(k) == beta_one_run.labels_by_id.get(k) for k in baseline_run.labels_by_id)
        same_first_tokens = all(
            baseline_run.first_token_by_id.get(k) == beta_one_run.first_token_by_id.get(k)
            for k in baseline_run.first_token_by_id
        )
        out["beta_one_identical_to_baseline"] = bool(same_labels and same_first_tokens)
    return out


def update_target_heads_from_validation(base_target_heads: dict[int, list[int]], validation_summary: dict[str, Any]) -> dict[int, list[int]]:
    out = {int(layer): sorted(set(int(h) for h in heads)) for layer, heads in base_target_heads.items()}
    if not validation_summary.get("stable", False):
        return out
    for row in validation_summary.get("top5_validation_faithful", []):
        layer = int(row["layer"])
        head = int(row["head"])
        value = float(row["value"])
        if value < 0.10:
            continue
        out.setdefault(layer, [])
        out[layer].append(head)
        out[layer] = sorted(set(out[layer]))
    return out


def run_hook_smoke_test(
    model,
    tokenizer,
    device: torch.device,
    examples: list[PreparedExample],
    target_heads: dict[int, list[int]],
    beta: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    changed_logits = 0
    changed_top1 = 0
    non_empty_outputs = 0
    rows: list[dict[str, Any]] = []
    amp = EvidenceHeadAmplifier(model=model, target_heads=target_heads, beta=beta)
    for ex in examples:
        input_ids, attention_mask = to_tensors(ex, device=device)
        with torch.inference_mode():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits_base = out_base.logits[0, -1, :].float()
        top1_base = int(torch.argmax(logits_base).item())
        amp.attach()
        try:
            with torch.inference_mode():
                out_amp = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits_amp = out_amp.logits[0, -1, :].float()
        finally:
            amp.detach()
        top1_amp = int(torch.argmax(logits_amp).item())
        l2 = float(torch.linalg.vector_norm(logits_amp - logits_base, ord=2).item())
        if l2 > 1e-6:
            changed_logits += 1
        if top1_amp != top1_base:
            changed_top1 += 1
        amp.attach()
        try:
            gen_ids, _, _ = decode_greedy(
                model,
                tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        finally:
            amp.detach()
        if gen_ids:
            non_empty_outputs += 1
        rows.append({"id": ex.id, "top1_base": top1_base, "top1_amp": top1_amp, "logit_l2_delta": l2, "generated_len": len(gen_ids)})
    return {"n_examples": len(examples), "changed_logits_count": changed_logits, "changed_top1_count": changed_top1, "non_empty_outputs_count": non_empty_outputs, "rows": rows}


def choose_random_head_control(model, excluded_heads: dict[int, list[int]], count: int, seed: int) -> dict[int, list[int]]:
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    excluded = {(int(l), int(h)) for l, hs in excluded_heads.items() for h in hs}
    candidates = [(l, h) for l in range(num_layers) for h in range(num_heads) if (l, h) not in excluded]
    if len(candidates) < count:
        raise RuntimeError("Not enough candidate random heads for control.")
    rng = random.Random(seed + 19)
    chosen = rng.sample(candidates, k=count)
    out: dict[int, list[int]] = {}
    for layer, head in chosen:
        out.setdefault(layer, [])
        out[layer].append(head)
    for layer in out:
        out[layer] = sorted(set(out[layer]))
    return out


def run_ablation(
    model,
    tokenizer,
    device: torch.device,
    examples: list[PreparedExample],
    max_new_tokens: int,
    baseline_run: EvalRun,
    best_beta: float,
    target_heads: dict[int, list[int]],
    random_control_heads_count: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_configs: dict[str, dict[int, list[int]]] = {
        "L27H20_only": {27: [20]},
        "L24H3_only": {24: [3]},
        "L24H4_only": {24: [4]},
        "L24H3_4_only": {24: [3, 4]},
        "all_three": {24: [3, 4], 27: [20]},
    }

    rows: list[dict[str, Any]] = []
    for name, heads in ablation_configs.items():
        run = evaluate_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=examples,
            max_new_tokens=max_new_tokens,
            mode_name=f"ablation_{name}",
            decode_mode="erha",
            target_heads=heads,
            beta=best_beta,
            mature_layer=31,
            premature_layers=[14, 16, 18, 20],
        )
        rows.append(summarize_mode_metrics(run=run, baseline_run=baseline_run, mode_name=name))

    random_heads = choose_random_head_control(
        model=model,
        excluded_heads=target_heads,
        count=random_control_heads_count,
        seed=seed,
    )
    random_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=examples,
        max_new_tokens=max_new_tokens,
        mode_name="ablation_random_head_control",
        decode_mode="erha",
        target_heads=random_heads,
        beta=best_beta,
        mature_layer=31,
        premature_layers=[14, 16, 18, 20],
    )
    random_metrics = summarize_mode_metrics(run=random_run, baseline_run=baseline_run, mode_name="random_head_control")
    random_metrics["random_heads"] = random_heads

    table = {
        "best_beta": best_beta,
        "ablation_by_head": rows,
        "random_head_control": random_metrics,
    }
    write_json(out_dir / "ablation_by_head.json", rows)
    write_json(out_dir / "random_head_control.json", random_metrics)
    write_json(out_dir / "ablation_table.json", table)
    return table


def create_plots(
    out_dir: Path,
    baseline_metrics: dict[str, Any],
    erha_metrics_rows: list[dict[str, Any]],
    dola_metrics: dict[str, Any] | None,
    best_erha: dict[str, Any] | None,
    ablation_table: dict[str, Any] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted([r for r in erha_metrics_rows if r.get("beta") is not None], key=lambda x: float(x["beta"]))
    betas = [1.0] + [float(r["beta"]) for r in rows_sorted if float(r["beta"]) != 1.0]
    metric_map = {float(r["beta"]): r for r in rows_sorted}
    metric_map[1.0] = baseline_metrics

    def line_plot(metric_key: str, name: str, ylabel: str, title: str) -> None:
        ys = [float(metric_map[b].get(metric_key, 0.0)) for b in betas]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(betas, ys, marker="o", linewidth=2.0)
        ax.set_xlabel("beta")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=220)
        plt.close(fig)

    line_plot("accuracy", "accuracy_vs_beta.png", "accuracy", "Accuracy vs beta")
    line_plot("restoration_rate", "restoration_vs_beta.png", "restoration rate", "Restoration vs beta")
    line_plot("stability_rate", "stability_vs_beta.png", "stability rate", "Stability vs beta")
    line_plot("harm_rate", "harm_vs_beta.png", "harm rate", "Harm vs beta")
    line_plot("restoration_improvement", "restoration_improvement_vs_beta.png", "restoration improvement", "Restoration Improvement vs beta")

    comparison = [("baseline", baseline_metrics)]
    if best_erha is not None:
        comparison.append((f"best_erha_beta_{best_erha['beta']}", best_erha))
    if dola_metrics is not None:
        comparison.append(("dola", dola_metrics))
    labels = [x[0] for x in comparison]
    xs = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(xs - width, [float(x[1].get("accuracy", 0.0)) for x in comparison], width, label="accuracy")
    ax.bar(xs, [float(x[1].get("restoration_rate", 0.0)) for x in comparison], width, label="restoration")
    ax.bar(xs + width, [float(x[1].get("stability_rate", 0.0)) for x in comparison], width, label="stability")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Baseline vs Best ERHA vs DoLa")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bar_chart.png", dpi=220)
    plt.close(fig)

    if ablation_table is not None:
        rows = list(ablation_table.get("ablation_by_head", []))
        rnd = ablation_table.get("random_head_control")
        if rnd is not None:
            rows.append(rnd)
        labels = [str(r.get("mode", "unknown")) for r in rows]
        vals = [float(r.get("restoration_improvement", 0.0)) for r in rows]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("restoration improvement")
        ax.set_title("Ablation by Head Configuration")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_bar_chart.png", dpi=220)
        plt.close(fig)

    lat_rows = [("baseline", baseline_metrics)]
    if best_erha is not None:
        lat_rows.append((f"best_erha_beta_{best_erha['beta']}", best_erha))
    if dola_metrics is not None:
        lat_rows.append(("dola", dola_metrics))
    labels = [x[0] for x in lat_rows]
    ms = [float(x[1]["latency"]["mean_ms_per_token_mode"]) for x in lat_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(np.arange(len(labels)), ms)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean ms/token")
    ax.set_title("Latency Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=220)
    plt.close(fig)
