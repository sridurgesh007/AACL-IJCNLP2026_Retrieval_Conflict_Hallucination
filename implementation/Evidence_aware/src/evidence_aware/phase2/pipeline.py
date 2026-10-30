from __future__ import annotations

from typing import Any

from evidence_aware.phase2.common import Phase2Config, ensure_dirs, seed_everything, write_json
from evidence_aware.phase2.data import leak_check_or_raise, load_examples, load_phase1_example_ids, make_disjoint_splits, prepare_examples
from evidence_aware.phase2.evaluation import (
    build_comparison_table,
    create_plots,
    evaluate_mode,
    run_ablation,
    run_hook_smoke_test,
    run_sanity_checks,
    select_best_erha,
    summarize_mode_metrics,
    update_target_heads_from_validation,
    write_mode_outputs,
)
from evidence_aware.phase2.modeling import load_model_and_tokenizer
from evidence_aware.phase2.validation import run_validation


def run_phase2_pipeline(config: Phase2Config) -> dict[str, Any]:
    seed_everything(config.seed)
    ensure_dirs(config.out_dir)

    examples = load_examples(config.cache_path)
    phase1_ids = load_phase1_example_ids(config.phase1_traces_path)
    validation_rows, eval_rows = make_disjoint_splits(
        examples=examples,
        phase1_ids=phase1_ids,
        validation_size=config.validation_size,
        eval_size=config.eval_size,
        seed=config.seed,
    )
    leak_check_or_raise(phase1_ids, validation_rows, eval_rows)

    model, tokenizer, device = load_model_and_tokenizer(
        model_id=config.model_id,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )

    validation_examples = prepare_examples(validation_rows, tokenizer, max_length=config.max_length)
    eval_examples = prepare_examples(eval_rows, tokenizer, max_length=config.max_length)
    if len(validation_examples) < config.validation_size:
        raise RuntimeError(
            f"Validation set too small after preprocessing: got={len(validation_examples)}, required={config.validation_size}"
        )
    if len(eval_examples) < config.min_eval_examples:
        raise RuntimeError(
            f"Eval set too small after preprocessing: got={len(eval_examples)}, required>={config.min_eval_examples}"
        )

    validation_summary = run_validation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=validation_examples[: config.validation_size],
        phase1_aggregate_path=config.phase1_aggregate_path,
        out_dir=config.out_dir / "validation",
        root_out_dir=config.out_dir,
    )
    if not validation_summary["stable"] and not config.allow_unstable_validation:
        raise RuntimeError(
            "Validation verdict is UNSTABLE. Stopping before intervention as required by phase2.md. "
            "Use --allow_unstable_validation 1 to override."
        )

    target_heads = update_target_heads_from_validation(config.target_heads, validation_summary)
    smoke_summary = run_hook_smoke_test(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples[: max(1, config.smoke_examples)],
        target_heads=target_heads,
        beta=2.0,
        max_new_tokens=min(8, config.max_new_tokens),
    )
    write_json(config.out_dir / "main_eval" / "hook_smoke_test.json", smoke_summary)

    baseline_run = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=eval_examples,
        max_new_tokens=config.max_new_tokens,
        mode_name="baseline",
        decode_mode="baseline",
        target_heads=None,
        beta=None,
        mature_layer=config.mature_layer,
        premature_layers=config.premature_layers,
    )
    baseline_metrics = summarize_mode_metrics(run=baseline_run, baseline_run=baseline_run, mode_name="baseline")
    write_mode_outputs(config.out_dir / "main_eval", baseline_run, baseline_metrics)

    betas = sorted(set(config.beta_values))
    if config.include_beta_one_sanity and 1.0 not in betas:
        betas = sorted(betas + [1.0])

    erha_runs = []
    erha_metrics_rows = []
    for beta in betas:
        run = evaluate_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=eval_examples,
            max_new_tokens=config.max_new_tokens,
            mode_name=f"erha_beta_{beta}",
            decode_mode="erha",
            target_heads=target_heads,
            beta=beta,
            mature_layer=config.mature_layer,
            premature_layers=config.premature_layers,
        )
        erha_runs.append(run)
        metrics = summarize_mode_metrics(run=run, baseline_run=baseline_run, mode_name=run.mode)
        erha_metrics_rows.append(metrics)
        write_mode_outputs(config.out_dir / "main_eval", run, metrics)

    dola_run = None
    dola_metrics = None
    if config.run_dola:
        dola_run = evaluate_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=eval_examples,
            max_new_tokens=config.max_new_tokens,
            mode_name="dola",
            decode_mode="dola",
            target_heads=None,
            beta=None,
            mature_layer=config.mature_layer,
            premature_layers=config.premature_layers,
        )
        dola_metrics = summarize_mode_metrics(run=dola_run, baseline_run=baseline_run, mode_name="dola")
        write_mode_outputs(config.out_dir / "main_eval", dola_run, dola_metrics)

    sanity_checks = run_sanity_checks(
        baseline_run=baseline_run,
        erha_runs=erha_runs,
        target_heads=target_heads,
    )
    comparison_table = build_comparison_table(
        baseline_metrics=baseline_metrics,
        erha_metrics_rows=erha_metrics_rows,
        dola_metrics=dola_metrics,
        sanity_checks=sanity_checks,
        target_heads=target_heads,
    )
    write_json(config.out_dir / "main_eval" / "comparison_table.json", comparison_table)

    best_erha = select_best_erha(erha_metrics_rows)
    ablation_table = None
    if config.run_ablation and best_erha is not None:
        ablation_table = run_ablation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=eval_examples,
            max_new_tokens=config.max_new_tokens,
            baseline_run=baseline_run,
            best_beta=float(best_erha["beta"]),
            target_heads=target_heads,
            random_control_heads_count=config.random_control_heads_count,
            seed=config.seed,
            out_dir=config.out_dir / "ablation",
        )

    create_plots(
        out_dir=config.out_dir / "plots",
        baseline_metrics=baseline_metrics,
        erha_metrics_rows=erha_metrics_rows,
        dola_metrics=dola_metrics,
        best_erha=best_erha,
        ablation_table=ablation_table,
    )

    final_summary = {
        "model_id": config.model_id,
        "num_phase1_ids": len(phase1_ids),
        "num_validation_examples": len(validation_examples[: config.validation_size]),
        "num_eval_examples": len(eval_examples),
        "validation_summary": validation_summary,
        "target_heads_used": target_heads,
        "hook_smoke_test": smoke_summary,
        "baseline_metrics": baseline_metrics,
        "erha_metrics_rows": erha_metrics_rows,
        "dola_metrics": dola_metrics,
        "best_erha": best_erha,
        "sanity_checks": sanity_checks,
        "ablation_table": ablation_table,
    }
    write_json(config.out_dir / "final_summary.json", final_summary)
    return final_summary
