# Reproduction runbook

All commands below are run from the repository root. `tools/run.py` fixes cwd and source paths. Its default is a dry run: it prints the exact command. `--execute` checks declared inputs and starts the script. Paths inside profiles resolve relative to `implementation/medrag`, `implementation/Evidence_aware` or `implementation/conflictqa`, as printed by the runner. Absolute overrides are also supported.

## 1. Verify the code-only checkout

```bash
python tools/check_release.py
python -m unittest discover -s tests -v
python tools/run.py --list
```

These checks require no dataset, model weights or CUDA. They validate Python syntax, profile CLI flags, frozen selection counts, statistical helpers, duplicate-ID failures and disruption outcome accounting. They do not establish GPU numerical reproducibility.

## 2. Prepare or import data

Follow [DATA.md](DATA.md). For exact paper samples, import the original prepared files and enforce recorded SHA256/counts where available. For newly generated data, run the author builders and retain a separate audit. Exact frozen metadata must not be applied to different questions or contexts.

## 3. MedQA headline and dose-response

For the original prepared MedQA corpus:

```bash
python tools/bootstrap.py
python tools/run.py medqa.decomposition --execute
python tools/run.py medqa.sweep --execute
python tools/run.py medqa.killtest --execute
python tools/run.py medqa.headline --execute
python tools/run.py medqa.subthreshold --execute
python tools/audit_medqa.py \
  --results-dir implementation/medrag/results/medqa_final_v3/killtest \
  --output implementation/medrag/results/medqa_final_v3/killtest/audited_comparison.json
```

The scalar frozen settings installed by bootstrap are the published `E_letter`, tau=2.0, lambda_pos=0.0, lambda_neg=2.5 operating point. The decomposition/sweep regenerate calibration outputs and may select a different point in a different environment. Inspect and report that result rather than relabelling it as the historical setting. To apply the frozen point after calibration without silently overwriting results, keep a separate copy of the original scalar configs or run the headline profile with explicit overrides. Bootstrap refuses to replace different existing metadata.

For a new corpus, run `medqa.preflight` first, complete the label audit, and use the new matched/audited IDs. The historical source's preflight template is not an automatic gold audit. The source also contains scripts for optional ITI controls (`51_validate_iti_truthfulqa.py`, `52_run_medqa_iti.py`); no TruthfulQA corpus or external probe checkpoint is bundled.

The clean-subset restoration and matched baseline-correct harm populations differ. Use `tools/audit_medqa.py`, not a rounded number copied from an old plot. It reports clean and matched C4-vs-C5 comparisons separately. C5 restoration is against C2; a shared percentage denominator must not be assumed.

## 4. Causal localization and matched controls

After `medqa.headline` has produced H4 per-item rows:

```bash
python tools/run.py medqa.localize --execute
python tools/run.py medqa.pairs --execute
python tools/run.py medqa.matched-controls --execute
```

The controls read the fixed `patching_pairs.json` and head scores. The script verifies the frozen 35-pair design and checks the selected-five 31/35 replication before interpreting other arms. The disruption pair set differs from the matched-source pair set and is saved separately. Outcomes use exclusive unchanged/source/destination/third categories. The corrected disruption result may differ from the legacy 22/35 expectation; the script records the discrepancy instead of tuning toward the old result.

Do not generate a new random pair set independently for each arm. Attribution alone is not the causal endpoint. Joint patching, matched controls and exact paired tests must use identical destination IDs.

## 5. Llama CounterFact OV steering (2,000 evaluation items)

```bash
python tools/run.py counterfact.prepare --execute
python tools/run.py counterfact.trace --execute
python tools/run.py counterfact.baseline --execute
python tools/run.py counterfact.ov --execute
```

The baseline stage creates disjoint trace, validation and evaluation populations and produces baseline rows needed by OV steering. Preserve the full normalized cache. The OV profile uses seed 42 as in this historical pipeline; do not substitute MedQA's seed 1234. The seed/source revision/population must be verified against the original run before comparing the exact 17.2% point.

The pipeline exports gate statistics and gain diagnostics. For the full-set belief audit, point the following profiles at the actual regenerated **non-pilot** gate-statistics file:

```bash
python tools/run.py counterfact.belief \
  --set ov_results_path=../Evidence_aware/results/true_multi_head/gate_stats/ACTUAL_FULL_FILE.jsonl --execute
python tools/run.py counterfact.stratify \
  --set ov_results_path=../Evidence_aware/results/true_multi_head/gate_stats/ACTUAL_FULL_FILE.jsonl \
  --set gate_stats_dir=../Evidence_aware/results/true_multi_head/gate_stats --execute
```

Replace `ACTUAL_FULL_FILE` with the filename printed by your run. The historical audit default points to `exp_b_corrected`; a new OV output root is not automatically the same directory. The strict audit expects 2,000 unique IDs and the frozen baseline-label counts; a mismatch is a stop for investigation, not permission to weaken the check. Baseline hallucination versus belief and restoration versus belief are different analyses.

## 6. PopQA balanced intervention and ablation

The published Llama full profile expects the prepared 3,141-item natural file:

```bash
python tools/run.py popqa.balanced --execute
python tools/run.py popqa.suppression --execute
python tools/run.py popqa.plot --execute
python tools/run.py popqa.analyze --execute
```

Promotion uses all correct aliases' first-token sets, and the suppression-only condition has a separate output directory. Do not overwrite either condition or interpret canonical-only promotion as the alias experiment. The `paper.mistral-popqa` and `paper.qwen-popqa` profiles preserve their historical supplementation/pilot parameters and model-specific input paths. Recreate those inputs with scripts `62`, `67` and `68`, or supply the frozen prepared input. The original Llama 3,141-item input is not automatically recovered by a 500-item builder run.

## 7. Mistral/Qwen cross-model runs

Use model-specific localization, then calibration, then full inference. The `paper.*` profiles expose the full-run archived CLI parameters. They may reference historically named pilot/localization directories; inspect the dry run and override the paths to your regenerated prerequisites.

```bash
python tools/run.py mistral.medqa-localize --execute
python tools/run.py mistral.medqa-pilot --execute
python tools/run.py paper.mistral-medqa \
  --set localization_path=results/cross_model/mistral_medqa_localization/head_importance.json \
  --set pilot_summary_path=results/cross_model/mistral_medqa_pilot/pilot/margin_auc.json --execute

python tools/run.py qwen.medqa-localize --execute
python tools/run.py qwen.medqa-pilot --execute
python tools/run.py paper.qwen-medqa \
  --set localization_path=results/cross_model/qwen_medqa_localization/layer_importance.json \
  --set pilot_summary_path=results/cross_model/qwen_medqa_pilot/pilot/margin_auc.json --execute
```

The Qwen wrapper emits layer aggregation as well as head scores. Do not use only the base head-localization script and then point a layer profile at a nonexistent `layer_importance.json`.

CounterFact:

```bash
python tools/run.py mistral.counterfact-localize --execute
python tools/run.py mistral.counterfact-pilot --execute
python tools/run.py paper.mistral-counterfact \
  --set localization_path=results/cross_model/mistral_counterfact_localization/head_importance.json \
  --set pilot_summary_path=results/cross_model/mistral_counterfact_v2_pilot/pilot/margin_auc.json --execute

python tools/run.py qwen.counterfact-localize --execute
python tools/run.py qwen.counterfact-pilot --execute
python tools/run.py paper.qwen-counterfact \
  --set localization_path=results/cross_model/qwen_counterfact_localization/head_importance.json \
  --set pilot_summary_path=results/cross_model/qwen_counterfact_v2_pilot/pilot/margin_auc.json --execute
```

These new calibrations are method reproductions, not a guarantee of the archived selected lambdas. Preserve the archived profile settings and report any difference. Pilot-go checks are intentional: do not force full runs from a missing or failed pilot file.

Prompt-only C4/C5 extensions are scripts `74_run_medqa_cross_model_c45.py` and `76_run_counterfact_cross_model_c45.py`. They write to a separate output directory and consume existing baseline rows. The CounterFact script uses entity evaluation, not the MedQA letter metric. Inspect their CLI contracts in `docs/cli_contracts.json` before supplying model-specific inputs; the two condition namespaces do not mean the same thing as C4/C5 in the MedQA kill-test.

## 8. Selective prediction, deployment and optional scope studies

```bash
python tools/run.py paper.selective              # inspect all required result paths
python tools/run.py paper.deployment
```

Override the MedQA/PopQA summary and site paths to the freshly generated final runs before `--execute`. In particular, the selective profile's historical default is the suppression-only PopQA file; choose the intended condition explicitly. Deployment also needs all three model summaries and the matching intervention-site files. The cost measurement assumes E/P candidates already exist; it does not time candidate extraction for an automatic RAG system.

ClashEval scripts `56`, `57`, `59` and `60` reproduce the optional scope analysis with a separately obtained input. Never invent replacement rows when upstream response files are unavailable.

## 9. Report what was actually reproduced

Keep source revision, input hashes and counts, model IDs/revisions, actual package versions, tokenizer identity, run arguments, selected hyperparameters, per-item outcomes and numerator/denominator summaries in local or external storage. Git holds code/configuration only. Hardware-dependent latency and nondeterministic numerical boundaries may differ. Do not claim every paper table was rerun when only syntax, CPU statistics or a pilot was tested.
