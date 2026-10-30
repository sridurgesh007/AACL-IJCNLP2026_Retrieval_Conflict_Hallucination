# Paper-to-code map

Paths below are relative to `implementation/`. All outputs are regenerated locally and ignored by Git.

| Paper component | Source / entry point | Required predecessor inputs |
|---|---|---|
| CounterFact head decomposition and margin trajectory | `Evidence_aware/scripts/01_phase1_diagnostics.py` | normalized text cache, active model |
| CounterFact OV steering and gain diagnostics | `Evidence_aware/scripts/09_run_true_multi_head.py`; `true_multi_head/pipeline.py` | phase-1 traces and phase-2 baseline rows |
| MedQA controlled input construction | `medrag/scripts/22_build_medqa_conflict.py` | upstream MedQA and active model |
| MedQA matched subset / audit | `29_preflight_medqa_v3.py`; `medrag/matched_subset.py` | prepared conflicts; manual audit or frozen labels |
| Content-vs-letter / promotion-vs-suppression | `30_run_medqa_decomposition.py`, `31_run_medqa_pr_sweep.py` | matched IDs, audit, active model |
| MedQA prompt kill-test | `32_run_medqa_killtest.py`; `tools/audit_medqa.py` at repository root | matched and clean IDs, selected scalar settings |
| H1-H4 headline verification | `34_run_medqa_balanced_headline.py` | same prepared MedQA and selected gate |
| Localization and single-head causal checks | `43_run_medqa_localization.py`, `40_run_causal_patching.py` | H4 per-item rows and clean inputs |
| Joint top-five patching | `47_run_medqa_joint_patching.py` | fixed pairs, head importance and baseline control outputs |
| Matched random / attribution / disruption controls | `83_run_matched_head_controls.py` | identical fixed pairs, head scores and source inputs |
| PopQA confidence filtering / C2 conflicts | `conflictqa/scripts/62_build_popqa_conflict.py`, `67_rebuild_popqa_c2_conflicts.py` | official raw data; model-specific generation |
| Balanced alias promotion and suppression | `conflictqa/scripts/63_run_popqa_intervention.py` | prepared model-specific conflicts; aliases |
| PopQA stratification and plots | `64_plot_popqa_results.py`, `65_analyze_popqa_results.py` | regenerated per-item rows and summaries |
| Cross-model MedQA pilots / full / prompt controls | `67`, `68`, `74` scripts in `medrag/scripts/` | model-specific localization, pilot and shared prepared inputs |
| Cross-model CounterFact v2 | `70`, `71`, `72`, `76` scripts | relation-aware text cache, active tokenizer, pilot |
| Full CounterFact belief verification | `81_run_premise_check_full.py`, `82_run_stratification_full.py` | frozen 2,000-ID OV rows, text cache; strict semantic joins |
| Subthreshold additivity / dose response | `33_run_medqa_decomposition_subthreshold.py`, `paired_bootstrap_subthreshold.py`, `plot_dose_response_v4.py` | per-cell outputs on identical IDs |
| Selective prediction | `63_run_selective_prediction.py` | final MedQA/PopQA outputs and self-consistency sampling |
| Deployment cost | `62_run_deployment_characterization.py` | model summaries, matching gate/intervention sites |
| ClashEval scope study | `56`, `57`, `59`, `60` scripts | external upstream data/response file |

## Keep these result populations distinct

- Llama CounterFact OV evaluation: 2,000 items; restoration denominator is baseline hallucinated items, not 2,000.
- Cross-model CounterFact: 500 input items, with model-specific baseline hallucination denominators.
- MedQA: 473 matched IDs; clean headline restoration on 40 items; larger baseline-correct harm controls.
- Matched causal patching: 35 source-destination pairs, not the MedQA 40-item restoration population.
- PopQA: Llama full input 3,141, with model-specific genuine conflict and faithful-control populations.

The repository's exact statistical audit prints the populations rather than relying on captions or rounded percentages. The matched-control disruption and joint matched-source paths use different pair sets.
