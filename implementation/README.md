# Implementation guide

The three sibling directories retain the layout used in the research code. Several scripts dynamically import other numbered scripts, so renaming files or flattening these directories changes their dependencies. `tools/run.py` supplies the correct working directory and source paths.

| Package | Main responsibilities |
|---|---|
| `Evidence_aware/src/evidence_aware` | CounterFact normalization, model loading, per-head decomposition, margin gates, OV steering and gain checks |
| `medrag/src/medrag` | MedQA prompt construction, token sets, activation caching, gated interventions, causal patching, cross-model evaluation, statistics and selective prediction |
| `conflictqa/src/conflictqa_pipeline` | Entity normalization/matching, chat prompts, token sets, PopQA preparation, promotion and suppression |

`medrag` is the authors' local package name, not a redistributed copy of an external MedRAG benchmark. Third-party datasets and frameworks are not included. Some older helpers remain because later scripts import them; use the release profiles and paper-to-code map to choose the intended experiment.

## Mechanics

The prefill margin compares token-set probability mass for known evidence and parametric answers. It is evaluated at the answer position, and the configured threshold gates the intervention. The Llama CounterFact path writes through the selected OV contribution; MedQA and cross-model alternatives use the explicitly selected implementation. PopQA balances first-token alias promotion and parametric suppression over the configured initial decoding steps. This does not teacher-force a whole entity string.

Head attribution, causal patching and output correction are distinct endpoints. The matched-control experiment uses selected five-head patching, random same-layer and late-layer draws, attribution-matched heads and bottom heads. Its disruption outcomes form a mutually exclusive partition. The zero-third-letter pattern does not alone identify special decision-carrying heads.

## Corrections and release-only changes

The imported sources are listed with their original hashes in `docs/source_manifest.json`. The following changes are explicit:

Line endings and trailing blank lines are normalized for Git/Linux launchers; the source manifest hashes refer to the original source files before normalization.

1. A portable runner replaces cluster-specific launch assumptions. No old private `sbatch` files are published.
2. `01_phase1_diagnostics.resolve_cache_path` fails on a missing explicit cache instead of searching an unrelated `hallucination_clean` directory.
3. The MedQA kill-test's prompt restoration and paired comparison helper now selects `in_clean_subset`. Paired comparisons require nonempty identical unique IDs and record their population and paired count. This fixes the historical clean-vs-matched reporting ambiguity without changing model inference.
4. NumPy is imported lazily inside the bootstrap-statistics function, allowing exact-binomial and Wilson tests without a GPU environment.
5. Frozen selection metadata is stripped of question text and personal filesystem metadata. It contains only IDs/statuses and scalar configurations.
6. `tools/audit_medqa.py` provides an independent per-item reporting path for clean and full matched comparisons; C5's restoration baseline is C2. It does not label a baseline-correct item as part of a restoration denominator.
7. The archived Qwen MedQA metadata omits an intervention implementation field. The source evaluator `evaluate_cross_model_intervention` discards `site_mode` and applies token-logit bias. The release therefore names `token_logit_bias` explicitly rather than inferring a residual hook from the text "layer=24, residual-stream".

No activation tensor, checkpoint, benchmark row or generated output is included in the release. `configs/experiments.json` includes recorded historical model profiles and new preparation/analysis profiles. A profile's `scope` field explains whether it preserves archived CLI settings or starts a regenerated method run.

## Data preparation

Read [the data guide](../docs/DATA.md). All normalization/builders live under the corresponding `scripts/` directory. Exact paper inputs require the original prepared data and hashes; source-data regeneration can change candidate filtering, evidence text, order and resulting IDs. The release deliberately refuses to claim that any arbitrary fresh sample equals the paper's sample.

Model-specific token sets must be constructed with the active tokenizer. A Llama vocabulary index is not a reusable Qwen or Mistral answer identity. Keep answer strings and retokenize at inference; do not uppercase CounterFact entity names as if they were MedQA A-D labels.

## Reproducibility boundaries

- Frozen MedQA audit/selection IDs are supplied as small configuration metadata; data rows are acquired separately.
- Localization, pilot selection, baseline rows and causal pairs are outputs regenerated by predecessor scripts. The causal controls require the same fixed pair set, not newly sampled pairs per arm.
- The 2,000-item Llama CounterFact evaluation and the 500-item Mistral/Qwen CounterFact evaluations are separate populations.
- `81` and `82` intentionally verify the frozen 2,000-row OV population and label distribution. They fail for a different population instead of silently weakening the check.
- The release does not claim a recovered historical package lock or GPU rerun. Save the environment for each new run and compare input hashes and sample counts before comparing numbers.
