# Verification against historical runs

The profiles below were compared field by field with the `metadata.cli_args` stored in the actual local result archives. Each archive's SHA256 was also checked. This is a read-only settings audit, not a GPU experiment. The machine-readable evidence is `recorded_settings_audit.json`; applied historical scalar settings are in `configs/recorded_operating_points.json`.

| Release profile | Historical record | Result |
|---|---|---|
| `medqa.preflight` | `medqa_final_v3/preflight.json` | All recorded CLI arguments match |
| `medqa.killtest` | `medqa_final_v3/killtest/comparison.json` | All recorded CLI arguments match |
| `medqa.matched-controls` | `matched_head_controls_fixed_disruption_v2/summary.json` | All recorded CLI arguments match |
| `popqa.balanced` | `full_3141_balanced/summary.json` | All recorded CLI arguments match |
| `paper.selective` | `selective_prediction_20260521_v3/aurc_comparison.json` | All recorded CLI arguments match |
| `paper.deployment` | `deployment_20260521_v3/latency.json` | All recorded CLI arguments match |
| `paper.mistral-medqa` | `mistral_medqa_L29H2_tlb_full/summary.json` | All recorded CLI arguments match |
| `paper.qwen-medqa` | `qwen_intervention_20260515/summary.json` | All recorded CLI arguments match; explicit implementation annotation below |
| `paper.mistral-counterfact` | `mistral_counterfact_L29H2_v2_full/summary.json` | All recorded CLI arguments match |
| `paper.qwen-counterfact` | `qwen_counterfact_v2_full/summary.json` | All recorded CLI arguments match |
| `paper.mistral-popqa` | `mistral_popqa_c2_balanced_ablation/summary.json` | All recorded CLI arguments match |
| `paper.qwen-popqa` | `qwen_popqa_c2_balanced_ablation/summary.json` | All recorded CLI arguments match |

The full archive-relative paths and argument counts are in the JSON audit. No question text, generated answer, activation cache or model checkpoint is copied into this report.

## Corrections made during verification

- Deployment now points explicitly to the final Mistral token-bias summary and `qwen_intervention_20260515`, matching the v3 run, instead of stale convenience defaults.
- Selective prediction now uses the recorded v3 output/figure/cache directories.
- The matched-control output directory is the recorded `matched_head_controls_fixed_disruption_v2` directory.
- Llama PopQA was already configured correctly: seed 1, aliases, three bias steps, lambda_neg=5.0, lambda_pos grid `2.5,5.0,7.5,10.0`, and applied fixed tau=0.0. The raw pilot's approximately 0.0856 threshold was overridden in the historical full run; it must not replace the recorded fixed threshold.
- Qwen MedQA's archived arguments omit `intervention_impl`. The release names `token_logit_bias` explicitly because the source evaluator discards `site_mode` and applies output-token bias. This is documented source interpretation, not an extra historically logged argument.

## What this establishes

The recorded model IDs, seeds, dtype, sequence limits, prompts, path selections, grids and other explicitly logged CLI settings agree for these twelve profiles. Scalar settings selected by pilots are separately transcribed as reference operating points; running a new pilot can still select a different point.

The Llama CounterFact trace/baseline/OV preparation chain has no complete historical CLI manifest in this local archive. Its profile settings follow the source workflow and stated operating point but are **not fully verified** against an original invocation. Other preparation/helper profiles are likewise not covered by this historical comparison.

The audit does not recover an absent original package lock, immutable model/tokenizer revisions, upstream dataset revisions, or the original prepared input bytes. Matching CLI settings does not prove an independently rebuilt dataset will have the same rows, order, contexts or hash. These remain prerequisites for exact numerical reproduction and are not claimed to be verified.

## Repeat the check without GPU inference

```bash
python tools/verify_recorded_settings.py \
  --archive-root /path/to/original/experiment \
  --output run_records/recorded_settings_audit.json
```

The archive root contains both `medqa_final_v3/` and `hallucination_experiment/`. A missing source, changed source hash or differing recorded setting produces a failure status. Profiles without attached historical metadata are labelled `not_verified_against_recorded_run` rather than silently declared matches.
