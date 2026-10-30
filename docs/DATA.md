# Data preparation (no benchmark data is distributed)

Datasets and model weights are obtained by the user from their upstream maintainers under their own terms. Never commit downloaded files to this repository. Runtime inputs are placed under `implementation/<family>/data/`, which is ignored.

## Exact paper inputs versus regenerated method inputs

The frozen MedQA IDs/audit labels are small metadata, not the data itself. To reproduce the paper's exact samples you must supply the same prepared questions, contexts, answers, order and model-specific candidate files. The paper's original prepared corpus and every raw-data revision have not all been recovered into this checkout. **There is no claim that rebuilding from an arbitrary current upstream revision will regenerate identical paper inputs.**

The provided builders allow a new, reproducible method experiment. Record the upstream revision, file hash, preparation command and output hash. A new sample should be reported as a new sample. If exact frozen files are available on your cluster, import them without transformation:

```bash
python tools/prepare_data.py medqa --input /scratch/frozen/medqa_conflict_500.jsonl --expect-count 500
python tools/prepare_data.py counterfact --input /scratch/frozen/counterfact_solvent.json
python tools/prepare_data.py popqa --input /scratch/frozen/popqa_conflict_3141_natural.jsonl --expect-count 3141
```

Add `--expect-sha256 HASH` to enforce an archived fingerprint. Importing refuses to replace different existing data. `--validate-only` checks schema and count without copying. Model-specific PopQA candidate/supplement files may instead be placed at the explicit paths recorded in their profiles; do not use a Llama conflict file as a Qwen conflict set.

## MedQA

**Citation:** Jin et al. (2021), *What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams*. [Paper](https://www.mdpi.com/2076-3417/11/14/6421), [maintainer repository](https://github.com/jind11/MedQA).

The original normalizer is `medrag/src/medrag/medical_rag/medqa_loader.py`. The author builder is `medrag/scripts/22_build_medqa_conflict.py`. Its supported Hugging Face candidates/splits are explicit in that loader; inspect them and pin the upstream dataset version for a new experiment. `medqa.prepare` explicitly disables the optional external retrieval fallback and constructs gold-option evidence. Such gold-derived contexts are controlled inputs, not retrieval success measurements or deployment evidence.

```bash
python tools/run.py medqa.prepare                 # inspect command
python tools/run.py medqa.prepare --execute       # GPU filtering + controlled context construction
python tools/run.py medqa.preflight --execute
```

Prepared JSONL schema:

```text
question: string
options: mapping with A, B, C, D
context: string
correct_answer: A/B/C/D
parametric_answer: A/B/C/D
```

Preflight exports `matched_subset.json` and a label-audit template. For a newly built corpus, manually audit the labels and use that corpus's selections. Do not assign "clean" to every row or reuse historical labels against newly ordered/different questions. For the original prepared corpus, `tools/bootstrap.py` installs the frozen selections. Verify 473 matched IDs and 40 clean IDs before headline reporting.

## CounterFact

**Citation:** Meng et al. (2022), *Locating and Editing Factual Associations in GPT*. [Paper](https://arxiv.org/abs/2202.05262), [dataset project](https://rome.baulab.info/).

```bash
python tools/run.py counterfact.prepare --execute
```

The loader in `Evidence_aware/src/evidence_aware/dataset/counterfact.py` downloads and normalizes text data. It has explicit upstream candidate and fallback URL arguments. Select and record the intended source; the tracing adaptation and original ROME file have different schemas and must not be assumed interchangeable. Preserve the complete cache for the disjoint trace/validation/evaluation split; do not truncate the cache to 2,000 before splitting.

The normalized JSON is a list with `id`, `question`, `evidence`, `answer_liquid` (evidence target), `answer_solid` (parametric target), and optional `subject`, `relation_id`, `prompt_template`. The source's model-independent answer strings are retokenized with the active tokenizer. No prepared token-ID cache is redistributed.

The cross-model v2 profiles select `evidence_mode=relation_or_question_answer`. The primary Llama OV path uses the cache evidence. Keep these construction modes explicit; modifying evidence templates changes the experiment. The Llama OV evaluation has 2,000 items; Mistral/Qwen cross-model runs have 500. A 2,000-row file is not a 500-row four-threshold sweep without checking unique IDs and configuration fields.

## PopQA

**Citation:** Mallen et al. (2023), *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories*. [Paper](https://aclanthology.org/2023.acl-long.546/), [maintainer data/code](https://github.com/AlexTMallen/adaptive-retrieval).

Obtain the maintainer's `popQA.tsv` and place it at `implementation/conflictqa/data/raw/popQA.tsv`. The author builder and C2-resistant filter are separate stages:

```bash
python tools/run.py popqa.prepare --execute
python tools/run.py popqa.rebuild --set model_id=meta-llama/Llama-3.1-8B-Instruct \
  --set candidates_path=data/popqa_conflict_candidates.jsonl \
  --set out_dir=results/popqa/rebuilt --execute
```

The builder requires a positive target sample size and writes candidates plus a sampled file. It is not the recipe for automatically recovering the historical Llama 3,141-item natural file; import that frozen prepared file for the published full-run profile. For a fresh experiment, choose an attainable positive target size and use the rebuilt file explicitly, recording its new count.

Prepared rows contain `question`, `context`, `correct_answer`, `correct_aliases`, and `parametric_answer`. C1 candidates are filtered by model confidence, then by continued parametric answering under C2 context. Rebuild separately for each model. Where archived pilot supplementation was used, `68_supplement_popqa_for_pilot.py` reconstructs the required input structure; supplement markers must remain intact so they are not silently included in headline evaluation populations.

## ClashEval (optional scope analysis)

**Citation:** Wu et al. (2024), *ClashEval: Quantifying the Tug-of-War Between an LLM's Internal Prior and External Evidence*, [paper](https://arxiv.org/abs/2404.10198).

Acquire the upstream response/data file separately and pass it to the `56`, `57` or `59` scripts; consult their CLI contracts and `medical_rag/clasheval_loader.py` for required columns. No third-party benchmark repository, response parquet, or medical corpus is vendored here. This is a scope/negative-transfer analysis, not evidence of reliable free-form medical correction.
