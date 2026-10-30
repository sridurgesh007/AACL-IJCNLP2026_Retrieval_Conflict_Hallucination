# Mechanistic Evidence Routing in Transformers

**Margin-Gated Intervention for Retrieval-Conflict Hallucination**

Sridurgesh Ravichandran, National University of Singapore

AACL-IJCNLP 2026

This is the **code-only reproduction repository** for the paper. It contains the authors' localization, margin gating, OV steering, token-bias interventions, causal patching, ablations, data preparation and analysis scripts. It does **not** redistribute benchmark datasets, model weights, activation caches, per-item outputs, plots, third-party benchmark repositories, or large artifacts.

> **Scope:** evidence and parametric answer identities are constructed from known benchmark answers. The experiments are oracle-conditioned mechanistic analyses, not a deployed RAG system that discovers the correct answer automatically. Following CounterFact evidence measures compliance with a controlled, sometimes false, assertion.

## Start here

```bash
git clone https://github.com/sridurgesh007/AACL-IJCNLP2026_Retrieval_Conflict_Hallucination.git
cd AACL-IJCNLP2026_Retrieval_Conflict_Hallucination
python -m venv .venv
source .venv/bin/activate
# Install PyTorch for your cluster's CUDA version, then:
python -m pip install -r requirements.txt
python tools/check_release.py
python -m unittest discover -s tests -v
python tools/bootstrap.py
python tools/run.py --list
```

Use Python 3.10 or newer. CUDA is required for the large-model experiments. Accept the relevant model terms and authenticate locally for gated model access. Never put tokens in scripts or commit them. Use a scratch directory for `HF_HOME`, caches and outputs when working under a home-directory quota.

`requirements.txt` gives supported dependency bounds; it is **not** a recovered lock of the historical cluster. Every executed profile records its actual installed package versions and full command in ignored `run_records/`. We have validated CPU-side contracts and statistics; this release has not been rerun end-to-end on a GPU in the packaging environment.

Twelve metadata-backed profiles have also been checked against the archived runs' explicit CLI arguments and source hashes: see [Recorded settings verification](docs/SETTINGS_VERIFICATION.md). Preparation and helper profiles without archived metadata are labelled separately; they are not presented as historically verified runs.

## Reproduction paths

1. Read [Data preparation](docs/DATA.md). Acquire upstream data under its own terms, or import your own frozen prepared inputs. No benchmark is bundled.
2. Read [Reproduction runbook](docs/REPRODUCE.md). It distinguishes a fresh method run from exact reproduction with the original frozen inputs.
3. Use `python tools/run.py PROFILE` to print the command. Add `--check` to check declared inputs, or `--execute` to run it. Override a parameter with `--set key=value`.
4. Read [Implementation](implementation/README.md) for code structure, assumptions and documented release patches.
5. Read [Paper-to-code mapping](docs/PAPER_TO_CODE.md) and [script index](docs/SCRIPTS.md) to find individual experiments.

For example, after importing the prepared MedQA file:

```bash
python tools/bootstrap.py
python tools/run.py medqa.killtest --check
python tools/run.py medqa.killtest --execute
python tools/audit_medqa.py \
  --results-dir implementation/medrag/results/medqa_final_v3/killtest \
  --output implementation/medrag/results/medqa_final_v3/killtest/audited_comparison.json
```

The small frozen files in `configs/frozen/` contain **selection IDs, audit labels and scalar method settings**, not benchmark content or model outcomes. They preserve 473 matched MedQA IDs and the 40 clean IDs. They cannot reconstruct missing questions, evidence passages, or model-specific pilot outputs. Exact numerical reproduction requires the same prepared inputs and model/software environment; a newly prepared corpus must not be presented as the published sample.

## Models and intervention scope

| Model | Method configuration |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | CounterFact OV steering; MedQA letter-space intervention; PopQA alias promotion/suppression |
| `mistralai/Mistral-7B-Instruct-v0.3` | Model-specific localization; explicit token-bias configuration for the published MedQA run |
| `Qwen/Qwen2-7B-Instruct` | Layer/head localization and explicit output-token bias for the recorded MedQA path; a layer-localization label does not imply a residual-stream hook |

Profiles named `paper.*` preserve archived CLI settings and identify their source metadata by SHA256. Prerequisite localization and pilot outputs are regenerated locally. Seeds, subset definitions, generation lengths, evidence templates and intervention implementations differ between experiments; do not replace them with one universal setting. The three models' conflict subsets are not necessarily identical.

The main generation paths use greedy decoding. Selective-prediction self-consistency is an explicit exception: it samples with the script's recorded temperature/top-p and sample count. Per-run parameters are visible in `configs/experiments.json` and `docs/cli_contracts.json`.

## Cluster execution

The launcher contains no private username, absolute Conda installation, partition, or GPU type. Supply cluster-specific resources when submitting from the repository root:

```bash
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:YOUR_GPU_TYPE:1 \
  sbatch/run.sbatch medqa.killtest

# Optional existing Conda environment:
CONDA_ENV=hallucination sbatch --partition=YOUR_GPU_PARTITION \
  --gres=gpu:YOUR_GPU_TYPE:1 sbatch/run.sbatch popqa.balanced

# A dependent stage starts only after its prerequisite succeeds:
sbatch --dependency=afterok:JOB_ID --partition=YOUR_GPU_PARTITION \
  --gres=gpu:YOUR_GPU_TYPE:1 sbatch/run.sbatch medqa.matched-controls
```

The supplied time/memory values are resource requests, not runtime guarantees. Latency and GPU memory results depend on hardware, software, sequence lengths and cache state.

## Datasets and attribution

Please cite the original dataset papers as well as this work. Download from the original maintainers; the repository's MIT license applies only to our code and does not relicense data or models.

| Dataset | Paper | Maintainer source |
|---|---|---|
| MedQA | Jin et al. (2021), [What Disease Does This Patient Have?](https://www.mdpi.com/2076-3417/11/14/6421) | [jind11/MedQA](https://github.com/jind11/MedQA) |
| CounterFact | Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) | [ROME project](https://rome.baulab.info/) |
| PopQA | Mallen et al. (2023), [When Not to Trust Language Models](https://aclanthology.org/2023.acl-long.546/) | [adaptive-retrieval](https://github.com/AlexTMallen/adaptive-retrieval) |
| ClashEval, optional scope study | Wu et al. (2024), [ClashEval](https://arxiv.org/abs/2404.10198) | Obtain the upstream response/data file separately; see the loader schema in the implementation |

Dataset availability and licenses are governed by their maintainers. Check the applicable upstream terms before downloading, redistributing or using any derived examples. We do not bundle third-party benchmark suites. Prompt controls and the paper's optional ITI comparison are author-written experiment code; the core method does not require third-party baseline checkpoints.

## Statistics and storage policy

Restoration and harm use different populations. The MedQA headline restoration is measured on 40 initially wrong clean items; its harm population is the larger matched set of initially correct controls. The causal experiment uses 35 pairs. `tools/audit_medqa.py` reports numerator, denominator, Wilson interval and exact paired McNemar statistics separately for clean and full matched populations. Duplicate IDs and partial joins fail instead of silently reducing the denominator.

All runtime datasets, caches, outputs and logs are Git-ignored. A CI check rejects staged benchmark/generated files and files over 2 MB. See [Storage policy](docs/STORAGE.md) before adding material. Reproduction scripts create their outputs locally; no result archive is required in Git.

## Citation

```bibtex
@inproceedings{ravichandran2026mechanistic,
  title = {Mechanistic Evidence Routing in Transformers: Margin-Gated Intervention for Retrieval-Conflict Hallucination},
  author = {Ravichandran, Sridurgesh},
  booktitle = {Proceedings of AACL-IJCNLP},
  year = {2026},
  url = {https://github.com/sridurgesh007/AACL-IJCNLP2026_Retrieval_Conflict_Hallucination}
}
```

The citation intentionally does not invent an Anthology identifier, DOI, page range or publication URL. Update it to the publisher's final record when available.
