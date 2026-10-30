# Storage and release policy

Git is for source code, documentation, small method configurations and ID-only selection/audit manifests. It is not the experiment disk.

Excluded by `.gitignore`: benchmark data, generated JSONL/CSV/TSV, model weights, checkpoint files, activation caches, binary arrays, plots, PDFs, logs, Conda/virtual environments, credentials and run records. `tools/check_release.py` also inspects staged/tracked names and refuses forbidden runtime directories, generated file extensions and individual files over 2 MB.

For large reproducibility material, use your research storage or a separately managed archive. A code release can regenerate outputs without putting them in Git, but exact sample reproduction still requires the original prepared input or a complete deterministic recipe with revision/hash. Do not call a missing input publicly released merely because its filename is documented.

Use scratch paths for Hugging Face caches and generated run directories when your cluster home quota is limited. Never delete the only frozen input or the only per-item result needed for an audit. Move or archive material according to your research retention policy.

Do not commit tokens or export environment variables containing secrets into metadata. The runner records package versions and CLI arguments; pass secrets through the normal local authentication/environment mechanism, never through `--set`.
