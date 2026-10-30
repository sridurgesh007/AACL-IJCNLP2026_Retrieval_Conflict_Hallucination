#!/usr/bin/env bash
#SBATCH --job-name=evidence-routing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: sbatch --partition=YOUR_PARTITION --gres=gpu:YOUR_TYPE:1 sbatch/run.sbatch PROFILE [--set KEY=VALUE ...]' >&2
  exit 2
fi
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "${REPO_ROOT}"
if [[ ! -f tools/run.py ]]; then echo 'Repository root must contain tools/run.py' >&2; exit 2; fi
if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_SH:-}" ]]; then source "${CONDA_SH}"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
  conda activate "${CONDA_ENV}"
fi
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
python tools/run.py "$@" --execute
