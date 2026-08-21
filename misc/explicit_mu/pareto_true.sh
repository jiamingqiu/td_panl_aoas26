#!/usr/bin/env bash
# Submit from misc/. Each array task evaluates one row of the retained grid.
#SBATCH --job-name=pareto_true
#SBATCH --output=explicit_mu/pareto_true/pareto_true_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-714
#SBATCH --mem=20G
#SBATCH --time=99:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# This module name records the original cluster environment.
module load R/4.3.2-gfbf-2023a

mkdir -p explicit_mu/pareto_true

Rscript explicit_mu/compute_true.R \
  --integral_maxEval 20 \
  --which_batch "${SLURM_ARRAY_TASK_ID}" \
  --total_batch "${SLURM_ARRAY_TASK_COUNT}" \
  --outdir explicit_mu/pareto_true
