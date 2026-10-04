#!/bin/bash
#SBATCH -J spinovl
#SBATCH -o spinovl.%j.log
#SBATCH -e spinovl.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# alpha/beta overlays for the spin-extended pn_anchored model.  Runs AFTER the fit
# (submitted with --dependency), so it reads the converged coeffs_spin.json.
python -u spin_alpha_beta_overlay.py
