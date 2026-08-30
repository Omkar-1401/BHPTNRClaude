#!/bin/bash
#SBATCH -J ab_vs_x
#SBATCH -o ab_vs_x.%j.log
#SBATCH -e ab_vs_x.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# model alpha(x), beta(x) vs the parameter-free measurement, plus the Ehat(x)
# universality test that decides whether an E-driven model already IS an x-model.
python -u NRBHP_alpha_beta_model_vs_x.py \
    --model E_anchored flux_anchored inspiral_anchored
