#!/bin/bash
#SBATCH -J jncdiag
#SBATCH -o jncdiag.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26026
unset PYTHONPATH
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WORK/BHPTNRClaude/gwClaude_NRBHP/spin_dep
python junction_diag.py --q 3.0 5.0 8.0 \
    --chi -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 --tag grid
