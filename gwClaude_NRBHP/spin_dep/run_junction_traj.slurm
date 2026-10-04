#!/bin/bash
#SBATCH -J jnctraj
#SBATCH -o jnctraj.%j.log
#SBATCH -e jnctraj.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
python junction_floor.py --mode traj --model junction \
    --q 3.0 3.5128205128 3.8974358974 4.6 5.4358974359 6.5897435897 8.0 \
    --maxiter 60
