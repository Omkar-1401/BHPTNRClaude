#!/bin/bash
#SBATCH -J insp_grid
#SBATCH -o insp_grid.%j.log
#SBATCH -e insp_grid.%j.log
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/inspiral_only

unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude

# one BLAS thread per worker: parallelism comes from the process pool over q
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# full 64-q [3,8] grid.  The --check q values are already cached and are skipped.
python -u fit_scaling_inspiral_fluxanchored.py --nproc 24
