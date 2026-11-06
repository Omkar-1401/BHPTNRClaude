#!/bin/bash
# ml_bench: nn_extra (MLP-w extrapolation + T96-only training)
#SBATCH -J mb_nnx
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/nn_extra.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/nn_extra.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26029
unset PYTHONPATH LD_LIBRARY_PATH   # TACC intel libs break Julia PCRE
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_DEPOT_PATH=/work2/11702/omkarsridhar1401/stampede3/julia_depot
export PYTHON_JULIAPKG_PROJECT=/work2/11702/omkarsridhar1401/stampede3/julia_env_pysr
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench
python -u nn_extra.py --minutes 25 --nproc 128 --threads 64
