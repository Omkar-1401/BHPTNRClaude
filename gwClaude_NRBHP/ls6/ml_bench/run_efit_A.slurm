#!/bin/bash
# ml_bench: efit A (SR smoke, E-fitted MLP + GRU on T96, then PySR forms)
#SBATCH -J mb_efit
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/efit_A.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/efit_A.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26029
unset PYTHONPATH LD_LIBRARY_PATH   # TACC intel libs break Julia PCRE
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_DEPOT_PATH=/work2/11702/omkarsridhar1401/stampede3/julia_depot
export PYTHON_JULIAPKG_PROJECT=/work2/11702/omkarsridhar1401/stampede3/julia_env_pysr
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench
python -u efit.py train --arch mlp --train T96 --minutes 30 --nproc 96 --threads 64
python -u efit.py train --arch gru --train T96 --minutes 30 --batch 32 --nproc 96 --threads 64
python -u efit.py sr --minutes 8 --refit-minutes 3 --lr 5e-3 --procs 120 --nproc 96 --threads 32
