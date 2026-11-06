#!/bin/bash
# training_size: lean9 on G24, start pert2 (user 2026-10-28)
#SBATCH -J lean9_24_pert2
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/lean9_G24_pert2.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/lean9_G24_pert2.%j.log
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:45:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TS_FRESH=1          # T96 from cache_fresh/ (reproduces stored numbers to 1e-6; old cache 1e-4)
run() { echo "=== $(date +%T) $*"; timeout 100m python -u "$@" 2>&1 | grep -v -E "NOT NR calibrated|Surrogate loaded"; echo "=== $(date +%T) exit ${PIPESTATUS[0]}"; }
run fit_p40.py lean9 --grid 24 --budget 75 --start pert2
date
