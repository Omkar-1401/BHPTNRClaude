#!/bin/bash
# training_size: split A/B G96 reproduction diagnosis (stage objectives; B refit on the old cache)
#SBATCH -J ts_chksplit
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/chksplit.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/chksplit.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { echo "=== $(date +%T) $*"; timeout 25m python -u "$@" 2>&1 | grep -v -E "NOT NR calibrated|Surrogate loaded"; echo "=== $(date +%T) exit ${PIPESTATUS[0]}"; }
TS_FRESH=1 run check_split_stage.py
run check_split_stage.py
mkdir -p results/oldcache && cp -p results/splitB__G96__t96.json results/splitB__G96__t96.fresh.bak
run fit_split.py splitB --grid 96 --start t96
mv results/splitB__G96__t96.json results/oldcache/ && mv results/splitB__G96__t96.fresh.bak results/splitB__G96__t96.json
date
