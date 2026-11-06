#!/bin/bash
# training_size: gates + timing of one short G24 fit per model (results moved to results/smoke/)
#SBATCH -J ts_smoke
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/smoke.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/smoke.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:58:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { echo "=== $(date +%T) $*"; timeout 25m python -u "$@" 2>&1 | grep -v -E "NOT NR calibrated|Surrogate loaded"; echo "=== $(date +%T) exit ${PIPESTATUS[0]}"; }
run fit_rink.py --grid 24
run fit_simple.py pn13 --grid 24 --budget 8
run fit_simple.py flux_orbit --grid 24 --budget 8
run fit_p39.py --grid 24 --budget 8
run fit_split.py splitA --grid 24
run fit_split.py splitB --grid 24
mkdir -p results/smoke && mv results/*.json results/smoke/ 2>/dev/null
date
