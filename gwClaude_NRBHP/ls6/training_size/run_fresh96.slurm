#!/bin/bash
# training_size: fresh-G96 check -- regenerate T96 from scratch, compare with the old cache, run every gate on it
#SBATCH -J ts_fresh96
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/fresh96.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/fresh96.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { echo "=== $(date +%T) $*"; timeout 25m python -u "$@" 2>&1 | grep -v -E "NOT NR calibrated|Surrogate loaded"; echo "=== $(date +%T) exit ${PIPESTATUS[0]}"; }
export TS_FRESH=1
run check_fresh96.py
run fit_rink.py --grid 96 --gate-only
run fit_simple.py pn13 --grid 96 --gate-only
run fit_simple.py flux_orbit --grid 96 --gate-only
run fit_p39.py --grid 96 --gate-only
run fit_split.py splitA --grid 96 --gate-only
run fit_split.py splitB --grid 96 --gate-only
date
