#!/bin/bash
# training_size: gate lean12 + lean9 (stored p49b coefficients must reproduce stored T96 cases) (user 2026-10-28)
#SBATCH -J lean129_gate
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/lean129_gate.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size/logs/lean129_gate.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/training_size
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TS_FRESH=1
rc=0
for m in lean12 lean9; do
  echo "=== $(date +%T) gate $m"
  python -u fit_p40.py $m --grid 96 --gate-only 2>&1 | grep -v -E "NOT NR calibrated|Surrogate loaded"
  s=${PIPESTATUS[0]}; echo "=== exit $s"; [ $s -ne 0 ] && rc=1
done
date; exit $rc
