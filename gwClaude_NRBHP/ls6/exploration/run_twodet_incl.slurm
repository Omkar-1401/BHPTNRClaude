#!/bin/bash
# Two-detector FD mismatch at iota = pi/3, pi/2 (8 modes, 8 signal orbital phases, SNR-weighted), all models
#SBATCH -J twodetI
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/twodet_incl.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/twodet_incl.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:58:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in p40 rink p40lean p49b12 p49b9 flux_orbit; do
  [ -f fdm/results/twodet_incl_$m.json ] && continue
  python -u fdm/twodet_incl_run.py $m || exit 1; date
done
