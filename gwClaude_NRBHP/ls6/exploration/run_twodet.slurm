#!/bin/bash
# Rink et al. 2024 Fig. 12 style: two-detector (h+ / hx) FD mismatch vs total mass, face-on, all models
#SBATCH -J twodet
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/twodet.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/twodet.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in p40 rink p40lean p49b12 p49b9 flux_orbit; do
  [ -f fdm/results/twodet_$m.json ] && continue
  python -u fdm/twodet_run.py $m || exit 1; date
done
