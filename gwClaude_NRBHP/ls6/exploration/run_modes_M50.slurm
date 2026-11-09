#!/bin/bash
# Zhong et al. Fig. 14 layout at M = 50 Msun: per-mode (2,2) (3,3) (4,4) FD match, chi1 = 0, f_low raised per mode (window -5000..+100 M)
#SBATCH -J modesM50
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/modesM50.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/modesM50.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in p40 p49b12; do
  FDM_MTOT=50 python -u fdm/mode_match.py $m || exit 1; date
done
cd fdm && python -u fig14_compare.py 50; date
