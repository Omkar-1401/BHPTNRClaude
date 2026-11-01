#!/bin/bash
# Zhong et al. 2025 coefficients on our grids (+ one sealed-test look) and the frequency-domain (aLIGO ZDHP) mismatch study
#SBATCH -J fdm
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/fdm.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/fdm.%j.log
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
python -u fdm/zhong_score.py && python -u t100_score.py zhong; date
for m in zhong flux_orbit p49b9 p49b12 p40lean p40 rink; do
  [ -f fdm/results/fd_$m.json ] && continue
  python -u fdm/fd_run.py $m; date
done
