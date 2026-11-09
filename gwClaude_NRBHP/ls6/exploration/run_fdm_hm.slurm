#!/bin/bash
# multimode (8 modes) FD mismatch: HM cache, then our transferred models and the published Rink modes
#SBATCH -J fdm_hm
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/fdm_hm.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/exploration/fdm/fdm_hm.%j.log
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
python -u fdm/hm_cache.py || exit 1; date
for m in p49b12 p40 p40lean p49b9 flux_orbit rink; do
  [ -f fdm/results/hm_$m.json ] && continue
  python -u fdm/hm_run.py $m; date
done
