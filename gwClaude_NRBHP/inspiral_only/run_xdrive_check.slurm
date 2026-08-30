#!/bin/bash
#SBATCH -J xdrv_chk
#SBATCH -o xdrv_chk.%j.log
#SBATCH -e xdrv_chk.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 3-q probe across the degree ladder before committing to the 64-q grid.
# ka=kb=1 is the exact structural analogue of the energy models (4 params/q).
for KA in 1 2 3; do
  echo "############ alpha K=$KA, beta K=1 ############"
  python -u fit_scaling_x_drive.py --check --ka $KA --kb 1 --nproc 3
  echo
done
echo "############ alpha K=2, beta K=2 ############"
python -u fit_scaling_x_drive.py --check --ka 2 --kb 2 --nproc 3
