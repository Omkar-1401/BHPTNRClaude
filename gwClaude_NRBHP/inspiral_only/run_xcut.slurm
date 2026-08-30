#!/bin/bash
#SBATCH -J xcut
#SBATCH -o xcut.%j.log
#SBATCH -e xcut.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:45:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "################ 1. WHERE DOES x LOSE VALIDITY (measured) ################"
python -u diag_x_validity.py

# Two candidate cuts, both run rather than picking one by judgement:
#   -200 M : the window you named
#      0 M : merger, i.e. drop the ringdown where x has saturated
for TC in -200 0; do
  echo
  echo "################ per-q, t_cut = $TC M ################"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -u fit_scaling_x_drive.py --ka 1 --kb 1 --t-cut $TC --nproc 24
  python -u fit_scaling_x_drive.py --ka 2 --kb 2 --t-cut $TC --nproc 24
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

  echo
  echo "################ regression, t_cut = $TC M ################"
  python -u fit_scaling_x_drive_regress.py --ka 1 --kb 1 --t-cut $TC
  python -u fit_scaling_x_drive_regress.py --ka 1 --kb 1 --t-cut $TC --free-shape
  python -u fit_scaling_x_drive_regress.py --ka 2 --kb 2 --t-cut $TC
  python -u fit_scaling_x_drive_regress.py --ka 2 --kb 2 --t-cut $TC --free-shape
done

echo
echo "ALL STAGES ATTEMPTED"
