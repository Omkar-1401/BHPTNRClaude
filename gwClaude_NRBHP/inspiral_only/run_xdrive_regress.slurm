#!/bin/bash
#SBATCH -J xdrv_reg
#SBATCH -o xdrv_reg.%j.log
#SBATCH -e xdrv_reg.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# nu-regression (master layer) for the x-driven model.  Seeded first (cheap), then the
# global joint refit.  Ordered so a timeout still leaves complete seeded coefficients.
for CFG in "1 1" "2 2"; do
  set -- $CFG
  echo "################ SEEDED  ka=$1 kb=$2 ################"
  python -u fit_scaling_x_drive_regress.py --ka $1 --kb $2
  echo
done

for CFG in "1 1" "2 2"; do
  set -- $CFG
  echo "################ GLOBAL  ka=$1 kb=$2 ################"
  python -u fit_scaling_x_drive_regress.py --ka $1 --kb $2 --global --maxiter 80
  echo
done

echo "################ ka=1 kb=1, shape coeffs FREE (PP anchor dropped) ################"
python -u fit_scaling_x_drive_regress.py --ka 1 --kb 1 --free-shape

echo "ALL STAGES ATTEMPTED"
