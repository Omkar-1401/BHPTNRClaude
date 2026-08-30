#!/bin/bash
#SBATCH -J mr_only
#SBATCH -o mr_only.%j.log
#SBATCH -e mr_only.%j.log
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 3-q probe first, both anchors, so a bug shows up in a minute rather than an hour.
for A in qnm x1; do
  echo "################ PROBE anchor=$A ################"
  python -u fit_scaling_mr_only.py --check --anchor $A --nproc 3
  echo
done

# full grid, both anchors: qnm imposes B1 (3 params/q), x1 is the control (4 params/q)
for A in qnm x1; do
  echo "################ GRID anchor=$A, t_cut=-200 ################"
  python -u fit_scaling_mr_only.py --anchor $A --nproc 24
  echo
done

echo "ALL STAGES ATTEMPTED"
