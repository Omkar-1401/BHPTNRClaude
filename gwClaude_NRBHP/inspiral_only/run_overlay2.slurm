#!/bin/bash
#SBATCH -J insp_plot2
#SBATCH -o insp_plot2.%j.log
#SBATCH -e insp_plot2.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/inspiral_only

unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "########## t_cut gauge check (diagnostics only, does not touch the fits) ##########"
python -u check_tcut_gauge.py

echo
echo "########## alpha/beta overlays, both models ##########"
python -u NRBHP_inspiral_alpha_beta_overlay.py
