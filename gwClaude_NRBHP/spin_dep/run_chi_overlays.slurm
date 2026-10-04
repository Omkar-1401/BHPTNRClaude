#!/bin/bash
#SBATCH -J chiovl
#SBATCH -o chiovl.%j.log
#SBATCH -e chiovl.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# CONSTANT-q / VARYING-chi1 overlays for all four spin models -- the complement to the
# existing constant-chi1 figures, which are now in each model's chi_const/ subfolder.
# Reuses each model's own plot-module curve(), so the legend E values are identical to
# the constant-chi1 ones; this is a re-slicing of the same data, not a re-fit.
# Emits 5 figures x 2 formats per model (q = 2,3,5,8 plus a 4x2 contact sheet) into
# Agentic_spin_plots/<model>/q_const/, then re-runs the folder sort.
python -u plot_spin_chi_overlays.py
