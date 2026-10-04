#!/bin/bash
#SBATCH -J physprom
#SBATCH -o physprom.%j.log
#SBATCH -e physprom.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# Promote the finished pn15 fit into the pn_anc_spin_physical model, then draw the
# alpha/beta overlays for chi1 in {-0.6, -0.2, +0.2, +0.6}.  Submitted with
# --dependency=afterany on the pn15 pass-2 job, so it reads converged coefficients.
set -o pipefail
echo "=== PROMOTE ==="
python -u fit_pn_anc_spin_physical.py --promote || exit 1
echo
echo "=== OVERLAYS ==="
python -u plot_pn_anc_spin_physical_overlay.py || exit 1
echo
echo "ALL DONE"
