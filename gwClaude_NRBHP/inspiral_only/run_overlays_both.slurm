#!/bin/bash
#SBATCH -J insp_ovl
#SBATCH -o insp_ovl.%j.log
#SBATCH -e insp_ovl.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -A PHY26026

# Regenerate both routes' overlays with the fixed, route-distinguishing filenames.
# No fitting -- reads coeffs[_global].json and does one 1-D t0 solve per q.
# Measured: ~30 s per route.  The -t is a kill ceiling, not an estimate.

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/inspiral_only

unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "######## seeded (regressed per-q optima) ########"
python -u NRBHP_inspiral_alpha_beta_overlay.py

echo
echo "######## global (joint refit) ########"
python -u NRBHP_inspiral_alpha_beta_overlay.py --global
