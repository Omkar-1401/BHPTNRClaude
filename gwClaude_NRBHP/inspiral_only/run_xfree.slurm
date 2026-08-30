#!/bin/bash
#SBATCH -J xfree
#SBATCH -o xfree.%j.log
#SBATCH -e xfree.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# The missing cell: full-window ka2/kb2 with FREE shape coefficients.  With this,
# cut vs shape-anchoring separate cleanly against t<=+25 ka2/kb2 free (3.667e-04).
echo "######## full window, ka=2 kb=2, shape FREE ########"
python -u fit_scaling_x_drive_regress.py --ka 2 --kb 2 --free-shape

echo
echo "######## same, --global ########"
python -u fit_scaling_x_drive_regress.py --ka 2 --kb 2 --free-shape --global --maxiter 80
