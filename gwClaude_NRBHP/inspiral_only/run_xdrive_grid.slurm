#!/bin/bash
#SBATCH -J xdrv_grid
#SBATCH -o xdrv_grid.%j.log
#SBATCH -e xdrv_grid.%j.log
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

# Full 64-q grid at the degrees the 3-q probe selected (ka=2, kb=2).  The three
# --check mass ratios are already cached and are skipped.  ka=2/kb=1 is also run as
# the control: same total parameters as ka=3/kb=1, and the probe says the beta term
# is worth ~33% while the third alpha term is worth ~2% -- worth confirming on the grid.
echo "############ ka=2 kb=2, 64-q grid ############"
python -u fit_scaling_x_drive.py --ka 2 --kb 2 --nproc 24

echo
echo "############ ka=1 kb=1, 64-q grid (4 params/q, energy-model analogue) ############"
python -u fit_scaling_x_drive.py --ka 1 --kb 1 --nproc 24
