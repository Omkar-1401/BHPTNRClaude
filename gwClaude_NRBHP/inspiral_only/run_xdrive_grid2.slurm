#!/bin/bash
#SBATCH -J xdrv_g2
#SBATCH -o xdrv_g2.%j.log
#SBATCH -e xdrv_g2.%j.log
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

# Remaining rungs of the degree ladder on the full 64-q grid, so the whole comparison
# exists at grid resolution.  The 3 --check mass ratios are cached and skipped.
echo "############ ka=2 kb=1, 64-q grid ############"
python -u fit_scaling_x_drive.py --ka 2 --kb 1 --nproc 24

echo
echo "############ ka=3 kb=1, 64-q grid ############"
python -u fit_scaling_x_drive.py --ka 3 --kb 1 --nproc 24

echo
echo "############ ka=2 kb=2, inspiral-only window (t_nr < -200 M) ############"
python -u fit_scaling_x_drive.py --ka 2 --kb 2 --t-cut -200 --nproc 24
