#!/bin/bash
#SBATCH -J xcut25
#SBATCH -o xcut25.%j.log
#SBATCH -e xcut25.%j.log
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

# t_cut = +25 M: the MEASURED saturation point (diag_x_validity.py -> t_sat_05 runs
# 24.2..26.6 M across q; +25 sits inside that band and keeps every q inside its own
# valid range to within a fraction of a percent of x's total span).  This is the cut
# the instruction asked for; -200 and 0 are run in job 3435927 for comparison.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u fit_scaling_x_drive.py --ka 1 --kb 1 --t-cut 25 --nproc 24
python -u fit_scaling_x_drive.py --ka 2 --kb 2 --t-cut 25 --nproc 24
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

python -u fit_scaling_x_drive_regress.py --ka 1 --kb 1 --t-cut 25
python -u fit_scaling_x_drive_regress.py --ka 1 --kb 1 --t-cut 25 --free-shape
python -u fit_scaling_x_drive_regress.py --ka 2 --kb 2 --t-cut 25
python -u fit_scaling_x_drive_regress.py --ka 2 --kb 2 --t-cut 25 --free-shape
echo "ALL STAGES ATTEMPTED"
