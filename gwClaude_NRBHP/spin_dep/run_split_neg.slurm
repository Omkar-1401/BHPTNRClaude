#!/bin/bash
#SBATCH -J spl_neg
#SBATCH -o split_neg.%j.log
#SBATCH -e split_neg.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# build the 96-point cache, evaluate the seed, run the chi1=0 NULL CHECK,
# then fit the 4 spin coefficients against waveform mismatch
python -u fit_pn_anchored_spin_split.py --side neg --refit --budget 95
