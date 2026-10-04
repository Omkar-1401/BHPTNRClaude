#!/bin/bash
#SBATCH -J spineps
#SBATCH -o spineps.%j.log
#SBATCH -e spineps.%j.log
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
# eps = 1/q weighting on the spin terms.  Budget 50 min so BOTH arms fit in one job.
echo "################ W = eps = 1/q  (the new model) ################"
python -u fit_pn_anchored_spin_eps.py --weight eps --refit --budget 50
