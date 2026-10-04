#!/bin/bash
#SBATCH -J spinnu
#SBATCH -o spinnu.%j.log
#SBATCH -e spinnu.%j.log
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
# The CONTROL: identical script, identical seed scaling, identical budget, W = nu.
# The existing nu numbers came from runs that differed in seed/budget/quarantine, so
# this is the only like-for-like comparison.
echo "################ W = nu  (CONTROL) ################"
python -u fit_pn_anchored_spin_eps.py --weight nu --refit --budget 50
mv -f pn_anchored_spin_eps_results/coeffs_spin_eps.json pn_anchored_spin_eps_results/coeffs_nu_50min.json
