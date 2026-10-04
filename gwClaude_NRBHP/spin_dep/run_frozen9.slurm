#!/bin/bash
#SBATCH -J frozen9
#SBATCH -o frozen9.%j.log
#SBATCH -e frozen9.%j.log
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

# BASIS-VS-BASIS TEST.  Freeze the 9 nonspinning coefficients at the shipped
# nonspinning pn_anchored values and fit ONLY (sSO, sSS).
#
# WHY.  The all-11-fitted run confounds two changes against the predecessor: the spin
# basis (4 polynomial -> 2 PN-ordered) AND freeing the 9 onto the 2d grid.  The control
# (test_pn_spin_joint2d.py, polynomial basis, all 13 on 2d) shows the refit alone is
# worth -16.4% on the mean, so most of the apparent gain was never the PN re-ordering.
# With the 9 pinned, the ONLY difference from the predecessor is the basis: 2 derivable
# PN terms against 4 fitted polynomial ones, same grid, objective, drive and base.
#
# It also restores strict-extension behaviour -- bit-identical to nonspinning
# pn_anchored at chi1 = 0, which the all-11 fit does not satisfy.
#
# Only 2 free parameters, so Powell converges in a few hundred evals (~3.7 s/eval over
# the 96-point grid); 2h is ample.  Writes coeffs_frozen9.json -- it does NOT touch the
# promoted coeffs_physical.json.
python -u fit_pn_anc_spin_physical.py --freeze9 --maxiter 400
