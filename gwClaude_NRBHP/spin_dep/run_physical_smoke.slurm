#!/bin/bash
#SBATCH -J physsmoke
#SBATCH -o physsmoke.%j.log
#SBATCH -e physsmoke.%j.log
#SBATCH -p skx
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

# SMOKE TEST, run immediately rather than waiting for the chain.
# fit_pn_anc_spin_physical.py and plot_pn_anc_spin_physical_overlay.py have never been
# executed.  The overnight chain is afterok-linked, so a runtime bug in either would
# kill it silently.  This exercises both end to end against the IN-FLIGHT pn15
# checkpoint (--promote falls back to partial_pn15.json when coeffs_pn15.json is absent),
# so any bug surfaces now.  Its coeffs_physical.json is overwritten by job 3512764.
set -o pipefail
echo "=== PROMOTE (from the in-flight pn15 checkpoint) ==="
python -u fit_pn_anc_spin_physical.py --promote || exit 1
echo
echo "=== OVERLAYS ==="
python -u plot_pn_anc_spin_physical_overlay.py || exit 1
echo
echo "SMOKE TEST PASSED"
