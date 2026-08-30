#!/bin/bash
#SBATCH -J insp_fit
#SBATCH -o insp_fit.%j.log
#SBATCH -e insp_fit.%j.log
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A PHY26026

# Overnight, unattended.  The 64-q PER-Q grids at t_cut=-200 M are already complete for
# both models (jobs 3431452 flux, 3431501 anchored) and are NOT refitted here -- this job
# builds the FITTED (nu-regressed) master layer they lack.
#
# ORDERING IS DELIBERATE: everything cheap and everything that produces a usable
# deliverable runs FIRST, so a timeout or a failure in the long Powell stage still leaves
# complete seeded coefficients and finished overlays.  No `set -e` for the same reason.
#
# The -t above is a KILL CEILING, not an estimate.  Measured costs: the seeded regression
# is minutes (least-squares on cached per-q numbers + ~10 waveform loads); the --global
# Powell refit is the unknown, each objective call scoring 12 truncated windows with an
# inner t0 minimisation.  Nothing here is parallel -- the regression is serial, unlike the
# per-q grids which used --nproc 24.

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/inspiral_only

unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "############################################################"
echo "# 0. repair the stale flux summary.json (report only, NO refit)"
echo "############################################################"
# The --check --force run at 18:08 overwrote the 64-q summary with a 3-q one.  Re-running
# WITHOUT --force refits nothing (all 64 keys are cached) and rewrites the summary from
# the full grid.  Every fitted value in the cache is left untouched, so the numbers
# already quoted in the md files stay valid.
cp inspiral_fluxanchored_results/per_q_cache.json \
   inspiral_fluxanchored_results/per_q_cache.json.presummary.bak
python -u fit_scaling_inspiral_fluxanchored.py --nproc 1

echo
echo "############################################################"
echo "# 1. FITTED master, seeded (regress the per-q optima) -- CHEAP"
echo "############################################################"
for M in anchored fluxanchored; do
  echo "---- $M ----"
  python -u fit_scaling_inspiral_regress.py --model $M
  echo
done

echo
echo "############################################################"
echo "# 2. overlays from the seeded coefficients (clipped at t_cut)"
echo "############################################################"
python -u NRBHP_inspiral_alpha_beta_overlay.py

echo
echo "############################################################"
echo "# 3. FITTED master, --global joint refit -- THE LONG STAGE"
echo "############################################################"
for M in anchored fluxanchored; do
  echo "---- $M ----"
  python -u fit_scaling_inspiral_regress.py --model $M --global --maxiter 80
  echo
done

echo
echo "############################################################"
echo "# 4. overlays from the global coefficients"
echo "############################################################"
python -u NRBHP_inspiral_alpha_beta_overlay.py --global

echo
echo "ALL STAGES ATTEMPTED -- check above for any that failed"
