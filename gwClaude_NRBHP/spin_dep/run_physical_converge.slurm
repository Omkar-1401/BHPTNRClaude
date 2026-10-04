#!/bin/bash
#SBATCH -J physconv
#SBATCH -o physconv.%j.log
#SBATCH -e physconv.%j.log
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# OVERNIGHT CONVERGENCE for pn_anc_spin_physical.
#
# The dev queue caps wall time at 2h, which is why this fit has been running as a chain
# of 2h passes.  skx allows 48h, so this single job runs a 6h budget instead -- the
# pass-1 trace was still descending (1.2166e-03 at 35 min -> 1.1499e-03 at 61 min,
# decelerating but not flat), so convergence is the open question.
#
# --refit AUTO-RESUMES from coeffs_physical.json, which job 3512764 (--promote) writes.
# It does NOT restart from the poor x=0.12 seed projection (mean 6.06e-03).
# On budget exhaustion it writes coefficients and prints the full tables anyway, so a
# non-converged run is still a usable deliverable.
#
# Chained afterok on 3512764, so the early deliverable (coefficients + overlays from the
# pass-2 point) is already on disk before this starts.  Overlays are redrawn at the end
# from the converged coefficients.
set -o pipefail

echo "=== resuming point ==="
python -c "
import json,pathlib
p=pathlib.Path('pn_anc_spin_physical_results/coeffs_physical.json')
d=json.loads(p.read_text())
print(' ', ', '.join(f'{n}={v:+.4f}' for n,v in zip(d['names'],d['theta'])))
print(f\"  mean {d['mean']:.4e}  median {d['median']:.4e}  max {d['max']:.4e}\")
" || { echo "FATAL: no coeffs_physical.json -- did 3512764 run?"; exit 1; }

echo
echo "=== CONVERGENCE REFIT (6h budget) ==="
python -u fit_pn_anc_spin_physical.py --refit --budget 360 || exit 1

echo
echo "=== OVERLAYS from the converged coefficients ==="
python -u plot_pn_anc_spin_physical_overlay.py || exit 1

echo
echo "ALL DONE"
