#!/bin/bash
#SBATCH -J pn_verify
#SBATCH -o pn_verify.%j.log
#SBATCH -e pn_verify.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "###### 1. EVALUATE the SHIPPED coefficients (no refit) ######"
echo "# documented: in-range median 6.16e-4, max 7.67e-4, q2 2.55e-3"
python -u fit_scaling_pn_anchored.py

echo
echo "###### 2. FRESH REFIT (shipped coeffs.json backed up first) ######"
cp pn_anchored_results/coeffs.json pn_anchored_results/coeffs_shipped_backup.json
python -u fit_scaling_pn_anchored.py --refit --maxiter 40
# preserve the refit output separately, restore the shipped file as canonical
cp pn_anchored_results/coeffs.json pn_anchored_results/coeffs_refit_20260826.json
cp pn_anchored_results/coeffs_shipped_backup.json pn_anchored_results/coeffs.json

echo
echo "###### 3. theta comparison ######"
python3 - <<'PYEOF'
import json
old = json.loads(open("pn_anchored_results/coeffs_shipped_backup.json").read())
new = json.loads(open("pn_anchored_results/coeffs_refit_20260826.json").read())
names = old["param_names"]
print(f"{'param':>6} {'shipped':>12} {'refit':>12} {'rel diff':>10}")
for n, a, b in zip(names, old["theta"], new["theta"]):
    rd = (b - a) / a if a else float("inf")
    flag = "  <-- CHECK" if abs(rd) > 0.05 else ""
    print(f"{n:>6} {a:>12.4f} {b:>12.4f} {rd:>+9.1%}{flag}")
print()
print("the three that matter for the Wardell comparison: b1, b2, a2")
PYEOF
echo "DONE"
