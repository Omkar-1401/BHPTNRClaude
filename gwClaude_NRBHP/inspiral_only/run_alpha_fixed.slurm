#!/bin/bash
#SBATCH -J insp_afix
#SBATCH -o insp_afix.%j.log
#SBATCH -e insp_afix.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/inspiral_only
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# alpha inherited from the parent full-window fit (falls monotonically at every q);
# only beta_PP and P are fitted on the inspiral.  3 free coefficients.
for M in anchored fluxanchored; do
  echo "############ $M : alpha from parent, beta on inspiral (seeded) ############"
  python -u fit_scaling_inspiral_regress.py --model $M
  echo
  echo "############ $M : same, --global joint refit of the 3 ############"
  python -u fit_scaling_inspiral_regress.py --model $M --global --maxiter 80
  echo
done

echo "############ overlays ############"
python -u NRBHP_inspiral_alpha_beta_overlay.py
python -u NRBHP_inspiral_alpha_beta_overlay.py --global
