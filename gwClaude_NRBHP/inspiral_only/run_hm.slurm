#!/bin/bash
#SBATCH -J insp_hm
#SBATCH -o insp_hm.%j.log
#SBATCH -e insp_hm.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26026

# Higher modes for the two inspiral-only bases, via the EXISTING zero-parameter backbone
# transfer (fit_scaling_hm_backbone.py, note Eq. 43).  Nothing new is fitted: alpha_lm is
# alpha_22 * C_lm(nu) * beta^(-(l+eps-2)/3), rho=1, and the best-constant rho is closed
# form.  Only the (2,2) base changes.
#
# --t-cut defaults to -200 M for these bases, so the modes are scored on the window the
# base was fitted on and t0_nr is aligned there too; the full-window number is also
# reported per mode for orientation.
#
# q = 2,3,4,5,8 are the mass ratios already in .cache/hm/, so no surrogate generation.

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP

unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

for M in inspiral_anchored inspiral_fluxanchored; do
  echo "################ higher modes: $M ################"
  python -u fit_scaling_hm_backbone.py --model $M --q 3 5 8 2 4
  echo
done

echo "################ Fig-8 plots, both bases, q=2 and 4 ################"
python -u NRBHP_hm_backbone_fig8_plots.py --q 2 4 \
    --model inspiral_anchored inspiral_fluxanchored \
    --outdir inspiral_hm
