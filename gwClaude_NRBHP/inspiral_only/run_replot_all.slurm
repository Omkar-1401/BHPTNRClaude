#!/bin/bash
#SBATCH -J replot
#SBATCH -o replot.%j.log
#SBATCH -e replot.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A PHY26026

# Regenerate every figure produced this session, now with mathtext labels.
# Mathtext renders with text.usetex=False, so no LaTeX install is involved.

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "######## 1. alpha/beta vs x (master) + Ehat(x) ########"
python -u NRBHP_alpha_beta_model_vs_x.py \
    --model E_anchored flux_anchored inspiral_anchored

echo
echo "######## 2. inspiral alpha/beta overlays, both routes ########"
cd inspiral_only
python -u NRBHP_inspiral_alpha_beta_overlay.py
python -u NRBHP_inspiral_alpha_beta_overlay.py --global
cd ..

echo
echo "######## 3. higher-mode Fig-8 plots ########"
python -u NRBHP_hm_backbone_fig8_plots.py --q 2 4 \
    --model inspiral_anchored inspiral_fluxanchored \
    --outdir inspiral_hm
