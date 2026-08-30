#!/bin/bash
#SBATCH -J wrd_meas
#SBATCH -o wrd_meas.%j.log
#SBATCH -e wrd_meas.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
python -u wardell_flux_ratio_measured.py
