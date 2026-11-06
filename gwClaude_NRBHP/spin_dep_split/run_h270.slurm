#!/bin/bash
# Overnight: H270 held-out scores (no fitting).
#SBATCH -J h270
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep_split/logs/h270.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep_split/logs/h270.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26029
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep_split
unset PYTHONPATH
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname; date
python -u score_h270.py
echo "exit $?"; date
