#!/bin/bash
# Slim MR branch: greedy selection on S18 (staged in scratch).
#SBATCH -J slimsel
#SBATCH -o /scratch/11702/omkarsridhar1401/spin_dep_split_tmp/logs/slim_select.%j.log
#SBATCH -e /scratch/11702/omkarsridhar1401/spin_dep_split_tmp/logs/slim_select.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26029
cd /scratch/11702/omkarsridhar1401/spin_dep_split_tmp
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname; date
python -u slim_split.py --select
echo "exit $?"; date
