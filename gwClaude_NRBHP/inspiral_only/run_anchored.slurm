#!/bin/bash
#SBATCH -J insp_anch
#SBATCH -o insp_anch.%j.log
#SBATCH -e insp_anch.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A PHY26026

cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/inspiral_only

unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- 0. protect the completed 64-q flux grid before any --force ---
cp inspiral_fluxanchored_results/per_q_cache.json \
   inspiral_fluxanchored_results/per_q_cache.json.bak

echo "############ REGRESSION TEST: flux path after the --model refactor ############"
# --force refits these 3 q with the refactored code; the built-in validation compares
# them against beta_drift_tests/inspiral_only_sign.py.  Must still reproduce.
python -u fit_scaling_inspiral_fluxanchored.py --check --force --nproc 3

echo
echo "############ NEW MODEL: inspiral_anchored (E-driven alpha), 64-q grid ##########"
python -u fit_scaling_inspiral_anchored.py --nproc 24
