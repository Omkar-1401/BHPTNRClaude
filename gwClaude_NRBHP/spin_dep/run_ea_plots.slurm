#!/bin/bash
#SBATCH -J eaplots
#SBATCH -o eaplots.%j.log
#SBATCH -e eaplots.%j.log
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY26026
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
unset PYTHONPATH
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
echo "=== 1/3 chi-const alpha/beta overlays ==="
python plot_energy_anchored_spin_overlay.py
echo "=== 2/3 q-const overlays ==="
python plot_spin_chi_overlays.py --models energy_anchored
echo "=== 3/3 dense mismatch heatmap ==="
python mismatch_heatmap_dense.py --models energy_anchored
echo "=== DONE ==="
ls -la Agentic_spin_plots/energy_anchored_spin/ Agentic_spin_plots/energy_anchored_spin/*/ 2>/dev/null
