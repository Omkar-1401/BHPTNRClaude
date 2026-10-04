#!/bin/bash
#SBATCH -J hmdense
#SBATCH -o hmdense.%j.log
#SBATCH -e hmdense.%j.log
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
# one BLAS thread per worker: the parallelism is the Pool over (q, chi1) points
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Dense 20 q x 20 chi1 E(q, chi1) heatmaps for all four spin models (1600 mismatches,
# Pool over the node's cores).  Grids -> <model>_mismatch_grid_dense.json, figures ->
# <model>_mismatch_heatmap_dense.{pdf,png}.  Redraw only: add --plot-only.
python -u mismatch_heatmap_dense.py
