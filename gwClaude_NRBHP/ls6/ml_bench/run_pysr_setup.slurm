#!/bin/bash
# ml_bench: one-off Julia download + PySR precompile (depot on $WORK, not $HOME)
#SBATCH -J pysr_setup
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/pysr_setup.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/pysr_setup.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A PHY26029
unset PYTHONPATH LD_LIBRARY_PATH   # TACC intel libs break Julia PCRE
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export JULIA_DEPOT_PATH=/work2/11702/omkarsridhar1401/stampede3/julia_depot
export PYTHON_JULIAPKG_PROJECT=/work2/11702/omkarsridhar1401/stampede3/julia_env_pysr
export PYTHON_JULIACALL_THREADS=8
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench
python - <<'PY'
import time, numpy as np
t = time.time()
from pysr import PySRRegressor
print("import", round(time.time() - t, 1), "s", flush=True)
X = np.random.default_rng(0).uniform(-1, 1, (96, 2)); y = 2.5 * np.cos(X[:, 0]) + X[:, 1] ** 2
m = PySRRegressor(niterations=20, binary_operators=["+", "*", "-", "/"], unary_operators=["cos"],
                  verbosity=0, progress=False, random_state=0, deterministic=True, parallelism="serial")
m.fit(X, y); print(m.sympy(), flush=True)
print("total", round(time.time() - t, 1), "s")
PY
