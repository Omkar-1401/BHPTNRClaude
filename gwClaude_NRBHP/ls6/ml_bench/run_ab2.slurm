#!/bin/bash
# ml_bench: paradigm B rerun (clock target abtau; ab at K 40, 60)
#SBATCH -J mb_ab2
#SBATCH -o /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/ab2.%j.log
#SBATCH -e /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench/logs/ab2.%j.log
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:45:00
#SBATCH -A PHY26029
unset PYTHONPATH LD_LIBRARY_PATH   # TACC intel libs break Julia PCRE
source /work2/11702/omkarsridhar1401/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11702/omkarsridhar1401/stampede3/envs/ut_claude
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_DEPOT_PATH=/work2/11702/omkarsridhar1401/stampede3/julia_depot
export PYTHON_JULIAPKG_PROJECT=/work2/11702/omkarsridhar1401/stampede3/julia_env_pysr
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/ml_bench
python -u ab_targets.py extract --nproc 64 && python -u bench_direct.py cv --target abtau --ks 6,12,20 --nproc 128 && python -u bench_direct.py final --target abtau --nproc 128; python -u bench_direct.py cv --target ab --ks 40,60 --nproc 128 && python -u bench_direct.py final --target ab --nproc 128
