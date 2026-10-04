#!/bin/bash
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
until out=$(sbatch run_fa_plots.slurm 2>&1) && echo "$out" | grep -q Submitted; do sleep 60; done
echo "$(date +%H:%M) fa plots -> $(echo "$out" | grep -o '[0-9]\+$')"
