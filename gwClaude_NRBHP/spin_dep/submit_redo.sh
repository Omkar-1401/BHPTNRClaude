#!/bin/bash
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
until out=$(sbatch run_redo.slurm 2>&1) && echo "$out" | grep -q Submitted; do sleep 90; done
echo "$(date +%H:%M) redo -> $(echo "$out" | grep -o '[0-9]\+$')"
