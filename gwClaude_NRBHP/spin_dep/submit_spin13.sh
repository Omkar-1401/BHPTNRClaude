#!/bin/bash
cd /work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep
# wait for the corrected nonspinning fit (spin13 freezes its nine coefficients)
until [ -f ../pn_anchored_results/coeffs_corrected_betar_multistart.json ]; do sleep 60; done
until out=$(sbatch run_spin13.slurm 2>&1) && echo "$out" | grep -q Submitted; do sleep 60; done
echo "$(date +%H:%M) spin13 -> $(echo "$out" | grep -o '[0-9]\+$')"
