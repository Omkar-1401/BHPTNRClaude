"""Overnight: score the flux split variant on H270 (EXPLORATION.md held-out set: q in linspace(2,8,20) with 3 < q < 8,
chi1 in linspace(-0.6,0.8,20) without the two ends; 270 interior points off the training grid; caches read-only).
Models: flux split T96-fitted and S18-fitted (results/flux_spin_split_*_T-1.json), installed pn_anchored_spin and
fluxanchored_spin.  Both alignments; no fitting.  -> results/h270_scores.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_flux_spin_split as FF
M = FF.M

QS = np.linspace(2, 8, 20); CS = np.linspace(-0.6, 0.8, 20)
H270 = [(float(q), float(c)) for q in QS for c in CS[1:-1] if 3 < q < 8]


def main():
    assert len(H270) == 270
    FF.KEYS = H270
    for k in H270:
        M.CASES[k] = M.prepare(*k)
    fits = {name: json.loads((HERE / "results" / f"flux_spin_split_{name}_T-1.json").read_text()) for name in ("T96", "S18")}
    out = {}
    with get_context("fork").Pool(128) as pool:
        for name, d in fits.items():
            out[f"flux_split_{name}"] = FF.final(pool, "split", np.array(d["theta_insp"]), np.array(d["theta_mr"]))
        for kind in ("pn13", "flux9"):
            out[f"installed_{kind}"] = FF.final(pool, kind, None, None)
    for name, rows in out.items():
        FF.summary(rows, name)
    v = np.median([r["full_E_full"] for r in out["installed_pn13"]])
    print(f"check: installed pn_anchored_spin H270 median (full-aligned) {v:.3e}; EXPLORATION.md baseline 7.10e-4")
    (HERE / "results" / "h270_scores.json").write_text(json.dumps(out, indent=1))
    print("wrote results/h270_scores.json")


if __name__ == "__main__":
    main()
