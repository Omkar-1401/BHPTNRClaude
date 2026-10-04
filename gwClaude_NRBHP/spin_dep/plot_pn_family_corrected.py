"""Redraw pn_anchored_spin and pn_anc_spin_physical figures on the CORRECTED coefficients.

The plot scripts read shipped coefficient files (shipped9 -> pn_anchored_results/
coeffs.json, load_theta_spin -> coeffs_spin.json, shipped11 -> coeffs_physical.json),
which still hold the PUBLISHED values.  The code now carries the corrected beta_r, so a
plain rerun would draw published coefficients through corrected physics -- a model that
does not exist.  This wrapper injects the corrected coefficients IN MEMORY and touches no
coefficient file:
  pn_anchored_spin      nine = redo_properly_results/spin13.json frozen9_corrected
                        four = spin13.json steps['2'] theta_spin
  pn_anc_spin_physical  11   = redo_properly_results/physical_partial.json (mode 2)
A known cell is scored first and must match the corrected grid, else it stops.
Pre-correction figures are kept in Agentic_spin_plots/*_PREFIX_BACKUP_20260924/.
"""
import sys, json, tempfile

# RETIRED 2026-09-26.  This wrapper injected Thursday's intermediate "corrected" coefficients
# (redo_properly_results/), which predate the 2026-09-26 fixes (x on the note's clock, exact Kerr
# QNMs, new nonspinning 9) and were themselves unconverged.  The shipped coefficient files are now
# refit and stamped, so the ordinary plot scripts (plot_spin_chi_overlays.py,
# plot_pn_anc_spin_physical_overlay.py, plot_junction_spin_overlay.py) draw the shipped model
# directly.  Refusing to run so it cannot draw a superseded model.
if __name__ == "__main__":
    raise SystemExit("plot_pn_family_corrected.py is retired (2026-09-26): use the ordinary spin plot "
                     "scripts, which read the refit, stamped coefficient files.")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as P

R = HERE / "redo_properly_results"
s13 = json.loads((R / "spin13.json").read_text())
TH9 = np.array(s13["frozen9_corrected"], float)
THS = np.array(s13["steps"]["2"]["best"]["theta_spin"], float)
TH11 = np.array(json.loads((R / "physical_partial.json").read_text())["rows"][0]["theta"], float)
TAG = R / "CORRECTED(in-memory)"

S.shipped9 = lambda *a, **k: TH9.copy()
P.shipped11 = lambda *a, **k: (TH11.copy(), TAG)
tmp = Path(tempfile.mkdtemp()) / "pred_corrected.json"
tmp.write_text(json.dumps({"theta_spin": THS.tolist()}))
P.PRED = tmp                                   # predecessor diff sheet -> corrected 13c

import spin_alpha_beta_overlay as A
A.load_theta_spin = lambda *a, **k: (THS.copy(), TAG)
import plot_pn_anc_spin_physical_overlay as B

# ---- verify before drawing: known cells from today's corrected grids ----
ea = S.full_mm(TH9, THS, 5.0, 0.0, S.build_case(5.0, 0.0))
eb, _ = P.mismatch(TH11, 5.0, 0.0, S.build_case(5.0, 0.0), None)
print(f"verify pn_anchored_spin q=5 chi=0: {ea:.4e} (grid 0.61e-03)")
print(f"verify physical        q=5 chi=0: {eb:.4e} (grid 0.64e-03)")
assert abs(ea / 6.1e-4 - 1) < 0.03 and abs(eb / 6.4e-4 - 1) < 0.03, "coefficients not injected"

which = sys.argv[1]
if which == "A":
    sys.argv = ["x"]; A.main()
elif which == "B":
    sys.argv = ["x"]; B.main()
elif which == "Q":
    import plot_spin_chi_overlays as O
    sys.argv = ["x", "--models", "pn_anchored", "physical"]; O.main()
elif which == "H":
    import mismatch_heatmap_dense as H
    sys.argv = ["x", "--models", "pn_anchored", "physical", "--nproc", "8"]; H.main()
