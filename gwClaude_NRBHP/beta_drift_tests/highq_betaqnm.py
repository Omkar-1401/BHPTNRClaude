"""Evaluate the gwr_betaqnm --bfree model ABOVE its training range.

Training range is IN_Q = (3..8); LOW_Q = (2.75..2.0) is the documented downward
extrapolation.  This asks the opposite question: what happens at q > 8?

CAVEAT, printed again at the end: NRHybSur3dq8 is trained to q <= 8, so for q > 8 the
NR *reference* is extrapolated too.  These numbers are model-vs-extrapolated-surrogate,
not model-vs-NR.  Distinct from (and on the opposite side of) the q < 2.5 BHPT caveat.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_betaqnm as BQ

TH = np.asarray(json.loads((ROOT/"gwr_betaqnm_results"/"coeffs_bfree.json").read_text())["theta"], float)
print("theta (c0 c1 A m b) =", "  ".join(f"{v:.4f}" for v in TH), flush=True)

QS = [5.0, 6.0, 7.0, 8.0, 8.4, 8.8, 9.0, 9.4, 9.8, 10.0]

print(f"\n{'q':>6s} {'in range?':>10s} {'mismatch':>12s} {'B0':>9s} {'Bm':>9s} "
      f"{'B1(QNM)':>9s} {'gb %':>8s} {'insp rise':>10s}")
rows = []
for q in QS:
    case = FL.add_flux(G.load_case(q, 6, 10))
    P = BQ.prep(q, case)
    e = BQ.err(q, P, case, TH, True)
    c0, c1, A, m, b = TH
    _, beta, B0, Bm, gb = BQ.shape(P, c0, c1, A, m, b)
    tag = "train" if q <= 8.0 else "EXTRAP"
    print(f"{q:6.2f} {tag:>10s} {e:12.4e} {B0:9.5f} {Bm:9.5f} {P['B1']:9.5f} "
          f"{gb*100:+8.3f} {(Bm-B0)/B0*100:+9.2f}%", flush=True)
    rows.append(dict(q=q, mismatch=e, B0=B0, Bm=Bm, B1=P["B1"], gb=gb))

(Path(__file__).parent/"highq_betaqnm.json").write_text(json.dumps(rows, indent=2, default=float))
print("\nCAVEAT: q > 8 is outside NRHybSur3dq8's training range, so the NR reference is")
print("itself extrapolated there.  Compare across the q > 8 rows, not against q <= 8.")
