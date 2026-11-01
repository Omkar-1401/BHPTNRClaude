"""EXPLORATION pass 47 -- closed-form shape functions on top of pass 46 (closed-form QNM, remnant, ringdown decay).

The four spline shapes (phi_B, phi_A: pass 12; psi_B, psi_A: pass 17) are replaced by the closed forms of
fit_shapes_closedform.py (shapes_closedform.json): A0 exp(min(s,0)/tau) [A0 for s >= 0] + sum_{k=1..4} A_k sech((s-mu_k)/w_k),
s clipped to each spline's own range.  Then every numerical input of the model is a closed form (the inputs are only the
ppBHPT waveform and its measured drives E(t), x_pp(t)).
    S0   closed-form shapes, pass-40 coefficients frozen          S0r   the 18 coefficients refit on T96
Usage: python p47_shapescf.py [budget].  Writes exploration/p47/p47.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BUD = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
sys.argv = [sys.argv[0], "40"]
import p46_closedform as B                      # noqa: E402  (pass-40 chain + closed-form QNM / remnant / decay)
from fit_shapes_closedform import shape_cf      # noqa: E402
C, L, P6, A, PR, CF, S, PC = B.C, B.L, B.P6, B.A, B.PR, B.CF, B.S, B.PC
M = L.G.M                                       # p25: its ab reads M.A._PB / _PA (phi shapes) and M.L.SH (psi shapes)
OUT = HERE / "p47"; OUT.mkdir(exist_ok=True)
SH = json.loads((HERE / "shapes_closedform.json").read_text())
TH = np.array(B.R["theta"])


def install_shapes():
    pa, pb = SH["phiA"]["params"], SH["phiB"]["params"]
    M.A._PB = lambda s, p=pb: shape_cf(p, s)     # p25 clips s to M.A._KB / _KA before calling (unchanged ranges)
    M.A._PA = lambda s, p=pa: shape_cf(p, s)
    new = []
    for field, sp, rng, amps in M.L.SH:
        key = "psiB" if field == "dlnB" else "psiA"
        new.append((field, (lambda s, p=SH[key]["params"]: shape_cf(p, s)), rng, amps))
    M.L.SH[:] = new


def main():
    B.OUT = OUT                                  # B.score writes its running json here (never over p46/p46.json)
    P6.get = B.get; P6.ab = L.ab; C.use("O2n")
    CF.install(S, PC, qnm=True, remnant=True); B._USE_G["v"] = True
    out = {}
    ref = B.score("Call", TH, out)                      # pass 46 Call, reproduced before the swap
    install_shapes()
    B.score("S0", TH, out)
    pool = mp.get_context("fork").Pool(96)
    th, ef = A.refit(pool, TH, list(B.R["free"]), BUD, "S0r"); pool.close()
    out["S0r_theta"] = list(map(float, th)); B.score("S0r", th, out)
    (OUT / "p47.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
