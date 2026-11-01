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
_CLI = list(sys.argv)
sys.argv = [sys.argv[0], "40"]
import p46_closedform as B                      # noqa: E402  (pass-40 chain + closed-form QNM / remnant / decay)
from fit_shapes_closedform import shape_cf      # noqa: E402
from fit_shapes_cf2 import f2                   # noqa: E402
C, L, P6, A, PR, CF, S, PC = B.C, B.L, B.P6, B.A, B.PR, B.CF, B.S, B.PC
M = L.G.M                                       # p25: its ab reads M.A._PB / _PA (phi shapes) and M.L.SH (psi shapes)
OUT = HERE / "p47"; OUT.mkdir(exist_ok=True)
VER = sys.argv[2] if len(sys.argv) > 2 else "v1"            # kept for the record; set below from the CLI
SH = None
TH = np.array(B.R["theta"])


def _cf(key):
    d = SH[key]
    if d.get("form") == "sech2":
        return lambda s, p=d["params"], nb=d["nb"]: f2(p, s, nb)
    return lambda s, p=d["params"]: shape_cf(p, s)


def install_shapes(alpha=True, beta=True):
    if beta:
        M.A._PB = _cf("phiB")
    if alpha:
        M.A._PA = _cf("phiA")
    new = []
    for field, sp, rng, amps in M.L.SH:
        key = "psiB" if field == "dlnB" else "psiA"
        use = beta if field == "dlnB" else alpha
        new.append((field, _cf(key) if use else sp, rng, amps))
    M.L.SH[:] = new


def main():
    global SH
    SH = json.loads((HERE / "shapes_closedform_v2.json").read_text())
    B.OUT = OUT                                  # B.score writes its running json here (never over p46/p46.json)
    P6.get = B.get; P6.ab = L.ab; C.use("O2n")
    CF.install(S, PC, qnm=True, remnant=True); B._USE_G["v"] = True
    out = {}
    pb, pa, sh = M.A._PB, M.A._PA, list(M.L.SH)
    for tag, al, be in (("Salpha", True, False), ("Sbeta", False, True), ("Sboth", True, True)):
        M.A._PB, M.A._PA = pb, pa; M.L.SH[:] = sh      # reset to the splines
        install_shapes(alpha=al, beta=be); B.score(tag, TH, out)
    pool = mp.get_context("fork").Pool(96)
    th, ef = A.refit(pool, TH, list(B.R["free"]), BUD, "Sboth_r"); pool.close()
    out["Sboth_r_theta"] = list(map(float, th)); B.score("Sboth_r", th, out)
    (OUT / "p47b.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
