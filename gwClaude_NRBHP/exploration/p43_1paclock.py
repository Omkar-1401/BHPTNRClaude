"""EXPLORATION pass 43 -- the inspiral clock from 1PA self-force theory (theory sub-model; user rule 2026-10-18).

Pass 40's inspiral beta level and drift, X1^(6/5)(1 + b nu + b2 nu^2) + (P0 + P1 nu) E(t)  (4 fitted, rho(b, b2) = -0.97,
rho(P0, P1) = -0.94 in pass 41), is replaced by the phase-aligned 1PA clock of `sf1pa_clock.py`:
    beta_th(x_pp; q, c) = (x_pp / x_NR)^(3/2),  Phi_pp(x_pp) = Phi_NR(x_NR) + c,
    Phi_pp = (1/eps) int Omega/F0 dOmega  (ppBHPT, 0PA, m1),   Phi_NR = int Omega/(nu F0 + nu^2 F1) dOmega  (binary, 1PA, M)
    F0, F1 = WaSABI 1PAT1 forcing (Schwarzschild circular 1SF + 2SF fluxes, first-law binding energy)
    c(nu) = c0 + c1 (nu/nu5 - 1)   (2 fitted; c is the alignment phase, not fixed by theory; pass 43a: c ~ 1.2 / 3.5 / 5.1
                                     at q = 3 / 5 / 8)
Validity: 1PA breaks down near the ISCO; beta_th is frozen from the first x_NR where |nu F1/F0| >= 0.25 on (a fixed
criterion, not fitted).  The spin terms are kept: beta_I = [beta_th + g P_spin E] (1 + kappa_SO chi1 x^(3/2)),
P_spin = g(nu) [s2 nu5 chi1^2 + s3 (nu5/nu) chi1] as in pass 40 (the theory data are non-spinning).  x_pp is the ppBHPT's own
(unscaled) x.  c0, c1 live in the (now unused) b, b2 slots; P0, P1 = 0.
    H1  pass-40 onset (t1a + t1b chi1)          16c
    H2  Ori–Thorne onset, T_start = -1 (pass 42 K1)  14c
    H3, H4  (pass 50b) H2 with the freeze at |nu F1/F0| >= 0.10 / 0.05 (pass 43a: the 1PA clock leaves the fitted one where
            |nu F1/F0| ~ 0.1, well before 0.25)                                                                       14c
    H5  (pass 50c) H3 + the energy drift P0 E(t) kept free (the frozen clock has no late-inspiral drift): c0, c1, P0
            replace b, b2, P0, P1                                                                                     15c
Base = pass 40 (L4 coefficients, closed-form R2 O2).  Usage: python p43_1paclock.py H1|H2 [budget].  -> p43/p43_<var>.json
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; THR = {"H3": 0.10, "H4": 0.05, "H5": 0.10}.get(VAR, 0.25); BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
sys.argv = [sys.argv[0], "K1"]
import p42_otonset as K42                       # noqa: E402  (pass-40 machinery, widened onset guard, OT onset hook)
import sf1pa_clock as SF                        # noqa: E402
import fit_pn_anchored_spin as S                # noqa: E402
import fit_flux_anc_spin_with_orbit as FO       # noqa: E402
C = K42.C; L = K42.L; M = K42.M; P6 = K42.P6; A = K42.A; PR = K42.PR
OUT = HERE / "p43"; OUT.mkdir(exist_ok=True)
NAMES = K42.NAMES; IX = {n: i for i, n in enumerate(NAMES)}
R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
NU5 = 5.0 / 36.0
_orig_flux13 = M.flux13
_BT = {}


def freeze_x(nu):
    x, _, _, _ = SF.phase_tables(round(nu, 12))
    F0, F1 = SF.F0F1(1 / x)
    bad = np.abs(nu * F1 / F0) >= THR
    return float(x[np.argmax(bad)]) if bad.any() else float(x[-1])


def beta_clock(xpp, q, c):
    nu = q / (1 + q) ** 2
    b, xn = SF.beta_th(xpp, q, c)
    xf = freeze_x(nu)
    over = xn >= xf
    if over.any():
        i = int(np.argmax(over)); b = b.copy(); b[i:] = b[i]     # freeze from the first x_NR beyond the 1PA validity
    return b


def flux13_th(f, q, chi1, case, key):
    """p14 flux13 with b_pp -> the 1PA clock; f[4] = c0, f[9] = c1 (the b, b2 slots); P0 = P1 = 0 by construction."""
    c0, c1 = f[4], f[9]
    nu = q / (1 + q) ** 2; c = c0 + c1 * (nu / NU5 - 1)
    pre = "fit_" if key == "fit" else ""
    k = (q, chi1, key, round(float(c), 10))
    if k not in _BT:
        if len(_BT) > 200:
            _BT.clear()
        _BT[k] = beta_clock(np.asarray(case[pre + "x"]), q, c)
    g = np.array(f, float); g[4] = 0.0; g[9] = 0.0; g[6] = 0.0     # b, b2, P1 off in the empirical form
    if VAR != "H5":
        g[5] = 0.0                                                # P0 off too (H1-H4)
    alpha, beta_emp = _orig_flux13(g, q, chi1, case, key)
    # beta_emp = (X1^(6/5) + P_spin E)(1 + SO);  swap its level X1^(6/5) for the clock
    X1 = q / (1 + q)
    xs = S.PC.x_scale(q) * np.minimum(case[pre + "x"], S.XCLIP)
    sof = 1.0 + FO.kappa_so(q) * chi1 * xs ** 1.5             # the same spin-orbit factor flux13 applies
    beta = beta_emp + (_BT[k] - X1 ** 1.2) * sof
    return alpha, beta


def main():
    P6.get = L.get; P6.ab = L.ab; C.use("O2n")
    M.flux13 = flux13_th                         # p25's ab calls flux13 through its module global
    if VAR in ("H2", "H3", "H4", "H5"):
        M.S1FUN = K42.onset
    th0 = np.array(R["theta"])
    th0[IX["flux_b"]] = 3.5; th0[IX["flux_b2"]] = -5.0; th0[IX["flux_P0"]] = 0.0; th0[IX["flux_P1"]] = 0.0
    if VAR == "H5":
        th0 = np.array(json.loads((OUT / "p43_H3.json").read_text())["theta"])   # warm start: H3 (P0 = 0 there)
    drop = {IX["flux_P1"]} if VAR == "H5" else {IX["flux_P0"], IX["flux_P1"]}
    if VAR in ("H2", "H3", "H4", "H5"):
        th0[IX["t1a"]] = -1.0; th0[IX["t1b"]] = 0.0; drop |= {IX["t1a"], IX["t1b"]}
    free = [i for i in R["free"] if i not in drop]
    pool = mp.get_context("fork").Pool(96)
    e0 = float(np.mean(pool.map(P6._fit, [(list(th0), k) for k in P6.T96])))
    print(f"{VAR}: {len(free)}c; start fit-mean {e0:.4e} (pass 40: 1.983e-4)", flush=True)
    th, ef = A.refit(pool, th0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, freeze=THR, n_coef=len(free), free=free, theta=list(map(float, th)), c0=float(th[IX["flux_b"]]),
               c1=float(th[IX["flux_b2"]]), fit_mean=ef, start_fit_mean=e0, **sc)
    (OUT / f"p43_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, hh = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c (c0 {th[IX['flux_b']]:+.3f}, c1 {th[IX['flux_b2']]:+.3f}): T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | "
          f"H270 {hh[0]:.4e} / {hh[1]:.4e} / {hh[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
