"""EXPLORATION pass 42 -- the blend onset from Ori–Thorne theory (user rule 2026-10-18: theory sub-models are free).

Pass 40 (leader) has a fitted onset s1 = t1a + t1b chi1 (2 coefficients; -150 / -102 / -39 m1 at chi1 = -0.6 / 0 / +0.8).
Here the onset sits at a fixed Ori–Thorne time T_start of the ppBHPT's own transition (spin_dep_split/ot_join.py):
    s1 = s_isco + (T_start - T*) u_OT,
    s_isco = (ppBHPT time where its (2,2) frequency first reaches 2 Omega_isco(chi1)) - t_pk          (measured, m1)
    u_OT   = q^(1/5) tau_o(chi1) / (dtau/dt)_isco(chi1)                                              (OT unit, m1)
    T*     = 0.7154 (OT universal solution crosses the ISCO)
Login-node check (2026-10-18): a single T_start = -0.40 reproduces the fitted onsets; T_start = -1 (the OT departure from
adiabatic inspiral, the split models' theory value) puts them 19 m1 earlier on average.
    K1  T_start = -1 fixed (0 onset coefficients; t1a, t1b removed): 16c
    K2  T_start fitted (1 coefficient, stored in the t1a slot; t1b removed): 17c
    K1n as K1 with the OT small parameter eta = nu instead of 1/q (u_OT ~ nu^(-1/5)): pass 42 K1 brought back the rise past
        q = 5 (q8/q5 1.95); nu^(-1/5) grows more slowly with q than q^(1/5) at large q relative to q = 3 (16c)
The fitted form's validity window (-150 <= s1) was a guard for the linear onset; for the OT onset it is widened to
-1000 <= s1 (s1 + 3 <= s2 = 0 is kept).  Base = pass 40 (pass 39 L4 + closed-form R2, O2); the other 16 coefficients are
refit on T96.  Usage: python p42_otonset.py K1|K2 [budget_min].  Writes exploration/p42/p42_<var>.json.
"""
from __future__ import annotations
import inspect, json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "spin_dep_split"))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
import p40_r2closed as C                        # noqa: E402  (pass 39 L4 machinery + closed-form R2)
import ot_join as OT                            # noqa: E402
if VAR == "K1n":
    OT.ETA = "nu"
L = C.L; G = L.G; M = G.M; P6 = C.P6; A = C.A; PR = C.PR
OUT = HERE / "p42"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
NAMES = json.loads((HERE / "p32" / "p32_G2.json").read_text())["names"]
IA, IB = NAMES.index("t1a"), NAMES.index("t1b")

# p25's ab with the onset guard widened (-150 -> -1000); everything else verbatim
_src = inspect.getsource(M.ab).replace("-150 <= s1 <= 30", "-1000 <= s1 <= 30")
assert "-1000 <= s1 <= 30" in _src
exec(compile(_src, M.__file__, "exec"), M.__dict__)
G.base_ab = M.ab                                # the p31 / p39 wrappers call G.base_ab

_ON = {}


def onset(k, d):
    """s1 = s_isco + (T_start - T*) u_OT; T_start = theta[t1a]."""
    if k not in _ON:
        c = P6.get(k); case = c["case"]; t, h = case["t_bhpt"], case["h_bhpt"]
        ti = OT.t_isco_pp(t, h, k[1]); _, unit = OT.ot_time(t, ti, k[0], k[1])
        _ON[k] = (ti - c["tpk"], unit)
    si, u = _ON[k]
    return si + (d["t1a"] - OT.T_STAR) * u


def main():
    P6.get = L.get; P6.ab = L.ab; C.use("O2n")  # BEFORE the fork (pass 40 base)
    pool = mp.get_context("fork").Pool(96)
    th40 = np.array(R["theta"])
    chk = P6.T96[::8]
    ref = json.loads((HERE / "p40" / "p40.json").read_text())["O2n"]["T96_cases"]
    e = np.array(pool.map(P6._full, [(list(th40), k) for k in chk]))
    r = e / np.array([ref[f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: widened guard reproduces pass 40, ratio {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    pool.close()
    M.S1FUN = onset                              # the onset hook of p25's ab
    th0 = th40.copy(); th0[IA] = -0.40 if VAR == "K2" else -1.0; th0[IB] = 0.0
    free = [i for i in R["free"] if i != IB and (VAR == "K2" or i != IA)]
    pool = mp.get_context("fork").Pool(96)
    e0 = float(np.mean(pool.map(P6._fit, [(list(th0), k) for k in P6.T96])))
    print(f"   start (pass 40 coefficients, OT onset T_start = {th0[IA]:+.2f}): fit-mean {e0:.4e}; {len(free)}c", flush=True)
    th, ef = A.refit(pool, th0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, T_start=float(th[IA]), n_coef=len(free), free=free, theta=list(map(float, th)), fit_mean=ef,
               start_fit_mean=e0, **sc)
    (OUT / f"p42_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, hh = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c (T_start {th[IA]:+.3f}): T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {hh[0]:.4e} / {hh[1]:.4e} / {hh[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
