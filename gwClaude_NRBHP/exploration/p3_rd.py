"""EXPLORATION pass 3 -- anchored ringdown on top of the installed flux_anc_spin_with_orbit (frozen).

    s = t - t_peak,pp              (ppBHPT clock, m1; pass 0: merger features sit at fixed s, q-independent)
    W(s) = smoothstep5((s - s1)/(s2 - s1))                    (C2, 0 before s1, 1 after s2)
    ln beta  = (1 - W) ln beta_F  + W ln beta_r               beta_r = w220(chi1) M_f / w220(chi_f)   (exact)
    ln alpha = (1 - W) ln alpha_F + W ln alpha_RD
    alpha_RD(t) = alpha_F(t1) exp[ kappa_RD beta_r (t - t1) ],  t1 = t_peak,pp + s1
    kappa_RD = gamma_pp / beta_r - gamma_NR    (B5: the damping-rate mismatch; gamma = |Im w220|, pp per m1, NR per M)

Fitted: s1, s2 only (2 global numbers), on S18 (mean full-window E).  The 9 inspiral/merger coefficients are the
installed ones.  Arm "RDb": beta anchor only (alpha_RD = alpha_F), to separate the two pieces.
Then all T96 scored at full resolution.  Writes exploration/p3/p3_rd.json.
"""
from __future__ import annotations
import json, sys, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import fit_pn_anchored_spin as S              # noqa: E402
import fit_flux_anc_spin_with_orbit as FO     # noqa: E402
import fit_fluxanchored_spin as FS            # noqa: E402
import test_guard                              # noqa: E402,F401  (refuses sealed test points)
import physics_common as PC                   # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p3"; OUT.mkdir(exist_ok=True)
S18 = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
_C = {}


def anchors(q, chi1):
    s1 = [0, 0, float(chi1)]
    mf = float(S._SFBH.mf(q, s1, [0, 0, 0])[0]); chif = float(S._SFBH.chif(q, s1, [0, 0, 0])[0][2])
    g_pp = PC.w220_exact(chi1) / (2 * PC.Q220_exact(chi1))
    g_nr = PC.w220_exact(chif) / (2 * PC.Q220_exact(chif)) / mf
    br = float(S.beta_r_phys(q, chi1))
    return br, g_pp / br - g_nr


def get(k):
    if k not in _C:
        if len(_C) >= 20:          # bound per-worker memory (a case is ~20 MB)
            _C.clear()
        case = FS.build_case(*k)
        th = FO.load_theta()
        aF, bF = FO.model_ab(th, k[0], k[1], case, "full")
        afF, bfF = FO.model_ab(th, k[0], k[1], case, "fit")
        tpk = float(case["t_bhpt"][int(np.argmax(np.abs(case["h_bhpt"])))])
        _C[k] = dict(case=case, aF=aF, bF=bF, afF=afF, bfF=bfF, tpk=tpk, anc=anchors(*k))
    return _C[k]


def smooth5(u):
    u = np.clip(u, 0, 1); return u ** 3 * (10 - 15 * u + 6 * u * u)


def ab(k, s1, s2, key, arm):
    c = get(k)
    pre = "fit_" if key == "fit" else ""
    t = c["case"][pre + "t_bhpt"]
    aF, bF = (c["afF"], c["bfF"]) if key == "fit" else (c["aF"], c["bF"])
    br, kap = c["anc"]
    s = t - c["tpk"]
    W = smooth5((s - s1) / max(s2 - s1, 1e-3))
    beta = np.exp((1 - W) * np.log(bF) + W * np.log(br))
    if arm == "RD":
        t1 = c["tpk"] + s1
        a1 = np.interp(t1, t, aF)
        aRD = a1 * np.exp(kap * br * np.clip(t - t1, 0, None))
        alpha = np.exp((1 - W) * np.log(aF) + W * np.log(aRD))
    else:
        alpha = aF
    return alpha, beta


def err(k, s1, s2, arm, key, seed=None):
    c = get(k); pre = "fit_" if key == "fit" else ""
    case = c["case"]
    t, h, tn, hn = case[pre + "t_bhpt"], case[pre + "h_bhpt"], case[pre + "t_nr"], case[pre + "h_nr"]
    alpha, beta = ab(k, s1, s2, key, arm)
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
    if seed is None:
        grid = np.linspace(-300, 200, 126); seed = grid[int(np.argmin([f(g) for g in grid]))]
    r = minimize_scalar(f, bounds=(seed - 6, seed + 6), method="bounded", options={"xatol": 1e-3})
    return float(r.fun), float(r.x)


def full(k, s1, s2, arm):
    e, t0 = err(k, s1, s2, arm, "fit")
    return err(k, s1, s2, arm, "full", seed=t0)[0]


def _w(a):
    return full(*a)


def main():
    pool = mp.get_context("fork").Pool(96)
    out = {}
    for arm in ("RDb", "RD"):
        obj = lambda p: float(np.mean(pool.map(_w, [(k, p[0], p[1], arm) for k in S18]))) if p[1] > p[0] + 1 else 1.0
        best = None
        for x0 in ([0.0, 15.0], [5.0, 25.0], [-5.0, 10.0]):
            r = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 0.2, "fatol": 1e-7, "maxiter": 80})
            if best is None or r.fun < best.fun:
                best = r
        s1, s2 = map(float, best.x)
        e18 = pool.map(_w, [(k, s1, s2, arm) for k in S18])
        e96 = pool.map(_w, [(k, s1, s2, arm) for k in T96])
        out[arm] = dict(s1=s1, s2=s2, S18_mean=float(np.mean(e18)), T96=dict(zip([f"{q:.4f}_{c:+.1f}" for q, c in T96], e96)),
                        T96_mean=float(np.mean(e96)), T96_median=float(np.median(e96)), T96_max=float(np.max(e96)))
        print(arm, f"s1 {s1:.2f} s2 {s2:.2f}  S18 mean {np.mean(e18):.4e}  T96 mean {np.mean(e96):.4e} "
              f"median {np.median(e96):.4e} max {np.max(e96):.4e}", flush=True)
        for ch in S.CHIS:
            v = [e for (q, c), e in zip(T96, e96) if c == ch]
            print(f"   chi1 {ch:+.1f}: median {np.median(v):.3e} mean {np.mean(v):.3e}", flush=True)
    ref = json.loads(FO.COEFFS.read_text())["mismatches"]
    out["F_installed"] = dict(T96_mean=float(np.mean(list(ref.values()))), T96_median=float(np.median(list(ref.values()))))
    (OUT / "p3_rd.json").write_text(json.dumps(out, indent=1))
    print("installed F:", out["F_installed"])


if __name__ == "__main__":
    main()
