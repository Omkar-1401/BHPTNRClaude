"""EXPLORATION pass 4 (derived from p3_rd.py) -- anchored ringdown on top of the installed flux_anc_spin_with_orbit (frozen).

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

OUT = HERE / "p4"; OUT.mkdir(exist_ok=True)
S18 = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
_C = {}


def pp_ring_freq(t, h):
    """ppBHPT (2,2) ringdown frequency, measured: median of d(phase)/dt over s in [15, 50] m1 after the |h| peak."""
    from scipy.signal import savgol_filter
    dt = float(np.median(np.diff(t))); n = int(round(10 / dt)) | 1
    ph = np.unwrap(np.angle(h)); ph = ph if ph[-1] > ph[0] else -ph
    w = savgol_filter(ph, n, 3, deriv=1, delta=dt, mode="interp")
    ip = int(np.argmax(np.abs(h))); s = t - t[ip]
    return float(np.median(w[(s >= 15) & (s <= 50)]))


def anchors(q, chi1, case=None):
    s1 = [0, 0, float(chi1)]
    mf = float(S._SFBH.mf(q, s1, [0, 0, 0])[0]); chif = float(S._SFBH.chif(q, s1, [0, 0, 0])[0][2])
    g_pp = PC.w220_exact(chi1) / (2 * PC.Q220_exact(chi1))
    g_nr = PC.w220_exact(chif) / (2 * PC.Q220_exact(chif)) / mf
    if case is None:
        br = float(S.beta_r_phys(q, chi1))
    else:      # measured ppBHPT ringdown frequency over the NR remnant's w220
        br = pp_ring_freq(case["t_bhpt"], case["h_bhpt"]) / (PC.w220_exact(chif) / mf)
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
        _C[k] = dict(case=case, aF=aF, bF=bF, afF=afF, bfF=bfF, tpk=tpk, anc=anchors(*k, case=case))
    return _C[k]


def smooth5(u):
    u = np.clip(u, 0, 1); return u ** 3 * (10 - 15 * u + 6 * u * u)


def ab(k, s1, s2, key, arm, bump=None):
    c = get(k)
    pre = "fit_" if key == "fit" else ""
    t = c["case"][pre + "t_bhpt"]
    aF, bF = (c["afF"], c["bfF"]) if key == "fit" else (c["aF"], c["bF"])
    br, kap = c["anc"]
    s = t - c["tpk"]
    W = smooth5((s - s1) / max(s2 - s1, 1e-3))
    beta = np.exp((1 - W) * np.log(bF) + W * np.log(br))
    if bump is not None:       # merger bump on beta: nu (b0 + b1 chi1) exp(-((s - sc)/w)^2/2)
        b0, b1, sc, w = bump
        nu = k[0] / (1 + k[0]) ** 2
        beta = beta * (1 + nu * (b0 + b1 * k[1]) * np.exp(-0.5 * ((s - sc) / max(abs(w), 0.5)) ** 2))
    if arm == "RD":
        t1 = c["tpk"] + s1
        a1 = np.interp(t1, t, aF)
        aRD = a1 * np.exp(kap * br * np.clip(t - t1, 0, None))
        alpha = np.exp((1 - W) * np.log(aF) + W * np.log(aRD))
    else:
        alpha = aF
    return alpha, beta


def err(k, s1, s2, arm, key, seed=None, bump=None):
    c = get(k); pre = "fit_" if key == "fit" else ""
    case = c["case"]
    t, h, tn, hn = case[pre + "t_bhpt"], case[pre + "h_bhpt"], case[pre + "t_nr"], case[pre + "h_nr"]
    alpha, beta = ab(k, s1, s2, key, arm, bump)
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
    if seed is None:
        grid = np.linspace(-300, 200, 126); seed = grid[int(np.argmin([f(g) for g in grid]))]
    r = minimize_scalar(f, bounds=(seed - 6, seed + 6), method="bounded", options={"xatol": 1e-3})
    return float(r.fun), float(r.x)


def full(k, s1, s2, arm, bump=None):
    e, t0 = err(k, s1, s2, arm, "fit", bump=bump)
    return err(k, s1, s2, arm, "full", seed=t0, bump=bump)[0]


def _w(a):
    return full(*a)


def main():
    pool = mp.get_context("fork").Pool(96)
    out = {}
    def report(name, s1, s2, bump):
        e96 = pool.map(_w, [(k, s1, s2, "RDb", bump) for k in T96])
        out[name] = dict(s1=s1, s2=s2, bump=bump, T96=dict(zip([f"{q:.4f}_{c:+.1f}" for q, c in T96], e96)),
                         T96_mean=float(np.mean(e96)), T96_median=float(np.median(e96)), T96_max=float(np.max(e96)))
        print(name, f"s1 {s1:.2f} s2 {s2:.2f} bump {bump}  T96 mean {np.mean(e96):.4e} median {np.median(e96):.4e} "
              f"max {np.max(e96):.4e}", flush=True)
        print("   chi1 medians: " + " ".join(f"{np.median([e for (q, c), e in zip(T96, e96) if c == ch]):.2e}" for ch in S.CHIS), flush=True)
    # A: measured ppBHPT ringdown anchor, beta only
    objA = lambda p: float(np.mean(pool.map(_w, [(k, p[0], p[1], "RDb", None) for k in S18]))) if p[1] > p[0] + 1 else 1.0
    rA = min((minimize(objA, x0, method="Nelder-Mead", options={"xatol": 0.2, "fatol": 1e-7, "maxiter": 80})
              for x0 in ([-9.6, 33.0], [0.0, 15.0])), key=lambda r: r.fun)
    s1, s2 = map(float, rA.x)
    report("A_measured_ring_anchor", s1, s2, None)
    # B: + merger bump on beta (b0, b1, sc, w), s1, s2 held
    objB = lambda p: float(np.mean(pool.map(_w, [(k, s1, s2, "RDb", tuple(p)) for k in S18])))
    rB = min((minimize(objB, x0, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-7, "maxiter": 200})
              for x0 in ([0.2, 0.0, -10.0, 8.0], [-0.2, 0.0, -10.0, 8.0], [0.2, 0.2, -30.0, 15.0])), key=lambda r: r.fun)
    report("B_plus_beta_bump", s1, s2, tuple(map(float, rB.x)))
    (OUT / "p4_ring_bump.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
