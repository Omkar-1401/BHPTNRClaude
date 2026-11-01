"""EXPLORATION pass 6 -- M2, global fit (first regressed version).

ppBHPT clock t (m1); s = t - t_peak,pp.  FLUX branch = the installed flux_anc_spin_with_orbit form
(9 coefficients, FO.params_at, kappa_SO factor on beta).  Merger-ringdown branch, blended in ln over [s1, s2]
with a C2 smoothstep W(s):
    ln beta_MR  = ln beta_r + rho,                    rho = r0 + r1 chi1 + r2 nu
    ln alpha_MR = ln X1^(6/5) + l0 + l1 chi1 + l2 nu + kappa_RD beta_r max(s - 15, 0)     (B5 slope, ringdown only)
    beta_r = w220(chi1) M_f / w220(chi_f)  (exact);   kappa_RD = gamma_pp / beta_r - gamma_NR  (exact)
17 coefficients: 9 FLUX + s1, s2 + r0, r1, r2 + l0, l1, l2.
Objective: mean full-window mathcalE (fit-stride arrays, grid + bounded t0 search per case), Powell.
Stage 1: fit on S18 (screen).  Stage 2: refit on T96 from stage 1.  Both scored at full resolution on T96;
H270 (held-out, interior) scored for the stage-2 coefficients.  Writes exploration/p6/p6_m2.json.
"""
from __future__ import annotations
import json, sys, time, warnings
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
import p3_rd as P3                            # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p6"; OUT.mkdir(exist_ok=True)
S18 = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
NAMES = list(FO.NAMES) + ["s1", "s2", "r0", "r1", "r2", "l0", "l1", "l2"]
S_RING = 15.0
_C = {}


def get(k):
    if k not in _C:
        if len(_C) >= 24:
            _C.clear()
        case = FS.build_case(*k)
        tpk = float(case["t_bhpt"][int(np.argmax(np.abs(case["h_bhpt"])))])
        _C[k] = dict(case=case, tpk=tpk, anc=P3.anchors(*k))
    return _C[k]


def ab(th, k, key):
    q, chi1 = k; c = get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    t = case[pre + "t_bhpt"]
    aF, bF = FO.model_ab(th[:9], q, chi1, case, key)
    s1, s2, r0, r1, r2, l0, l1, l2 = th[9:]
    if not (-90 <= s1 <= 30 and s1 + 3 <= s2 <= 90):
        return None, None
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    br, kap = c["anc"]
    s = t - c["tpk"]
    W = P3.smooth5((s - s1) / (s2 - s1))
    lbm = np.log(br) + r0 + r1 * chi1 + r2 * nu
    lam = 1.2 * np.log(X1) + l0 + l1 * chi1 + l2 * nu + kap * br * np.clip(s - S_RING, 0, None)
    if np.any(aF <= 0) or np.any(bF <= 0):
        return None, None
    beta = np.exp((1 - W) * np.log(bF) + W * lbm)
    alpha = np.exp((1 - W) * np.log(aF) + W * lam)
    return alpha, beta


def err(th, k, key, seed=None):
    c = get(k)["case"]; pre = "fit_" if key == "fit" else ""
    alpha, beta = ab(th, k, key)
    if alpha is None:
        return 1.0, 0.0
    t, h, tn, hn = c[pre + "t_bhpt"], c[pre + "h_bhpt"], c[pre + "t_nr"], c[pre + "h_nr"]
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    if np.min(np.diff(ts)) <= 0:
        return 1.0, 0.0
    f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
    if seed is None:
        grid = np.linspace(-300, 200, 126); seed = grid[int(np.argmin([f(g) for g in grid]))]
    r = minimize_scalar(f, bounds=(seed - 6, seed + 6), method="bounded", options={"xatol": 1e-3})
    return float(r.fun), float(r.x)


def _fit(a):
    th, k = a; return err(np.asarray(th), k, "fit")[0]


def _full(a):
    th, k = a; th = np.asarray(th)
    e, t0 = err(th, k, "fit"); return err(th, k, "full", seed=t0)[0]


def run_fit(pool, x0, keys, budget_min, tag):
    st = dict(n=0, best=np.inf, bx=np.array(x0, float), t=time.time())
    class Stop(Exception):
        pass
    def obj(x):
        v = float(np.mean(pool.map(_fit, [(list(x), k) for k in keys])))
        st["n"] += 1
        if v < st["best"]:
            st["best"], st["bx"] = v, np.array(x, float)
            (OUT / f"{tag}_partial.json").write_text(json.dumps(dict(names=NAMES, theta=list(map(float, x)), mean=v, n=st["n"])))
            print(f"  [{tag} {st['n']:5d}] {(time.time()-st['t'])/60:5.1f} min  mean {v:.4e}", flush=True)
        if time.time() - st["t"] > budget_min * 60:
            raise Stop
        return v
    try:
        minimize(obj, x0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 40})
        note = "converged"
    except Stop:
        note = "stopped on budget"
    print(f"  {tag}: {note}, {st['n']} evals", flush=True)
    return st["bx"], note


def summary(pool, th, keys):
    e = pool.map(_full, [(list(th), k) for k in keys])
    return dict(zip([f"{q:.4f}_{c:+.1f}" for q, c in keys], e)), np.array(e)


def main():
    pool = mp.get_context("fork").Pool(96)
    th9 = FO.load_theta()
    # start the alpha level at the FLUX alpha at the ppBHPT peak (mean over S18)
    l0s = []
    for k in S18:
        c = get(k); a, _ = FO.model_ab(th9, k[0], k[1], c["case"], "full")
        l0s.append(np.log(np.interp(c["tpk"], c["case"]["t_bhpt"], a)) - 1.2 * np.log(k[0] / (1 + k[0])))
    x0 = np.r_[th9, -20.0, 40.0, -0.034, 0.075, 0.371, float(np.mean(l0s)), 0.0, 0.0]
    out = dict(names=NAMES)
    th1, n1 = run_fit(pool, x0, S18, 40, "S18")
    d1, e1 = summary(pool, th1, T96)
    out["stage1_S18"] = dict(theta=list(map(float, th1)), note=n1, T96=d1)
    print(f"stage 1 (S18-fitted) on T96: mean {e1.mean():.4e} median {np.median(e1):.4e} max {e1.max():.4e}", flush=True)
    th2, n2 = run_fit(pool, th1, T96, 55, "T96")
    d2, e2 = summary(pool, th2, T96)
    out["stage2_T96"] = dict(theta=list(map(float, th2)), note=n2, T96=d2)
    print(f"stage 2 (T96-fitted) on T96: mean {e2.mean():.4e} median {np.median(e2):.4e} max {e2.max():.4e}", flush=True)
    for ch in S.CHIS:
        v = [e for (q, c), e in zip(T96, e2) if c == ch]
        print(f"   chi1 {ch:+.1f}: median {np.median(v):.3e} mean {np.mean(v):.3e} max {np.max(v):.3e}", flush=True)
    for q in S.TRAIN_Q:
        v = [e for (qq, c), e in zip(T96, e2) if qq == q]
        print(f"   q {q:6.3f}: median {np.median(v):.3e}", flush=True)
    try:
        sys.path.insert(0, str(REPO / "spin_dep"))
        import plot_spin_paper_figs as PF
        H = PF.heldout_points()
        dh, eh = summary(pool, th2, H)
        out["stage2_H270"] = dh
        print(f"H270: mean {eh.mean():.4e} median {np.median(eh):.4e} max {eh.max():.4e}  (n={len(eh)})", flush=True)
    except Exception as ex:
        print("H270 skipped:", repr(ex), flush=True)
    (OUT / "p6_m2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
