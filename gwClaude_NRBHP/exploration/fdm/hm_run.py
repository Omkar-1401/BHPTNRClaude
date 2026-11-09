"""Multimode FD mismatch vs NRHybSur3dq8 (8 modes, hm_cache.MODES), same points / masses as fd_run.py.

Our (2,2) models -> higher modes by the zero-parameter transfer of fit_scaling_hm_backbone.py (note Eq. 43): the time
map tau(t) is common to all modes and
    alpha_lm(t) = alpha_22(t) C_lm(nu) beta(t)^(-(l + eps_lm - 2)/3),   rho_lm = 1 (nothing fitted per mode),
applied to the raw ppBHPT modes.  Rink et al. 2024: the published calibrated BHPTNRSur2dq1e3 modes (their own alpha_l
per l, shared beta), placed on the NR axis with the t0 of their (2,2) score.  Zhong et al. is not included: their
phase-scaling map exp(-i phi/beta') is not defined consistently across modes (it rescales each mode's constant phase).
Gates (before any output): the model's stored T96 scores (fd_run's gate); NR vs NR with all modes = 0; (2,2)-only
through hm_core = fd_core's single-mode match at the equivalent delta = arg K (iota = pi/3).
    python fdm/hm_run.py MODEL     MODEL in p40 | p40lean | p49b12 | p49b9 | flux_orbit | rink
    -> fdm/results/hm_<MODEL>.json
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
import numpy as np

import fd_run as FR                             # model setup + T96 gate inputs (reads sys.argv[1])
import fd_core as FC
import hm_core as HC
from hm_cache import MODES, path as hm_path
S = FR.S; MODEL = FR.MODEL


def C_lm(l, m, q):                              # verbatim from fit_scaling_hm_backbone.py (note Eq. 39)
    X1, X2 = q / (1.0 + q), 1.0 / (1.0 + q); k = l + (l + m) % 2 - 1
    return abs(X2 ** k + (-1.0) ** m * X1 ** k)


def beta_exponent(l, m):                        # verbatim from fit_scaling_hm_backbone.py (note Eq. 43)
    return -(l + (l + m) % 2 - 2) / 3.0
INCS = {"pi/3": np.pi / 3, "pi/2": np.pi / 2}


def maps(k):
    """(tc, nr {mode: arr}, model {mode: arr}) on the common NR window."""
    q, c = k; d = np.load(hm_path(q, c))
    tn = d["t_nr"]
    if MODEL == "rink":
        import fd_models as FM
        ts, h, al, case, grid = FM.wave("rink", q, c)
        _, t0 = FM.fit_t0(ts, h, al, case, grid)
        tr = d["t_rink"]; tau = t0 + (tr - S.T_ANCHOR)
        assert len(tr) == len(ts) and np.allclose(tr - S.T_ANCHOR, ts), "rink grid"
        src = {mo: d[f"rink_{mo[0]}{mo[1]}"] for mo in MODES}; amp = {mo: np.ones_like(tr) for mo in MODES}
    else:
        if MODEL == "flux_orbit":
            M = FR.PF.MODELS[FR.NAME]; case = M["build"](q, c); t = case["t_bhpt"]
            _, alpha, tau = M["shape"](q, c, case); beta = np.gradient(tau, t)
        else:
            P6, TH = FR.P6, FR.TH
            _, t0 = P6.err(TH, k, "fit"); _, t1 = P6.err(TH, k, "full", seed=t0)
            case = P6.get(k)["case"]; t = case["t_bhpt"]
            alpha, beta = P6.ab(TH, k, "full")
            bc = P6.creative.cumulative_trapezoid(beta, t); tau = t1 + bc - np.interp(S.T_ANCHOR, t, bc)
        assert np.array_equal(d["t_pp"], t), "pp grid"
        src = {mo: d[f"pp_{mo[0]}{mo[1]}"] for mo in MODES}
        amp = {(l, m): alpha * C_lm(l, m, q) * beta ** beta_exponent(l, m) for (l, m) in MODES}
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc = tn[m]
    ii = lambda x: np.interp(tc, tau, x)
    mod = {mo: ii(amp[mo]) * (ii(src[mo].real) + 1j * ii(src[mo].imag)) for mo in MODES}
    nr = {mo: d[f"nr_{mo[0]}{mo[1]}"][m] for mo in MODES}
    return tc, nr, mod


def _work(k):
    try:
        tc, nr, mod = maps(k)
        row = dict(q=k[0], chi1=k[1])
        for name, io in INCS.items():
            for M in FR.MTOT:
                row[f"{name}_M{int(M)}"] = HC.mm_match(tc, nr, mod, M, io)
        return row
    except Exception as exc:
        print(f"  [fail] {k}: {exc!r}", flush=True); return dict(q=k[0], chi1=k[1], error=repr(exc))


def selftest(k):
    tc, nr, mod = maps(k)
    z = HC.mm_match(tc, nr, nr, 100.0, np.pi / 3)["mm_worst"]
    io = np.pi / 3
    a = HC.mm_match(tc, {(2, 2): nr[(2, 2)]}, {(2, 2): mod[(2, 2)]}, 100.0, io)["mm_mean"]
    Y, Ym = HC.sylm(io, 2, 2), HC.sylm(io, 2, -2)
    b = np.mean([FC.match(tc, nr[(2, 2)], mod[(2, 2)], 100.0, deltas=(float(np.angle(
        np.exp(1j * ka) * Y * np.exp(2j * ps) + np.conj(np.exp(1j * ka) * Ym * np.exp(-2j * ps)))),))["mm"]
        for ps in HC.PHIS for ka in HC.KAPPAS])
    print(f"selftest {k}: NR vs NR (8 modes) {z:.1e}; (2,2)-only multimode {a:.6e} vs single-mode {b:.6e}", flush=True)
    assert z < 1e-8 and abs(a / b - 1) < 1e-4, (z, a, b)


def main():
    t0 = time.time()
    with mp.get_context("fork").Pool(96) as pool:
        e = np.array(pool.map(FR._E, FR.CHK))
        r = e / np.array([FR.REF[k] for k in FR.CHK])
        print(f"{MODEL}: reproduce stored T96 on {len(FR.CHK)} points: {r.min():.5f}..{r.max():.5f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        selftest(FR.PTS[len(FR.QS) * 2 + 8])                              # chi1 = 0, q = 5
        rows = pool.map(_work, FR.PTS, chunksize=1)
    ok = [x for x in rows if "error" not in x]
    for name in INCS:
        for M in FR.MTOT:
            v = np.array([x[f"{name}_M{int(M)}"]["mm"] for x in ok]); fl = sum(x[f"{name}_M{int(M)}"]["flagged"] for x in ok)
            print(f"  iota {name}, M = {M:.0f}: median {np.median(v):.3e} max {v.max():.2e}; f_low raised {fl}/{len(ok)}", flush=True)
    (FR.OUT / f"hm_{MODEL}.json").write_text(json.dumps(dict(
        model=MODEL, label=FR.LABEL, modes=[list(m) for m in MODES], inclinations=list(INCS), MTOT=FR.MTOT,
        kappas=list(HC.KAPPAS), phis=list(HC.PHIS), transfer="alpha_22 C_lm beta^-(l+eps-2)/3, rho = 1" if MODEL != "rink"
        else "published calibrated modes", rows=rows), indent=1))
    print(f"{MODEL}: {len(ok)}/{len(rows)} points, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
