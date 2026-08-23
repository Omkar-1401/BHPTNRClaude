"""Global joint fit for the PN-anchored ppBHPT->NR (2,2) model.

Implements the "Physical Anchors" note (alpha_beta_pn_scaling_note_revised.pdf): the
q-dependence of the leading scaling is IMPOSED analytically (base = X1^(6/5), plus the
fixed 1PN amplitude slope 55/42) rather than fitted, so only 9 nu-suppressed residual
constants are calibrated -- once, jointly, against the even-nu [3,8] waveform set (Powell).

    beta_insp = base*(1 + nu*(b1*xc + b2*xc^2)),  base = (q/(1+q))^(6/5)
    alpha_insp= base*(1 + (55/42)*nu*xc + nu*a2*xc^2)          # 55/42 FIXED
    beta_r    = beta_r_physical(q)*r0   (QNM, NRSur3dq8Remnant); alpha_mr = base*(m0+m1*nu)
    S = sigmoid((p_loss-(p0_0+p0_1*nu))/w0);  blend (1-S)*insp + S*MR
    x = (omega_gw/2)^(2/3) from the ppBHPT phase, xc = min(x, 0.26); p_loss = wf-flux coord

theta = [b1, b2, a2, p0_0, p0_1, w0, r0, m0, m1]. Writes pn_anchored_results/coeffs.json.

Run `python fit_scaling_pn_anchored.py`            -> validate the shipped coeffs.
Run `python fit_scaling_pn_anchored.py --refit`    -> re-run the global fit and rewrite coeffs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTutils"))

import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_hybrid_q_dep as H
import surfinBH

_SFBH = surfinBH.LoadFits("NRSur3dq8Remnant")
QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292
PN_SLOPE = 55.0 / 42.0
XCLIP = 0.26
T_ANCHOR = -100.0
NAMES = ["b1", "b2", "a2", "p0_0", "p0_1", "w0", "r0", "m0", "m1"]
THETA0 = np.array([0.0, 0.0, 0.0, -0.9, 0.0, 0.05, 0.89, 1.3, -3.5])
RESULTS = ROOT / "pn_anchored_results"
COEFFS = RESULTS / "coeffs.json"
# even-nu 12 q training set (matches wf_nu_hybrid_global)
TRAIN_Q = [3.0, 3.2564102564, 3.5128205128, 3.8974358974, 4.2, 4.6, 5.0,
           5.4358974359, 5.9487179487, 6.5897435897, 7.2307692308, 8.0]


def get_x(t, h):
    ph = np.unwrap(np.angle(h)); n = len(ph)
    win = min(401, n - (1 - n % 2)); win = win - 1 if win % 2 == 0 else win
    if win >= 11:
        ph = savgol_filter(ph, win, 3, mode="interp")
    return np.clip((0.5 * np.abs(np.gradient(ph, t))) ** (2.0 / 3.0), 1e-8, 0.6)


def beta_r_phys(q):
    mf, _ = _SFBH.mf(q, [0, 0, 0], [0, 0, 0]); chif, _ = _SFBH.chif(q, [0, 0, 0], [0, 0, 0])
    chif = float(chif[2]); mf = float(mf)
    return (0.3683 * (1 + q) / q) / ((QF1 + QF2 * (1 - chif) ** QF3) / mf)


def model_ab(theta, q, x, p_loss):
    b1, b2, a2, p00, p01, w0, r0, m0, m1 = theta
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    xc = np.minimum(x, XCLIP)
    S = creative.sigmoid((p_loss - (p00 + p01 * nu)) / max(w0, 1e-3))
    beta = (1 - S) * base * (1 + nu * (b1 * xc + b2 * xc ** 2)) + S * beta_r_phys(q) * r0
    alpha = (1 - S) * base * (1 + PN_SLOPE * nu * xc + nu * a2 * xc ** 2) + S * base * (m0 + m1 * nu)
    return alpha, beta


def _err_at_t0(t0, alpha, ts, t, h, tn, hn):
    tau = t0 + ts; lo = max(tn[0], tau[0]); hi = min(tn[-1], tau[-1])
    m = (tn >= lo) & (tn <= hi)
    if m.sum() / len(tn) < H.MIN_COVERAGE:
        return 50.0
    tc = tn[m]; r = hn[m]
    g = np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag)
    a = np.interp(tc, tau, alpha)
    gg = a * g
    n1 = np.sum(np.abs(r) ** 2); n2 = np.sum(np.abs(gg) ** 2); Z = np.abs(np.sum(r * np.conj(gg)))
    return float((n1 + n2 - 2 * Z) / (2 * n1))


def mismatch(theta, q, case, key, seed):
    if key == "fit":
        t, h, tn, hn, pl, x = (case["fit_t_bhpt"], case["fit_h_bhpt"], case["fit_t_nr"],
                               case["fit_h_nr"], case["fit_losses"]["p_loss"], case["fit_x"])
    else:
        t, h, tn, hn, pl, x = (case["t_bhpt"], case["h_bhpt"], case["t_nr"],
                               case["h_nr"], case["losses"]["p_loss"], case["x"])
    alpha, beta = model_ab(theta, q, x, pl)
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(T_ANCHOR, t, bc)
    d = np.min(np.diff(ts))
    if d <= 0:
        return 50.0 + 1e3 * (-d), seed
    # wide coarse + refine (t0 optimum can be tens of M)
    grid = np.linspace(seed - 200, seed + 200, 81)
    e = [_err_at_t0(x0, alpha, ts, t, h, tn, hn) for x0 in grid]; c = grid[int(np.argmin(e))]
    r = minimize_scalar(lambda x0: _err_at_t0(x0, alpha, ts, t, h, tn, hn),
                        bounds=(c - 6, c + 6), method="bounded", options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def add_x(case):
    case["x"] = get_x(case["t_bhpt"], case["h_bhpt"])
    case["fit_x"] = get_x(case["fit_t_bhpt"], case["fit_h_bhpt"])
    return case


def full_mm(theta, q, case, seed=-50.0):
    e, t0 = mismatch(theta, q, case, "fit", seed)
    e, _ = mismatch(theta, q, case, "full", t0)
    return e


def write_coeffs(theta, in_range_median, in_range_max, q_extrap):
    RESULTS.mkdir(exist_ok=True)
    out = {
        "param_names": NAMES,
        "theta": [float(v) for v in theta],
        "fixed": {"beta_prefactor": "X1^(6/5)", "alpha_prefactor": "X1^(6/5)",
                  "alpha_pn_slope": PN_SLOPE, "alpha_pn_slope_expr": "55/42 (1PN, fixed)",
                  "x_clip": XCLIP},
        "qnm": {"F1": QF1, "F2": QF2, "F3": QF3, "remnant_fit": "NRSur3dq8Remnant"},
        "_meta": {"fit": "global joint (Powell) over even-nu 12 q in [3,8]",
                  "in_range_median": in_range_median, "in_range_max": in_range_max,
                  "q_extrap": q_extrap,
                  "note": "inspiral PN-anchored (X1^6/5 + 55/42 nu x); MR branch present but "
                          "under-engaged -- see scaling_pn_anchored.md caveats"},
    }
    COEFFS.write_text(json.dumps(out, indent=2))
    print(f"wrote {COEFFS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--maxiter", type=int, default=40)
    args = ap.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train = {q: add_x(H.load_case(q, 6, 10)) for q in TRAIN_Q}
        low = {q: add_x(H.load_case(q, 6, 10)) for q in [2.75, 2.5, 2.25, 2.0]}

        if args.refit:
            def obj(theta):
                errs = [mismatch(theta, q, train[q], "fit", -50.0)[0] for q in TRAIN_Q]
                obj.last = (float(np.mean(errs)), float(np.max(errs)))
                return np.mean(errs)
            obj.last = (np.nan, np.nan)
            t0 = time.time()
            res = minimize(obj, THETA0, method="Powell",
                           options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
            print(f"opt {time.time()-t0:.0f}s  mean={obj.last[0]:.3e} max={obj.last[1]:.3e}")
            theta = res.x
        else:
            theta = np.array(json.loads(COEFFS.read_text())["theta"], float)

        ir = [full_mm(theta, q, train[q]) for q in TRAIN_Q]
        print(f"in-range [3,8]: median={np.median(ir):.3e}  max={np.max(ir):.3e}")
        qext = {}
        for q in [2.75, 2.5, 2.25, 2.0]:
            e = full_mm(theta, q, low[q]); qext[str(q)] = e
            print(f"  q={q}: mathcalE={e:.3e}" + ("" if e < 1e-2 else "  (>1e-2)"))
        print("theta =", np.array2string(theta, precision=4))
        if args.refit:
            write_coeffs(theta, float(np.median(ir)), float(np.max(ir)), qext)


if __name__ == "__main__":
    main()
