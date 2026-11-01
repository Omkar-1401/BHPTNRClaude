"""EXPLORATION pass 3a -- where in time do the installed models lose?  (diagnostic, no fitting)

For pn_anchored_spin and flux_anc_spin_with_orbit on all T96 cases: the full-window error with the
full-window alignment (t0, phi0), decomposed exactly into NR-time bands
    E = sum_b |h - e^{i phi0} g|^2_b / (2 |h|^2_total)
bands (NR time from the NR peak, M): [-5000,-1000], [-1000,-150], [-150,-50], [-50,-15], [-15,0], [0,+15], [+15,+100].
Also: the per-band split of the error into amplitude and phase parts, using
    |h - g|^2 ≈ A^2 (d^2 + dphi^2)  (reported as the band means of d^2 and dphi^2, A^2-weighted).
Writes exploration/p3/p3_where.json.
"""
from __future__ import annotations
import json, sys, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import fit_pn_anchored_spin as S              # noqa: E402
import fit_flux_anc_spin_with_orbit as FO     # noqa: E402
import fit_fluxanchored_spin as FS            # noqa: E402
import test_guard                              # noqa: E402,F401  (refuses sealed test points)
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p3"; OUT.mkdir(exist_ok=True)
BANDS = [(-5000.1, -1000), (-1000, -150), (-150, -50), (-50, -15), (-15, 0), (0, 15), (15, 100.1)]


def model_g(alpha, beta, case, t0):
    t, h, tn = case["t_bhpt"], case["h_bhpt"], case["t_nr"]
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    tau = t0 + ts
    g = np.interp(tn, tau, h.real) + 1j * np.interp(tn, tau, h.imag)
    g = np.interp(tn, tau, alpha) * g
    ok = (tn >= tau[0]) & (tn <= tau[-1])
    return g, ok


def decompose(alpha, beta, case):
    t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
    grid = np.linspace(-300, 200, 126)
    c = grid[int(np.argmin([f(g) for g in grid]))]
    r = minimize_scalar(f, bounds=(c - 6, c + 6), method="bounded", options={"xatol": 1e-3})
    t0 = float(r.x)
    g, ok = model_g(alpha, beta, case, t0)
    r_, g_ = hn[ok], g[ok]; tt = tn[ok]
    ph0 = np.angle(np.sum(r_ * np.conj(g_)))
    g_ = g_ * np.exp(1j * ph0)
    n1 = np.sum(np.abs(r_) ** 2)
    tp = tn[int(np.argmax(np.abs(hn)))]
    s = tt - tp
    out = dict(E=float(r.fun), bands=[])
    A = np.abs(r_); d = np.abs(g_) / np.maximum(A, 1e-30) - 1
    dphi = np.angle(g_ * np.conj(r_))
    for lo, hi in BANDS:
        m = (s >= lo) & (s < hi)
        w = A[m] ** 2
        out["bands"].append(dict(lo=lo, hi=hi, E_part=float(np.sum(np.abs(r_[m] - g_[m]) ** 2) / (2 * n1)),
                                 power=float(np.sum(w) / n1),
                                 d2=float(np.sum(w * d[m] ** 2) / max(np.sum(w), 1e-300)),
                                 dphi2=float(np.sum(w * dphi[m] ** 2) / max(np.sum(w), 1e-300))))
    return out


def one(k):
    q, chi1 = k
    th9 = S.model9(); ths = np.array(json.loads(S.COEFFS.read_text())["theta_spin"], float)
    case = FS.build_case(q, chi1)
    aL, bL = S.model_ab(th9, ths, q, chi1, case["x"], case["p_loss"])
    aF, bF = FO.model_ab(FO.load_theta(), q, chi1, case, "full")
    return dict(q=q, chi1=chi1, L=decompose(aL, bL, case), F=decompose(aF, bF, case))


def main():
    keys = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
    with mp.get_context("fork").Pool(96) as pool:
        rows = pool.map(one, keys)
    (OUT / "p3_where.json").write_text(json.dumps(rows, indent=1))
    for m in ("L", "F"):
        E = np.array([r[m]["E"] for r in rows])
        P = np.array([[b["E_part"] for b in r[m]["bands"]] for r in rows])
        print(f"\n{m}: mean E {E.mean():.3e} (stored-equivalent check), share of the T96-mean error by band:")
        for j, (lo, hi) in enumerate(BANDS):
            d2 = np.mean([r[m]["bands"][j]["d2"] for r in rows]); dp = np.mean([r[m]["bands"][j]["dphi2"] for r in rows])
            pw = np.mean([r[m]["bands"][j]["power"] for r in rows])
            print(f"  [{lo:7.1f},{hi:6.1f}]  share {P[:, j].mean() / E.mean():6.1%}  power {pw:6.1%}  "
                  f"<d^2> {d2:.2e}  <dphi^2> {dp:.2e}")
        for ch in S.CHIS:
            ii = [i for i, r in enumerate(rows) if r["chi1"] == ch]
            sh = P[ii].mean(0) / E[ii].mean()
            print(f"  chi1 {ch:+.1f}: mean E {E[ii].mean():.2e}  shares " + " ".join(f"{v:5.1%}" for v in sh))


if __name__ == "__main__":
    main()
