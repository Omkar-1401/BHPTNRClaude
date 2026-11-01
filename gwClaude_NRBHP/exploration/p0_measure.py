"""EXPLORATION pass 0 -- measurements only, no fitting.

For every T96 case (q in TRAIN_Q x chi1 in CHIS), from the cached waveform pair:

1. Measured alpha(t), beta(t) by the merger-aligned equal-phase construction
   (spin_dep/spin_measure_alpha_beta.py): (2,2) phases zeroed at each |h| peak, paired at
   equal phase;  beta = omega_pp / omega_NR,  alpha = |h_NR| / |h_pp|  at the paired points.
   A gauge choice (the phase offset is fixed at the peak), the same for every case.
2. Landmarks on the ppBHPT clock (m1 units): t_isco (omega_22/2 reaches the Kerr ISCO
   frequency of chi1), the |h| peak, the leader's switch centre (S = 0.5), and two
   features of the measured beta (steepest rise, maximum).  Also in Ori-Thorne time T.
3. B5 check: slope of ln alpha_meas over NR tau in [tau_pk + 15, tau_pk + 60] M against
   the zero-parameter prediction  gamma_pp / beta - gamma_NR  (gamma = |Im omega_220|;
   pp side Kerr(chi1) in m1 units, NR side the remnant, NRSur3dq8Remnant).
4. Frequency-domain diagnostic (alignment-free): omega^2/omegadot against omega for NR and
   for the leader model (pn_anchored_spin, installed coefficients).
Collapse tests run on the saved curves afterwards (p0_collapse.py).

Run on a compute node:  python p0_measure.py --nproc 48
Writes exploration/p0/p0_measure.json (scalars) and p0_curves.npz (curves on fixed grids).
"""
from __future__ import annotations
import argparse, json, sys, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "spin_dep")); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spin_dep_split"))
import fit_pn_anchored_spin as S          # noqa: E402  (build_case, model_ab, beta_r_phys, grid)
import test_guard                              # noqa: E402,F401  (refuses sealed test points)
import physics_common as PC               # noqa: E402
import ot_join as OT                      # noqa: E402  (kerr_isco, ot_time, t_isco_pp)
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p0"; OUT.mkdir(exist_ok=True)
WIN = 20.0                                  # savgol span for phase derivatives, in each side's own time unit
UG = np.linspace(-6.0, 3.0, 181)            # u_T grid (scaled orbital frequency about the ISCO)
XG = np.linspace(0.06, 0.26, 101)           # note's-clock x grid (inspiral)
TG = np.linspace(-400.0, 60.0, 231)         # ppBHPT time from the |h| peak, m1
OG = np.linspace(0.035, 0.30, 120)          # NR-clock GW omega grid for omega^2/omegadot


def phase_freq(t, h, win):
    dt = float(np.median(np.diff(t)))
    n = int(round(win / dt)); n += (n % 2 == 0); n = max(n, 7)
    ph = np.unwrap(np.angle(h))
    if ph[-1] < ph[0]:
        ph = -ph
    w = savgol_filter(ph, n, 3, deriv=1, delta=dt, mode="interp")
    wd = savgol_filter(ph, n, 3, deriv=2, delta=dt, mode="interp")
    return ph, w, wd


def gammas(q, chi1):
    """|Im omega_220| on each side's own clock: pp per m1 (Kerr chi1), NR per M (remnant)."""
    s1 = [0, 0, float(chi1)]
    mf = float(S._SFBH.mf(q, s1, [0, 0, 0])[0]); chif = float(S._SFBH.chif(q, s1, [0, 0, 0])[0][2])
    g_pp = PC.w220_exact(chi1) / (2 * PC.Q220_exact(chi1))
    g_nr = PC.w220_exact(chif) / (2 * PC.Q220_exact(chif)) / mf
    return g_pp, g_nr, mf, chif


def one(args):
    q, chi1, th9, ths = args
    case = S.build_case(q, chi1)
    t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    # --- phases, frequencies, peaks
    pp, wp, wpd = phase_freq(t, h, WIN)
    pn, wn, wnd = phase_freq(tn, hn, WIN)
    ip, inr = int(np.argmax(np.abs(h))), int(np.argmax(np.abs(hn)))
    Pp, Pn = pp - pp[ip], pn - pn[inr]
    # --- equal-phase pairing (NR phase must be monotone on the used span)
    good_n = np.r_[True, np.diff(Pn) > 0]
    Pn_m, tn_m = Pn[good_n], tn[good_n]
    sel = (Pp >= Pn_m[0]) & (Pp <= Pn_m[-1]) & (t <= t[ip] + 60.0)
    tt, Pq = t[sel], Pp[sel]
    tau = np.interp(Pq, Pn_m, tn_m)
    beta = wp[sel] / np.interp(tau, tn, wn)
    alpha = np.interp(tau, tn, np.abs(hn)) / np.abs(h[sel])
    # --- landmarks (ppBHPT clock, m1)
    t_isco = OT.t_isco_pp(t, h, chi1)
    T_all, T_unit = OT.ot_time(t, t_isco, q, chi1)
    _, Om_isco = OT.kerr_isco(chi1)
    # leader switch
    b1, b2, a2, p00, p01, w0, r0, m0, m1 = th9
    Sw = creative.sigmoid((case["p_loss"] - (p00 + p01 * nu)) / max(w0, 1e-3))
    iS = np.flatnonzero(Sw >= 0.5)
    t_S50 = float(t[iS[0]]) if len(iS) else float("nan")
    S_at_peak = float(Sw[ip])
    # measured-beta features in [t_pk - 300, t_pk + 15]
    m = (tt >= t[ip] - 300) & (tt <= t[ip] + 15)
    bsm = savgol_filter(beta, 51, 3) if len(beta) > 60 else beta
    db = np.gradient(bsm, tt)
    m_pre = m & (tt <= t[ip])
    t_bmax = float(tt[m][np.argmax(bsm[m])])
    t_bsteep = float(tt[m_pre][np.argmax(db[m_pre])])
    Tof = lambda x: float(np.interp(x, t, T_all))
    # --- B5: late-ringdown slope of ln alpha vs NR tau
    g_pp, g_nr, mf, chif = gammas(q, chi1)
    br = float(S.beta_r_phys(q, chi1))
    rd = (tau >= tn[inr] + 15) & (tau <= tn[inr] + 60)
    if rd.sum() > 20:
        sl = float(np.polyfit(tau[rd], np.log(alpha[rd]), 1)[0])
        b_rd = float(np.median(beta[rd]))
    else:
        sl, b_rd = float("nan"), float("nan")
    pred_br = g_pp / br - g_nr
    pred_bm = g_pp / b_rd - g_nr if np.isfinite(b_rd) else float("nan")
    # --- frequency-domain: omega^2/omegadot vs omega (NR clock), NR and leader model
    aL, bL = S.model_ab(th9, ths, q, chi1, case["x"], case["p_loss"])
    tauL = creative.cumulative_trapezoid(bL, t)
    wL = wp / bL
    wLd = np.gradient(wL, tauL)
    def adia(w, wd_, tm):
        mm = (tm <= 0) & (w > 0) & (wd_ > 0)
        w, r = w[mm], (w[mm] ** 2 / wd_[mm])
        o = np.argsort(w)
        return np.interp(OG, w[o], r[o], left=np.nan, right=np.nan)
    A_nr = adia(wn, np.gradient(wn, tn), tn - tn[inr])
    A_L = adia(wL, wLd, t - t[ip])
    # --- curves on fixed grids
    uT = (wp[sel] / 2 - Om_isco) / (1.0 / q) ** 0.4
    xc = PC.x_scale(q) * (wp[sel] / 2) ** (2 / 3)
    pre = tt <= t[ip]
    def on(grid, c, y, msk):
        c, y = c[msk], y[msk]
        o = np.argsort(c); c, y = c[o], y[o]
        return np.interp(grid, c, y, left=np.nan, right=np.nan)
    rb = beta / X1 ** 1.2 - 1.0
    ra = alpha / X1 ** 1.2 - 1.0
    curves = dict(
        rb_u=on(UG, uT, rb, pre), ra_u=on(UG, uT, ra, pre),
        rb_x=on(XG, xc, rb, pre & (xc <= 0.26)), ra_x=on(XG, xc, ra, pre & (xc <= 0.26)),
        rb_t=on(TG, tt - t[ip], rb, np.ones_like(tt, bool)), ra_t=on(TG, tt - t[ip], ra, np.ones_like(tt, bool)),
        A_nr=A_nr, A_L=A_L)
    sc = dict(q=q, chi1=chi1, nu=nu, X1=X1, t_peak=float(t[ip]), t_isco=t_isco, T_unit=float(T_unit),
              T_peak=Tof(t[ip]), t_S50=t_S50, T_S50=Tof(t_S50) if np.isfinite(t_S50) else None,
              S_at_peak=S_at_peak, t_bsteep=t_bsteep, T_bsteep=Tof(t_bsteep), t_bmax=t_bmax, T_bmax=Tof(t_bmax),
              beta_rd_meas=b_rd, beta_r=br, slope_meas=sl, slope_pred_beta_r=pred_br, slope_pred_beta_meas=pred_bm,
              g_pp=g_pp, g_nr=g_nr, Mf=float(mf), chif=float(chif), Om_isco=Om_isco,
              nr_monotone_frac=float(good_n.mean()))
    return sc, curves


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--check", action="store_true", help="reproduction gate on S18 only")
    a = ap.parse_args()
    th9 = S.model9()
    d = json.loads(S.COEFFS.read_text()); ths = np.array(d["theta_spin"], float)
    if a.check:
        keys = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
        with mp.get_context("fork").Pool(min(a.nproc, len(keys))) as pool:
            e = pool.starmap(_score, [(q, c, th9, ths) for q, c in keys])
        st = [d["mismatches"][f"{q:.4f}_{c:+.1f}"] if f"{q:.4f}_{c:+.1f}" in d["mismatches"] else None for q, c in keys]
        print(json.dumps({"keys": keys, "rescored": e, "stored": st}, indent=1))
        r = [x / y for x, y in zip(e, st) if y]
        print(f"ratio min {min(r):.6f} max {max(r):.6f}")
        return
    keys = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
    with mp.get_context("fork").Pool(min(a.nproc, len(keys))) as pool:
        res = pool.map(one, [(q, c, th9, ths) for q, c in keys])
    scal = [r[0] for r in res]
    names = list(res[0][1])
    arr = {k: np.array([r[1][k] for r in res]) for k in names}
    np.savez_compressed(OUT / "p0_curves.npz", q=np.array([k[0] for k in keys]),
                        chi=np.array([k[1] for k in keys]), UG=UG, XG=XG, TG=TG,
                        OG=OG, **arr)
    (OUT / "p0_measure.json").write_text(json.dumps(scal, indent=1))
    print("wrote", OUT / "p0_measure.json")


def _score(q, c, th9, ths):
    return float(S.full_mm(th9, ths, q, c, S.build_case(q, c)))


if __name__ == "__main__":
    main()
