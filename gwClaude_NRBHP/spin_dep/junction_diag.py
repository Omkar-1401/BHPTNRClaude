"""junction_diag.py -- model-free tests of the C1 JUNCTION construction, incl. spin.

Run BEFORE building the junction model.  Nothing here is fitted to a waveform
mismatch; everything is read off the cached (ppBHPT, NR) pairs and compared to
QNM-derived predictions.  If T1 or T3 fails the construction is dead and the
model is not worth writing.

===========================================================================
WHAT IS BEING TESTED
===========================================================================
The proposal replaces pn_anchored's logistic switch with a C1 junction at t_*:

    inspiral  t < t_* :  beta_I = base[1 + nu(b1 xc + b2 xc^2 + sSO chi1 xc^1.5
                                                              + sSS chi1^2 xc^2)]
                         alpha_I= base[1 + (55/42) nu xc + nu a2 xc^2]
    merger-rd t > t_* :  beta_M  = beta_r + D_b exp(-dt/tau_b)
                         alpha_M = exp(kappa dt)[A + B exp(-dt/tau_a)]

with [alpha]=[alpha']=[beta]=[beta']=0 fixing D_b, A, B and LOCATING t_*.
beta_r, kappa, tau are all derived from QNM theory + the remnant fit.

T1  kappa.  beta matches the two QNM *real* frequencies; it cannot also match
    the imaginary parts, so the leftover decay-rate mismatch must be carried by
    alpha:   alpha ~ exp(kappa dt),  kappa = (w_pp/2)(1/Q_pp - 1/Q_f).
    Measured: the late post-merger log-slope of alpha_emp = |h_NR|/|h_pp|.
    FAILS if kappa_meas has the wrong sign or is order-of-magnitude off.

T2  tau.  The relaxation timescale, the one arguable input.  Measured: fit
    beta_emp(post-merger) = beta_r + D exp(-dt/tau).  Compared against the
    fundamental damping time tau_f and the overtone timescale tau_ov.
    Does not FAIL; it DECIDES whether tau can stay derived or needs 1 coef.

T3  root.  t_* solves G(t) = tau_b beta_I'(t) + beta_I(t) - beta_r = 0.
    Measured: sign change of G over the inspiral, and where.
    FAILS if there is no root, or the root is not near merger.

Also reported as a by-product:
  * beta_r check -- beta_emp in ringdown vs derived beta_r (independent of any fit)
  * peak-match quality, which is expected to degrade at chi1 = -0.6 where the
    post-merger |h| envelope carries 15-16 genuine nulls (see FREEZE_XC note in
    fit_pn_anchored_spin.py); those cases are FLAGGED, not silently averaged in.

===========================================================================
PROVENANCE OF EVERY CONSTANT
===========================================================================
QF1,QF2,QF3 = 1.5251,-1.1568,0.1292   Berti-Cardoso-Will (2,2,0) REAL part.
    UNUSED since 2026-09-26: w220/Q220/w221/Q221 now come from physics_common (exact
    Kerr, qnm package).  The BCW rows below are kept only as a record.
QQ1,QQ2,QQ3 = 0.7000,1.4187,-0.4990   BCW (2,2,0) QUALITY FACTOR.  *** NEW to
    this workspace, NOT independently verified against the paper. ***  Internal
    consistency check: Q220(0)=2.1187 -> tau=2Q/w=11.50 M vs the textbook
    Schwarzschild 11.24 M, a 2% fit error, which is the right size.  CHECK
    AGAINST THE PAPER BEFORE THIS GOES IN A MODEL.
OF*,OQ* = the n=1 overtone row.  *** UNVERIFIED, quoted from memory. ***  Used
    ONLY to print a comparison number in T2; nothing depends on it.  T2's
    deliverable is the MEASURED tau.

pp-side spin: in ppBHPT the background IS the primary, mass m1 spin chi1, so
its QNM is w_pp(chi1) = w220_exact(chi1) (exact Kerr; was 0.3737*BCW(chi1)/BCW(0) until
2026-09-26) in primary-mass units -- NOT
Schwarzschild, and NOT times (1+q)/q (that was the units bug, UNITS_AUDIT.md A1).  Getting only the NR side to
move with spin was a real bug, caught 2026-08-30 (diag_beta_r.py).  Both Q_pp
and w_pp therefore carry chi1 here, which is also why kappa is spin-dependent
with no fitted coefficient.
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import fit_pn_anchored_spin as S          # build_case, get_x, w220, beta_r_phys, _SFBH

QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292
QQ1, QQ2, QQ3 = 0.7000, 1.4187, -0.4990          # UNVERIFIED -- see docstring
OF1, OF2, OF3 = 1.3673, -1.0260, 0.1628          # UNVERIFIED n=1
OQ1, OQ2, OQ3 = 0.1000, 0.5436, -0.4731          # UNVERIFIED n=1

PN_SLOPE = 55.0 / 42.0
RESULTS = HERE / "junction_diag_results"; RESULTS.mkdir(exist_ok=True)


# Level-anchored to the EXACT Schwarzschild (2,2,0) value.  The BCW fit gives
# w220(0)=0.3683, but the true value is 0.3737 (1.5% apart) and measurement
# (0.3734-0.3763) favours the latter; the gwr_* family already uses 0.3737.  This
# MUST match fit_pn_anchored_spin.w_pp_qnm or the two disagree by 1.5% -- which is
# exactly what units_selftest caught on 2026-09-24 after they were patched apart.
W_SCHW = 0.3737
RESCALE_QNM = True


import physics_common as PC       # exact Kerr QNMs (2026-09-26); the BCW rows above are unused
def w220(chi):  return PC.w220_exact(chi)


def w_pp_qnm(chi):
    """pp-side (2,2,0) QNM frequency, in units of the PRIMARY's mass: exact Kerr, no rescale
    (the W_SCHW/RESCALE_QNM rescale only made sense on top of the BCW fit; 2026-09-26)."""
    return w220(chi)
def Q220(chi):  return PC.Q220_exact(chi)
def w221(chi):  return PC.w221_exact(chi)
def Q221(chi):  return PC.Q221_exact(chi)


def derived(q, chi1):
    """Every QNM/remnant quantity the junction needs.  Zero fitted parameters."""
    s1 = [0, 0, float(chi1)]
    mf, _ = S._SFBH.mf(q, s1, [0, 0, 0])
    chif, _ = S._SFBH.chif(q, s1, [0, 0, 0])
    mf = float(mf); chif = float(chif[2])
    # pp background = the primary, mass m1, spin chi1.  NO (1+q)/q: the ppBHPT
    # surrogate's time is already in units of the PRIMARY's mass, verified by direct
    # measurement of the late-ringdown frequency (2026-09-23):
    #   w_pp_meas = 0.3763/0.3746/0.3734 at q=3/5/8, chi1=0  vs  w220(0)=0.3683,
    #   and 0.4974/0.4954/0.4933 at chi1=+0.6 vs w220(0.6)=0.4975.
    # (fit_pn_anchored_spin.beta_r_phys carried an extra (1+q)/q until 2026-09-24; both
    # now use this form, with exact Kerr QNMs since 2026-09-26.)
    w_pp = w_pp_qnm(chi1)
    Q_pp = Q220(chi1)
    w_f = w220(chif) / mf                    # remnant
    Q_f = Q220(chif)
    beta_r = w_pp / w_f
    tau_f = 2 * Q_f / w_f                                        # 1/w_I,f
    wI0 = w220(chif) / (2 * Q220(chif)) / mf
    wI1 = w221(chif) / (2 * Q221(chif)) / mf
    tau_ov = 1.0 / max(wI1 - wI0, 1e-12)                         # overtone relaxation
    kappa = 0.5 * w_pp * (1.0 / Q_pp - 1.0 / Q_f)
    # TIME-AXIS CONVERSION (found 2026-09-24).  dt_ in the model is ppBHPT time, and
    # kappa is already per unit pp time by construction (kappa = w_I,pp - beta_r w_I,f).
    # tau_f = 1/w_I,f is NOT: w_f is on the NR/total-mass axis, so tau_f is an NR-time
    # duration.  Converting a duration NR -> pp divides by beta_r.  Same species as the
    # beta_r_phys bug: two time axes, the conversion applied to one quantity and not the
    # other.  16-34% and q-dependent.  T2 could not resolve it (measured/tau_f = 0.97
    # median but 0.74-1.23 spread, several fits railed), so it was not caught there.
    tau_f_pp = tau_f / beta_r
    tau_ov_pp = tau_ov / beta_r          # same conversion for the overtone time (2026-09-26)
    return dict(mf=mf, chif=chif, w_pp=w_pp, Q_pp=Q_pp, w_f=w_f, Q_f=Q_f,
                beta_r=beta_r, tau_f=tau_f, tau_f_pp=tau_f_pp, tau_ov=tau_ov,
                tau_ov_pp=tau_ov_pp, kappa=kappa)


def phase_peaks_2s(t, h, dphi=np.pi):
    """GW-phase extrema BOTH SIDES of merger.  j<0 pre, j=0 merger, j>0 post.

    fit_scaling_peaks.phase_peaks only walks backward from merger; the ringdown
    tests need the forward half, so this is a two-sided version of it."""
    amp = np.abs(h)
    t_m = float(t[int(np.argmax(amp))])
    psi = np.unwrap(np.angle(h))
    sgn = 1.0 if psi[-1] > psi[0] else -1.0
    pm = sgn * psi
    mono = float(np.min(np.diff(pm)))            # >0 => phase strictly monotone
    psi_m = float(np.interp(t_m, t, pm))
    kb = int((psi_m - pm[0]) / dphi)
    kf = int((pm[-1] - psi_m) / dphi)
    j = np.arange(-kb, kf + 1)
    targets = psi_m + dphi * j
    t_pk = np.interp(targets, pm, t)
    om = np.abs(np.gradient(psi, t))
    return t_m, j, t_pk, np.interp(t_pk, t, amp), np.interp(t_pk, t, om), mono


def empirical(case):
    """Model-free alpha(t), beta(t) from matched phase peaks, both sides of merger."""
    tb, hb, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    tmb, jb, tpb, ab, ob, mb = phase_peaks_2s(tb, hb)
    tmn, jn, tpn, an, on, mn = phase_peaks_2s(tn, hn)
    j = np.intersect1d(jb, jn)
    ib = np.searchsorted(jb, j); iN = np.searchsorted(jn, j)
    dtb = tpb[ib] - tmb; dtn = tpn[iN] - tmn          # merger-relative peak times
    alpha = an[iN] / ab[ib]
    beta_map = np.gradient(dtn, dtb)                  # local d t_NR / d t_bhpt
    beta_frq = ob[ib] / np.maximum(on[iN], 1e-12)     # w_pp/w_NR, the direct estimate
    return dict(j=j, dt_b=dtb, dt_n=dtn, alpha=alpha, beta_map=beta_map,
                beta_frq=beta_frq, amp_n=an[iN], amp_b=ab[ib],
                mono_b=mb, mono_n=mn)


def t1_kappa(emp, der, amp_floor=1e-2, dt_lo=15.0):
    """Late post-merger log-slope of alpha vs the derived kappa."""
    m = (emp["j"] > 0) & (emp["amp_n"] > amp_floor * emp["amp_n"].max())
    if m.sum() < 4:
        return dict(ok=False, reason="too few post-merger peaks", n=int(m.sum()))
    dt, al = emp["dt_b"][m], emp["alpha"][m]
    late = dt >= dt_lo
    out = dict(n_post=int(m.sum()), n_late=int(late.sum()),
               dt_range=[float(dt.min()), float(dt.max())],
               kappa_der=der["kappa"], alpha_at_merger=float(al[0]))
    if late.sum() >= 3:                                    # asymptotic slope
        p = np.polyfit(dt[late], np.log(al[late]), 1)
        out["kappa_late"] = float(p[0])
    # full form alpha = exp(k dt)(A + B exp(-dt/ta)), k free
    def res(p):
        k, A, B, lta = p
        return np.log(np.maximum(np.exp(k * dt) * (A + B * np.exp(-dt / np.exp(lta))),
                                 1e-12)) - np.log(al)
    try:
        r = least_squares(res, [der["kappa"], al[-1], al[0] - al[-1],
                                np.log(max(der["tau_f_pp"], 1.0))],   # dt is pp time
                          bounds=([-1, 1e-6, -10, np.log(0.5)], [1, 10, 10, np.log(200)]),
                          max_nfev=6000)
        out.update(kappa_fit=float(r.x[0]), A_fit=float(r.x[1]), B_fit=float(r.x[2]),
                   tau_a_fit=float(np.exp(r.x[3])), fit_rms=float(np.sqrt(np.mean(r.fun**2))))
    except Exception as e:
        out["fit_error"] = str(e)
    out["ok"] = True
    return out


def t2_tau(emp, der, amp_floor=1e-2):
    """Post-merger beta relaxation timescale, and a beta_r sanity check."""
    m = (emp["j"] > 0) & (emp["amp_n"] > amp_floor * emp["amp_n"].max())
    if m.sum() < 4:
        return dict(ok=False, reason="too few post-merger peaks", n=int(m.sum()))
    dt = emp["dt_b"][m]
    # dt is ppBHPT (m1) time, so the candidates are the pp-axis durations (2026-09-26 fix:
    # this used the NR-axis tau_f/tau_ov, 16-34% off, after the model file was fixed).
    out = dict(n_post=int(m.sum()), beta_r_der=der["beta_r"],
               tau_f=der["tau_f_pp"], tau_ov=der["tau_ov_pp"])
    for tag, b in (("frq", emp["beta_frq"][m]), ("map", emp["beta_map"][m])):
        out[f"beta_late_{tag}"] = float(np.median(b[dt >= dt.max() * 0.5]))
        def res(p):                          # beta_r FIXED at derived; D, tau free
            D, lt = p
            return der["beta_r"] + D * np.exp(-dt / np.exp(lt)) - b
        try:
            r = least_squares(res, [b[0] - der["beta_r"], np.log(max(der["tau_f_pp"], 1.0))],
                              bounds=([-2, np.log(0.5)], [2, np.log(300)]), max_nfev=6000)
            out[f"tau_meas_{tag}"] = float(np.exp(r.x[1]))
            out[f"D_meas_{tag}"] = float(r.x[0])
            out[f"rms_{tag}"] = float(np.sqrt(np.mean(r.fun ** 2)))
        except Exception as e:
            out[f"err_{tag}"] = str(e)
    out["ok"] = True
    return out


def t3_root(case, q, chi1, der, theta, tau_b):
    """Does G(t) = tau_b beta_I'(t) + beta_I(t) - beta_r have a sane root?"""
    b1, b2, a2, p00, p01, w0, r0, m0, m1, sSO, sSS = theta
    t, h = case["t_bhpt"], case["h_bhpt"]
    x = S.PC.x_scale(q) * S.get_x(t, h)        # note's x; NO clip: the junction replaces it
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    beta_I = base * (1 + nu * (b1 * x + b2 * x ** 2
                               + sSO * chi1 * x ** 1.5 + sSS * chi1 ** 2 * x ** 2))
    dbeta = np.gradient(beta_I, t)
    G = tau_b * dbeta + beta_I - der["beta_r"]
    i_m = int(np.argmax(np.abs(h)))
    w = (t >= t[i_m] - 3000.0) & (t <= t[i_m])          # inspiral side only
    tw, Gw = t[w], G[w]
    s = np.where(np.sign(Gw[:-1]) != np.sign(Gw[1:]))[0]
    out = dict(tau_b=float(tau_b), beta_r=der["beta_r"], base=float(base),
               gap_pct=float(100 * (der["beta_r"] - base) / base),
               G_early=float(Gw[0]), G_merger=float(Gw[-1]),
               n_sign_changes=int(len(s)))
    if len(s):
        i = s[-1]                                        # latest root before merger
        t0 = float(np.interp(0.0, [Gw[i], Gw[i + 1]], [tw[i], tw[i + 1]]))
        out.update(t_star_rel=t0 - float(t[i_m]),
                   x_star=float(np.interp(t0, t, x)),
                   beta_I_star=float(np.interp(t0, t, beta_I)),
                   sane=bool(-400.0 <= t0 - float(t[i_m]) <= 0.0))
    else:
        out["sane"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    ap.add_argument("--chi", type=float, nargs="+",
                    default=[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--coeffs", default="physical",
                    choices=["physical", "frozen9"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cj = HERE / "pn_anc_spin_physical_results" / f"coeffs_{a.coeffs}.json"
    theta = np.array(json.loads(cj.read_text())["theta"], float)
    print(f"coefficients: {cj.name}  b1={theta[0]:.4f} b2={theta[1]:.4f} "
          f"sSO={theta[9]:.4f} sSS={theta[10]:.4f}\n")

    rows = []
    for q in a.q:
        for c in a.chi:
            d = derived(q, c)
            case = None
            try:
                case = S.build_case(q, c)
                emp = empirical(case)
                t1 = t1_kappa(emp, d); t2 = t2_tau(emp, d)
            except Exception as e:
                print(f"q={q} chi={c:+.2f}  EMPIRICAL FAILED: {e}")
                emp, t1, t2 = None, {"ok": False, "reason": str(e)}, {"ok": False}
            # tau_b multiplies d(beta_I)/dt with t = ppBHPT time, so it must be a pp-axis
            # duration (2026-09-26 fix).  No case -> no T3 (was: silently the previous one).
            t3f = t3_root(case, q, c, d, theta, d["tau_f_pp"]) if case is not None else {"sane": False}
            t3o = t3_root(case, q, c, d, theta, d["tau_ov_pp"]) if case is not None else {"sane": False}
            rows.append(dict(q=q, chi1=c, derived=d, T1=t1, T2=t2,
                             T3_tau_f=t3f, T3_tau_ov=t3o,
                             mono_b=None if emp is None else float(emp["mono_b"]),
                             mono_n=None if emp is None else float(emp["mono_n"])))

    # ---------------- report ----------------
    print("=" * 108)
    print("T1  kappa = (w_pp/2)(1/Q_pp - 1/Q_f):  alpha's DERIVED ringdown log-slope [per m1, ppBHPT time]")
    print("=" * 108)
    print(f"{'q':>4} {'chi1':>6} {'chi_f':>7} {'Q_pp':>6} {'Q_f':>6} {'kappa_der':>10} "
          f"{'kappa_late':>11} {'kappa_fit':>10} {'ratio':>7} {'npost':>6} {'flag':>6}")
    for r in rows:
        d, t1 = r["derived"], r["T1"]
        kl = t1.get("kappa_late", np.nan); kf = t1.get("kappa_fit", np.nan)
        rat = kl / d["kappa"] if np.isfinite(kl) and d["kappa"] else np.nan
        flag = "NULLS" if (r["mono_n"] is not None and r["mono_n"] <= 0) else ""
        print(f"{r['q']:>4g} {r['chi1']:>+6.2f} {d['chif']:>7.4f} {d['Q_pp']:>6.3f} "
              f"{d['Q_f']:>6.3f} {d['kappa']:>+10.5f} {kl:>+11.5f} {kf:>+10.5f} "
              f"{rat:>7.2f} {t1.get('n_post', 0):>6d} {flag:>6}")

    print("\n" + "=" * 108)
    print("T2  relaxation timescale [m1, ppBHPT time]:  measured vs derived candidates")
    print("=" * 108)
    print(f"{'q':>4} {'chi1':>6} {'beta_r':>8} {'b_late_f':>9} {'b_late_m':>9} "
          f"{'tau_f':>7} {'tau_ov':>7} {'tau_frq':>8} {'tau_map':>8} {'tau_a_fit':>10}")
    for r in rows:
        d, t2, t1 = r["derived"], r["T2"], r["T1"]
        g = lambda k: t2.get(k, np.nan)
        print(f"{r['q']:>4g} {r['chi1']:>+6.2f} {d['beta_r']:>8.4f} "
              f"{g('beta_late_frq'):>9.4f} {g('beta_late_map'):>9.4f} "
              f"{d['tau_f_pp']:>7.2f} {d['tau_ov_pp']:>7.2f} {g('tau_meas_frq'):>8.2f} "
              f"{g('tau_meas_map'):>8.2f} {t1.get('tau_a_fit', np.nan):>10.2f}")

    print("\n" + "=" * 108)
    print("T3  junction root  G(t) = tau_b beta_I' + beta_I - beta_r = 0")
    print("=" * 108)
    print(f"{'q':>4} {'chi1':>6} {'base':>7} {'beta_r':>8} {'gap%':>7} "
          f"{'|tau_f: t*':>10} {'x*':>6} {'sane':>5} {'|tau_ov: t*':>11} {'x*':>6} {'sane':>5}")
    for r in rows:
        f_, o_ = r["T3_tau_f"], r["T3_tau_ov"]
        print(f"{r['q']:>4g} {r['chi1']:>+6.2f} {f_['base']:>7.4f} {f_['beta_r']:>8.4f} "
              f"{f_['gap_pct']:>+7.2f} {f_.get('t_star_rel', np.nan):>10.1f} "
              f"{f_.get('x_star', np.nan):>6.3f} {str(f_['sane']):>5} "
              f"{o_.get('t_star_rel', np.nan):>11.1f} {o_.get('x_star', np.nan):>6.3f} "
              f"{str(o_['sane']):>5}")

    # ---------------- verdicts ----------------
    ks = [(r["T1"].get("kappa_late", np.nan), r["derived"]["kappa"]) for r in rows
          if r["mono_n"] is not None and r["mono_n"] > 0]
    ks = [(a_, b_) for a_, b_ in ks if np.isfinite(a_)]
    print("\n" + "=" * 108)
    print("VERDICTS  (cases with non-monotone NR phase -- the chi1<0 nulls -- excluded)")
    print("=" * 108)
    if ks:
        sign_ok = sum(1 for a_, b_ in ks if np.sign(a_) == np.sign(b_))
        rr = np.array([a_ / b_ for a_, b_ in ks if b_ != 0])
        print(f"T1  sign agrees in {sign_ok}/{len(ks)} clean cases; "
              f"kappa_meas/kappa_der median {np.median(rr):.2f} "
              f"[{np.min(rr):.2f}, {np.max(rr):.2f}]")
        print(f"    -> {'PASS' if sign_ok == len(ks) and 0.3 < np.median(rr) < 3 else 'FAIL / NEEDS WORK'}")
    else:
        print("T1  no clean cases")
    tf = [r["T3_tau_f"]["sane"] for r in rows]; to = [r["T3_tau_ov"]["sane"] for r in rows]
    print(f"T3  sane root with tau_f:  {sum(tf)}/{len(tf)};  with tau_ov: {sum(to)}/{len(to)}")
    print(f"    -> {'PASS' if max(sum(tf), sum(to)) == len(tf) else 'FAIL / NEEDS WORK'}")

    out = RESULTS / f"junction_diag{('_' + a.tag) if a.tag else ''}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
