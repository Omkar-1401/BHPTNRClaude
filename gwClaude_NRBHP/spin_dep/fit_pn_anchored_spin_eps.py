"""pn_anchored + spin, with the spin correction weighted by eps = 1/q, not by nu.

    beta_insp = X1^(6/5) [ 1 + nu*(b1*xc + b2*xc^2)                       <- nonspinning
                             + W *((s11 chi1 + s12 chi1^2)*xc
                                 + (s21 chi1 + s22 chi1^2)*xc^2) ]        <- spin
    W = eps = 1/q  (this model)      vs      W = nu  (the original)

Everything else is IDENTICAL to fit_pn_anchored_spin.py: same frozen 9, same 96-point
grid, same waveform-mismatch objective, same derived beta_r, alpha still carries NO spin
terms.  `--weight nu` reproduces the original exactly, so the two run back to back under
identical conditions -- which matters, because the existing nu-weighted numbers come from
runs that differed in seed, budget and drive-quarantine setting and are not directly
comparable to each other.

WHY.  In the original the spin terms sat INSIDE the nonspinning drift's nu*(...) bracket
and so inherited a nu prefactor by construction -- never chosen, never tested.  The
symptom is a U-shape in q at every chi1 != 0 (at chi1 = +0.8: 1.09e-2 at q=3, 2.40e-3 at
q=5, 7.43e-3 at q=8) while the chi1 = 0 column falls monotonically.  The spin term is the
only difference between those columns, so the mass-ratio weighting is wrong.

MEASURED, from the parameter-free step-1 grid (spin_measure_results/curves_win20.json),
fitting |R_beta - 1| ~ V^k across q = 3..8 at fixed chi1, over x <= 0.09:

    V = nu = X1*X2       k = 1.45 +- 0.37
    V = X2 = 1/(1+q)     k = 1.15 +- 0.29
    V = eps = 1/q        k = 0.95 +- 0.24     <- consistent with EXACTLY 1
    V = nu*X1 (chi_eff)  k = 1.94 +- 0.49     <- chi_eff is the WRONG direction

Decline over q = 3 -> 8: measured 2.626 +- 0.633, against 2.667 for 1/q.

So this is not a fitted exponent.  eps = m2/m1 is the SELF-FORCE expansion parameter and
eps^1 is first-order self-force: the ppBHPT waveform is the adiabatic test-particle
solution around Kerr of spin chi1, exact at leading order, so what the calibration
corrects -- spin included -- should enter at O(eps).  An integer power of the natural
variable rather than a free real exponent.

ALPHA STAYS SPIN-BLIND, and re-checking strengthened it: R_alpha - 1 CHANGES SIGN across
q at chi1 = +0.3, +0.6, +0.8 (at +0.8: -0.43%, -0.46%, -0.33%, +0.01%, +0.14% for q=3..8).
No power law in any mass-ratio variable can change sign at fixed chi1, so alpha's residual
is not a mis-weighted physical term and no prefactor would rescue it; its size is also
below the within-case x-spread (0.7% vs 0.66-2.4%).  Both checks are INSPIRAL ONLY --
alpha's merger spin dependence remains untested (open item 2).

Usage:
    python fit_pn_anchored_spin_eps.py --weight eps --refit --budget 95
    python fit_pn_anchored_spin_eps.py --weight nu  --refit --budget 95   # control
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT.parent / "BHPTutils"))
sys.path.insert(0, "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/surrogates")
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative as creative
import physics_common as PC          # shared QNMs, x clock, window, stamps (2026-09-26)
import fit_scaling_wf_nu_hybrid_q_dep as H
import fit_scaling_wf_nu_q_dep as WF
import surfinBH

_SFBH = surfinBH.LoadFits("NRSur3dq8Remnant")
QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292
# ---------------------------------------------------------------------------
# UNITS FIX 2026-09-24 -- see UNITS_AUDIT.md
#
# beta_r = dtau_NR/dt_pp, and the two waveforms live on DIFFERENT time axes:
#   ppBHPT  t_bhpt : PRIMARY-mass units (calibrated=False returns the raw training
#                    grid, common_utils/utils.py:176 -- no mass rescaling)
#   NR      t_nr   : TOTAL-mass units
# So beta_r must carry the m1 -> M unit conversion.  The old form computed the QNM
# ratio with BOTH holes in total-mass units -- correct physics, wrong object: it is
# the frequency ratio, not the time-map factor, and is too large by 1/X1 (33% at
# q=3, 12.5% at q=8).  Measured w_pp: 0.3763/0.3746/0.3734 at q=3/5/8 chi1=0 vs
# w220(0)=0.3683 and w220(0)(1+q)/q = 0.4911/0.4420/0.4143.
#   OLD: (w220(chi1)*(1+q)/q) / (w220(chi_f)/M_f)
#   NEW:  w_pp(chi1) * M_f / w220(chi_f)          == OLD * X1
# Note Eq. (7) already uses the m1 axis for the pp side (eta_pp = m2/m1), so this
# restores consistency with the derivation the model is built on.
#
# W_SCHW: the BCW FIT gives w220(0) = 0.3683, but the exact Schwarzschild (2,2,0)
# value is 0.3737 (1.5% apart) and the gwr_* family already uses 0.3737.  Measured
# 0.3734-0.3763 favours 0.3737.  RESCALE_QNM=True anchors the fit's level to the
# exact value while keeping its chi dependence.  Set False to reproduce the old
# level.  NOTE: at chi1=+0.6 the unrescaled fit is already within 0.4% of measured,
# so this 1.5% choice sits inside the ~1% scatter of that measurement -- it is
# flagged, not settled.  A1 (the X1 factor) does NOT depend on it.
W_SCHW = 0.3737
RESCALE_QNM = True


def w_pp_qnm(chi):
    """pp-side (2,2,0) QNM frequency in units of the PRIMARY's mass: exact Kerr
    (physics_common.w220_exact; counter-rotating branch for chi < 0).  Replaces the BCW fit
    rescaled to 0.3737 (2026-09-26): that was 2.35% high at chi=+0.6 and, extrapolated to
    chi < 0, 3% low at -0.2 and 7% low at -0.6.  W_SCHW / RESCALE_QNM above are unused."""
    return PC.w220_exact(chi)

PN_SLOPE, XCLIP, T_ANCHOR = 55.0 / 42.0, 0.26, -100.0

# TEMPORAL DOMAIN DECOMPOSITION on the DRIVE (Rink et al., adapted).  Their surrogate
# partitions the time domain at merger, [-13500, 0] U [0, 125], and trains the two
# subdomains separately -- because a global fit including the RETROGRADE ringdown modes
# excited at chi1 <~ -0.6 could not be made accurate.  So the chi1 < 0 ringdown we read
# x(t) from is a quarantined subdomain model.
#
# Measured consequence here (2026-08-31): at chi1 = -0.6 the post-merger |h| envelope
# carries 15-16 genuine nulls (1-5 at -0.2, ZERO at chi1 > 0); each null wraps the phase
# and spikes x = (omega/2)^(2/3) to 1.4-1.8x the pp-side QNM frequency, which is
# physically impossible.  The spikes reach alpha and beta because pn_anchored's switch
# saturates at S <= 0.19, so the x-driven INSPIRAL branch still supplies ~83% of both
# through the whole ringdown.
#
# The fix is H(t) applied to the drive: freeze x at its merger value for t > 0.  This is
# not a new mechanism -- XCLIP already does exactly this for chi1 >= 0, where x at merger
# (0.27-0.33) sits ABOVE the clip so xc is pinned at 0.26 for the entire ringdown.  It
# fails on retrograde only because x at merger there is 0.233-0.235, BELOW the clip, so
# the dips pass through unclipped and xc oscillates.  Verified: turning this on changes
# NOTHING at chi1 >= 0 (max |dxc| = 0), so the chi1 = 0 null check is preserved exactly.
FREEZE_XC = False

RESULTS = HERE / "pn_anchored_spin_eps_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs_spin_eps.json"
WEIGHT = "eps"      # "eps" = 1/q (this model); "nu" = the original, as a control
PARTIAL = COEFFS.with_name("coeffs_spin_partial.json")
CACHE = HERE / "spin_wf_cache"; CACHE.mkdir(exist_ok=True)

TRAIN_Q = [3.0, 3.2564102564, 3.5128205128, 3.8974358974, 4.2, 4.6, 5.0,
           5.4358974359, 5.9487179487, 6.5897435897, 7.2307692308, 8.0]
CHIS = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
SNAMES = ["s11", "s12", "s21", "s22"]
# Seed from the measured-ratio fit, CORRECTLY mapped to this convention.
# Measured (p fixed at 1, spin_extension_plan.md sec 8):
#     R_beta - 1 = nu [ (a1 + a2 x) chi + (a3 + a4 x) chi^2 ],
#     a1=-0.010, a2=+1.107, a3=-0.160, a4=+2.882
# Here:
#     R_beta - 1 ~ nu [ (s11 chi + s12 chi^2) x + (s21 chi + s22 chi^2) x^2 ]
# so the x-LINEAR terms map directly: s11 <- a2, s12 <- a4; the x^2 terms have no
# counterpart in the measured form (start at 0), and the measured CONSTANTS a1, a3 have
# no home here at all -- pn_anchored's spin correction carries no level shift by
# construction (deliberate: sec 8 found a level shift is only needed at chi1 = -0.6).
SEED = np.array([1.107, 2.882, 0.0, 0.0])
SRC_STRIDE, NR_STRIDE, F_LOW = 6, 10, 5e-3


MODEL_FORM = PC.PNA_FORM


def shipped9():
    """The nonspinning 9.  Refuses a coeffs.json fitted for a different beta_r form: the
    pre-fix 9 with the corrected beta_r scored median 2.16e-2 (audit 2026-09-26)."""
    d = json.loads((PARENT / "pn_anchored_results" / "coeffs.json").read_text())
    if d.get("model_form") != MODEL_FORM:
        raise RuntimeError("pn_anchored_results/coeffs.json was fitted for model_form="
                           f"{d.get('model_form', 'unstamped')!r}, not {MODEL_FORM!r}.  Refit "
                           "the nonspinning model first: fit_scaling_pn_anchored.py --refit")
    return np.array(d["theta"], float)


def get_x(t, h):
    ph = np.unwrap(np.angle(h)); n = len(ph)
    win = PC.savgol_window(float(np.median(np.diff(t))), 80.0, n)   # 80 m1 in time (was 401 samples = 40 m1 here)
    if win >= 11:
        ph = savgol_filter(ph, win, 3, mode="interp")
    return np.clip((0.5 * np.abs(np.gradient(ph, t))) ** (2.0 / 3.0), 1e-8, 0.6)


def w220(chi):
    """Kerr (2,2,0) QNM frequency in units of the hole's mass, exact (0.37367 at chi=0)."""
    return PC.w220_exact(chi)          # exact Kerr (2026-09-26); was the BCW fit


def beta_r_phys(q, chi1):
    """QNM remnant time-stretch = omega_QNM(pp side) / omega_QNM(NR side).

    Spin dependence DERIVED from RemS + the QNM fit; NO free parameter.

    BOTH sides move with spin, and getting only one of them was a real bug (caught
    2026-08-30, diag_beta_r.py):
      * pp side: in ppBHPT the 'remnant' IS the primary, spin chi1 (the secondary is
        negligible).  Its QNM is w220(chi1) IN UNITS OF THE PRIMARY'S MASS -- and that is
        what we need, because the ppBHPT time axis is itself in primary-mass units
        (calibrated=False returns the raw training grid).  NO (1+q)/q.  CORRECTED
        2026-09-24: this used to convert to total-mass units with (1+q)/q, which double-
        counts and made beta_r 17-29% too large (UNITS_AUDIT.md A1).  Both sides exact
        Kerr since 2026-09-26 (physics_common.w220_exact).  Leaving the pp side at
        w220(0) while the NR side moved made beta_r fall with positive spin (2026-08-30).
      * NR side: w220(chi_f)/M_f with M_f, chi_f from NRSur3dq8Remnant.
    At chi1 = 0 this reduces exactly to the (corrected) nonspinning expression."""
    s1 = [0, 0, float(chi1)]
    mf, _ = _SFBH.mf(q, s1, [0, 0, 0]); chif, _ = _SFBH.chif(q, s1, [0, 0, 0])
    return w_pp_qnm(float(chi1)) * float(mf) / w220(float(chif[2]))


def model_ab(th9, ths, q, chi1, x, p_loss):
    b1, b2, a2, p00, p01, w0, r0, m0, m1 = th9
    s11, s12, s21, s22 = ths
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    # THE CHANGE: the spin terms carry their OWN mass-ratio weight instead of
    # inheriting the nonspinning drift's nu.  W = 1/q is first-order self-force;
    # measured k = 0.95 +- 0.24 against 1.45 +- 0.37 for nu.  --weight nu recovers
    # the original model exactly.
    W = (1.0 / q) if WEIGHT == "eps" else nu
    sp1 = s11 * chi1 + s12 * chi1 ** 2
    sp2 = s21 * chi1 + s22 * chi1 ** 2
    xc = PC.x_scale(q) * np.minimum(x, XCLIP)   # note's x (Eq. 6); clip on x_pp (2026-09-26)
    S = creative.sigmoid((p_loss - (p00 + p01 * nu)) / max(w0, 1e-3))
    beta = ((1 - S) * base * (1 + nu * (b1 * xc + b2 * xc ** 2)
                                + W * (sp1 * xc + sp2 * xc ** 2))
            + S * beta_r_phys(q, chi1) * r0)
    alpha = ((1 - S) * base * (1 + PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
             + S * base * (m0 + m1 * nu))
    return alpha, beta


def _err_at_t0(t0, alpha, ts, t, h, tn, hn):
    tau = t0 + ts; lo = max(tn[0], tau[0]); hi = min(tn[-1], tau[-1])
    m = (tn >= lo) & (tn <= hi)
    if m.sum() / len(tn) < H.MIN_COVERAGE:
        return 50.0
    tc = tn[m]; r = hn[m]
    g = np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag)
    gg = np.interp(tc, tau, alpha) * g
    n1 = np.sum(np.abs(r) ** 2); n2 = np.sum(np.abs(gg) ** 2)
    Z = np.abs(np.sum(r * np.conj(gg)))
    return float((n1 + n2 - 2 * Z) / (2 * n1))


def mismatch(th9, ths, q, chi1, case, key, seed):
    p = "fit_" if key == "fit" else ""
    t, h, tn, hn = case[p + "t_bhpt"], case[p + "h_bhpt"], case[p + "t_nr"], case[p + "h_nr"]
    pl, x = case[p + "p_loss"], case[p + "x"]
    alpha, beta = model_ab(th9, ths, q, chi1, x, pl)
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(T_ANCHOR, t, bc)
    d = np.min(np.diff(ts))
    if d <= 0:
        return 50.0 + 1e3 * (-d), seed
    grid = np.linspace(seed - 200, seed + 200, 81)
    e = [_err_at_t0(x0, alpha, ts, t, h, tn, hn) for x0 in grid]
    c = grid[int(np.argmin(e))]
    r = minimize_scalar(lambda x0: _err_at_t0(x0, alpha, ts, t, h, tn, hn),
                        bounds=(c - 6, c + 6), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def full_mm(th9, ths, q, chi1, case, seed=-50.0):
    e, t0 = mismatch(th9, ths, q, chi1, case, "fit", seed)
    e, _ = mismatch(th9, ths, q, chi1, case, "full", t0)
    return e


# ---------------------------------------------------------------------------

def cache_path(q, chi1):
    return CACHE / f"wf_q{q:.6f}_chi{chi1:+.3f}.npz"


def build_case(q, chi1):
    """Generate/load the (ppBHPT 2d, NRHybSur3dq8) pair and everything derived."""
    p = cache_path(q, chi1)
    if not p.exists():
        import BHPTNRSur2dq1e3 as ppsur, gwsurrogate
        global _NR
        try:
            _NR
        except NameError:
            _NR = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
        t_pp, hd = ppsur.generate_surrogate(q=q, spin1=chi1, calibrated=False,
                                            modes=[(2, 2)], neg_modes=False)
        h_pp = np.asarray(hd[(2, 2)])
        t_nr, hn, _ = _NR(q, [0, 0, chi1], [0, 0, 0.0], dt=0.1, f_low=F_LOW)
        h_nr = np.asarray(hn[(2, 2)])
        m = (t_nr >= WF.qdep.NR_T_START) & (t_nr <= WF.qdep.NR_T_END)
        np.savez_compressed(p, t_bhpt=np.asarray(t_pp, float),
                            h_bhpt_re=h_pp.real, h_bhpt_im=h_pp.imag,
                            t_nr=np.asarray(t_nr, float)[m],
                            h_nr_re=h_nr.real[m], h_nr_im=h_nr.imag[m])
    d = np.load(p)
    t = d["t_bhpt"]; h = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    tn = d["t_nr"]; hn = d["h_nr_re"] + 1j * d["h_nr_im"]
    si = np.unique(np.r_[np.arange(0, len(t), SRC_STRIDE), len(t) - 1])
    ni = np.unique(np.r_[np.arange(0, len(tn), NR_STRIDE), len(tn) - 1])
    x = get_x(t, h)
    if FREEZE_XC:                      # H(t) on the drive -- see the FREEZE_XC note
        ipk = int(np.argmax(np.abs(h)))
        x = x.copy(); x[t > t[ipk]] = x[ipk]
    # p_loss = 0.5*(e_hat + j_hat), the SAME wf-flux coordinate the nonspinning
    # pn_anchored uses (fit_scaling_wf_nu_q_dep.wf_loss_coordinates; its `meta`
    # argument is unused inside, so an empty dict is safe).
    pl = WF.wf_loss_coordinates(t, h, {})["p_loss"]
    return {"t_bhpt": t, "h_bhpt": h, "t_nr": tn, "h_nr": hn, "x": x, "p_loss": pl,
            "fit_t_bhpt": t[si], "fit_h_bhpt": h[si], "fit_t_nr": tn[ni],
            "fit_h_nr": hn[ni], "fit_x": x[si], "fit_p_loss": pl[si]}


def main():
    import fit_pn_anchored_spin as _S0; _S0.refuse_frozen9_protocol(__file__)   # 2026-09-27
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", choices=["eps", "nu"], default="eps",
                    help="mass-ratio weight on the SPIN terms: eps=1/q, or nu for the "
                         "original as a like-for-like control")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--freeze-xc", action="store_true",
                    help="freeze the drive x at its merger value for t > 0 (Rink et al. "
                         "temporal domain decomposition applied to the drive); affects "
                         "chi1 <~ -0.2 only, see the FREEZE_XC note")
    ap.add_argument("--resume", action="store_true",
                    help="with --refit, start the optimiser from coeffs_spin.json instead "
                         "of SEED (the 2026-08-30 run stopped on budget, not convergence)")
    ap.add_argument("--ab-test", action="store_true",
                    help="score the CURRENT coefficients both ways first, to isolate how "
                         "much of each chi1 column is the retrograde x artefact")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--budget", type=float, default=95.0,
                    help="wall-clock minutes for the optimiser; on expiry it stops "
                         "GRACEFULLY and the best-so-far theta is scored and written. "
                         "Keep this below the SLURM -t ceiling minus ~15 min.")
    args = ap.parse_args()

    global FREEZE_XC, WEIGHT
    FREEZE_XC = args.freeze_xc
    WEIGHT = args.weight
    print(f"spin weight W = {'eps = 1/q' if WEIGHT == 'eps' else 'nu  (CONTROL)'}")
    th9 = shipped9()
    print(f"drive: x {'FROZEN at merger (H(t) quarantine)' if FREEZE_XC else 'unmodified'}")
    print(f"frozen 9: {dict(zip(['b1','b2','a2','p00','p01','w0','r0','m0','m1'], np.round(th9,4)))}")
    print(f"grid: {len(TRAIN_Q)} q x {len(CHIS)} chi1 = {len(TRAIN_Q)*len(CHIS)} points",
          flush=True)

    t0 = time.time(); cases = {}
    for q in TRAIN_Q:
        for c in CHIS:
            cases[(q, c)] = build_case(q, c)
        print(f"  cached q={q:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)
    if args.cache_only:
        return

    # An A/B on the drive is only meaningful at coefficients that FIT: at SEED the
    # mismatch is dominated by gross coefficient error (3e-2 at chi1 = -0.6) and swamps
    # any artefact contribution.  So --ab-test and --resume both read the fitted values.
    use_fitted = (not args.refit) or args.resume or args.ab_test
    if use_fitted and COEFFS.exists():
        ths = np.array(json.loads(COEFFS.read_text())["theta_spin"], float)
        print(f"  theta_spin loaded from {COEFFS.name}")
    else:
        # Seed from the nu-weighted fit, rescaled so the spin correction STARTS THE SAME
        # SIZE under the new weight: nu*s_old = W*s_new, i.e. s_new = s_old*mean(nu/W).
        # Both --weight arms then start from an equivalent correction, so the comparison
        # is not confounded by seed quality.
        prev = HERE / "pn_anchored_spin_results" / "coeffs_spin.json"
        base_s = (np.array(json.loads(prev.read_text())["theta_spin"], float)
                  if prev.exists() else SEED.copy())
        qs = np.array(TRAIN_Q, float)
        scale = float(np.mean((qs / (1 + qs) ** 2) * qs)) if WEIGHT == "eps" else 1.0
        ths = base_s * scale
        print(f"  seeded from the nu-weighted fit x {scale:.4f} -> {np.round(ths, 4)}")

    # Every objective evaluation is ~96 waveform mismatches (~1 min).  Powell is
    # silent and scipy only returns at convergence, so without the bookkeeping below
    # a wall-clock kill loses the entire run.  Each eval therefore (a) prints, and
    # (b) checkpoints the best-so-far to disk, and the optimiser stops itself before
    # the SLURM ceiling so the final 96-point table still gets written.
    class _Budget(Exception):
        pass

    st = {"n": 0, "best": np.inf, "bx": np.array(ths, float), "t0": time.time(),
          "budget": args.budget * 60.0, "stop": False}

    def objective(ths_):
        errs = [full_mm(th9, ths_, q, c, cases[(q, c)])
                for q in TRAIN_Q for c in CHIS]
        mean, mx = float(np.mean(errs)), float(np.max(errs))
        objective.last = (mean, mx)
        st["n"] += 1
        el = time.time() - st["t0"]
        tag = ""
        if mean < st["best"]:
            st["best"] = mean
            st["bx"] = np.array(ths_, float)
            tag = "  <-- best"
            PARTIAL.write_text(json.dumps({
                "status": "in progress", "n_eval": st["n"],
                "elapsed_min": round(el / 60.0, 2),
                "spin_param_names": SNAMES,
                "theta_spin": [float(x) for x in st["bx"]],
                "mean": mean, "max": mx}, indent=2))
        print(f"  [{st['n']:4d}] {el/60:6.1f}m  mean={mean:.4e} max={mx:.4e}  "
              f"best={st['best']:.4e}  th=" +
              " ".join(f"{n}={v:+.4f}" for n, v in zip(SNAMES, ths_)) + tag, flush=True)
        if el > st["budget"]:
            st["stop"] = True
            raise _Budget
        return mean
    objective.last = (np.nan, np.nan)

    if args.ab_test:
        print("\n===== A/B: same coefficients, drive quarantine OFF then ON =====",
              flush=True)
        print(f"{'chi1':>6} {'median OFF':>12} {'median ON':>12} {'max OFF':>11} "
              f"{'max ON':>11} {'gain':>7}")
        base = {}
        for tag, flag in (("off", False), ("on", True)):
            FREEZE_XC = flag
            # rebuild BOTH ways -- `cases` was built under the CLI flag, so reusing it
            # for the "off" arm would silently compare a branch against itself
            cs = {k: build_case(*k) for k in cases}
            base[tag] = {c: np.array([full_mm(th9, ths, q, c, cs[(q, c)])
                                      for q in TRAIN_Q]) for c in CHIS}
        FREEZE_XC = args.freeze_xc
        for c in CHIS:
            o, n = base["off"][c], base["on"][c]
            print(f"{c:+6.1f} {np.median(o):12.4e} {np.median(n):12.4e} "
                  f"{o.max():11.4e} {n.max():11.4e} {np.median(o)/np.median(n):7.2f}x",
                  flush=True)
        if not args.refit:
            return

    print(f"\nseed theta_spin = {np.round(ths,4)}")
    objective(ths); print(f"  seed: mean={objective.last[0]:.4e} max={objective.last[1]:.4e}",
                          flush=True)
    print(f"  chi1=0 reference (shipped model on the 1d surrogate gave 6.148e-04; "
          f"this uses the 2d surrogate at chi1=0, so exact equality is NOT expected): "
          f"{np.median([full_mm(th9, ths, q, 0.0, cases[(q,0.0)]) for q in TRAIN_Q]):.4e}",
          flush=True)

    if args.refit:
        st["t0"] = time.time()          # do not charge the seed sweep to the budget
        st["best"], st["bx"] = objective.last[0], np.array(ths, float)
        try:
            r = minimize(objective, ths, method="Powell",
                         options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
            ths = np.array(r.x, float)
            if objective.last[0] > st["best"]:      # Powell returned a worse point
                ths = st["bx"]
            print(f"  opt CONVERGED in {time.time()-st['t0']:.0f}s / {st['n']} evals: "
                  f"{r.message}", flush=True)
        except _Budget:
            ths = st["bx"]
            print(f"  opt STOPPED ON BUDGET after {args.budget:.0f} min / {st['n']} "
                  f"evals -- reporting best-so-far (mean={st['best']:.4e}). "
                  f"Rerun without --refit to resume from it.", flush=True)

    print(f"\nfitted theta_spin = " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(SNAMES, ths)))
    print(f"\n{'q':>8} " + " ".join(f"{c:>+8.1f}" for c in CHIS))
    allm = {}
    for q in TRAIN_Q:
        row = [full_mm(th9, ths, q, c, cases[(q, c)]) for c in CHIS]
        for c, v in zip(CHIS, row):
            allm[f"{q:.4f}_{c:+.1f}"] = v
        print(f"{q:>8.4f} " + " ".join(f"{v:8.2e}" for v in row))
    v = np.array(list(allm.values()))
    print(f"\nALL 96: median {np.median(v):.4e}  max {np.max(v):.4e}")
    for c in CHIS:
        sub = np.array([allm[f"{q:.4f}_{c:+.1f}"] for q in TRAIN_Q])
        print(f"  chi1={c:+.1f}: median {np.median(sub):.4e}  max {np.max(sub):.4e}")

    COEFFS.write_text(json.dumps({
        "model": "pn_anchored_spin", "spin_param_names": SNAMES,
        "theta_spin": [float(x) for x in ths],
        "theta9_frozen": [float(x) for x in th9],
        "grid": {"q": TRAIN_Q, "chi1": CHIS},
        "objective": "waveform mismatch (user choice 2026-08-30)",
        "alpha_spin_terms": "NONE (measured consistent with zero, see spin_extension_plan.md sec 7)",
        "beta_r_spin": "DERIVED from NRSur3dq8Remnant mf/chif(q,[0,0,chi1]) -- no free parameter",
        "mismatches": allm,
        "median": float(np.median(v)), "max": float(np.max(v)),
    }, indent=2))
    print(f"\nwrote {COEFFS}")


if __name__ == "__main__":
    main()
