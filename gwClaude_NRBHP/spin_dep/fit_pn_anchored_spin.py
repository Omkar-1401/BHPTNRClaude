"""pn_anchored extended to aligned primary spin: 4 spin coefficients on the 9 nonspinning ones.

    STATUS 2026-09-27: the installed coefficients (coeffs_spin.json, key spin_base) have
    b1, b2, a2 from nonspinning pn_anchored and p00, p01, w0, r0, m0, m1 FITTED on the 2d spin
    grid (still spin-independent), with the 4 spin terms (test_spin_free_switch.py --free
    merger).  The frozen-9 text below is the original (2026-08-30) protocol, kept as history;
    the model no longer reduces to nonspinning pn_anchored at chi1 = 0.

    beta_insp = X1^(6/5) [ 1 + nu ( b1(chi1) xc + b2(chi1) xc^2 ) ]
        b1(chi1) = b1 + s11 chi1 + s12 chi1^2
        b2(chi1) = b2 + s21 chi1 + s22 chi1^2
    alpha_insp= X1^(6/5) [ 1 + (55/42) nu xc + nu a2 xc^2 ]      # NO spin terms
    beta_r    = beta_r_phys(q, chi1) * r0                        # spin DERIVED from RemS
    alpha_mr  = X1^(6/5) (m0 + m1 nu)
    S         = sigmoid((p_loss - (p0_0 + p0_1 nu))/w0)

theta_spin = [s11, s12, s21, s22];  [HISTORY, superseded 2026-09-27] the 9 nonspinning
constants were held at their SHIPPED values (verified 2026-08-29 to reproduce their documented mismatches to 3-4
digits, job 3448537).  At chi1 = 0 the model is bit-identical to the shipped one, so the
chi1 = 0 rows are a NULL CHECK, not a refit.

WHY freeze: an independent refit of the 9 (job 3448537) landed in a WORSE basin with
b1 -42%, so re-opening them would entangle the spin answer with basin choice.  The
frozen b2 is known to be merger-contaminated (~2x the second-order-self-force x^2
content, wardell_ideation.md 4h), but that bias is chi1-INDEPENDENT while the spin terms
are strictly odd/even in chi1, so over the symmetric grid they are near-orthogonal.

WHY alpha gets no spin terms: measured (spin_extension_plan.md section 7) --
R_alpha = 1.000 +- 0.005 across the whole grid in the inspiral band, i.e. consistent
with zero, and of OPPOSITE sign to Islam Eq. 8's prediction.

OBJECTIVE: waveform mismatch (user's choice, 2026-08-30) -- the paradigm the
nonspinning models were built under, and directly comparable to Islam Eq. 7.  The
measured-ratio fit of section 8 is used only as the SEED.

GRID: 12 even-nu q in [3,8] (the same TRAIN_Q as the nonspinning fit) x 8 chi1 in
[-0.6, +0.8] = 96 points.  chi1 >= -0.6 avoids the retrograde-QNM regime.

Usage:
    python fit_pn_anchored_spin.py --cache-only      # build the waveform cache
    python fit_pn_anchored_spin.py --refit           # fit the 4 spin coefficients
    python fit_pn_anchored_spin.py                   # evaluate whatever is in coeffs
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
QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292   # BCW fit coefficients -- UNUSED since 2026-09-26 (w220 = exact Kerr)
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

RESULTS = HERE / "pn_anchored_spin_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs_spin.json"
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

# The 9 non-spin coefficients of THIS model (installed 2026-09-27, user decision).  b1, b2, a2
# (inspiral shape) are the nonspinning pn_anchored values; p00, p01, w0 (switch) and r0, m0,
# m1 (merger branch) are fitted on the 2d spin grid together with the 4 spin terms.  All 9
# stay spin-INDEPENDENT (functions of nu only).  Consequence: at chi1 = 0 the model no longer
# reduces exactly to nonspinning pn_anchored.  Before 2026-09-27 all 9 were frozen at the
# nonspinning values (median 8.93e-4 on the fixed code; the switch+merger fit gives 8.65e-4,
# mean 1.17e-3 vs 1.68e-3).
NAMES9 = ["b1", "b2", "a2", "p00", "p01", "w0", "r0", "m0", "m1"]
FREE9 = [3, 4, 5, 6, 7, 8]                  # fitted here; 0, 1, 2 from the nonspinning fit
SPIN_BASE = ("pn_anchored_spin base 2026-09-27: b1,b2,a2 = nonspinning pn_anchored; "
             "p00,p01,w0,r0,m0,m1 fitted on the 2d spin grid (spin-independent)")


def model9():
    """The 9 non-spin coefficients this model runs with, from coeffs_spin.json `theta9`.
    Refuses a file not stamped with SPIN_BASE / MODEL_FORM, or whose b1, b2, a2 no longer
    equal the nonspinning coeffs.json (that would pair new and old coefficients)."""
    d = json.loads(COEFFS.read_text())
    if d.get("model_form") != MODEL_FORM or d.get("spin_base") != SPIN_BASE:
        raise RuntimeError(f"{COEFFS.name} is stamped model_form={d.get('model_form')!r}, "
                           f"spin_base={d.get('spin_base')!r}; expected the current "
                           "MODEL_FORM and SPIN_BASE.  Refit: fit_pn_anchored_spin.py --refit")
    th9 = np.array(d["theta9"], float)
    if not np.allclose(th9[:3], shipped9()[:3], rtol=0, atol=1e-12):
        raise RuntimeError(f"{COEFFS.name} b1, b2, a2 differ from the nonspinning "
                           "pn_anchored_results/coeffs.json -- refit the spin model")
    return th9


def refuse_frozen9_protocol(script):
    """Old scripts that pair shipped9() with coeffs_spin.json's theta_spin (the pre-
    2026-09-27 frozen-9 protocol) would now combine new spin terms with the old switch and
    merger -- and some write coeffs_spin*.json.  They stop here instead."""
    if COEFFS.exists() and json.loads(COEFFS.read_text()).get("spin_base") == SPIN_BASE:
        sys.exit(f"{script}: retired 2026-09-27 -- it assumes the frozen-9 pn_anchored_spin, "
                 f"but {COEFFS.name} now holds the switch+merger model (see "
                 "scaling_pn_anchored_spin.md).  Use fit_pn_anchored_spin.py / model9().")


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
    b1c = b1 + s11 * chi1 + s12 * chi1 ** 2
    b2c = b2 + s21 * chi1 + s22 * chi1 ** 2
    xc = PC.x_scale(q) * np.minimum(x, XCLIP)   # note's x (Eq. 6); clip on x_pp (2026-09-26)
    S = creative.sigmoid((p_loss - (p00 + p01 * nu)) / max(w0, 1e-3))
    beta = ((1 - S) * base * (1 + nu * (b1c * xc + b2c * xc ** 2))
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
    ap = argparse.ArgumentParser()
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

    global FREEZE_XC
    FREEZE_XC = args.freeze_xc
    th9 = model9() if COEFFS.exists() else shipped9()
    print(f"drive: x {'FROZEN at merger (H(t) quarantine)' if FREEZE_XC else 'unmodified'}")
    print(f"base 9 (b1,b2,a2 nonspinning; rest fitted on 2d): "
          f"{dict(zip(NAMES9, np.round(th9, 4)))}")
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
        _d = json.loads(COEFFS.read_text())
        if _d.get("model_form") != MODEL_FORM and not args.refit:
            raise SystemExit(f"{COEFFS.name} was fitted for model_form="
                             f"{_d.get('model_form', 'unstamped')!r}, not {MODEL_FORM!r}; "
                             "refusing to score it.  Refit first (--refit).")
        ths = np.array(_d["theta_spin"], float)
        print(f"  theta_spin loaded from {COEFFS.name}"
              + ("" if _d.get("model_form") == MODEL_FORM else "  (as a SEED only: other model_form)"))
    else:
        ths = SEED.copy()
        if use_fitted:
            print(f"  WARNING: {COEFFS.name} absent -- falling back to SEED, so any "
                  f"A/B below is NOT diagnostic")

    # Every objective evaluation is ~96 waveform mismatches (~1 min).  Powell is
    # silent and scipy only returns at convergence, so without the bookkeeping below
    # a wall-clock kill loses the entire run.  Each eval therefore (a) prints, and
    # (b) checkpoints the best-so-far to disk, and the optimiser stops itself before
    # the SLURM ceiling so the final 96-point table still gets written.
    class _Budget(Exception):
        pass

    st = {"n": 0, "best": np.inf, "bx": np.r_[th9[FREE9], ths], "t0": time.time(),
          "budget": args.budget * 60.0, "stop": False}

    FNAMES = [NAMES9[i] for i in FREE9] + SNAMES

    def unpack(xv):
        t9 = th9.copy(); t9[FREE9] = xv[:len(FREE9)]
        return t9, np.asarray(xv[len(FREE9):], float)

    def objective(xv):
        t9_, ths_ = unpack(xv)
        errs = [full_mm(t9_, ths_, q, c, cases[(q, c)])
                for q in TRAIN_Q for c in CHIS]
        mean, mx = float(np.mean(errs)), float(np.max(errs))
        objective.last = (mean, mx)
        st["n"] += 1
        el = time.time() - st["t0"]
        tag = ""
        if mean < st["best"]:
            st["best"] = mean
            st["bx"] = np.array(xv, float)
            tag = "  <-- best"
            PARTIAL.write_text(json.dumps({
                "status": "in progress", "n_eval": st["n"],
                "elapsed_min": round(el / 60.0, 2),
                "spin_param_names": SNAMES, "spin_base": SPIN_BASE,
                "theta9": [float(x) for x in t9_],
                "theta_spin": [float(x) for x in ths_],
                "mean": mean, "max": mx}, indent=2))
        print(f"  [{st['n']:4d}] {el/60:6.1f}m  mean={mean:.4e} max={mx:.4e}  "
              f"best={st['best']:.4e}  th=" +
              " ".join(f"{n}={v:+.4f}" for n, v in zip(FNAMES, xv)) + tag, flush=True)
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
    x0 = np.r_[th9[FREE9], ths]
    objective(x0); print(f"  seed: mean={objective.last[0]:.4e} max={objective.last[1]:.4e}",
                          flush=True)
    print(f"  chi1=0 reference (shipped model on the 1d surrogate gave 6.148e-04; "
          f"this uses the 2d surrogate at chi1=0, so exact equality is NOT expected): "
          f"{np.median([full_mm(th9, ths, q, 0.0, cases[(q,0.0)]) for q in TRAIN_Q]):.4e}",
          flush=True)

    if args.refit:
        st["t0"] = time.time()          # do not charge the seed sweep to the budget
        st["best"], st["bx"] = objective.last[0], x0.copy()
        try:
            minimize(objective, x0, method="Powell",
                     options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
            th9, ths = unpack(st["bx"])             # best point seen (Powell's last may be worse)
            print(f"  opt CONVERGED in {time.time()-st['t0']:.0f}s / {st['n']} evals: "
                  f"{r.message}", flush=True)
        except _Budget:
            th9, ths = unpack(st["bx"])
            print(f"  opt STOPPED ON BUDGET after {args.budget:.0f} min / {st['n']} "
                  f"evals -- reporting best-so-far (mean={st['best']:.4e}). "
                  f"Rerun without --refit to resume from it.", flush=True)

    print(f"\nfitted: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(FNAMES, np.r_[th9[FREE9], ths])))
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

    if not args.refit:            # scoring only: never rewrite the installed file (2026-09-27)
        print(f"\n(scored only; {COEFFS.name} left unchanged -- stored mean "
              f"{json.loads(COEFFS.read_text()).get('mean', float('nan')):.4e}, this run "
              f"{float(np.mean(v)):.4e})")
        return
    COEFFS.write_text(json.dumps({
        "model": "pn_anchored_spin", "model_form": MODEL_FORM, "spin_param_names": SNAMES,
        "theta_spin": [float(x) for x in ths],
        "spin_base": SPIN_BASE, "names9": NAMES9,
        "theta9": [float(x) for x in th9],
        "theta9_fitted_here": [NAMES9[i] for i in FREE9],
        "grid": {"q": TRAIN_Q, "chi1": CHIS},
        "objective": "waveform mismatch (user choice 2026-08-30)",
        "alpha_spin_terms": "NONE (measured consistent with zero, see spin_extension_plan.md sec 7)",
        "beta_r_spin": ("DERIVED: beta_r = w220(chi1) * M_f / w220(chi_f), w220 = exact Kerr "
                        "(2,2,0) QNM frequency (qnm package; counter-rotating for chi < 0); pp "
                        "side in m1 units, NR side in M units; M_f, chi_f from NRSur3dq8Remnant. "
                        "No free parameter (r0 scales it)."),
        "mismatches": allm,
        "mean": float(np.mean(v)), "median": float(np.median(v)), "max": float(np.max(v)),
    }, indent=2))
    print(f"\nwrote {COEFFS}")


if __name__ == "__main__":
    main()
