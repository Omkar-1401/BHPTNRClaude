"""PARAMETRIC DOMAIN DECOMPOSITION in chi1: separate spin coefficients either side of 0.

Rink et al.'s second strategy, adapted.  They could not fit retrograde and prograde
systems with one global parametric model and built TWO, split at chi1 = 0.  Two
independent lines of evidence say the same is true here:

  * their own finding, that retrograde ringdown modes at chi1 <~ -0.6 defeat a global fit;
  * OUR step-1 measurement (spin_extension_plan.md sec 8): the measured beta spin ratio
    sits at 0.83-1.29 x Islam's prediction on retrograde spin but 1.5-3.7 x on prograde.
    A single quadratic in chi1 is being asked to span two regimes.

and the symptom is the U-shape of the joint fit: medians 1.28e-3 (-0.6) and 3.50e-3
(+0.8) against 4.86e-4 at -0.2.

MODEL.  theta_spin is now 8 long, [neg(4) | pos(4)], and `pick_spin` selects by the sign
of chi1:

    b1(chi1) = b1 + s11 chi1 + s12 chi1^2       (s11, s12) from the chi1 < 0 half
    b2(chi1) = b2 + s21 chi1 + s22 chi1^2       or the chi1 >= 0 half

Both halves reduce to the frozen nonspinning model at chi1 = 0 IDENTICALLY -- every spin
term carries a factor of chi1 -- so the null check is preserved and the model is
CONTINUOUS at the split.  Its DERIVATIVE is not, which is what a domain decomposition
buys and is exactly the freedom the joint fit lacked.  No attempt is made to impose C1.

FITTING.  Each side is fitted on its own sub-grid with the other side held fixed, and
chi1 = 0 is EXCLUDED from both objectives: the nonspinning column is independent of every
spin coefficient, so it contributes an exactly-zero gradient and only dilutes the mean.
That takes the objective from 96 points to 36 (neg) and 48 (pos), so evaluations are
2.0-2.7x faster and the same budget buys correspondingly more of them -- the previous two
runs both stopped on budget rather than converging.

The final table is still scored over ALL 96 including chi1 = 0, for comparability.

DRIVE QUARANTINE stays OFF by default: the A/B of 2026-08-31 (job 3453596) showed
freezing x at merger is 0.97x, i.e. slightly WORSE, on the two retrograde columns and
exactly neutral everywhere else.  The x spikes are cosmetic -- they sit where |h| is
5e-5..3e-4 of peak and carry no weight in an L2 mismatch.

Usage:
    python fit_pn_anchored_spin_split.py --side neg --refit
    python fit_pn_anchored_spin_split.py --side pos --refit
    python fit_pn_anchored_spin_split.py            # score both halves, all 96
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
# UNITS FIX 2026-09-24 (UNITS_AUDIT.md A1/A2) -- this file was MISSED in the first pass.
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
SNAMES8 = [f"{n}-" for n in SNAMES] + [f"{n}+" for n in SNAMES]
CHIS_NEG = [-0.6, -0.4, -0.2]          # chi1 = 0 excluded: zero gradient, see docstring
CHIS_POS = [0.2, 0.4, 0.6, 0.8]
SPLIT = RESULTS / "coeffs_spin_split.json"


def pick_spin(ths, chi1):
    """The 4 spin coefficients in force at this chi1.  Accepts a 4-vector (both sides
    the same, i.e. the old joint model) or an 8-vector [neg | pos]."""
    ths = np.asarray(ths, float)
    if ths.size == 4:
        return ths
    return ths[:4] if chi1 < 0 else ths[4:]
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
    s11, s12, s21, s22 = pick_spin(ths, chi1)
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


def load_split():
    """[neg(4) | pos(4)].  Seeds both halves from the joint fit if no split exists yet."""
    if SPLIT.exists():
        return np.array(json.loads(SPLIT.read_text())["theta_spin8"], float)
    if COEFFS.exists():
        j4 = np.array(json.loads(COEFFS.read_text())["theta_spin"], float)
        print(f"  no split yet -- seeding BOTH halves from the joint fit {np.round(j4,4)}")
        return np.r_[j4, j4]
    print("  no coefficients at all -- seeding both halves from SEED")
    return np.r_[SEED, SEED]


def main():
    import fit_pn_anchored_spin as _S0; _S0.refuse_frozen9_protocol(__file__)   # 2026-09-27
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["neg", "pos"], default=None,
                    help="which half to fit; omit to only SCORE both halves")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--freeze-xc", action="store_true")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--budget", type=float, default=95.0)
    args = ap.parse_args()

    global FREEZE_XC
    FREEZE_XC = args.freeze_xc
    th9 = shipped9()
    print(f"drive: x {'FROZEN at merger' if FREEZE_XC else 'unmodified (A/B says this is better)'}")
    print(f"grid: {len(TRAIN_Q)} q x {len(CHIS)} chi1 = {len(TRAIN_Q)*len(CHIS)} points",
          flush=True)

    t0 = time.time(); cases = {}
    for q in TRAIN_Q:
        for c in CHIS:
            cases[(q, c)] = build_case(q, c)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)

    ths8 = load_split()
    print(f"  start: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(SNAMES8, ths8)),
          flush=True)

    if args.side and args.refit:
        chis = CHIS_NEG if args.side == "neg" else CHIS_POS
        sl = slice(0, 4) if args.side == "neg" else slice(4, 8)
        pts = [(q, c) for q in TRAIN_Q for c in chis]
        print(f"\n===== fitting the chi1 {'< 0' if args.side=='neg' else '>= 0'} half "
              f"on {len(pts)} points (chi1 = {chis}) =====", flush=True)

        class _Budget(Exception):
            pass
        st = {"n": 0, "best": np.inf, "bx": ths8[sl].copy(), "t0": time.time(),
              "budget": args.budget * 60.0}

        def objective(th4):
            full = ths8.copy(); full[sl] = th4
            errs = [full_mm(th9, full, q, c, cases[(q, c)]) for q, c in pts]
            mean, mx = float(np.mean(errs)), float(np.max(errs))
            st["n"] += 1; el = time.time() - st["t0"]; tag = ""
            if mean < st["best"]:
                st["best"] = mean; st["bx"] = np.array(th4, float); tag = "  <-- best"
                out = ths8.copy(); out[sl] = st["bx"]
                PARTIAL.write_text(json.dumps({
                    "status": "in progress", "side": args.side, "n_eval": st["n"],
                    "elapsed_min": round(el / 60.0, 2), "names": SNAMES8,
                    "theta_spin8": [float(x) for x in out], "mean": mean, "max": mx},
                    indent=2))
            print(f"  [{st['n']:4d}] {el/60:6.1f}m  mean={mean:.4e} max={mx:.4e}  "
                  f"best={st['best']:.4e}  th=" +
                  " ".join(f"{n}={v:+.4f}" for n, v in zip(SNAMES, th4)) + tag, flush=True)
            if el > st["budget"]:
                raise _Budget
            return mean

        objective(ths8[sl])
        st["t0"] = time.time(); st["n"] = 0
        try:
            r = minimize(objective, ths8[sl], method="Powell",
                         options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
            # every trial point passed through objective(), so st["bx"] IS the
            # minimum seen -- use it rather than r.x, which Powell can return worse
            ths8[sl] = st["bx"]
            print(f"  opt CONVERGED / {st['n']} evals (best mean={st['best']:.4e}): "
                  f"{r.message}", flush=True)
        except _Budget:
            ths8[sl] = st["bx"]
            print(f"  opt STOPPED ON BUDGET after {args.budget:.0f} min / {st['n']} "
                  f"evals (best mean={st['best']:.4e})", flush=True)

    # ---- score BOTH halves over all 96, chi1 = 0 included, for comparability -----
    print(f"\nfitted theta_spin8:")
    for lab, sl in (("chi1 <  0", slice(0, 4)), ("chi1 >= 0", slice(4, 8))):
        print(f"  {lab}: " + ", ".join(f"{n}={v:+.4f}"
                                       for n, v in zip(SNAMES, ths8[sl])))
    print(f"\n{'q':>8} " + " ".join(f"{c:>+8.1f}" for c in CHIS))
    allm = {}
    for q in TRAIN_Q:
        row = [full_mm(th9, ths8, q, c, cases[(q, c)]) for c in CHIS]
        for c, v in zip(CHIS, row):
            allm[f"{q:.4f}_{c:+.1f}"] = v
        print(f"{q:>8.4f} " + " ".join(f"{v:8.2e}" for v in row))
    v = np.array(list(allm.values()))
    print(f"\nALL 96: median {np.median(v):.4e}  max {np.max(v):.4e}")
    print("  (joint 4-coefficient reference, job 3453596: median 7.8628e-04  max 1.0907e-02)")
    for c in CHIS:
        sub = np.array([allm[f"{q:.4f}_{c:+.1f}"] for q in TRAIN_Q])
        print(f"  chi1={c:+.1f}: median {np.median(sub):.4e}  max {np.max(sub):.4e}")

    SPLIT.write_text(json.dumps({
        "model": "pn_anchored_spin, parametric domain decomposition at chi1 = 0",
        "rationale": "Rink et al. two-model strategy; see module docstring",
        "names": SNAMES8, "theta_spin8": [float(x) for x in ths8],
        "theta9_frozen": [float(x) for x in th9],
        "grid": {"q": TRAIN_Q, "chi1": CHIS,
                 "fit_neg": CHIS_NEG, "fit_pos": CHIS_POS,
                 "chi0_excluded_from_objectives": True},
        "freeze_xc": bool(FREEZE_XC),
        "objective": "waveform mismatch",
        "mismatches": allm,
        "median": float(np.median(v)), "max": float(np.max(v)),
    }, indent=2))
    print(f"\nwrote {SPLIT}")


if __name__ == "__main__":
    main()
