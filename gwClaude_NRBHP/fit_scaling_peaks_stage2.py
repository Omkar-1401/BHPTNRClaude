"""Stage 2 of the peaks-based BHPT->NR (2,2) calibration model.

Turns the Stage-1 peak point clouds (fit_scaling_peaks.py -> peaks_results/per_q/)
into a q-dependent master model  q -> (alpha(x;nu), beta(x;nu))  that reconstructs
the calibrated waveform on the NR time map.

ARCHITECTURE (see peaks_results/RESUME.md for the full derivation / dead-ends)
-----------------------------------------------------------------------------
Stage A (per q): fit the measured peak clouds
    alpha(x) - 1  and  beta_abs(x) - 1
  as CHEBYSHEV polynomials in a centered/scaled frequency coordinate
    u = (clip(x, XLO, XHI) - XMID) / XHALF ,  x = (M*omega_orb)^(2/3).
  Centering is essential: on the raw x-range [0.05,0.22] a degree-5 monomial fit
  has O(1e4) coefficients with catastrophic cancellation; the centered Chebyshev
  coefficients are O(0.1) and regress smoothly in nu.

Stage B (across q): regress each Chebyshev coefficient in nu = q/(1+q)^2,
  PP-ANCHORED so every correction vanishes in the point-particle limit:
    coeff_m(nu) = sum_{k>=1} g[m,k] * nu^k          (coeff_m(0) = 0  =>  alpha,beta -> 1)

RECONSTRUCTION = TIME MAP (not the per-sample frequency map).
  Evaluate the model alpha,beta ONLY at the BHPT phase peaks (where x is clean and
  monotone), set trel_nr(k) = beta_k * trel_bhpt(k), and build the BHPT->NR time map
  by interpolating THROUGH the peaks in time; interpolate alpha in time as well.
  This avoids the near-merger x-scrambling that caps a per-sample x-map at ~2% at q=2.

TRAINING RANGE.  The [3,8] grid alone extrapolates to q=2 at only ~2-5% (regression
wall on beta), and that number is fragile to the x-clip bound (q=2's long early
inspiral falls below the training x-range).  We therefore train on the cached low-q
anchors as well, [2.25, 8], so the x-normalisation lower bound is REAL training data
(q=2.25) and q=2 becomes a robust one-step extrapolation (~0.55%).  The model is
honestly a LOW-Q peak-ratio model calibrated on [2.25, 8], not a pure [3,8]
extrapolator.  (Recalibrating with q=2 included pushes q=2 to ~0.34%, but then it is
no longer a held-out prediction.)

COVERAGE.  The peaks method is an inspiral method: peaks run from the early inspiral
through the merger (k=0).  Reconstruction covers [~-5000 M, merger] ~ 98% of the NR
(2,2) energy.  The post-merger ringdown (~2% of energy) is NOT modelled here; it is a
separate QNM/remnant branch (planned, see RESUME.md).  Reported mismatches are over
the covered inspiral-through-merger window and are labelled with their coverage.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTNRSurrogate" / "surrogates"))

import fit_scaling_PN_opt_creative as creative
import fit_scaling_peaks as peaks

WAVE = ROOT / ".cache" / "q_dep"
PER_Q = ROOT / "peaks_results" / "per_q"
COEFF_JSON = ROOT / "peaks_results" / "stage2_coeffs.json"

# ---- model hyperparameters (chosen by bake-off; see RESUME.md 2026-07-22) ----
DEG_X_ALPHA = 5      # Chebyshev degree of alpha(x)
DEG_X_BETA = 4       # Chebyshev degree of beta_abs(x)
DEG_NU_ALPHA = 4     # nu-polynomial degree for alpha coefficients
DEG_NU_BETA = 4      # nu-polynomial degree for beta coefficients
TRAIN_Q_LO = 2.25    # low-q anchor: train on [TRAIN_Q_LO, 8]
LOW_Q_ANCHORS = (2.25, 2.5, 2.75)   # cached q's below the [3,8] grid, added to training


# ============================================================ Stage-1 clouds
def load_cases() -> dict[float, dict]:
    """[3,8] point clouds from disk + low-q anchors extracted from the cache."""
    cases = {float(p.stem.replace("peaks_q", "")): dict(np.load(p))
             for p in sorted(PER_Q.glob("peaks_q*.npz"))}
    for q in LOW_Q_ANCHORS:
        if q not in cases:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cases[q] = peaks.extract(q)
    return cases


# ============================================================ x-basis (centered Chebyshev)
def x_norm_range(cases: dict, q_lo: float) -> tuple[float, float]:
    xs = [cases[q]["x"] for q in cases if q >= q_lo - 1e-9]
    xall = np.concatenate(xs)
    return float(xall.min()), float(xall.max())


def u_of(x, xlo, xhi):
    xmid, xhalf = 0.5 * (xlo + xhi), 0.5 * (xhi - xlo)
    return (np.clip(x, xlo, xhi) - xmid) / xhalf


def cheb_fit(u, y, deg):
    A = np.polynomial.chebyshev.chebvander(u, deg)
    c, *_ = np.linalg.lstsq(A, y - 1.0, rcond=None)
    return c


def cheb_eval(c, u):
    return 1.0 + np.polynomial.chebyshev.chebval(u, c)


# ============================================================ Stage A + Stage B fit
def fit_master(cases: dict) -> dict:
    xlo, xhi = x_norm_range(cases, TRAIN_Q_LO)
    qs = sorted(q for q in cases if q >= TRAIN_Q_LO - 1e-9)
    nus = np.array([q / (1.0 + q) ** 2 for q in qs])

    # Stage A: per-q Chebyshev coefficients
    A = np.array([cheb_fit(u_of(cases[q]["x"], xlo, xhi), cases[q]["alpha"], DEG_X_ALPHA)
                  for q in qs])
    B = np.array([cheb_fit(u_of(cases[q]["x"], xlo, xhi), cases[q]["beta_abs"], DEG_X_BETA)
                  for q in qs])

    # Stage B: regress each column in nu, PP-anchored (coeff = sum_{k>=1} g_k nu^k)
    def regress(C, dnu):
        Anu = np.vstack([nus ** k for k in range(1, dnu + 1)]).T
        return np.array([np.linalg.lstsq(Anu, C[:, m], rcond=None)[0]
                         for m in range(C.shape[1])])

    GA = regress(A, DEG_NU_ALPHA)   # (DEG_X_ALPHA+1, DEG_NU_ALPHA)
    GB = regress(B, DEG_NU_BETA)    # (DEG_X_BETA+1, DEG_NU_BETA)
    return {"xlo": xlo, "xhi": xhi, "GA": GA, "GB": GB,
            "deg_x_alpha": DEG_X_ALPHA, "deg_x_beta": DEG_X_BETA,
            "deg_nu_alpha": DEG_NU_ALPHA, "deg_nu_beta": DEG_NU_BETA,
            "train_q_lo": TRAIN_Q_LO, "train_qs": qs}


def eval_coeffs(G, nu):
    """evaluate the x-Chebyshev coefficients at a given nu (PP-anchored nu-poly)."""
    return np.array([sum(G[m, k] * nu ** (k + 1) for k in range(G.shape[1]))
                     for m in range(G.shape[0])])


def alpha_beta_at(model, nu, x):
    """model alpha(x;nu), beta(x;nu) at frequency coord x for one mass ratio nu."""
    u = u_of(x, model["xlo"], model["xhi"])
    a = cheb_eval(eval_coeffs(model["GA"], nu), u)
    b = cheb_eval(eval_coeffs(model["GB"], nu), u)
    return a, b


# ============================================================ time-map reconstruction
def phase_peaks_full(t, h):
    """BHPT phase-extrema peaks incl. merger (k=0); returns (t_merger, t_pk, omega_pk)."""
    amp = np.abs(h)
    t_m = float(t[int(np.argmax(amp))])
    psi = np.unwrap(np.angle(h))
    sign = 1.0 if psi[-1] > psi[0] else -1.0
    psim = sign * psi
    psi_m = float(np.interp(t_m, t, psim))
    n = int((psi_m - psim[0]) / np.pi)
    tgt = psi_m - np.pi * np.arange(0, n + 1)
    tgt = tgt[tgt >= psim[0]]
    t_pk = np.interp(tgt, psim, t)
    om = np.abs(np.gradient(psi, t))
    return t_m, t_pk, np.interp(t_pk, t, om)


def build_time_map(model, q, t_bhpt, h_bhpt):
    """Return (trel_bhpt_nodes, trel_nr_nodes, alpha_nodes) defining the NR time map.

    Merger-relative (t=0 at each merger). Nodes are the BHPT phase peaks; the map is
    strictly increasing so np.interp gives a monotone BHPT->NR time transform.
    """
    nu = q / (1.0 + q) ** 2
    tm_b, tpk_b, om_pk = phase_peaks_full(t_bhpt, h_bhpt)
    trel_pk = tpk_b - tm_b               # <= 0, k=0 at merger
    x_pk = (om_pk / 2.0) ** (2.0 / 3.0)
    a_pk, b_pk = alpha_beta_at(model, nu, x_pk)
    trel_nr_pk = b_pk * trel_pk          # absolute-ratio time map
    o = np.argsort(trel_pk)
    tb, tn, a = trel_pk[o], trel_nr_pk[o], a_pk[o]
    keep = np.r_[True, np.diff(tn) > 0]  # enforce strictly increasing NR time
    return tb[keep], tn[keep], a[keep]


def reconstruct(model, q, t_bhpt, h_bhpt, t_nr_merger, phi0=0.0):
    """Reconstruct the calibrated (2,2) waveform on the BHPT sample grid.

    Returns (tau_abs, h_scaled, use) where tau_abs is NR-aligned time, h_scaled the
    calibrated waveform, and `use` the monotone-and-in-window mask.
    """
    tb, tn, a = build_time_map(model, q, t_bhpt, h_bhpt)
    m_b = int(np.argmax(np.abs(h_bhpt)))
    trel = t_bhpt - t_bhpt[m_b]
    tau = np.interp(trel, tb, tn)
    a_t = np.interp(trel, tb, a)
    tau_abs = tau + t_nr_merger
    h_scaled = a_t * np.exp(1j * phi0) * h_bhpt
    mono = np.r_[True, np.diff(tau_abs) > 0]
    return tau_abs, h_scaled, mono


# ============================================================ validation (needs NR)
def load_waveforms(q):
    d = np.load(WAVE / f"waveforms_q{q:.10f}.npz")
    return (d["t_bhpt"], d["h_bhpt_re"] + 1j * d["h_bhpt_im"],
            d["t_nr"], d["h_nr_re"] + 1j * d["h_nr_im"])


def mismatch_vs_nr(model, q):
    """Full recon + 1-D phi0 refinement against NR. Returns (mathcalE, coverage)."""
    t_b, h_b, t_n, h_n = load_waveforms(q)
    m_b = int(np.argmax(np.abs(h_b)))
    m_n = int(np.argmax(np.abs(h_n)))
    tb, tn, a = build_time_map(model, q, t_b, h_b)
    trel = t_b - t_b[m_b]
    tau_abs = np.interp(trel, tb, tn) + t_n[m_n]
    a_t = np.interp(trel, tb, a)
    mono = np.r_[True, np.diff(tau_abs) > 0]
    use = (tau_abs >= t_n[0]) & (tau_abs <= t_n[-1]) & mono
    phi0_guess = np.unwrap(np.angle(h_n))[m_n] - np.unwrap(np.angle(h_b))[m_b]

    def err(d):
        hs = a_t * np.exp(1j * (phi0_guess + d)) * h_b
        hi = creative.interp_complex(tau_abs[use], hs[use], t_n)
        cov = (t_n >= tau_abs[use][0]) & (t_n <= tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov], hi[cov]), np.mean(cov)

    r = minimize_scalar(lambda d: err(d)[0], bounds=(-np.pi, np.pi), method="bounded")
    e, cov = err(r.x)
    return float(e), float(cov)


# ============================================================ persistence
def save_coeffs(model) -> None:
    COEFF_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in model.items()}
    COEFF_JSON.write_text(json.dumps(out, indent=2))


def load_coeffs() -> dict:
    d = json.loads(COEFF_JSON.read_text())
    d["GA"] = np.array(d["GA"])
    d["GB"] = np.array(d["GB"])
    return d


# ============================================================ main
def main() -> None:
    print("loading Stage-1 peak clouds + low-q anchors ...")
    cases = load_cases()
    print(f"  {len(cases)} q cases; training on q >= {TRAIN_Q_LO} "
          f"({sum(q >= TRAIN_Q_LO - 1e-9 for q in cases)} cases)")

    model = fit_master(cases)
    print(f"  x-norm range [{model['xlo']:.3f}, {model['xhi']:.3f}]")
    print(f"  alpha: x-deg {DEG_X_ALPHA}, nu-deg {DEG_NU_ALPHA}  "
          f"({model['GA'].size} coeffs)")
    print(f"  beta : x-deg {DEG_X_BETA}, nu-deg {DEG_NU_BETA}  "
          f"({model['GB'].size} coeffs)")

    save_coeffs(model)
    print(f"  saved coefficients -> {COEFF_JSON.relative_to(ROOT)}")

    print("\nvalidation (mathcalE vs NRHybSur3dq8, inspiral-through-merger window):")
    print(f"{'q':>6} {'mathcalE':>11} {'%':>8} {'coverage':>9}   note")
    in_range = []
    for q in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e, cov = mismatch_vs_nr(model, q)
        in_range.append(e)
        print(f"{q:6.2f} {e:11.4e} {100*e:8.4f} {cov:9.3f}")
    print(f"  in-range [3,8] median {np.median(in_range):.3e}, "
          f"max {np.max(in_range):.3e}")

    print()
    for q in [2.75, 2.5, 2.25, 2.0, 1.75, 1.5]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e, cov = mismatch_vs_nr(model, q)
        tag = "anchor" if q in LOW_Q_ANCHORS else "extrap"
        gate = "  <-- q=2 GATE" if q == 2.0 else ""
        flag = "" if e < 0.01 else "  (>1%)"
        print(f"{q:6.2f} {e:11.4e} {100*e:8.4f} {cov:9.3f}   {tag}{gate}{flag}")


if __name__ == "__main__":
    main()
