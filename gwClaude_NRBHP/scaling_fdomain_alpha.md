# Frequency-domain alpha diagnostic

**Script:** `diag_alpha_fdomain.py`  
**Date:** 2026-08-13  
**Base model:** `gwr_energy_mult` per-q parameters (`gw_remnant_energy_results/per_q_cache_mult.json`)

## What this measures

Computes `alpha(f) = sqrt(P_NR(f) / P_pp(f)) / a_PP` directly from FFTs of h_NR and
h_ppBHPT (both on the NR time grid via the optimal time map), and asks whether the
shape is smooth enough to be parameterised as a polynomial in f or x = (πMf)^{2/3}.
This is a diagnostic, not a new fitted model: the per-q mult parameters are inputs
taken from the existing cache, not quantities being fitted here.

The question: is alpha smoother as a function of frequency than as a function of time?
Answer: **yes, about 3× smoother**, and the shape extrapolates cleanly to q < 3.

## Base model — gwr_energy_mult

The per-q cache `gw_remnant_energy_results/per_q_cache_mult.json` stores six parameters
per q:

```
params = [a_PP, a_E, b_PP, b_E, t0_nr, phi0]

alpha(t) = a_PP * (1 + a_E * E(t))
beta(t)  = b_PP * (1 + b_E * E(t))

E(t) = gw_remnant Eoft   (cumulative radiated energy, units of M, zero at window start)
tau(t_pp) = t0_nr + integral_{T_ANCHOR}^{t_pp} beta(t') dt'   (time map)
h_model(tau(t_pp)) = alpha(t_pp) * exp(i*phi0) * h_ppBHPT(t_pp)
```

All five of `a_PP, a_E, b_PP, b_E, t0_nr` are fitted **jointly** by minimising the
waveform mismatch against NRHybSur3dq8 at each q independently (Nelder-Mead, 120
random seeds + physical seeds, top-5 polished to fatol=1e-13).  `phi0` is solved
analytically at every function evaluation as `phi0 = arg(Σ h_NR * conj(h_model))` and
is never a free variable.

The five parameters cannot be separated: `b_E` determines the time map, which sets
where in the ppBHPT waveform each NR sample is taken, which changes what amplitude
ratio `a_E` needs to produce.  Pulling on one moves the other.

The cache covers q ∈ [3, 8].  For the held-out q < 3 numbers below, `optimize_case`
was re-run at each q from scratch to obtain the correct per-q params.

## Time map — relationship between t_pp and t_NR

The ppBHPT surrogate and the NR waveform use different time coordinates.

- **t_pp** — ppBHPT coordinate time [M].  The surrogate `BHPTNRSur1dq1e4` is
  evaluated on this axis; t_pp = 0 is defined internally by the surrogate.
- **t_NR** — NR coordinate time [M].  `NRHybSur3dq8` is evaluated on this axis;
  t_NR = 0 is set at the amplitude peak (merger).
- **β(t_pp)** — the time-stretch (dilation) factor evaluated at ppBHPT time.
  In the mult model: `β(t_pp) = b_PP · (1 + b_E · E(t_pp))`, where
  `b_PP` and `b_E` are fitted parameters and `E(t_pp)` is the cumulative
  radiated energy at that moment.
- **T_ANCHOR** — a fixed reference time in the ppBHPT coordinate at which the
  time-map integral is anchored (set to −2750 M in this workspace).
- **t0_nr** — the NR time corresponding to T_ANCHOR; a fitted parameter that
  sets the absolute origin of the map.

The map is defined by the ordinary differential equation

```
dt_NR / dt_pp = β(t_pp)
```

with the boundary condition `t_NR = t0_nr` when `t_pp = T_ANCHOR`.
Integrating gives the explicit map

```
τ(t_pp) = t0_nr + ∫_{T_ANCHOR}^{t_pp} β(t') dt'
```

where `τ(t_pp)` is the NR time that corresponds to ppBHPT time `t_pp`.
Because β ∈ [0.63, 0.90] < 1 across the inspiral, the ppBHPT waveform
evolves faster than the NR waveform: the same physical phase accumulates
over a shorter ppBHPT time interval than over the corresponding NR interval.

To evaluate the model at a given NR time grid point `t_NR`, the map is
inverted numerically (τ is monotone in t_pp, so the inversion is unique):
find `t_pp` such that `τ(t_pp) = t_NR`, then interpolate `h_ppBHPT` and
`α` at that `t_pp` value.

## Critical gotcha — h_22 sign convention

h_22 uses the PN convention `h = A exp(-iΦ)` with `Φ̇ > 0`.  The physical FFT signal
lives at **negative frequencies** (`f = -f_GW < 0`).  The positive-frequency half
contains only Fresnel-leakage sidelobes, roughly 30,000× less power than the physical
signal (Parseval check: two-sided power 27.6 ≈ time-domain power 28.1; positive-f
power 8.6e-4).

Taking the positive half (the natural first guess) gives `alpha_f ≈ 0.28`.  The fix:

```python
neg = f_full < 0
return -f_full[neg][::-1], H_full[neg][::-1]   # flip to positive apparent frequency
```

After fix: `alpha_f` mean = 0.986 ✓.

## Secondary gotcha — np.interp clamping

When the NR time grid extends beyond the ppBHPT coverage after the time map, `np.interp`
clamps out-of-range NR times to the boundary ppBHPT value, creating a spurious
constant-waveform segment in `h_pp_on_nr`.  Fix: restrict to the NR-time range where
the time map is valid before inverting:

```python
valid_tau = (tau >= t_nr[0]) & (tau <= t_nr[-1])
tau_lo, tau_hi = tau[valid_tau][0], tau[valid_tau][-1]
common = (t_nr >= tau_lo) & (t_nr <= tau_hi)
```

## Method

```python
# 1. Build the time map from per-q mult params
beta_t    = b_PP * (1 + b_E * E(t_pp))
tau(t_pp) = t0_nr + integral_{T_ANCHOR}^{t_pp} beta(t') dt'

# 2. Invert tau: for each t_NR in [tau_lo, tau_hi], find t_pp(t_NR)
# 3. Interpolate h_ppBHPT and alpha(t) onto the NR time grid -> h_pp_on_nr, alpha_t_on_nr

# 4. FFT both h_NR and h_pp_on_nr (same time grid -> same frequency axis)
#    Take the negative-frequency half, flip sign -> apparent positive f [cycles/M]

# 5. Cut to inspiral band f ∈ [0.004, 0.025] cycles/M
#    Smooth power spectra P = |H|^2 with a 30-bin boxcar -> P_NR_s, P_pp_s
#    Trim 15-bin edge artefacts from each end of the band

# 6. alpha(f) = sqrt(P_NR_s(f) / P_pp_s(f)) / a_PP
#    x(f)     = (pi * f)^{2/3}    (SPA orbital-frequency coordinate)
```

The 30-bin smooth spans ~30 Fresnel fringe periods (df ~ 2e-4 cycles/M, fringe
period 1/T ~ 2e-4 cycles/M for a ~5000 M window).  It kills the fringes while
leaving enough band to fit polynomials.  The residual ~1e-3 floor in polynomial fits
below is irreducible smoothing noise, not a model accuracy number.

## Results

### In-range q ∈ [3, 8]

Polynomial fit residuals = RMS of `alpha(f)/a_PP - polyval(deg, coord)`:

| q | deg-4 in f | deg-4 in x | alpha range | mean |
|---|---:|---:|:---|---:|
| 3 | 9.21e-4 | 9.26e-4 | [0.970, 1.007] | 0.986 |
| 4 | 9.33e-4 | 9.34e-4 | [0.972, 1.014] | 0.986 |
| 5 | 9.36e-4 | 9.71e-4 | [0.974, 1.015] | 0.986 |
| 6 | 9.98e-4 | 1.05e-3 | [0.975, 1.014] | 0.987 |
| 7 | 1.06e-3 | 1.11e-3 | [0.977, 1.011] | 0.986 |
| 8 | 9.90e-4 | 9.99e-4 | [0.977, 1.005] | 0.985 |

Comparison: deg-4 residuals in x are **~9e-4 here vs ~2.6e-3 time-domain** — ~3×
smoother.  f and x are equally good coordinates at this smoothing level.

### Held-out q < 3 (extrapolation)

These numbers were obtained via a one-off session script, not `diag_alpha_fdomain.py`.
The diagnostic script clamps to the nearest cache key (q=3.0) for any q < 3, giving
the wrong a_PP.  For each held-out q, `G.optimize_case(q, form='mult')` was called
fresh to fit per-q mult params, then the FFT ratio was computed with the same pipeline.
`diag_alpha_fdomain.py` needs a `--params` override or an automatic fallback to
`optimize_case` before it can handle q < 3 correctly.

| q | deg-4 in f | deg-4 in x | alpha range | mean |
|---|---:|---:|:---|---:|
| 2.75 | 8.68e-4 | 8.87e-4 | [0.970, 1.004] | 0.986 |
| 2.50 | 8.36e-4 | 8.74e-4 | [0.969, 1.001] | 0.986 |
| 2.25 | 7.44e-4 | 8.00e-4 | [0.968, 0.996] | 0.986 |
| 2.00 | 7.05e-4 | 7.49e-4 | [0.967, 0.995] | 0.986 |

Three things stand out:

1. **Residuals are smaller at low q, not larger.**  The S-curve is simpler to fit at
   q=2 (7e-4) than at q=8 (1e-3).  This is the opposite of the time-domain picture,
   where E-coupling degeneracy (large a_E) and the P sign flip near q≈4 make low-q
   the hard end.

2. **The shape compresses smoothly.**  The upper end of the S-curve shrinks
   monotonically from 1.015 (q=8) to 0.995 (q=2) as the merger frequency drops.
   No kinks, basin-jumps, or sign flips — the same S-curve family throughout.

3. **Mean is q-universal at 0.986 ± 0.001 across q ∈ [2, 8].**  a_PP mildly
   overestimates the inspiral amplitude because it is fit to the full time-domain window
   (including merger), not the inspiral band alone.

## Physical interpretation

The S-curve `alpha(f)/a_PP ∈ [0.97, 1.015]` is a ~4% drift over the inspiral band.
a_PP already captures all leading q-dependence; what remains is a universal frequency
variation of the ppBHPT/NR amplitude ratio.  The curve rises with f because ppBHPT
amplitude grows slightly faster than NR during late inspiral.  The upper end compresses
at low q because the merger frequency drops, so the band [0.004, 0.025] captures
less of the pre-merger rise.

The f-domain S-curve is smooth because frequency is a better coordinate than time for
the amplitude ratio: the ratio tracks orbital frequency, while time integrates beta
(carrying both amplitude and timing information simultaneously).

### Why the time-domain E-coupling pathologies do not appear here

In the mult model, a_E ≈ −5 to −8 (large and negative) and P = b_PP * b_E crosses
zero near q ≈ 4.  Both are consequences of fitting a time-domain coupling that must
simultaneously absorb amplitude residuals and phase/timing residuals.

In the frequency domain, P_NR(f) / P_pp_on_nr(f) measures amplitude only — the phase
cancels in the power ratio.  The result is the smooth S-curve, implying that the large
a_E and the P sign flip in the time-domain model are at least partly compensating phase
residuals rather than tracking the true amplitude ratio.

## Frequency-domain alpha model — BUILT, 18 regression coefficients

The diagnostic motivates a direct frequency-domain parameterisation of alpha.  The
shape at each q is well described by a low-degree polynomial:

```
alpha(f, q) = a_PP(q) * g(f, q)

where  g(f, q) = 1 + c1(q)*f + c2(q)*f**2 + c3(q)*f**3 + c4(q)*f**4
```

The degree-4 form is motivated directly by the residual table: going from deg-2 to
deg-4 reduces the RMS by ~26% (1.27e-3 → 9.36e-4 at q=5), while deg-3 adds almost
nothing over deg-2 (1.27→1.24e-3) — the S-curve has even-order curvature that deg-3
alone cannot capture.  The ~9e-4 floor after deg-4 is irreducible smoothing noise, not
remaining polynomial structure, so deg-4 is where to stop.

Note on coordinates: f and x = (πf)^{2/3} give the same residual floor (both ~9e-4
at deg-4); the choice is a matter of physical convention.  x is the SPA coordinate and
is more natural for connecting to PN expressions if that is wanted later.  The
coefficient names c1–c4 refer to whichever coordinate is chosen and are defined by:

```
g = sum_{k=1}^{4} c_k(q) * coord**k,    coord = f  or  x = (pi*f)^{2/3}
```

**STATUS CORRECTED 2026-08-18.**  This section previously said the nu-regression was
still to be done and that the per-q coefficients were "not currently saved to disk".
That is out of date: the regression is BUILT and shipped.  `fit_scaling_fdomain_alpha.py`
(2026-08-13) implements it and writes `fdomain_alpha_results/coeffs.json`.

**What is built.**  The regressed model, 18 coefficients:

```
alpha(t, q) = a_PP(nu) * g(z(t), nu)
z(t)        = (clip(x(t), X_LO, X_HI) - X_MID) / X_SCALE          in [-1, 1]
              X_LO=0.078493, X_HI=0.168415, X_MID=0.123454, X_SCALE=0.044961
a_PP(nu)    = 1 + p1*nu + p2*nu^2 + p3*nu^3                        3   [PP-anchored at 1]
ck(nu)      = dk0 + dk1*nu + dk2*nu^2,   k = 0..4                 15
                                                            total  18
```

`z` rather than raw `x`: over the fitted band x ∈ [0.079, 0.168] a raw polynomial in x
has O(1e4) coefficients that partly cancel and blow up outside the band; normalising to
[-1, 1] gives O(1) coefficients that regress stably and are clipped to the trained range.
Training is all 64 q in `per_q_cache_mult.json`; held out 2.75 / 2.5 / 2.25 / 2.0.

**Results — WAVEFORM MISMATCH, not the polynomial residuals tabulated above.**  Do not
conflate the two: the ~9e-04 numbers earlier in this file are RMS residuals of the
*alpha shape* against a polynomial; the numbers here are mismatch against NRHybSur3dq8.

| | in-range median | in-range max | q=2.75 | q=2.5 | q=2.25 | q=2 |
|:---|---:|---:|---:|---:|---:|---:|
| fdomain alpha (18c) | 1.171e-03 | 2.164e-03 | 2.260e-03 | 2.424e-03 | 2.767e-03 | 3.555e-03 |
| fluxanchored (7c) | 4.752e-04 | 7.150e-04 | 6.347e-04 | 5.937e-04 | 6.867e-04 | 1.306e-03 |

(in-range median/max are over the 64-q grid here, over 6 q for fluxanchored — not a
like-for-like grid, but the gap is far larger than that difference.)  Per-q mismatch falls
monotonically with q, 2.164e-03 at q=3 to 5.549e-04 at q=8.

**So the 3x smoother alpha SHAPE has not yet translated into a better waveform model** --
this is 2-3x worse in-band than fluxanchored with 18 coefficients against 7, and worse at
every held-out q < 3.  The most likely reason is the open question below.

**BETA IS NOT TOUCHED BY THIS MODEL.**  `fit_scaling_fdomain_alpha.py` takes
`b_PP, b_E, t0_nr` verbatim from the per-q `mult` cache, so this replaces alpha only and
inherits the unconstrained time-domain beta -- including its fall above q ~ 4 (see
`RESUME_gwremnant.md`, "THE KEY BETA RESULT").  It is therefore not a complete
frequency-domain calibration and is orthogonal to the beta-monotonicity work.

**What remains.**

1. *The `a_PP` consistency question (do this first).*  Whether the `a_PP(q)` fitted by the
   time-domain mismatch optimiser -- which also moves beta -- agrees with the `a_PP`
   implied by the FFT amplitude ratio.  The q-universal 0.986 mean offset is already the
   symptom, and this is the prime suspect for why the in-band numbers above are poor: the
   f-domain shape is being hung on a prefactor fitted under a different criterion.
2. *Global refit:* jointly optimise the 18 regression coefficients against the waveforms,
   as in the `gwr_energy_stiff --global` pass.  Given the seeded fit sits well above the
   per-q floor, this is the case where the joint fit is expected to help (that is the
   documented `stiff` lesson: the joint fit helps only when the form cannot reach the
   floor by regression).
3. *Diagnostic script q < 3 bug:* `diag_alpha_fdomain.py` clamps to the nearest cache key
   (q=3.0) for any q < 3, giving the wrong `a_PP`.  The held-out table earlier in this file
   came from a one-off session script, so it is NOT reproducible from the committed code.
   Needs a `--params` override or an automatic fallback to `G.optimize_case`.
4. *Beta in the f-domain:* the cross-spectrum-phase route sketched below, still
   unimplemented.

**What this is not.**  The freq-domain g(f, q) parameterises alpha only.  Beta still
requires time-domain treatment: it enters through the integrated time map tau(t_pp),
not through the power spectrum.  A complete frequency-domain calibration would therefore
pair freq-domain alpha with the existing time-domain beta (e.g. stiff or anchored).

**Key open question.**  Whether a_PP(q) fitted by the time-domain mismatch optimiser
(which also moves beta) agrees with the a_PP implied by the FFT ratio.  The mean
offset of 0.986 (FFT mean / a_PP) is already one symptom.  This should be checked
before combining the two.

## Beta in the frequency domain

Beta does not enter the power ratio P_NR / P_pp_on_nr (phase cancels).  It imprints
instead on the **cross-spectrum phase**:

```python
cross_phase(f) = angle(H_NR(f) * conj(H_pp_on_nr(f)))
               = phi0  +  (residual from time-map error)
```

If the time map is perfect, cross_phase is flat.  A frequency-dependent slope is a
beta residual — the group delay of h_pp_on_nr at frequency f does not match h_NR's.
This is not currently computed by the diagnostic but is a cheap fourth panel to add.

Extracting a corrected beta from cross_phase is substantially harder than reading alpha
from the amplitude ratio: phase unwrapping is sensitive to noise, and beta's key
quantity (its overall level, shown to match X1^{6/5} to 0.03-0.14%) is already
well-determined from the time-domain fit.

## Outputs

```
diag_alpha_fdomain.py --q {q}  ->  diag_alpha_fdomain_q{q}.png
    panel 1: smoothed |H_NR(f)| and a_PP*|H_pp_on_nr(f)| in inspiral band
    panel 2: alpha(f)/a_PP  [solid blue] and deg-3 poly fit [orange dashed] vs f
    panel 3: alpha(x)/a_PP  [solid blue] and deg-3 poly fit [orange dashed] vs x

diag_alpha_fdomain_multi_q.png  (--q list)
    same two alpha panels, one curve per q (viridis gradient low->high q)
    no poly-fit overlay on the multi-q plot
```
