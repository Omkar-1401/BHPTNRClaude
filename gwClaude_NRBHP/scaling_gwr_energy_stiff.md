# gw_remnant energy model, stiffened nu-structure (gwr_energy_stiff)

Switchless, energy-driven, **no PN expressions anywhere**, **9 coefficients**, **globally jointly fitted** on q in [3, 8] only.

## Model

```python
E(t) = gw_remnant Eoft          # radiated energy, units of M, E(t_start) = 0
nu   = q/(1+q)**2               X2 = 1/(1+q)

alpha(t) = alpha_PP(nu) * (1 + alpha_E(nu) * E(t))
beta (t) = beta_PP(X2)  +  P(nu) * E(t)
tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'
h_model(tau) = alpha * exp(i*phi0) * h_BHPT       # phi0 analytic

# --- the four q-dependent functions: 9 free coefficients in total ----------
alpha_PP(nu) = 1 + a1*nu + a2*nu**2 + a3*nu**3   # 3 free.  Leading 1 is the PP
                                                 # anchor: pinned, not fitted.

alpha_E(nu)  = A(nu) / nu                        # 2 free.  The 1/nu is IMPOSED by
A(nu)        = A0 + A1*nu                        # post-adiabatic counting, not fitted.

beta_PP(X2)  = 1 + b1*X2 + b2*X2**2              # 2 free.  Leading 1 is the PP anchor.
                                                 # X2 rather than nu because a
                                                 # polynomial in X2 IS the Taylor
                                                 # series of X1^(6/5) = (1-X2)^(6/5).

P(nu)        = P0 + P1*nu                        # 2 free.  P = beta_PP * beta_E, i.e.
                                                 # dbeta/dE -- the anchor-independent
                                                 # combination that enters beta linearly.

# shipped values (gwr_energy_stiff_results/coeffs.json)
a1, a2, a3 = -1.143934,   1.565901, -15.772983
A0, A1     = -0.887504,  -1.153702
b1, b2     = -1.186040,   0.095983
P0, P1     = -4.662603,  28.957652
```

**What is imposed and what is fitted.**  Imposed: the two PP anchors (`alpha_PP`,
`beta_PP` -> 1 in the test-mass limit), the `1/nu` in `alpha_E`, the choice of
coordinate for each function, and the degrees.  Fitted: the nine numbers above --
first per-q followed by independent regression (to *measure* the structure and seed
the optimiser), then all nine jointly against the waveforms in the `--global` pass.
Each function is linear in its own coefficients, so the seeded stage is plain
least-squares; only the joint refit is nonlinear.

**X1^(6/5) appears nowhere in the model.**  It is a *measured outcome* -- `beta_PP`
comes out equal to it to 0.03% without ever being told to -- not an input.  That is
what makes it independent confirmation of the note's anchor rather than an import of
it, and it is the difference between this model and `pn_anchored`, which imposes
X1^(6/5) analytically.

At evaluation `params_at` clips `alpha_PP` to [0.05, 2.5] and `beta_PP` to [0.2, 1.6]
as a guard against an extrapolation producing a non-monotonic time map.  Neither bound
is ever active in practice: over q in [2, 8] the two run 0.650-0.887 and 0.615-0.869.

The PP limit is carried by the coordinate, not imposed on the couplings: E_rad -> 0 as nu -> 0 (measured E_tot ~ nu^2.31), so alpha -> alpha_PP -> 1 and beta -> beta_PP -> 1 with X2 -> 0 being the same limit as nu -> 0.

## Is this model global?

**Yes.**  The shipped coefficients come from a **global joint fit**: all 9 are optimised simultaneously (Powell) against 12 waveforms sampled even in nu across [3, 8], with phi0 solved analytically at every evaluation and t0_nr the only per-q nuisance (a 1-D bounded search, warm-started between iterations).

A per-q fit of the same forms, followed by independent regression of each function, is used only to (a) *measure* the nu-structure that motivates the forms below and (b) seed the optimiser.  It is reported as the `seeded` column for reference.  No q < 3 data enters anywhere.

Worth noting *why* the joint fit earns its place here.  A global fit only recovers something when the parameterisation is rigid enough that independent per-coefficient regression cannot already reach the per-q floor.  With these stiff forms it cannot: at q=3 the seeded fit gives 1.10e-03 against a per-q floor of 9.6e-4, and the joint fit recovers that (9.74e-04) while simultaneously improving every q < 3 point (q=2: 7.10e-03 -> 3.78e-03).  Had the forms been flexible enough to sit at the floor already, the joint fit would have had no slack and would have changed nothing.

## nu-structure and why each degree is what it is

| function | form | coeffs |
|:---|:---|---:|
| `alpha_PP(nu)` | `1 + a1 nu + a2 nu^2 + a3 nu^3` | 3 |
| `alpha_E(nu)` | `A(nu)/nu`,  `A = A0 + A1 nu` | 2 |
| `beta_PP(X2)` | `1 + b1 X2 + b2 X2^2` | 2 |
| `P(nu)` | `P0 + P1 nu` | 2 |

These are not fit-quality choices.  Each coordinate and degree follows from what the two rescalings physically are; the measured residuals below are *confirmations*, not the reasons.

### The shared base: both prefactors carry the Newtonian chirp factor X1^(6/5)

ppBHPT evolves a test mass on a fixed background of the PRIMARY's mass m1, and its time and strain are in units of m1.  NR works in units of the total mass M = m1 + m2.  Converting a Newtonian chirp between those two mass units introduces X1 = m1/M, and the quadrupole/chirp scaling puts it at the 6/5 power.  So both prefactors should carry X1^(6/5).  They do, and beta_PP is *exactly* it:

| | measured / X1^(6/5) |
|:---|:---|
| `beta_PP`  | 1.0019 +- 0.0014 across [3, 8] (spread **0.14%**) |
| `alpha_PP` | 1.0247 .. 1.0401, with (ratio - 1)/nu = 0.214 .. 0.250 |

beta_PP is the Newtonian chirp scaling to a tenth of a percent, with nothing fitted.  On the four held-out mass ratios below q=3 -- never used anywhere -- X1^(6/5) alone predicts the per-q truth to -0.19%, -0.22%, -0.30%, -0.46% at q = 2.75, 2.5, 2.25, 2.0.  alpha_PP carries the same base times (1 + ~0.24*nu), i.e. the chirp factor plus a first-order finite-mass amplitude correction.

This is the same X1^(6/5) that `pn_anchored` *imposes* analytically.  Here it was not imposed: it fell out of an energy-driven fit that knows nothing about PN, which is an independent confirmation of that anchor rather than an import of it.

### Why X2 = 1/(1+q), and why degree 2

Since X1 = 1 - X2, a polynomial in X2 is precisely the Taylor expansion of X1^(6/5) = (1 - X2)^(6/5) about the test-mass limit X2 -> 0.  The expansion is `1 - (6/5) X2 + (6/5)(1/5)/2 X2^2 - ...`, and the fit returns:

```
fitted:  1 - 1.18604*X2 + 0.09598*X2^2
Taylor:  1 - 1.20000*X2 + 0.12000*X2^2
```

So X2 is the right coordinate because the physical scaling is a power of (1 - X2), and **degree 2 is second order in that expansion** -- not a residual argument.  It also explains why nu and 1/q are poor coordinates for beta_PP: neither is the variable the physical form is a power of.

Getting beta_PP right matters more than anything else here, because it multiplies the *whole integrated time map*: a constant relative error eps accumulates as eps x elapsed time over the ~3e4 M window, which t0_nr cannot absorb.  The sensitivity scan at q=2 puts the tolerance at **+-0.5%**.

### Why alpha_E goes as 1/nu

This is fixed by post-adiabatic counting.  ppBHPT is an adiabatic (0PA) calculation, linear in the mass ratio, so its *fractional* amplitude error is first post-adiabatic -- O(nu).  The correction this model writes is alpha_E * E(t), and E is radiated energy accumulated over a window fixed in TIME, so E ~ nu^2 (flux ~ nu^2, window length fixed; measured nu^2.31, the excess coming from the start-frequency drift).  For the product to be the required O(nu), the coupling must carry nu^-1:

```
alpha_E * E  ~  (1/nu) * nu^2  =  nu        <- 1PA, as required
```

So the 1/nu is dictated, not fitted, and `A = A0 + A1*nu` then carries the next post-adiabatic order.  Degree 1 means "keep 1PA and 2PA, stop".  The measured flatness of alpha_E*nu (-1.092 .. -1.165, 6.3% across [3, 8]) is the confirmed prediction of this counting.  It also explains the failure mode of the unstructured version: a polynomial in nu cannot represent nu^-1 at all, and fitting alpha_E as a free cubic mis-predicts alpha_E(q=2) by -57%.

The same counting is why **nu is the polynomial variable** for the couplings and for alpha_PP: nu is the post-adiabatic expansion parameter, so degree n means "through nPA".  A polynomial in q or 1/q has no such reading.

### Where the physics runs out: P, and alpha_PP's degree

> **CAVEAT added 2026-08-03 — this subsection over-reads a gauge-dependent quantity.**
> The *drift* of `beta(t)` is not physical: the exact phase-matching map is
> `tau(t) = phi_NR^{-1}(phi_pp(t) + dphi)` with `dphi` a free constant (the model's own
> analytic `phi0`), so "the empirical dbeta/dt" is a one-parameter family whose sign flips
> under a ~2 rad shift of `dphi`.  The shipped model's phase residual is 0.017-0.046 rad
> RMS over ~50 cycles, so its time map is right to ~1 part in 1e4 and there is no defect
> to explain.  Consequently the Bondi comparison below compares quantities in different
> gauges and the 1.1% agreement at q=3 may be coincidental; "something else dominates at
> larger q" should not be read as physics.  What IS gauge-robust is the LEVEL of beta,
> which matches `X1^(6/5)` to 0.03-0.04%.  See `RESUME_gwremnant.md`
> §"beta's LEVEL is physical; its DRIFT is gauge".

**`P` is the weakest-motivated form here, and this should be stated plainly.**  Its leading piece *is* physical and parameter-free: the clock tracks the Bondi mass, so with beta ~ (m1/M(t))^(6/5) and M(t) = M - E(t), expanding gives P = (6/5) * beta_PP.  At q=3 that predicts 0.851 against a measured 0.861 -- 1.1%, with nothing fitted.  But it fails immediately above: P crosses zero at q ~ 4.1 and P/[(6/5) beta_PP] runs 1.011, 0.125, -0.385, -0.818, -1.22 at q = 3, 3.77, 4.41, 5.18, 8.  Mass loss therefore accounts for the clock drift only at the bottom of the training range, and something else -- for which this model has no physical account -- dominates at larger q.  Linear-in-nu is a 1PA-order statement and nothing more.  Consistently, P is also the form whose extrapolation is worst (+17.8% at q=2), and it is the residual limit on q<3.

**`alpha_PP` at degree 3 is likewise unstructured.**  Given that alpha_PP is measurably X1^(6/5) * (1 + ~0.24 nu), the physical form is `X1^(6/5) * (1 + (c0 + c1 nu) nu)` -- 2 coefficients on a derived base, rather than 3 empirical ones on no base.  That refit has not been done here; the cubic in nu is an unstructured 3PA expansion that absorbs the same behaviour.  It is the clearest remaining improvement to this model.

### Two degree choices the measurements do NOT support (2026-08-03, open)

Both of the following are **regression-level** comparisons against the per-q values --
they change what the seeded stage produces, and a `--global` refit is needed before
either is adopted, since the joint fit partly compensates a poor form (it already pulls
the shipped `alpha_E` error at q=2 from -19.4% down to +14.4%).

**1. `alpha_E`: degree 1 is the worst of its neighbours.**  The `1/nu` factor is
genuinely dictated by post-adiabatic counting, but "degree 1 = keep 1PA and 2PA, stop"
labels the terms without justifying the truncation.  Measured, `A = alpha_E * nu` is
**non-monotonic** across the training range -- -1.1104 at q=3, a minimum of -1.1640 near
q=4, back to -1.0918 near q=6.8, -1.1062 at q=8 -- and the held-out truth keeps rising
below q=3 (-1.0851, -1.0582, -1.0289, -0.9998 at q = 2.75, 2.5, 2.25, 2).  A straight
line through a U-shaped curve tilts the wrong way:

| degree of `A` | coeffs | in-range max resid | RMS | alpha_E error at q=2.75 / 2.5 / 2.25 / 2 |
|---:|---:|---:|---:|:---|
| 0 | 1 | 3.35% | 2.22% | +3.7% / +6.4% / +9.4% / +12.6% |
| **1 (current)** | **2** | **5.07%** | **1.40%** | **+8.1% / +11.5% / +15.3% / +19.4%** |
| 2 | 3 | 2.45% | 1.03% | +3.9% / +5.3% / +6.7% / **+7.9%** |
| 3 | 4 | 0.85% | 0.32% | -3.4% / -8.2% / -16.2% / -28.2% |

Degree 2 dominates degree 1 on **both** axes, and even a constant extrapolates better.
Options: degree 0 (9 -> 8 coefficients, cleanest 1PA claim, gives up in-range RMS) or
degree 2 (9 -> 10, best of both, costs a coefficient).  Degree 3's superb in-range fit
with a -28% blow-up at q=2 is the signature of fitting scatter -- `alpha_PP` and
`alpha_E` are correlated in the per-q solutions, so part of the ~6% wiggle in `A` may be
degeneracy scatter rather than physics.

**2. `beta_PP`: the 2 coefficients buy in-range accuracy, not q=2.**  The X2 *coordinate*
is strongly justified (at degree 1 it gives 0.196% in-range residual against 3.375% for
nu and 3.780% for 1/q, and it wins at every degree, in-range and extrapolated).  The
*degree* is weaker.  Imposing `beta_PP = X1^(6/5)` exactly, with zero coefficients:

| | coeffs | in-range max | q=2.75 | q=2.5 | q=2.25 | q=2 |
|:---|---:|---:|---:|---:|---:|---:|
| fitted deg-2 in X2 | 2 | 0.079% | -0.03% | -0.10% | -0.23% | **-0.46%** |
| impose X1^(6/5) | **0** | 0.213% | -0.19% | -0.22% | -0.30% | **-0.46%** |

They are **identical at q=2** and both inside the +-0.5% tolerance.  So the two
coefficients buy nothing at the mass ratio that matters most; what they buy is in-range
accuracy, and that is not negligible -- 0.213% of a constant relative error in beta
accumulates to ~60 M of drift over the ~3e4 M window, which t0_nr cannot absorb.  Going
to zero coefficients (9 -> 7) is therefore a real trade, not a free win, and it would
also remove one side of the `beta_PP`/`P` degenerate pair -- which may clean that up or
may simply push the problem into `P`.  Only a joint refit will say.

### Which coefficient orders are physically motivated -- and the test that decides (2026-08-03)

The `1/nu` in `alpha_E` is presented above as dictated by post-adiabatic counting.  **That
justification does not survive measurement and should be restated as a normalisation
convention.**  What follows is the corrected account.

**The chain rule separates the physical from the conventional.**

```
alpha_E = (d ln alpha / dx) * (dx / dE)
```

The first factor is genuinely O(nu) by post-adiabatic counting.  The second depends on E
-- and **E is not a physical energy**: it is the ppBHPT's own radiated energy, computed
from the ppBHPT (2,2)+(2,-2) in the surrogate's normalisation (which carries its own
`norm = 1/q`).  So the nu-power of `alpha_E` is inherited from that convention.  Only the
product `alpha_E * E` is physical, and no nu-power for `alpha_E` alone is derivable --
including `1/nu`.

**Measured scalings** (over [3,8]): `alpha_E ~ nu^-0.903`, `E_tot ~ nu^2.449`, so the
product goes as `nu^1.55`, not `nu^1`.  The original counting -- "E ~ nu^2, require the
product to be O(nu), therefore alpha_E ~ nu^-1" -- would demand `nu^-1.449` given the
measured E-scaling.  The `1/nu` is empirically close (-0.903) but the stated derivation
is not what makes it work.

**The physical combination, measured directly.**  `|d ln alpha|`, the total fractional
amplitude change across the window, goes as `nu^1.635`:

| q | 3 | 4 | 5.4 | 6.8 | 8 |
|:---|---:|---:|---:|---:|---:|
| \|d ln alpha\| | 0.2677 | 0.1976 | 0.1392 | 0.1088 | 0.0965 |

The window is not the culprit: it covers a nearly nu-independent frequency range
(`dx ~ nu^0.226`, only 16% variation across [3,8], x_start 0.065 -> 0.074, x_end 0.392 ->
0.357), so 1PA alone would predict `nu^1.0`.  Fitting `d ln alpha = c1 nu + c2 nu^2` gives
`c1 = 0.3236`, `c2 = 5.7752`, i.e. a **2PA term 1.78 / 2.50 / 3.39 times LARGER than the
1PA term** at nu = 0.10 / 0.14 / 0.19 -- and still 8.5% residual.  **The post-adiabatic
series for this quantity is not in its asymptotic regime at q = 3-8.**  "Keep through 1PA,
stop" has no content when the next term is three times the one retained.

**The test.**  Compute the 2PA/1PA term ratio at the training nu.  If << 1 the order
counting is meaningful and truncation is justified; if >~ 1 it is not:

| quantity | 2PA/1PA | orders meaningful? |
|:---|---:|:---|
| `alpha_PP` residual over X1^(6/5) | ~0.15 | **yes** -- `(ratio-1)/nu` flat to 15% |
| `beta_PP` residual over X1^(6/5) | << 0.15 | **yes** -- residual is 0.19% in total |
| `d ln alpha` (the `alpha_E` sector) | **1.8 - 3.4** | **no** |
| `P` | changes sign in range | **no** |

So **3 of the proposed 6 coefficients stand on derivable physics**: the `X1^(6/5)` bases of
`alpha_PP` and `beta_PP` (Newtonian chirp rate + mass-unit conversion, exact at leading
order, zero fitted coefficients) plus the 1PA coefficient of each residual.  `alpha_E`'s
single coefficient and `P`'s two are empirical.  The only route to more is to couple to `x`
rather than `E`, where the leading coefficient is a derivable `55/42` -- which is
`pn_anchored`, and costs the PN-free property.

### Three negative results -- do not re-run these

1. **E-normalisation does not explain `A`'s non-monotonicity.**  Rescaling `E -> E/nu^p` is
   exactly equivalent to regressing `A_p = alpha_E * nu^p`.  Scanned p = 0.903, 1.0, 1.31,
   1.449: **`A_p` is non-monotonic at every power.**  The U-shape survives any monomial
   rescaling, so it is not a fractional-power artifact.  (p=1 is near-best in-range at
   3.35%; p=1.31 is better at q=2, -3.90% vs +12.60%, but far worse in-range at 11.67%.)

2. **Fitting the invariant `Q = alpha_PP * alpha_E = dalpha/dE` is much worse**, despite
   being the exact analogue of what the model already does on the beta side with
   `P = dbeta/dE`.  Degree 1 gives 5.98% in-range and **-29.2%** at q=2, against `A`'s
   3.35% / +12.60%.  The beta-side trick does not transfer.

3. **Pinning `alpha_PP` to its structured form does not de-degenerate `alpha_E`** (A'
   spread 6.36% vs A's 6.26%, same non-monotonicity, same extrapolation).  This is the
   informative one: `alpha_PP` and `alpha_E` are **-0.974 correlated** (deg-1-detrended
   residuals, n=64), with lag-1 autocorrelation **+0.974** -- smooth, not optimiser noise.
   But `alpha_PP`'s residual excursion is **0.17%** while `alpha_E`'s is **6.3%**.  The
   valley is extremely elongated and `alpha_E` is the soft direction, so pinning `alpha_PP`
   to 0.17% cannot fix a wiggle that corresponds to a sub-0.17% move in `alpha_PP`.

**Consequence:** `alpha_E`'s nu-structure beyond the leading constant is not a physical
quantity -- it is the shape of the floor of a soft valley, weakly determined by the data.
That is why adding orders makes extrapolation *worse*, and it converts degree 0 for
`alpha_E` from an empirical preference into a principled choice: higher orders are neither
identifiable nor meaningful.

`P`'s residuals are also -0.495 correlated with `beta_PP`'s (the documented degenerate
pair), and `corr(A, P) = -0.134`.  `A`'s quadratic vertex sits at q ~ 3.98 and `P`'s zero
crossing at q ~ 3.91 -- suggestive, but the weak `corr(A,P)` argues against a shared cause.

## Coefficients

- `alpha_PP` (degree 3 in `nu`, anchored at 1): [  1.        -1.143934   1.565901 -15.772983]
- `A` (degree 1 in `nu`, free): [-0.887504 -1.153702]
- `beta_PP` (degree 2 in `X2`, anchored at 1): [ 1.       -1.18604   0.095983]
- `P` (degree 1 in `nu`, free): [-4.662603 28.957652]

## Results

`seeded` = the same 9 forms fitted per-q then regressed independently; `global` = all 9 optimised jointly.  `per-q floor` is what the model *form* reaches when fitted at that single q alone -- the target a perfect nu-parameterisation would hit.

| q | seeded | global | per-q floor |
|---:|---:|---:|---:|
| 3 | 1.0953e-03 | **9.7359e-04** | 9.5679e-04 |
| 4 | **9.2649e-04** | 9.3518e-04 | 9.2614e-04 |
| 5 | **6.9700e-04** | 7.0098e-04 | 6.9648e-04 |
| 6 | **5.4570e-04** | 5.4634e-04 | 5.3977e-04 |
| 7 | 4.3363e-04 | **4.2661e-04** | 4.0610e-04 |
| 8 | **3.3374e-04** | 3.5418e-04 | 2.4581e-04 |
| **in-range median** | **6.2135e-04** | 6.2366e-04 | 6.1813e-04 |
| **in-range max** | 1.0953e-03 | **9.7359e-04** | 9.5679e-04 |

| q < 3 (held out) | seeded | global | per-q floor | BHPT input |
|---:|---:|---:|---:|:---|
| 2.75 | 1.2768e-03 | **9.8291e-04** | 9.1084e-04 | in domain |
| 2.5 | 1.7910e-03 | **1.1240e-03** | 8.5641e-04 | in domain |
| 2.25 | 3.2110e-03 | **1.7272e-03** | 8.1920e-04 | **extrapolated** |
| 2 | 7.1049e-03 | **3.7850e-03** | 8.4786e-04 | **extrapolated** |

q = 2.5 is the honest low-q gate: `BHPTNRSur1dq1e4` declares validity only for q >= 2.5 (`X_min = log10(2.5)`) and merely warns outside it, so at q = 2.25 and 2.0 the BHPT *input* is itself an extrapolation of the surrogate's splines in log q, on top of the coefficient extrapolation.

### Held-out low-q accuracy of each nu-form

Predicted vs the per-q truth at mass ratios never used in the fit:

| q | beta_PP | P | alpha_E |
|---:|---:|---:|---:|
| 2.75 | 0.69055 / 0.69051 (+0.01%) | 1.0002 / 1.0829 (-7.6%) | -5.6921 / -5.549 (+2.6%) |
| 2.5 | 0.66897 / 0.66930 (-0.05%) | 1.2471 / 1.2719 (-2.0%) | -5.5025 / -5.185 (+6.1%) |
| 2.25 | 0.64415 / 0.64516 (-0.16%) | 1.5059 / 1.4262 (+5.6%) | -5.3200 / -4.830 (+10.1%) |
| 2 | 0.61532 / 0.61756 (-0.36%) | 1.7724 / 1.5044 (+17.8%) | -5.1475 / -4.499 (+14.4%) |

beta_PP lands inside its +-0.5% tolerance at every held-out point, which is what the whole parameterisation was built to achieve.

## Comparison

| model | in-range median | q=2.5 | q=2 | PN used? |
|:---|---:|---:|---:|:---|
| **gwr_energy_stiff** | **6.24e-04** | **1.12e-03** | **3.78e-03** | **no** |
| pn_anchored | 6.1e-4 | 9.0e-4 | 2.55e-3 | yes (X1^6/5, 55/42, QNM) |
| PN_opt_remnant_partial | 7.48e-5 | — | 3.7e-3 | yes (+ surfinBH chi_f) |

Command: `python fit_scaling_gwr_energy_stiff.py --global --maxiter 80`

Plots: `python NRBHP_gwr_energy_stiff_plots.py` -> `Agentic_plots/gwr_energy_stiff/`
(q = 2, 3, 5, 8; waveform / zoom / stacked alpha-beta-E panels, master coefficients vs q,
alpha-beta overlay, E(t) across q).  Same figure set and styling as
`NRBHP_gw_remnant_energy_plots.py`, so the two models can be compared panel by panel.
The figures are generated with `fit_scaling_gw_remnant_energy.evaluate_model`, the same
evaluator behind the table above: it reproduces 9.7360e-04 / 7.0099e-04 / 3.5419e-04 at
q = 3 / 5 / 8 and 3.7850e-03 at q = 2.
