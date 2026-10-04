> **UNITS CORRECTION 2026-09-24 -- the numbers in the body of this file are PRE-CORRECTION.**  Current numbers are in the section "2026-09-24 units correction" at the END of this file.  The bugs are described in `UNITS_AUDIT.md` (repo root).

# fluxanchored + spin — TESTED AND REJECTED (9 coefficients, fitted on the 2d grid)

**Result: median 1.4120e-03 across the 96-point (q, chi1) grid, against
7.686e-04 for pn_anchored + spin.  1.8x worse, and CONVERGED — so this is the model's
answer, not an optimisation shortfall.**

Recorded because the reasoning that motivated it was sound and the way it fails is
specific and reusable.  fluxanchored is the better NONSPINNING model (median 4.694e-04
against pn_anchored's 6.148e-04) and needs only 2 spin coefficients where pn_anchored
needs 4, so it looked like the cheaper and more accurate base.  It is neither.

## Model

```python
# `calc` = the gw_remnant calculator object, used for E and Edot below:
#     from gw_remnant.remnant_calculators.remnant_mass_calculator
#         import RemnantMassCalculator
#     calc = RemnantMassCalculator(time=t, h_dict={(2,2): h_pp, (2,-2): conj(h_pp)},
#                                  q=q, E_initial=0.0, L_initial=0.0, M_initial=1.0,
#                                  use_filter=False)
#   constructed inside fit_scaling_gw_remnant_energy.gwr_energy_coordinates(t, h_pp, q).
#   (2,-2) is conj((2,2)) because the binary is non-precessing.
#
# `despike(y)` = the drive repair, physics_common.despike (since 2026-09-26).  Inside the
#   BHPTNRSur2dq1e3 join window t in [-9, -5] m1 only: flag samples with
#   |d2 y| > 1e-2 * max|y| * (dt/0.2)^2, dilate the flagged run by 0.8 m1, replace it by
#   linear interpolation between the nearest UNFLAGGED samples.  Applied to Edot inside
#   physics_common.energy_drive, so E below is the integral of the despiked Edot.
#   (Before 2026-09-26: whole waveform, unscaled threshold, 4 samples, E from raw Edot.)

h_pp(t) : ppBHPT (2,2), the model INPUT   -- BHPTNRSur2dq1e3 at spin1 = chi1
h_NR(t) : NR (2,2), the model TARGET      -- NRHybSur3dq8 at chi1 = [0,0,chi1]
q       : mass ratio m1/m2 >= 1     chi1 : aligned spin of the PRIMARY
nu      = q/(1+q)**2     X1 = q/(1+q)     base = X1**1.2
W       = nu  or  eps = 1/q       # mass-ratio weight on the SPIN terms only; a FLAG

Edot(t) = calc.E_dot      # taken DIRECTLY; NOT d/dt of E -- see Definitions
E(t)    = cumulative_trapezoid(despike(Edot), t, initial=0)   # per unit TOTAL mass M
                          # (physics_common.energy_drive, e_units="M"; until 2026-09-26
                          #  calc.Eoft, the integral of the raw Edot)
F(t)    = despike(Edot) / max(despike(Edot))     # the alpha drive, in [0,1]

alpha(t) = alpha_PP(q) * (1 + alpha_F(q) * F(t))    # NO spin terms anywhere in alpha
beta (t) = beta_PP(q)  +  P(nu, chi1) * E(t)        # ALL the spin lives here
tau(t)  = t0 + integral from T_ANCHOR to t of beta(t') dt'     # T_ANCHOR = -100.0 M [ppBHPT time, so m1 units; see Constants]
h_model(tau) = alpha(t) * exp(i*phi0) * h_pp(t)
#   t0   : per-case time offset, a fitted NUISANCE, not a model coefficient
#   phi0 : overall phase, solved ANALYTICALLY inside the mismatch (not fitted)

alpha_PP(q)   = base * (1 + c0*nu + c1*nu**2)                 # 2 coefficients
alpha_F(q)    = A0*nu + A1*nu**2                              # 2
beta_PP(q)    = base * (1 + b*nu)                             # 1
P(nu, chi1)   = P0 + P1*nu  +  W*(s1*chi1 + s2*chi1**2)       # 2 nonspinning
                                                              # + 2 spin
                                                              # ----> 9 total
```

**All 9 are fitted; nothing is frozen, and chi1 = 0 is INCLUDED in the objective.**  Both
differ from the pn_anchored spin extension deliberately -- see "Why nothing was frozen".

## Anchors (derived, not fitted)

Added 2026-09-27 and checked line by line against the code on that date (file:line
references are as of then).  This section lists every anchor, that is every fixed
ingredient that is derived or pinned rather than fitted, that the code of this model
uses.  Everything not listed here is either a fitted coefficient or a numerical setting
(see the constants notes).

### A1. Leading-order (Newtonian chirp) anchor X1^(6/5)

**What it is.**  The ppBHPT waveform describes a point particle of mass m2 moving on a
background black hole of mass m1, so its time axis and its strain are in units of the
PRIMARY mass m1.  The NR waveform is in units of the TOTAL mass M = m1 + m2.  The model
relates the two through a time map tau = beta t and an amplitude factor alpha (note Eq. 1;
"the note" is `alpha_beta_pn_scaling_note_revised.pdf`).  At leading post-Newtonian order
both are fixed, with no free parameter:

    beta_LO(q)    = (nu / eta_pp)^(3/5) = X1^(6/5) = (q/(1+q))^(6/5)     (note Eq. 12)
    alpha_2,LO(q) = X1^(6/5) = beta_LO                                  (note Eq. 16)

with X1 = m1/M = q/(1+q) (Eq. 4), nu = X1 X2 = q/(1+q)^2 and eta_pp = m2/m1 = 1/q (Eq. 5).
beta_LO is the leading-order time map between ppBHPT time in m1 units and NR time in
total-mass units.

**Derivation.**  With x = (M Omega)^(2/3) (Eq. 6), the adiabatic Newtonian inspiral obeys
dx_pp/dt = (64/5) eta_pp x_pp^5 for the ppBHPT and dx_NR/dtau = (64/5) nu x_NR^5 for the
comparable-mass binary (Eq. 7).  Requiring the orbital phase to agree under tau = beta t
gives Omega_NR = Omega_pp / beta, i.e. x_NR = beta^(-2/3) x_pp (Eq. 8).  Rewritten on the
NR clock, the ppBHPT chirp is dx_NR/dtau = (64/5) eta_pp beta^(5/3) x_NR^5 (Eq. 9).
Equating this with the NR chirp gives eta_pp beta^(5/3) = nu (Eq. 10), and since
nu / eta_pp = X1^2 (Eq. 11), beta = X1^(6/5) (Eq. 12).  The exponent is 3/5 (inverting
beta^(5/3)) times 2 (from nu/eta_pp = X1^2).  Matching the Newtonian (2,2) amplitude,
h22 proportional to eta_pp x_pp for the ppBHPT and to nu x_NR for NR (Eqs. 13-14), gives
alpha_2 = X1^2 beta^(-2/3) (Eq. 15), which with Eq. 12 is again X1^(6/5) (Eq. 16).  So
the amplitude and the clock carry the same leading factor.

**Why it is derived, not fitted.**  It follows from the leading-order chirp rate and
amplitude together with the change of mass unit from m1 to M; there is no adjustable
number in it.  It is also what the data give when nothing is imposed: in the PN-free
`gwr_energy_stiff` fit the per-q optimum of the clock prefactor is
beta_PP / X1^(6/5) = 1.0019 +- 0.0014 over q in [3, 8] (`scaling_gwr_energy_stiff.md`,
"The shared base"; measured with the pre-2026-09-26 energy drive).  The note (Sec. 3.2)
also observes that the fitted alpha_2(q) and beta(q) of BHPTNRSur1dq1e4 lie close to
X1^(6/5) over 3 <= q <= 10.  The anchor is exact only at leading order; the fitted factors
described below carry the finite-mass corrections.  As q -> infinity, X1 -> 1 and the
anchor tends to 1, so it is consistent with the test-particle limit on its own.

**Spin.**  The anchor does not depend on chi1.  The Newtonian chirp rate (Eq. 7) and
the Newtonian amplitude (Eq. 13) do not involve the spins; spin first enters the PN
phasing at 1.5PN order, through spin-orbit coupling (e.g. Blanchet, Living Rev. Relativ.
17, 2 (2014)).  The same X1^(6/5) is therefore used at every chi1, and all spin
dependence of this model is in the fitted coefficient P(nu, chi1) (below).

**How this model uses it.**  `bs = (q / (1 + q)) ** 1.2` (`spin_dep/fit_fluxanchored_spin.py:81`;
1.2 = 6/5 exactly) multiplies both prefactors in `params_at` (lines 79-87):

    alpha_PP(q)   = X1^(6/5) * (1 + c0 nu + c1 nu^2)                          (line 83)
    beta_PP(q)    = X1^(6/5) * (1 + b nu)                                     (line 84)
    P(nu, chi1)   = P0 + P1 nu + W (s1 chi1 + s2 chi1^2),  W = nu or 1/q      (lines 82, 86)
    alpha(t)      = X1^(6/5) (1 + c0 nu + c1 nu^2) * (1 + (A0 nu + A1 nu^2) F(t))
    beta(t)       = X1^(6/5) (1 + b nu) + P(nu, chi1) E(t)

The last entry returned is P / beta_PP (line 87), so beta_PP (1 + beta_E E) = beta_PP + P E.
All 9 coefficients are fitted on the 96-point (q, chi1) grid.  The anchor itself is never
rescaled or refitted, and the clips ([0.05, 2.5] and [0.2, 1.6], lines 83-84) are safety
bounds.

### A2. Test-particle and non-spinning limits (by construction; no pinned constant)

As nu -> 0, X1^(6/5) -> 1 and the O(nu) factors -> 1, so alpha_PP and beta_PP -> 1 without
any coefficient being pinned.  The flux coupling alpha_F = A0 nu + A1 nu^2 (line 85) has
no constant term, so it vanishes at nu = 0 by its form; this is needed because the
peak-normalised F is 1 at the peak for every q.  The whole clock correction P(nu, chi1) E,
spin terms included, multiplies E, which -> 0 in that limit.  (E_tot ~ nu^2.31 was
measured on the nonspinning 1d grid, `fit_scaling_gw_remnant_energy.py:190-193`; on
this grid E is per M, i.e. X1 times that.)  The spin terms W (s1 chi1 + s2 chi1^2) vanish
identically at chi1 = 0, so the chi1 = 0 column is the nonspinning form.  s1, s2, P0, P1
and the alpha coupling are NOT anchored.  The primary's spin chi1 enters no fixed
ingredient: there is no remnant or ringdown anchor (below).

### A3. The energy drive E(t) and its units (a fixed convention)

E(t) comes from `physics_common.energy_drive` (`physics_common.py:181-203`), called through
`fit_scaling_gw_remnant_energy.gwr_energy_coordinates(..., surrogate="2d", e_units="M")`
in `build_case` (`spin_dep/fit_fluxanchored_spin.py:107`).  It takes the gw_remnant
`RemnantMassCalculator` flux E_dot = (1/16 pi) sum_lm |dh_lm/dt|^2 over the ppBHPT (2,2)
mode and its (2,-2) = conj((2,2)) partner, with E_initial = 0.  E_dot is despiked only inside the
BHPTNRSur2dq1e3 join window, t in [-9, -5] m1 (`JOIN_WINDOWS["2d"]`,
`physics_common.py:153`; the threshold 1e-2 of the peak is scaled by (dt/0.2)^2 for this
surrogate's dt = 0.1 m1, line 166).  E(t) is the cumulative trapezoid integral of the
despiked E_dot, so it is zero at the first ppBHPT sample (t = -13500 m1).

**Units: E is per unit TOTAL mass M**, requested explicitly at
`spin_dep/fit_fluxanchored_spin.py:107` and independent of the nonspinning flag
`E_TOTAL_MASS_UNITS` (which is False).  With e_units = "M", t and h are multiplied by X1
before gw_remnant sees them (`physics_common.py:194-195`), so E is X1 times the
per-m1 energy.  The drive version is stamped `ENERGY_DRIVE_2D` (`physics_common.py:214-215`:
"E per M; E = cumtrapz(despiked E_dot); despike in join window only").  The per-(q, chi1)
cache `spin_dep/flux_loss_cache/*.npz` stores this string and is recomputed when it does
not match (`fit_fluxanchored_spin.py:103-111`).  The nonspinning family uses E per m1
instead.  Both are valid, because E enters only through fitted couplings; each unit was
measured to be the better one for its own models (comment at
`fit_fluxanchored_spin.py:95-97`).

**The flux drive F(t)** = E_dot / max(E_dot) (line 110), from the same despiked E_dot,
peak-normalised to [0, 1], so its units drop out.

### A4. Time-map reference point T_ANCHOR = -100 m1 (chosen, not physics)

`T_ANCHOR = -100.0` (`fit_scaling_PN_opt_creative.py:24`) is a ppBHPT time, so it is in m1
units: 100 m1 before the ppBHPT |h22| peak, which sits at t = 0 on the 1d surrogate
and within 3 m1 of it on the 2d one.  The time map is
tau(t) = t0 + integral_{T_ANCHOR}^{t} beta(t') dt': the running integral of beta is set to
zero at T_ANCHOR (`fit_scaling_gwr_energy_flux.py:131`, `model_shape`, reached through `FL.fast_mismatch` in `score`, lines 118-130) and the per-case offset t0 (NR time, M units, a fitted
nuisance) is added.  Moving T_ANCHOR only shifts tau by a constant, which t0 absorbs, so
it fixes where t0 is counted from and nothing else.

### Not used by this model

No ringdown anchor: no remnant mass or spin and no quasinormal-mode frequency enter
(the model is switchless).  Unlike `pn_anchored` + spin, whose ringdown clock beta_r(q, chi1)
comes from remnant fits and Kerr QNM frequencies, nothing in this model supplies spin
dependence at merger except the fitted P.  No PN frequency coordinate x is used, so
`physics_common.x_scale` does not apply.

## Definitions and sources — every quantity, and where it comes from

**Waveforms.  BOTH differ from the nonspinning pipeline's.**

| symbol | definition | SOURCE |
|:---|:---|:---|
| `h_pp(t)` | ppBHPT (2,2), the model INPUT | `BHPTNRSur2dq1e3.generate_surrogate(q=q, spin1=chi1, calibrated=False, modes=[(2,2)], neg_modes=False)`, in `fit_pn_anchored_spin.build_case`. **`calibrated=False` = raw 0PA**, no NR calibration inside the surrogate. Domain **q in [3, 1000]**, chi1 in [-0.8, +0.8]; grid t in [-13500, 125] M [ppBHPT time, so m1 units] at **dt = 0.1** |
| `h_NR(t)` | NR (2,2), the TARGET | `gwsurrogate.LoadSurrogate("NRHybSur3dq8")(q, [0,0,chi1], [0,0,0], dt=0.1, f_low=5e-3)`. Domain q <= 8, \|chi\| <= 0.8. **chi2 = 0 because the ppBHPT secondary is a non-spinning test particle** |
| window | scored NR span | masked to [`NR_T_START`, `NR_T_END`] = **[-5000.1, +100.0] M** |
| strides | subsampling for the fast inner fit | `SRC_STRIDE = 6`, `NR_STRIDE = 10` |

**This is a DIFFERENT source from the nonspinning models** (BHPTNRSur1dq1e4). At
chi1 = 0 the two surrogates differ from each other by a mismatch of **8.7e-05 to
1.2e-04** -- about 25% of the residual these models fit -- so coefficients are NOT
portable between them. That is why everything here is refitted rather than frozen.

**Energy and flux -- all from `gw_remnant`, none computed here.**

`RemnantMassCalculator(time=t, h_dict={(2,2): h, (2,-2): conj(h)}, q=q, E_initial=0,
L_initial=0, M_initial=1, use_filter=False)`, via
`fit_scaling_gw_remnant_energy.gwr_energy_coordinates`, cached per (q, chi1) in
`flux_loss_cache/`.

| symbol | definition | SOURCE |
|:---|:---|:---|
| `Edot(t)` | `(1/16 pi) sum_lm \|dh_lm/dt\|^2`, `dh/dt` by 2nd-order finite differences | `calc.E_dot`, `remnant_mass_calculator.py:136-164`. **Taken DIRECTLY (since 2026-09-10)**, NOT by differentiating `Eoft` -- `Eoft` IS `cumtrapz(E_dot)` (line 182), so that round trip is a [1,2,1]/4 binomial smoothing |
| `E(t)` | cumulative radiated energy, per unit TOTAL mass M | `physics_common.energy_drive(..., e_units="M", surrogate="2d")` (requested at `fit_fluxanchored_spin.py:107`): `cumulative_trapezoid` of the DESPIKED `E_dot` on the X1-scaled axis, initial 0.  (Until 2026-09-26 `calc.Eoft`, the integral of the raw `E_dot`; corrected here 2026-09-27) |
| `F(t)` | **the alpha drive**: `despike(Edot)/max(despike(Edot))` | despike = `physics_common.despike`: curvature detection (`\|d2 Edot\| > 1e-2 * peak * (dt/0.2)^2`) inside the 2d join window t in [-9, -5] m1 only, dilate 0.8 m1, replace by interpolation between the nearest CLEAN samples.  The old whole-waveform despike was measured to be **a NO-OP on this cache** (0.03% on the chi1 = 0 null check, 2026-09-09) |

**Mass ratio, spin, mismatch.**

| symbol | definition | SOURCE |
|:---|:---|:---|
| `nu`, `X1`, `base` | `q/(1+q)^2`, `q/(1+q)`, `X1^(6/5)` | derived; `base` is the Newtonian chirp anchor, so the model is **not PN-free** |
| `W` | mass-ratio weight on the SPIN terms only: `nu` or `eps = 1/q` | a FLAG, `--weight`. Measurement favours eps (k = 0.95 +- 0.24); the mismatch favours nu. See `scaling_pn_anchored_spin.md` |
| `chi1` | dimensionless aligned spin of the PRIMARY | grid `[-0.6, -0.4, -0.2, 0, +0.2, +0.4, +0.6, +0.8]`; chi1 < -0.6 excluded (retrograde QNM regime) |
| `T_ANCHOR` | time-map anchor, **-100.0 M** [ppBHPT time axis, so m1 units] | `fit_scaling_PN_opt_creative.T_ANCHOR` |
| `t0` | per-case time offset, a fitted NUISANCE (not a coefficient) | coarse scan over [-250, +150] M, then `minimize_scalar` |
| `phi0` | overall phase, **analytic** | `fit_scaling_gwr_energy_global.err_at_t0` |
| mismatch | `(n1 + n2 - 2\|Z\|)/(2 n1)`, relative L2 on the overlap; 50.0 if NR coverage < `MIN_COVERAGE = 0.88` | same |
| grid | **12 even-nu q in [3,8] x 8 chi1 = 96 points**, chi1 = 0 INCLUDED | `TRAIN_Q`, `CHIS` in `fit_pn_anchored_spin` |

## Constants and their sources — additions (2026-09-26)

Code references are to `spin_dep/fit_fluxanchored_spin.py` unless another file is named
(line numbers as of 2026-09-26).  **Units:** ppBHPT times (`t_bhpt`) are in units of the
PRIMARY mass m1; NR times (`t_nr`) in units of the TOTAL mass M.  Where the tables write
"M" for a ppBHPT time (the h_pp grid, `T_ANCHOR`), read m1.

**Derived (not fitted)**

* **`base = X1**1.2` = X1^(6/5).**  Leading-order time-map and amplitude scaling, from
  matching the Newtonian chirp rate and the Newtonian (2,2) amplitude between ppBHPT
  (mass ratio m2/m1 = 1/q) and the comparable-mass binary (nu).  Derivation:
  `../alpha_beta_pn_scaling_note_revised.pdf` Eqs. (7)-(12) and (13)-(16).

**Chosen settings (by hand, not fitted)**

* **despike:** threshold `DESPIKE_T = 1e-2` (|second difference| as a fraction of the
  peak, scaled by (dt/0.2)^2) and padding `DESPIKE_PAD_T = 0.8` m1, applied only inside the
  2d join window t in [-9, -5] m1 (`../physics_common.py:150-153`, since 2026-09-26; before
  that `DESPIKE_PAD = 4` samples over the whole waveform,
  `../fit_scaling_gwr_fluxanchored_reg.py:70-71`).  The docstring there explains the 1e-2: it sits ~25x above real merger structure (~4e-4 of
  peak) and ~50x below the 1d-surrogate glitch (~0.5 of peak) it was built for.
* **`T_ANCHOR = -100.0`** (`../fit_scaling_PN_opt_creative.py:24`): ppBHPT time (m1 units)
  where the running integral of beta is zeroed (`../fit_scaling_gwr_energy_flux.model_shape`).
* **Window `[-5000.1, +100.0]`** (NR time, M units): `../fit_scaling_PN_opt_creative.py:21-22`.
* **`MIN_COVERAGE = 0.88`** (`../fit_scaling_PN_opt_creative_q_dep.py:34`): below 88% NR
  coverage a case scores 50.0.  Non-monotone tau scores 50 + 1000*(backwards step).
* **t0 search:** first pass an 81-point scan over [-250, +150] M (`score`, line 117), then a
  bounded 1-D minimisation within +- 60 M of the current value
  (`../fit_scaling_gwr_energy_flux.fast_mismatch`, line 146), on the strided arrays; the
  reported mismatch is then recomputed on the FULL-resolution arrays at that t0.
* **Strides** `SRC_STRIDE = 6`, `NR_STRIDE = 10`: `fit_pn_anchored_spin.py:137`, used only
  for the fast t0 stage.
* **Safety bounds (not in the Model block):** `alpha_PP` clipped to [0.05, 2.5], `beta_PP`
  to [0.2, 1.6] (`params_at`, lines 83-84); inactive at the fitted values.
* **Optimiser:** Powell (line 183), xtol 1e-4, ftol 1e-6, `--maxiter` default 400,
  `--budget` default 100 minutes (lines 132-133).  Objective = MEAN mismatch over the 96
  cases.  Starting point (`seed_theta`, line ~75): the nonspinning fluxanchored
  coefficients from `../gwr_fluxanchored_reg_results/coeffs_aF2.json` plus s1 = s2 = 0.

**External inputs**

* **E(t), Edot(t):** `gw_remnant` `RemnantMassCalculator` via
  `../fit_scaling_gw_remnant_energy.gwr_energy_coordinates` -> `../physics_common.energy_drive`.
  **E is per unit TOTAL mass M**, requested explicitly (`e_units="M"`,
  `fit_fluxanchored_spin.py:107`) and independent of the nonspinning flag
  `E_TOTAL_MASS_UNITS` (now False, `../fit_scaling_gw_remnant_energy.py:100`); the drive is
  stamped `ENERGY_DRIVE_2D` in each `flux_loss_cache/` file (lines 103-111).  See "Anchors",
  A3.  *Dated history (corrected 2026-09-27):* before the 2026-09-24 fix E was on the ppBHPT
  (m1) axis, i.e. E/m1; the body's P0, P1, s1, s2 were fitted to E/m1.  From 2026-09-24 it was
  multiplied by X1 through `E_TOTAL_MASS_UNITS = True` (`UNITS_AUDIT.md` B1); since
  2026-09-26 that flag is False again and the spin models request M units themselves.
  F(t) is peak-normalised, so its units do not matter.

**Fitted coefficients** (9, all fitted on the 96-point grid; `NAMES`, line 69)

| name | multiplies |
|:---|:---|
| `c0`, `c1` | nu and nu^2 in alpha_PP = base*(1 + c0 nu + c1 nu^2) |
| `A0`, `A1` | nu*F(t) and nu^2*F(t) in alpha |
| `b` | nu in beta_PP = base*(1 + b nu) |
| `P0`, `P1` | E(t) and nu*E(t) in beta |
| `s1`, `s2` | W*chi1*E(t) and W*chi1^2*E(t) in beta |

Stored in `fluxanchored_spin_results/coeffs_{nu,eps}.json` (`theta`).  **Note:** the
W = nu row in the Coefficients block below is PRE-correction and is no longer in the
json; `coeffs_nu.json` now holds the 2026-09-24 corrected fit: c0 = +0.2017,
c1 = -0.9165, A0 = -1.6851, A1 = +0.9458, b = +0.0124, P0 = -6.3159, P1 = +39.7332,
s1 = +35.5488, s2 = +0.7588 (median 1.3012e-03).  The W = 1/q row still matches
`coeffs_eps.json`.

**Where the other numbers come from**

* Mismatch = (n1 + n2 - 2|Z|)/(2 n1) on the NR window, phase solved exactly, t0 per case,
  full resolution.  "median", "max" over the 96 cases; per-chi1 rows = median over the 12
  q at that chi1.
* Two generations of runs appear in this file.  First converged runs (Edot obtained by
  differentiating E): jobs 3483948 (nu, median **1.4120e-03**, chi1 = -0.6 column
  **4.11e-03**) and 3483949 (eps, median **1.6671e-03**), logs `faspin.348394{8,9}.log`.
  Refit on the direct Edot (2026-09-10): jobs 3485184 (nu, 1.4110e-03) and 3485185 (eps,
  1.6672e-03), logs `faspin2.348518{4,5}.log`; these are the Coefficients and Results
  tables.  So the headline, the "4.11e-03" in "WHY it loses" and the "1.6671e-03" in the
  eps section are first-generation numbers.
* "8.7e-05 to 1.2e-04", "2.8e-05", "4.694e-04 -> 7.453e-04", "6.148e-04 -> 5.543e-04": the
  2026-09-09 measurement between the two ppBHPT surrogates at chi1 = 0 (`../history.md`
  line 170); 7.453e-04 is the transplanted-coefficient null check,
  `fluxanchored_spin_nullcheck.json`.
* "0.03%": change in the chi1 = 0 null-check median with despike on vs off
  (`fluxanchored_spin_nullcheck.json`).  "28000x": largest |d2|h|| over its median, found
  at q = 5, chi1 = +0.6 (`null_check_fluxanchored_spin.py` docstring; `../history.md` line 321).
* "0.17% to 3.35%": within-case spread over x of the measured spin ratio R_beta
  (`spin_extension_plan.md` sec 8 table).  "k = 0.95 +- 0.24", "1.45 +- 0.37": fitted
  exponents of |R_beta - 1| ~ V^k across q, x <= 0.09 (see `scaling_pn_anchored_spin.md`).
  "0.7%", "0.66-2.4%": |R_alpha - 1| maximum and its within-case spread
  (`spin_extension_plan.md` sec 7).

## Coefficients

```
names   c0       c1       A0       A1       b        P0       P1       s1       s2

W = nu   (job 3485184, CONVERGED, 974 evals)
      +0.2018 -0.9265 -1.7204 +1.1502 +0.0122 -5.1323 +32.2112 +27.7901 +0.7345

W = 1/q  (job 3485185, CONVERGED, 1433 evals)
      +0.1846 -0.8384 -1.7293 +1.1526 +0.0117 -4.7396 +30.0999 +16.6686 +0.6348
```

Both refitted 2026-09-10 on the direct-`E_dot` drive (see Definitions).  That change
moved nothing: the W = nu median went 1.4120e-03 -> 1.4110e-03 (0.07%).  As with
fluxanchored and flux_n, the despike bridges the repaired window either way, so removing
the round trip's [1,2,1]/4 smoothing has no effect on a model that despikes.

## Results

| | coefficients | median | max |
|:---|---:|---:|---:|
| pn_anchored + spin | 13 (9 frozen + 4) | **7.686e-04** | 9.64e-03 |
| pn_anchored + chi1 split | 17 (9 + 8) | **7.245e-04** | 1.01e-02 |
| **fluxanchored + spin, W = nu** | **9, all fitted** | 1.4110e-03 | 1.5175e-02 |
| fluxanchored + spin, W = 1/q | 9, all fitted | 1.6672e-03 | 1.8481e-02 |

**1.8x worse than pn_anchored, and CONVERGED** (974 evals, "terminated successfully") --
so this is the model's answer, not an optimisation shortfall.

Per chi1, against pn_anchored:

| chi1 | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| pn_anchored | 1.32e-3 | 6.11e-4 | 4.81e-4 | 5.54e-4 | 5.88e-4 | 9.29e-4 | 1.17e-3 | 3.44e-3 |
| fluxanchored, nu | 4.12e-03 | 2.37e-03 | 9.95e-04 | 6.14e-04 | 7.79e-04 | 1.18e-03 | 1.84e-03 | 4.23e-03 |
| fluxanchored, eps | 5.81e-03 | 3.05e-03 | 1.33e-03 | 6.65e-04 | 8.03e-04 | 1.36e-03 | 2.66e-03 | 5.25e-03 |

Full grid (W = nu):

| q | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.000 | 5.23e-03 | 4.54e-03 | 2.42e-03 | 1.15e-03 | 1.10e-03 | 2.84e-03 | 4.80e-03 | 1.07e-02 |
| 3.256 | 3.13e-03 | 2.58e-03 | 1.45e-03 | 8.75e-04 | 1.01e-03 | 2.21e-03 | 2.86e-03 | 7.07e-03 |
| 3.513 | 2.14e-03 | 1.53e-03 | 9.30e-04 | 7.75e-04 | 9.76e-04 | 1.80e-03 | 1.73e-03 | 4.56e-03 |
| 3.897 | 1.82e-03 | 9.18e-04 | 6.02e-04 | 7.21e-04 | 9.25e-04 | 1.38e-03 | 9.41e-04 | 2.40e-03 |
| 4.200 | 2.10e-03 | 8.91e-04 | 5.37e-04 | 6.89e-04 | 8.77e-04 | 1.18e-03 | 7.55e-04 | 1.65e-03 |
| 4.600 | 2.80e-03 | 1.16e-03 | 5.76e-04 | 6.41e-04 | 8.11e-04 | 1.03e-03 | 8.19e-04 | 1.44e-03 |
| 5.000 | 3.65e-03 | 1.61e-03 | 6.86e-04 | 5.87e-04 | 7.47e-04 | 9.64e-04 | 1.07e-03 | 1.78e-03 |
| 5.436 | 4.59e-03 | 2.17e-03 | 8.46e-04 | 5.31e-04 | 6.81e-04 | 9.63e-04 | 1.45e-03 | 2.56e-03 |
| 5.949 | 5.63e-03 | 2.84e-03 | 1.06e-03 | 4.78e-04 | 6.08e-04 | 1.01e-03 | 1.95e-03 | 3.90e-03 |
| 6.590 | 6.76e-03 | 3.63e-03 | 1.35e-03 | 4.41e-04 | 5.22e-04 | 1.09e-03 | 2.62e-03 | 6.22e-03 |
| 7.231 | 7.71e-03 | 4.36e-03 | 1.66e-03 | 4.39e-04 | 4.42e-04 | 1.18e-03 | 3.38e-03 | 9.48e-03 |
| 8.000 | 8.68e-03 | 5.18e-03 | 2.08e-03 | 4.86e-04 | 3.59e-04 | 1.30e-03 | 4.50e-03 | 1.52e-02 |

**Read the q-columns, not just the medians.**  At chi1 = -0.6 the error RISES
monotonically with q, 5.23e-03 at q=3 to 8.68e-03 at q=8, and at chi1 = +0.8 it is
U-shaped with a minimum near q = 4.6 (1.44e-03) rising to 1.52e-02 at q=8.  Compare the
chi1 = 0 column, which falls from 1.15e-03 to a minimum at q = 7.2 and barely ticks up.
So the spin sector degrades with q on BOTH signs while the nonspinning part improves --
the opposite of what a well-specified spin term would do.

## WHY it loses, and both reasons were structural rather than bad luck

**1. No remnant anchor.**  pn_anchored's merger branch carries `beta_r(q, chi1)`, the
ratio of the two sides' QNM frequencies, derived from NRSur3dq8Remnant plus the Kerr
(2,2,0) fit -- spin dependence in the ringdown for ZERO fitted parameters.  fluxanchored
is switchless with no remnant anchor at all; its merger comes only from E(t) flattening.
The cost shows exactly where predicted: **the retrograde columns are 3-4x worse**
(4.11e-03 vs 1.32e-03 at chi1 = -0.6), and chi1 = -0.6 is where the remnant
differs most from the primary.

**2. One spin shape term, not two.**  pn_anchored's beta drift has two terms (`b1*x` and
`b2*x^2`), so a quadratic in chi1 on each gives 4 coefficients AND lets the spin
correction change its time PROFILE.  fluxanchored's drift is the single term `P*E`, so
the spin correction's profile is locked to E(t) and only its amplitude moves.  The
step-1 measurement had already shown that is not enough -- `R_beta`'s spin ratio varies
with x by 0.17% to 3.35% -- and the fit confirms it cannot use even the freedom it has:
**s1 = +27.79 against s2 = +0.72**, i.e. the chi1^2 term is
doing essentially nothing.

Note also that chi1 = 0 comes out at 6.14e-04, WORSE than pn_anchored's
5.543e-04 -- despite being fitted on this exact grid while pn_anchored's was frozen from
a 1d fit.  So the deficit is not a spin problem alone.

## Why nothing was frozen, and why chi1 = 0 was included

The freeze-then-add-spin pattern used for pn_anchored assumes the nonspinning
coefficients transfer between waveform sources.  **They do not.**  BHPTNRSur1dq1e4 and
BHPTNRSur2dq1e3 at chi1 = 0 differ from each other by a mismatch of 8.7e-05 to 1.2e-04
(2.8e-05 in the inspiral alone) -- small absolutely, but ~25% of the 4.7e-04 residual
being fitted.  Transplanting 1d coefficients onto 2d waveforms therefore moves the answer
by tens of percent with a sign set only by how the two residuals align: fluxanchored went
4.694e-04 -> 7.453e-04, pn_anchored went 6.148e-04 -> 5.543e-04.  **Neither is evidence
about the models** -- the same ~1e-04 perturbation landing opposite ways.  Fitting on the
grid the model is scored on removes the ambiguity entirely.

chi1 = 0 is in the objective for the same reason: with the nonspinning coefficients free,
that column carries real information about them.  (In the pn_anchored spin fits it was
excluded, correctly, because those coefficients were frozen and its gradient with respect
to the spin coefficients is identically zero.)

## eps = 1/q loses here too

median 1.6671e-03 against 1.4120e-03 for nu -- an 18% penalty, and worse in
every single chi1 column.  This is the **second independent model** to prefer nu: the
pn_anchored like-for-like test gave 7.686e-04 (nu) against 8.151e-04 (eps).  Both stand
against a parameter-free measurement that favours eps (exponent k = 0.95 +- 0.24 against
1.45 +- 0.37), which is inspiral-only.  The full-window mismatch is merger-weighted and
disagrees.  See `scaling_pn_anchored_spin.md` for the measurement.

## Caveats

* The drive despike was a NO-OP on this cache (0.03% on the chi1 = 0 null check, measured 2026-09-09 with the old whole-waveform despike; since 2026-09-26 it acts only in the 2d join window t in [-9, -5] m1).
  BHPTNRSur2dq1e3 carries no analogue of the 1d surrogate's 4-sample glitch AT THE ONE
  PLACE IT WAS CHECKED (t = -15 M); it has never been scanned for defects elsewhere.  One
  blemish is known: q = 5, chi1 = +0.6, where |d2|h|| reaches 28000x the median with a
  non-monotone dphase.
* alpha carries no spin, on the same evidence as pn_anchored: |R_alpha - 1| <= 0.7%
  against a within-case x-spread of 0.66-2.4%, and R_alpha - 1 CHANGES SIGN across q at
  chi1 = +0.3, +0.6, +0.8 -- no power law in any mass-ratio variable can do that.  Both
  checks are INSPIRAL ONLY.
* q < 3 is a double extrapolation: outside the fit's range AND outside
  BHPTNRSur2dq1e3's own domain (q_min = 3, warning only).
* Only the (2,2) mode; aligned primary spin only, secondary a non-spinning test particle.

## Files

- fit: `python fit_fluxanchored_spin.py --weight {nu,eps} --refit --budget 100`
  (`--resume` to continue; `--cache-only` to build the loss cache)
- results: `fluxanchored_spin_results/coeffs_{nu,eps}.json`
- loss cache: `flux_loss_cache/` (gw_remnant E and F per (q, chi1); the expensive part)
- overlays: `python plot_fluxanchored_spin_overlay.py` -> `Agentic_spin_plots/fluxanchored_spin/`
- chi1 = 0 null check of the TRANSPLANTED 1d coefficients:
  `null_check_fluxanchored_spin.py`, `fluxanchored_spin_nullcheck.json`
- nonspinning parent: `../scaling_gwr_energy_fluxanchored.md`
- the model that wins: `scaling_pn_anchored_spin.md`

---

## 2026-09-24 units correction

**Root cause (all models):** the ppBHPT waveform's time axis is in units of the PRIMARY
mass (`calibrated=False` returns the raw training grid); the NR waveform's is in TOTAL
mass.  Anything crossing between them needs an `X1 = q/(1+q)` conversion.
Mechanical check: `python units_selftest.py --q 3 5 8` (all pass).

**What affected this model:** beta uses raw E (the Eoft axis error).  Drive already despiked.

**Corrected:** median **1.3012e-03 (-7.8% vs 1.4110e-03, better)**, mean 2.1109e-03,
max 1.4030e-02, converged.  (The headline above says 1.4120e-03; the coefficient file
records 1.4110e-03, which is the value compared against.)
Worst region: chi1=+0.8 (column median 4.99e-03), same shape as energy_anchored_spin.
q=2 before -> after: median 2.847e-02 -> 2.800e-02 (-1.6%, no change).  q=2 is dominated
by the out-of-domain surrogate input (double extrapolation), not by the calibration.

**Status: within the bar (better).**  Plots REDRAWN: `Agentic_spin_plots/fluxanchored_spin/`;
pre-correction in `fluxanchored_spin_PREFIX_BACKUP_20260924/`.


## 2026-09-27 paper-style figures (after Islam+ arXiv:2407.18319v2 Figs 5, 7, 9a)

`plot_spin_paper_figs.py --models fluxanchored_spin` (job 3474067), in `Agentic_spin_plots/fluxanchored_spin/paper/`:
`fluxanchored_spin_hist_train_vs_heldout` (log10 mathcalE, training 96 vs held-out 270), `fluxanchored_spin_calibration_chi+0.6`
(Re h22 before / after calibration, q = 3, 5, 8), `fluxanchored_spin_mismatch_vs_chi1` (q = 3, 5, 8).
Scored with this model's own mismatch code on the current (09-26 fixed) inputs;
all 96 training values recomputed equal the stored ones (ratio 1.000000).
Training: mean 2.110e-3 / median 1.300e-3 / max 1.40e-2.  Held-out (270 points inside the training range, off the training grid:
3 < q < 8, chi1 in (-0.6, 0.8)): mean 1.442e-3 / median 1.001e-3 / max 6.37e-3 -- lower than training because the held-out set is interior
(no q = 3 / 8 row, no chi1 = -0.6 / +0.8 column), where the errors are largest; interpolation only.

## 2026-09-27 test: beta driven by J(t) + k*Jdot(t) instead of E(t) -- rejected

TEST, not a model change; installed coefficients untouched.  Question: the flux identity
2*Edot/(omega*Jdot) = 1 + (Adot/(omega A))^2 holds to 1e-6..5e-5 in the inspiral on every spin
case (the (2,2) J flux is Edot/Omega), so Jdot carries no new spin information -- but as a
merger-peaked profile next to a cumulative one it could give beta the ~10% merger bump that a
single monotone E(t) cannot.

1. Projection, no mismatch (`beta_shape_projection.py`, 8 cases q = 3, 8 x chi1 = -0.6, 0, 0.4,
   0.8, measured beta = omega_pp/omega_NR at equal phase, window -1000..+10 M): {1, J + k*Jdot}
   with one shared k = 0.70 cut the merger-band rms from 3.5% to 2.5% (overall 1.11% -> 0.95%);
   a rate term on top of E did nothing (best k -> 0).  The retrograde ringdown was excluded: its
   omega is not a single clean frequency (beta_meas 0.05-1.85 at q = 3, chi1 = -0.6).
2. Mismatch fit (`test_beta_jdrive.py`, jobs 3474486 / 3474490; 18 points q = 3, 5, 8 x chi1 =
   -0.6, -0.2, 0, 0.4, 0.6, 0.8; both arms refitted on the same subset from coeffs_nu.json):
   beta = b_pp*(1 + b_E*D), D = E_rad*[J/J_rad + k*Jdot/max Jdot] (J, Jdot per M, same scaling
   as E; this code's E_dot reproduces gw_remnant's to 2e-8).

   | arm | coefficients | mean | median | max |
   |---|---|---|---|---|
   | control, beta from E(t) | 9 | **3.206e-3** | **1.855e-3** | 1.195e-2 |
   | beta from J + k*Jdot | 10 | 4.199e-3 | 2.605e-3 | 1.445e-2 |

   Three J-arm starts (P x 0.1 / 0.1 / 0.2, k = 2.0 / 0.7 / 5.0) converge to the same point
   (k = 1.924, mean 4.1987e-3), so it is not a local minimum.  J wins only at q = 3, chi1 = +0.8
   (5.1e-3 vs 7.8e-3); it loses most at retrograde q = 5, 8 and at q = 8, chi1 = -0.2.

Why the projection misled: J/J_rad is already 0.16-0.26 at t = -5000 M and 0.88-0.92 at the peak
(E/E_rad: 0.02-0.09 and 0.63-0.73), so a J drive puts a strong drift into beta over the long
inspiral, which the -1000 M projection window never saw.  Verdict: keep E(t) as beta's drive.
Full physics write-up, with all three tests: `../angular_momentum_flux_failure.md`.
