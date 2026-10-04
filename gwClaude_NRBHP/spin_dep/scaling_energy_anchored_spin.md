> **UNITS CORRECTION 2026-09-24 -- the numbers in the body of this file are PRE-CORRECTION.**  Current numbers are in the section "2026-09-24 units correction" at the END of this file.  The bugs are described in `UNITS_AUDIT.md` (repo root).

# energy_anchored + spin — TESTED AND REJECTED (8 coefficients, fitted on the 2d grid)

**Result: median 1.7311e-03 across the 96-point (q, chi1) grid, against
1.411e-03 for fluxanchored + spin and 7.686e-04 for pn_anchored + spin.  CONVERGED
(1227 evals, reproduced to 4 decimals by a 164-eval resume), so this is the model's
answer, not an optimisation shortfall.**

**What it isolates.**  This differs from `fit_fluxanchored_spin.py` in EXACTLY ONE
THING: alpha reads `E(t)` instead of `F(t)` (and `alpha_E = A0/nu` rather than
`A0*nu + A1*nu^2`).  Switchless, no remnant anchor, beta driven by `E` with the spin on
`P`, one spin shape term -- all shared.  So the 23% gap between them
(1.7311e-03 vs 1.4110e-03) measures what alpha's DRIVE is worth under spin, with
nothing else varying.  **The flux is the better alpha coordinate**, and by almost exactly
the margin it wins by without spin (4.70e-04 vs 6.26e-04, 25%).  That the same ~25%
appears in a completely different regime is the most useful thing this run produced.

## Model

```python
# `calc` = the gw_remnant calculator object, used for E below:
#     from gw_remnant.remnant_calculators.remnant_mass_calculator
#         import RemnantMassCalculator
#     calc = RemnantMassCalculator(time=t, h_dict={(2,2): h_pp, (2,-2): conj(h_pp)},
#                                  q=q, E_initial=0.0, L_initial=0.0, M_initial=1.0,
#                                  use_filter=False)
#   constructed inside fit_scaling_gw_remnant_energy.gwr_energy_coordinates(t, h_pp, q).
#   (2,-2) is conj((2,2)) because the binary is non-precessing.

h_pp(t) : ppBHPT (2,2), the model INPUT   -- BHPTNRSur2dq1e3 at spin1 = chi1
h_NR(t) : NR (2,2), the model TARGET      -- NRHybSur3dq8 at [0,0,chi1]
q       : mass ratio m1/m2 >= 1     chi1 : aligned spin of the PRIMARY
nu      = q/(1+q)**2     X1 = q/(1+q)     base = X1**1.2
W       = nu  or  eps = 1/q       # mass-ratio weight on the SPIN terms only; a FLAG

E(t)    = cumulative_trapezoid(despike(calc.E_dot), t, initial=0)
                          # cumulative radiated energy, per unit TOTAL mass M, E(t_start) = 0
                          # (physics_common.energy_drive, e_units="M"; despike only in the
                          #  2d join window t in [-9, -5] m1).  This model never reads the
                          #  FLUX as a drive and has no peak normalisation.  (Until
                          #  2026-09-26: calc.Eoft of the raw E_dot, no despike.)

alpha(t) = alpha_PP(q) * (1 + alpha_E(q)   * E(t))     # E-driven, NO spin terms
beta (t) = beta_PP(q)  * (1 + b_E(q,chi1)  * E(t))     # ALL the spin lives here
tau(t)   = t0 + integral from T_ANCHOR to t of beta(t') dt'    # T_ANCHOR = -100.0 M [ppBHPT time, so m1 units; see Constants]
h_model(tau) = alpha(t) * exp(i*phi0) * h_pp(t)
#   t0   : per-case time offset, a fitted NUISANCE, not a model coefficient
#   phi0 : overall phase, solved ANALYTICALLY inside the mismatch

alpha_PP(q)  = base * (1 + c0*nu + c1*nu**2)                  # 2 coefficients
alpha_E(q)   = A0 / nu                                        # 1  (see below)
beta_PP(q)   = base * (1 + b*nu)                              # 1
P(nu,chi1)   = P0 + P1*nu + W*(s1*chi1 + s2*chi1**2)          # 2 nonspinning + 2 spin
b_E(q,chi1)  = P(nu,chi1) / beta_PP(q)                        # ----> 8 total
```

**On `alpha_E = A0/nu`.**  The `1/nu` is a NORMALISATION CONVENTION, not physics: `E` is
the ppBHPT's radiated energy in the surrogate's own units, so only the product
`alpha_E * E` is physical.  The parent's own notes record that higher orders in it are
neither identifiable (it is the soft direction of a -0.974-correlated degeneracy with
`alpha_PP`) nor meaningful.  Carried over unchanged.

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

**How this model uses it.**  `bs = (q / (1 + q)) ** 1.2` (`spin_dep/fit_energy_anchored_spin.py:78`;
1.2 = 6/5 exactly) multiplies both prefactors in `params_at` (lines 76-84):

    alpha_PP(q)   = X1^(6/5) * (1 + c0 nu + c1 nu^2)                          (line 80)
    beta_PP(q)    = X1^(6/5) * (1 + b nu)                                     (line 81)
    alpha_E(q)    = A0 / nu                                                   (line 82)
    P(nu, chi1)   = P0 + P1 nu + W (s1 chi1 + s2 chi1^2),  W = nu or 1/q      (lines 79, 83)
    alpha(t)      = X1^(6/5) (1 + c0 nu + c1 nu^2) * (1 + (A0/nu) E(t))       (line 91)
    beta(t)       = X1^(6/5) (1 + b nu) (1 + (P / beta_PP) E(t))
                  = X1^(6/5) (1 + b nu) + P(nu, chi1) E(t)                    (lines 84, 92)

All 8 coefficients are fitted on the 96-point (q, chi1) grid.  The anchor itself is never
rescaled or refitted, and the clips ([0.05, 2.5] and [0.2, 1.6], lines 80-81) are safety
bounds.

### A2. Test-particle and non-spinning limits (by construction; no pinned constant)

As nu -> 0, X1^(6/5) -> 1 and the O(nu) factors -> 1, so alpha_PP and beta_PP -> 1 without
any coefficient being pinned.  The amplitude term (A0/nu) E goes as E/nu, which -> 0 because E
falls faster than nu.  The whole clock correction P(nu, chi1) E,
spin terms included, multiplies E, which -> 0 in that limit.  (E_tot ~ nu^2.31 was
measured on the nonspinning 1d grid, `fit_scaling_gw_remnant_energy.py:190-193`; on
this grid E is per M, i.e. X1 times that.)  The spin terms W (s1 chi1 + s2 chi1^2) vanish
identically at chi1 = 0, so the chi1 = 0 column is the nonspinning form.  s1, s2, P0, P1
and the alpha coupling are NOT anchored.  The primary's spin chi1 enters no fixed
ingredient: there is no remnant or ringdown anchor (below).

### A3. The energy drive E(t) and its units (a fixed convention)

E(t) comes from `physics_common.energy_drive` (`physics_common.py:181-203`), called through
`fit_scaling_gw_remnant_energy.gwr_energy_coordinates(..., surrogate="2d", e_units="M")`
in `build_case` (`spin_dep/fit_fluxanchored_spin.py:107`, reached from this model through `FS.build_case`, `fit_energy_anchored_spin.py:131`).  It takes the gw_remnant
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

This model reads only E(t), not the flux F, but E itself is integrated from the despiked
E_dot.

### A4. Time-map reference point T_ANCHOR = -100 m1 (chosen, not physics)

`T_ANCHOR = -100.0` (`fit_scaling_PN_opt_creative.py:24`) is a ppBHPT time, so it is in m1
units: 100 m1 before the ppBHPT |h22| peak, which sits at t = 0 on the 1d surrogate
and within 3 m1 of it on the 2d one.  The time map is
tau(t) = t0 + integral_{T_ANCHOR}^{t} beta(t') dt': the running integral of beta is set to
zero at T_ANCHOR (`spin_dep/fit_energy_anchored_spin.py:95-96`, in `mismatch`) and the per-case offset t0 (NR time, M units, a fitted
nuisance) is added.  Moving T_ANCHOR only shifts tau by a constant, which t0 absorbs, so
it fixes where t0 is counted from and nothing else.

### Not used by this model

No ringdown anchor: no remnant mass or spin and no quasinormal-mode frequency enter
(the model is switchless).  Unlike `pn_anchored` + spin, whose ringdown clock beta_r(q, chi1)
comes from remnant fits and Kerr QNM frequencies, nothing in this model supplies spin
dependence at merger except the fitted P.  No PN frequency coordinate x is used, so
`physics_common.x_scale` does not apply.

## Definitions and sources — every quantity, and where it comes from

| symbol | definition | SOURCE |
|:---|:---|:---|
| `h_pp(t)` | ppBHPT (2,2), the INPUT | `BHPTNRSur2dq1e3.generate_surrogate(q=q, spin1=chi1, calibrated=False, modes=[(2,2)], neg_modes=False)`, in `fit_pn_anchored_spin.build_case`. **`calibrated=False` = raw 0PA.** Domain **q in [3, 1000]**, chi1 in [-0.8, +0.8]; t in [-13500, 125] M [ppBHPT time, so m1 units] at **dt = 0.1** |
| `h_NR(t)` | NR (2,2), the TARGET | `gwsurrogate.LoadSurrogate("NRHybSur3dq8")(q, [0,0,chi1], [0,0,0], dt=0.1, f_low=5e-3)`. Domain q <= 8, \|chi\| <= 0.8. **chi2 = 0: the ppBHPT secondary is a non-spinning test particle** |
| window | scored NR span | masked to [`NR_T_START`, `NR_T_END`] = **[-5000.1, +100.0] M** |
| strides | subsampling for the inner fit | `SRC_STRIDE = 6`, `NR_STRIDE = 10` |
| `E(t)` | cumulative radiated energy, per unit TOTAL mass M | `physics_common.energy_drive(..., e_units="M", surrogate="2d")`: `cumulative_trapezoid` of the E_dot despiked in the 2d join window, initial 0 (until 2026-09-26 `calc.Eoft` of the raw E_dot, `remnant_mass_calculator.py:182`; corrected here 2026-09-27). Cached per (q, chi1) in `flux_loss_cache/` (shared with the fluxanchored spin fit, `fit_fluxanchored_spin.py:103-111`) |
| `calc.E_dot` | `(1/16 pi) sum_lm \|dh_lm/dt\|^2`, `dh/dt` by 2nd-order finite differences | `remnant_mass_calculator.py:136-164`. **Used only to build E here** — this model has no flux drive |
| `nu`, `X1`, `base` | `q/(1+q)^2`, `q/(1+q)`, `X1^(6/5)` | derived; `base` is the Newtonian chirp anchor, so the model is **not PN-free** |
| `T_ANCHOR` | time-map anchor, **-100.0 M** [ppBHPT time axis, so m1 units] | `fit_scaling_PN_opt_creative.T_ANCHOR` |
| `t0` | per-case time offset, a NUISANCE | coarse scan over [-250, +150] M, then `minimize_scalar` |
| `phi0` | overall phase, **analytic** | `fit_scaling_gwr_energy_global.err_at_t0` |
| mismatch | `(n1 + n2 - 2\|Z\|)/(2 n1)`, relative L2 on the overlap; 50.0 if NR coverage < `MIN_COVERAGE = 0.88` | same |
| grid | **12 even-nu q in [3,8] x 8 chi1 = 96 points**, chi1 = 0 INCLUDED | `TRAIN_Q`, `CHIS` in `fit_pn_anchored_spin` |

**The waveform source differs from the nonspinning models'** (BHPTNRSur1dq1e4). At
chi1 = 0 the two surrogates differ from each other by a mismatch of 8.7e-05 to 1.2e-04 --
~25% of the residual being fitted -- so coefficients are NOT portable between them.  That
is why all 8 are refitted rather than the nonspinning 6 being frozen, and why chi1 = 0 is
in the objective (with the nonspinning coefficients free, that column constrains them).

## Constants and their sources (added 2026-09-26)

Code references are to `spin_dep/fit_energy_anchored_spin.py` unless another file is
named (line numbers as of 2026-09-26).  **Units:** ppBHPT times (`t_bhpt`, the input
waveform's axis) are in units of the PRIMARY mass m1; NR times (`t_nr`) are in units of
the TOTAL mass M.  Where the table above writes "M" for a ppBHPT time (the h_pp time
grid, `T_ANCHOR`), read m1.

**Derived (not fitted)**

* **`base = X1**1.2` = X1^(6/5).**  Leading-order time-map and amplitude scaling from
  matching the Newtonian chirp rate and Newtonian (2,2) amplitude between ppBHPT
  (mass ratio m2/m1 = 1/q) and the comparable-mass binary (nu):
  beta_LO = (nu * q)^(3/5) = X1^(6/5).  Derivation: `../alpha_beta_pn_scaling_note_revised.pdf`
  Eqs. (7)-(12) and (13)-(16).
* **`nu = q/(1+q)**2`, `X1 = q/(1+q)`, `eps = 1/q`.**  Mass-ratio variables; `W` is `nu` or
  `eps` according to the `--weight` flag (`params_at`, line ~79).

**Chosen settings (by hand, not fitted)**

* **`T_ANCHOR = -100.0`** (`../fit_scaling_PN_opt_creative.py:24`, read via
  `fit_scaling_gw_remnant_energy.T_ANCHOR`): ppBHPT time (m1 units) where the running
  integral of beta is zeroed (`mismatch`, line ~96).  Only sets where t0 is counted from.
* **Window `[-5000.1, +100.0]`** (NR time, M units): `NR_T_START`, `NR_T_END`,
  `../fit_scaling_PN_opt_creative.py:21-22`, applied in `fit_pn_anchored_spin.build_case`.
* **`MIN_COVERAGE = 0.88`** (`../fit_scaling_PN_opt_creative_q_dep.py:34`): below 88% NR
  coverage the case scores 50.0 (`../fit_scaling_gwr_energy_global.err_at_t0`).  A
  non-positive beta or non-monotone time map also scores 50.0 (`mismatch`).
* **t0 search** (`mismatch`, lines ~101-105): first pass an 81-point scan over [-250, +150]
  M (NR time offset), then a bounded 1-D minimisation within +- 60 M of the current seed;
  later passes start from the previous t0.
* **Safety bounds (not in the Model block above):** `alpha_PP` is clipped to [0.05, 2.5]
  and `beta_PP` to [0.2, 1.6] (`params_at`, lines 80-81).  They only stop the optimiser
  wandering into nonsense; at the fitted values they are not active (alpha_PP, beta_PP
  are near X1^1.2 = 0.71-0.87 on this grid).
* **Grid:** `TRAIN_Q`, `CHIS` from `fit_pn_anchored_spin.py` (lines 122-124): the 12 q
  values 3.0 ... 8.0 of the nonspinning fit, chi1 = -0.6 ... +0.8 in steps of 0.2.
* **Optimiser:** Powell (line 171), xtol 1e-4, ftol 1e-6, `--maxiter` default 400,
  `--budget` default 100 minutes (lines 114-115).  Objective = MEAN mismatch over the
  96 cases.  Starting point (`seed_theta`, line 69): the nonspinning 6 from
  `../gwr_energy_anchored_results/coeffs.json` plus s1 = s2 = 0.
* **Strides:** this script scores every case on the FULL-resolution arrays (`mismatch`
  uses `case["t_bhpt"]`, `case["t_nr"]`), so `SRC_STRIDE = 6`, `NR_STRIDE = 10` in the
  table above are NOT used by this model (they are used by the pn_anchored and
  fluxanchored spin fits).

**External inputs**

* **`E(t)`**: `gw_remnant` `RemnantMassCalculator` flux, integrated by
  `../physics_common.energy_drive` after the join-window despike, reached through
  `fit_fluxanchored_spin.build_case` (`fit_energy_anchored_spin.py:131`).  **E is per unit
  TOTAL mass M**, requested explicitly (`e_units="M"`, `fit_fluxanchored_spin.py:107`) and
  independent of the nonspinning flag `E_TOTAL_MASS_UNITS` (now False,
  `../fit_scaling_gw_remnant_energy.py:100`).  See "Anchors", A3.  *Dated history
  (corrected 2026-09-27):* before the 2026-09-24 fix E was on the ppBHPT (m1) axis, i.e.
  E/m1, so the body's coefficients (A0, P0, P1, s1, s2) were fitted to E/m1.  From
  2026-09-24 it was multiplied by X1 through `E_TOTAL_MASS_UNITS = True` (`UNITS_AUDIT.md`
  B1); since 2026-09-26 that flag is False again and the spin models request M units
  themselves.
* **Waveforms:** BHPTNRSur2dq1e3 (`calibrated=False`, raw ppBHPT) and NRHybSur3dq8 at
  `dt = 0.1`, `f_low = 5e-3` (`fit_pn_anchored_spin.py:137`).

**Fitted coefficients** (8, all fitted on the 96-point grid; `NAMES`, line 63)

| name | multiplies |
|:---|:---|
| `c0`, `c1` | nu and nu^2 in alpha_PP = base*(1 + c0 nu + c1 nu^2) |
| `A0` | E(t)/nu in alpha (alpha_E = A0/nu) |
| `b` | nu in beta_PP = base*(1 + b nu) |
| `P0`, `P1` | 1 and nu in the E-coefficient P of beta |
| `s1`, `s2` | W*chi1 and W*chi1^2 in P |

Stored in `energy_anchored_spin_results/coeffs_{nu,eps}.json` (`theta`).  **Note:** the
W = nu row in the Coefficients block below is PRE-correction and is no longer in any file
(the rerun overwrote it, see the correction section); `coeffs_nu.json` now holds the
corrected fit: c0 = +0.4031, c1 = -1.5846, A0 = -1.1557, b = +0.0116, P0 = -6.2616,
P1 = +40.0259, s1 = +35.2646, s2 = +1.1580 (median 1.5703e-03).  The W = 1/q row below
still matches `coeffs_eps.json` (that arm was not rerun).

**Where the other numbers come from**

* Mismatch = (n1 + n2 - 2|Z|)/(2 n1) on the NR window, phase solved exactly, t0
  optimised per case, full resolution.  "median", "max" = over the 96 cases; per-chi1 rows
  = median over the 12 q at that chi1.  Pre-correction values: the job logs
  `easpin.3487390/1/2.log`; corrected values: `coeffs_{nu,eps}.json` `mismatches`.
* Comparison rows: pn_anchored + spin from `pn_anchored_spin_results/coeffs_spin.json`;
  the chi1 split from `coeffs_spin_split.json`; fluxanchored + spin from
  `fluxanchored_spin_results/` (see `scaling_fluxanchored_spin.md`).  The fluxanchored
  per-chi1 row here (4.11e-3 at -0.6, 9.93e-4 at -0.2) is from its first converged run
  (job 3483948, log `faspin.3483948.log`); its median 1.4110e-03 is from the 2026-09-10
  refit (job 3485184), whose row reads 4.12e-03 and 9.95e-04.  `REF` in this script
  (line 66) hard-codes 7.686e-04 and 1.4110e-03 for printing only.
* Nonspinning margin "4.70e-04 vs 6.26e-04": the nonspinning fluxanchored median
  4.694e-04 (`../scaling_gwr_energy_fluxanchored.md`) and energy_anchored median
  6.2551e-04 (`../scaling_gwr_energy_anchored.md` line 68), rounded.
* "8.7e-05 to 1.2e-04": measured mismatch between BHPTNRSur1dq1e4 and BHPTNRSur2dq1e3 at
  chi1 = 0 (`../history.md`, 2026-09-09 entry, line 170).
* "-0.974": correlation between the fitted alpha_PP and alpha_E residuals across q
  (n = 64), recorded in `../scaling_gwr_energy_stiff.md` line 266 (not in
  `scaling_gwr_energy_anchored.md` as "the parent's own notes" suggests).
* "0.17% to 3.35%": the within-case spread over x of the measured spin ratio R_beta,
  step-1 measurement, `spin_extension_plan.md` sec 8 table.
* "k = 0.95 +- 0.24", "1.45 +- 0.37": fitted exponents of |R_beta - 1| ~ V^k over q at fixed
  chi1, x <= 0.09 (see `scaling_pn_anchored_spin.md`).
* "0.7%", "0.66-2.4%": |R_alpha - 1| maximum and within-case x-spread from the same
  step-1 measurement (`spin_extension_plan.md` sec 7).

## Coefficients

```
names   c0       c1       A0       b        P0       P1       s1       s2

W = nu   (jobs 3487390 + 3487391, CONVERGED, 1227 + 164 evals)
      +0.3449 -1.2165 -0.9363 +0.0113 -5.0487 +32.1994 +27.5702 +1.0729

W = 1/q  (job 3487392, stopped on budget at 1796 evals)
      +0.3316 -1.1386 -0.9512 +0.0109 -4.6798 +30.2163 +16.5436 +0.8328
```

## Results

| | coefficients | median | max |
|:---|---:|---:|---:|
| pn_anchored + spin | 13 (9 frozen + 4) | **7.686e-04** | 9.64e-03 |
| pn_anchored + chi1 split | 17 (9 + 8) | **7.245e-04** | 1.01e-02 |
| fluxanchored + spin, W = nu | 9, all fitted | 1.4110e-03 | 1.52e-02 |
| **energy_anchored + spin, W = nu** | **8, all fitted** | 1.7311e-03 | 1.5303e-02 |
| energy_anchored + spin, W = 1/q | 8, all fitted | 1.9934e-03 | 1.8645e-02 |

Per chi1:

| chi1 | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| pn_anchored | 1.32e-3 | 6.11e-4 | 4.81e-4 | 5.54e-4 | 5.88e-4 | 9.29e-4 | 1.17e-3 | 3.44e-3 |
| fluxanchored | 4.11e-3 | 2.37e-3 | 9.93e-4 | 6.14e-4 | 7.79e-4 | 1.18e-3 | 1.84e-3 | 4.23e-3 |
| **energy_anchored, nu** | 4.34e-03 | 2.64e-03 | 1.25e-03 | 8.41e-04 | 1.00e-03 | 1.42e-03 | 2.26e-03 | 4.41e-03 |
| energy_anchored, eps | 6.00e-03 | 3.29e-03 | 1.52e-03 | 8.78e-04 | 1.03e-03 | 1.63e-03 | 2.98e-03 | 5.44e-03 |

Full grid (W = nu):

| q | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.000 | 5.34e-03 | 4.83e-03 | 2.74e-03 | 1.45e-03 | 1.45e-03 | 3.33e-03 | 5.35e-03 | 9.67e-03 |
| 3.256 | 3.32e-03 | 2.88e-03 | 1.76e-03 | 1.17e-03 | 1.33e-03 | 2.65e-03 | 3.45e-03 | 6.71e-03 |
| 3.513 | 2.36e-03 | 1.83e-03 | 1.24e-03 | 1.05e-03 | 1.26e-03 | 2.18e-03 | 2.29e-03 | 4.61e-03 |
| 3.897 | 2.06e-03 | 1.21e-03 | 8.98e-04 | 9.82e-04 | 1.19e-03 | 1.71e-03 | 1.42e-03 | 2.75e-03 |
| 4.200 | 2.34e-03 | 1.17e-03 | 8.19e-04 | 9.39e-04 | 1.12e-03 | 1.47e-03 | 1.18e-03 | 2.06e-03 |
| 4.600 | 3.03e-03 | 1.43e-03 | 8.39e-04 | 8.75e-04 | 1.04e-03 | 1.29e-03 | 1.20e-03 | 1.86e-03 |
| 5.000 | 3.87e-03 | 1.86e-03 | 9.30e-04 | 8.07e-04 | 9.65e-04 | 1.21e-03 | 1.41e-03 | 2.17e-03 |
| 5.436 | 4.81e-03 | 2.40e-03 | 1.07e-03 | 7.37e-04 | 8.88e-04 | 1.20e-03 | 1.76e-03 | 2.91e-03 |
| 5.949 | 5.83e-03 | 3.04e-03 | 1.26e-03 | 6.69e-04 | 8.03e-04 | 1.23e-03 | 2.23e-03 | 4.21e-03 |
| 6.590 | 6.94e-03 | 3.81e-03 | 1.53e-03 | 6.14e-04 | 7.03e-04 | 1.30e-03 | 2.88e-03 | 6.49e-03 |
| 7.231 | 7.86e-03 | 4.52e-03 | 1.82e-03 | 5.97e-04 | 6.11e-04 | 1.38e-03 | 3.62e-03 | 9.69e-03 |
| 8.000 | 8.81e-03 | 5.32e-03 | 2.22e-03 | 6.28e-04 | 5.14e-04 | 1.48e-03 | 4.72e-03 | 1.53e-02 |

**chi1 = 0 is 8.41e-04**, against fluxanchored's 6.14e-04 and pn_anchored's
5.54e-04 -- so it starts behind before any spin is involved, and every spin column
inherits that.

**The q-structure repeats the fluxanchored pattern.**  At chi1 = -0.6 the error RISES
monotonically with q (5.34e-03 at q=3 to 8.81e-03 at q=8); at chi1 = +0.8 it is U-shaped
with a minimum near q = 4.6.  The chi1 = 0 column falls to a minimum near q = 7.2 and
barely ticks up.  So the spin sector degrades with q on BOTH signs while the nonspinning
part improves.

## WHY it loses — the two causes were predicted before the run

Both were written into the script's docstring in advance, and both are inherited from
the fluxanchored spin attempt because **energy_anchored shares them**:

**1. No remnant anchor.**  pn_anchored's merger branch carries `beta_r(q, chi1)`, the
ratio of the two sides' QNM frequencies, derived from NRSur3dq8Remnant plus the Kerr
(2,2,0) fit -- ringdown spin dependence for ZERO fitted parameters.  A switchless
E-driven model has nothing equivalent.  The cost lands where predicted: **chi1 = -0.6 is
4.34e-03 against pn_anchored's 1.32e-03, 3.3x worse.**

**2. One spin shape term.**  The spin enters only through `P`, so the correction's time
PROFILE is locked to `E(t)` and only its amplitude varies with chi1.  The step-1
measurement had already shown `R_beta`'s spin ratio varies with x by 0.17% to 3.35%, and
the fit confirms it cannot use even the freedom it has: **s1 = +27.57
against s2 = +1.07**, so the chi1^2 term does essentially nothing.

## eps = 1/q loses here too — the THIRD independent model to say so

median 1.9934e-03 against 1.7311e-03 for nu, and worse in every chi1
column.  Running tally, all on this grid:

| model | nu | eps = 1/q |
|:---|---:|---:|
| pn_anchored + spin | **7.686e-04** | 8.151e-04 |
| fluxanchored + spin | **1.4110e-03** | 1.6672e-03 |
| energy_anchored + spin | **1.7311e-03** | 1.9934e-03 |

All three prefer `nu`, against a parameter-free measurement that prefers `eps`
(exponent k = 0.95 +- 0.24 for eps against 1.45 +- 0.37 for nu).  That measurement is
**inspiral-only** (x <= 0.09); the mismatch is **full-window and merger-weighted**.  The
two are answering about different regions, and neither settles the weighting question.
See `scaling_pn_anchored_spin.md`.

## Caveats

* alpha carries no spin, on the measurement that settled it for pn_anchored:
  \|R_alpha - 1\| <= 0.7% against a within-case x-spread of 0.66-2.4%, and R_alpha - 1
  CHANGES SIGN across q at chi1 = +0.3, +0.6, +0.8 -- no power law in any mass-ratio
  variable can do that.  **Both checks are INSPIRAL ONLY**; alpha's merger spin
  dependence has never been measured.
* `alpha_E = A0/nu` is a normalisation convention and is the soft direction of a
  -0.974-correlated degeneracy with `alpha_PP`.  Carried over from the parent unchanged;
  not re-examined here.
* q < 3 is a double extrapolation -- outside the fit's range AND outside
  BHPTNRSur2dq1e3's own domain (q_min = 3, warning only).
* Only the (2,2) mode; aligned primary spin only, secondary a non-spinning test particle.
* The W = eps arm stopped on budget (1796 evals) rather than converging; the W = nu arm
  converged. Since eps loses by 15% that is very unlikely to reverse, but it is not a
  converged comparison.

## Files

- fit: `python fit_energy_anchored_spin.py --weight {nu,eps} --refit --budget 100`
  (`--resume` to continue from the checkpoint)
- results: `energy_anchored_spin_results/coeffs_{nu,eps}.json`
- energy cache: `flux_loss_cache/` (shared with `fit_fluxanchored_spin.py`)
- overlays: `python plot_energy_anchored_spin_overlay.py`
  -> `Agentic_spin_plots/energy_anchored_spin/`
- nonspinning parent: `../scaling_gwr_energy_anchored.md` (6 coefficients, median
  6.2551e-04)
- the one-variable-different sibling: `scaling_fluxanchored_spin.md`
- the model that wins: `scaling_pn_anchored_spin.md`

---

## 2026-09-24 units correction

**Root cause (all models):** the ppBHPT waveform's time axis is in units of the PRIMARY
mass (`calibrated=False` returns the raw training grid); the NR waveform's is in TOTAL
mass.  Anything crossing between them needs an `X1 = q/(1+q)` conversion.
Mechanical check: `python units_selftest.py --q 3 5 8` (all pass).

**What affected this model:** alpha AND beta use raw E (the Eoft axis error).

**Corrected:** median **1.5703e-03 (-9.3% vs 1.7311e-03, better)**, max 1.4221e-02.
Worst region: chi1=+0.8 (column median 5.14e-03).  q=2 row 1.6-4.0e-02 (double
extrapolation).  **The PRE-correction coefficients were LOST** -- overwritten by the rerun,
and `spin_dep/` is not in git.  The published number survives in this md and in history.md.

**Status: within the bar (better).**  Plots REDRAWN: `Agentic_spin_plots/energy_anchored_spin/`;
pre-correction in `energy_anchored_spin_PREFIX_BACKUP_20260924/`.


## 2026-09-27 paper-style figures (after Islam+ arXiv:2407.18319v2 Figs 5, 7, 9a)

`plot_spin_paper_figs.py --models energy_anchored_spin` (job 3474067), in `Agentic_spin_plots/energy_anchored_spin/paper/`:
`energy_anchored_spin_hist_train_vs_heldout` (log10 mathcalE, training 96 vs held-out 270), `energy_anchored_spin_calibration_chi+0.6`
(Re h22 before / after calibration, q = 3, 5, 8), `energy_anchored_spin_mismatch_vs_chi1` (q = 3, 5, 8).
Scored with this model's own mismatch code on the current (09-26 fixed) inputs;
all 96 training values recomputed equal the stored ones (ratio 1.000000).
Training: mean 2.359e-3 / median 1.574e-3 / max 1.42e-2.  Held-out (270 points inside the training range, off the training grid:
3 < q < 8, chi1 in (-0.6, 0.8)): mean 1.699e-3 / median 1.285e-3 / max 6.69e-3 -- lower than training because the held-out set is interior
(no q = 3 / 8 row, no chi1 = -0.6 / +0.8 column), where the errors are largest; interpolation only.
