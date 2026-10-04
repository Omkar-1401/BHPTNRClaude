> **STATUS 2026-09-27 -- current coefficients: CONVERGED full refit (Lonestar6 job 3473739), mean 1.1430e-03 / median 9.6416e-04 / max 6.2482e-03 on the 96-point grid.**  See "2026-09-27 full refit on Lonestar6" at the END of this file (full q-chi1 grid incl. q = 2, 2.5).  The Definitions and Model sections below are current (anchors and x updated 2026-09-27 to match the code); the RESULT / CONVERGENCE / PN-ordering numbers in the body are from 2026-09-17/18, before the 09-24 units correction and the 09-26 fixes (`UNITS_AUDIT.md`, `AUDIT_20260926.md`).


# pn_anc_spin_physical — the spin sector at derivable PN orders, 2 coefficients not 4

**Result: every spin term in this calibration sits at a PN order that can be derived
rather than fitted, and the spin sector costs 2 coefficients instead of 4.** The model
is `pn_anchored + spin` with the four-term polynomial `(s11, s12, s21, s22)` replaced by

    sSO * chi1    * xc**1.5     1.5PN spin-orbit
    sSS * chi1**2 * xc**2       2PN   spin-spin

Total 11 fitted coefficients against the predecessor's 13, **all 11 fitted on the same
96-point (q, chi1) grid** with chi1 = 0 included — nothing transplanted from a
nonspinning 1d fit.

**It was promoted on structural grounds, not on a mismatch win.** The justification is
the PN-ORDERING section below, which is a property of the *form* and holds regardless of
where the optimiser lands. The mismatch numbers are reported in RESULT for completeness
and the honest summary of them is: better on the objective (mean) and much better on the
max, worse on the median.

Files: `fit_pn_anc_spin_physical.py`,
`pn_anc_spin_physical_results/coeffs_physical.json`,
`plot_pn_anc_spin_physical_overlay.py`,
`Agentic_spin_plots/pn_anc_spin_physical/`.
Promoted 2026-09-17 from `fit_pn_anchored_spin_pn15.py` (jobs 3512518 / 3512521).
Predecessor, still on disk: `fit_pn_anchored_spin.py` +
`pn_anchored_spin_results/coeffs_spin.json` (13 coef, median 7.6858e-04).

---

## Definitions — every symbol, and where its value comes from

Read this before the Model block. Nothing in that block is defined only after it.

### What is being calibrated

    h_NR(tau) = alpha(t) * exp(i*phi0) * h_pp(t),        dtau/dt = beta(t)

| symbol | meaning | source |
|:---|:---|:---|
| `h_pp` | the ppBHPT waveform: (2,2) mode of **BHPTNRSur2dq1e3** at (q, chi1). Aligned-spin, q in [3, 1000], abs(chi) <= 0.8, dt = 0.1, t in [-13500, 125] M [ppBHPT time, so m1 units] | `BHPTNRSurrogate` package, via `fit_pn_anchored_spin.build_case` |
| `h_NR` | the calibration TARGET: (2,2) mode of **NRHybSur3dq8**. An NR *proxy*, not NR; valid only q <= 8 | `gwsurrogate`, same function |
| `t` | the ppBHPT time coordinate (units of the primary mass m1) | `case["t_bhpt"]` |
| `tau` | the NR time coordinate (units of the total mass M) | `case["t_nr"]` |
| `alpha(t)` | amplitude scaling, dimensionless, > 0 | fitted (see below) |
| `beta(t)` | time-coordinate rate, dimensionless, > 0. Non-positive beta makes tau non-monotone; the objective returns a 50.0 penalty there | fitted |
| `phi0` | a constant phase | solved **analytically**, not fitted, in `fit_pn_anchored_spin._err_at_t0` |
| `t0` | a constant time offset | a per-case **nuisance** parameter, minimised by `scipy.optimize.minimize_scalar` at each (q, chi1). NOT a model coefficient |

### Mass-ratio variables

`q = m1/m2 >= 1` is the mass ratio (an input, not fitted).

| symbol | definition | note |
|:---|:---|:---|
| `X1` | `q/(1+q)` | the larger body's mass fraction |
| `nu` | `q/(1+q)**2` | symmetric mass ratio |
| `base` | `X1**1.2` = `X1**(6/5)` | the **Newtonian chirp anchor**. DERIVED, not fitted. Its presence is why this model is not PN-free: it fixes the leading scaling so the fitted coefficients carry only corrections |

### Drive variables — the "where in the coalescence are we" coordinates

| symbol | definition | source |
|:---|:---|:---|
| `x_pp` | `(m1*omega/2)**(2/3)`, measured from the ppBHPT (2,2) phase on the ppBHPT time axis, i.e. on the PRIMARY-mass (m1) clock: `(0.5*abs(d(unwrap(angle(h)))/dt))**(2/3)`, with the phase Savitzky-Golay smoothed over **80 m1 in time** (polyorder 3) **before** differentiation, clipped to [1e-8, 0.6] | `fit_pn_anchored_spin.get_x`; window from `../physics_common.savgol_window` (was a fixed 401 samples = 40 m1 on this surrogate, corrected 2026-09-26) |
| `x` | `(M*Omega)**(2/3)`, the note's PN frequency variable (note Eq. 6), on the TOTAL-mass clock. Obtained from `x_pp` with the leading-order time map `beta_LO = X1**(6/5)` (note Eqs. 8, 12): `x = x_pp * X1**(-4/5)`. The fixed 1PN slope 55/42 is defined for this `x` (note Eq. 57) | `../physics_common.x_scale(q) = X1**-0.8` |
| | **`x ~ v**2`, so `x**n` corresponds to nPN. This single fact is what the whole model turns on.** | |
| `xc` | `X1**(-4/5) * min(x_pp, XCLIP)`, `XCLIP = 0.26` applied to `x_pp` | `model_ab`; `fit_pn_anchored_spin.XCLIP`. The clip stops the `x**1.5` and `x**2` terms running away through merger, where the PN series has no business being evaluated |
| `p_loss` | `0.5*(e_hat + j_hat)` — the mean of the normalised radiated energy and angular momentum, **each divided by its own final value** so both run 0 -> 1 [doc-audit 2026-09-26: the code does something else -- `../fit_scaling_wf_nu_q_dep.wf_loss_coordinates` measures E and J from the \|h\| peak and divides by their size at `T_REF = -100` (ppBHPT time, `../fit_scaling_PN_opt_creative.py:23`), so p_loss = -1 at t = -100, 0 at the peak, and positive after it; hence the fitted switch centre p00 + p01*nu is about 4.5-5.0, not inside 0..1] | `fit_pn_anchored_spin.build_case`. Used ONLY as the switch argument: it is a monotone progress variable, not an energy |

### Spin variable

`chi1` is the dimensionless spin of the **larger** body, projected on the orbital
angular momentum. The smaller body is non-spinning. Grid:
`fit_pn_anchored_spin.CHIS` = `[-0.6, -0.4, -0.2, 0.0, +0.2, +0.4, +0.6, +0.8]`.

### The merger-ringdown anchor — zero fitted parameters

    beta_r_phys(q, chi1) = w220(chi1) * M_f / w220(chi_f)          (current, 2026-09-26)

    Full explanation: "Anchors (derived, not fitted)" A2, after the Model section.

    history: until 2026-09-24 ( w220(chi1)*(1+q)/q ) / ( w220(chi_f)/M_f ) -- WRONG, 17-29% too
    large (UNITS_AUDIT.md A1); until 2026-09-26 w220 was the Berti-Cardoso-Will fit, with the pp
    side rescaled to 0.3737*w220_BCW(chi1)/w220_BCW(0) -- replaced by exact Kerr (below).

the ratio of the ppBHPT ringdown frequency to the true remnant's, i.e. the rate at which
ppBHPT time must be stretched so the two ring down together.
Defined in `fit_pn_anchored_spin.beta_r_phys`.

| symbol | definition | source |
|:---|:---|:---|
| `w220(chi)` | real part of the Kerr l = m = 2, n = 0 quasinormal-mode frequency, `M*omega_220`, in units of the hole's own mass, **exact** | Leaver's continued-fraction solution of the Teukolsky equation, via the `qnm` package (L. C. Stein 2019, JOSS 4, 1683), `../physics_common.w220_exact`. For `chi < 0` (hole spinning ANTI-aligned with the orbit) the (2,2) mode is the counter-rotating one, `abs(omega_{2,-2}(abs(chi)))`. w220(0) = 0.37367. Replaced (2026-09-26) the Berti–Cardoso–Will fit 1.5251 - 1.1568*(1-chi)**0.1292 (Phys. Rev. D 73, 064030, Table VIII), which is 1.5% low at chi = 0, fitted for chi >= 0 only, and 3% / 7% low when extrapolated to chi = -0.2 / -0.6. Checked against the measured ppBHPT ringdown frequency at q = 5: chi1 = -0.2 / 0 / 0.2 / 0.6 -> +0.09 / +0.26 / -0.67 / +0.18 % |
| pp side `w220(chi1)` | the ppBHPT waveform's own ringdown: in ppBHPT the ringing hole IS the primary (spin chi1, the secondary is negligible), so its QNM frequency is `w220(chi1)` in units of **m1** -- the same units as the ppBHPT time axis. No mass conversion | `fit_pn_anchored_spin.w_pp_qnm` |
| NR side `w220(chi_f)/M_f` | the real remnant's ringdown frequency in units of the TOTAL mass M, the NR time axis | `fit_pn_anchored_spin.beta_r_phys` |
| `M_f`, `chi_f` | remnant mass (units of M) and spin at `(q, [0,0,chi1], [0,0,0])` | `surfinBH.LoadFits("NRSur3dq8Remnant")` (Varma et al. 2019, Phys. Rev. Lett. 122, 011101, arXiv:1809.09125). **This is the ONLY NR-informed object in the model.** Everything else is derived analytically or fitted to surrogate waveforms |

### Fixed constants — not fitted

| symbol | value | meaning |
|:---|:---|:---|
| `PN_SLOPE` | `55/42` | the derived 1PN coefficient of alpha's inspiral expansion: the (2,2) amplitude is h22 = h22_Newtonian*[1 + (-107/42 + (55/42) nu) x + ...] (`../alpha_beta_pn_scaling_note_revised.pdf` Eq. 55); the -107/42 part is shared with the test-particle limit and cancels in h_NR/h_pp, leaving (55/42) nu x (note Eqs. 56-57). `fit_pn_anchored_spin.py:97` |
| `T_ANCHOR` | `-100.0` (ppBHPT time, m1 units; the "M" label used until 2026-09-27 was wrong) | the time at which tau is pinned to t, fixing the zero of the cumulative integral of beta. A chosen setting, `fit_pn_anchored_spin.py:97` (same as `../fit_scaling_PN_opt_creative.py:24`) |
| `XCLIP` | `0.26` | defined above. A chosen setting, inherited from nonspinning pn_anchored (`fit_pn_anchored_spin.py:97`, `../fit_scaling_pn_anchored.py:72`) |

### The 11 fitted coefficients — `NAMES`, in the order `theta` stores them

| name | what it does |
|:---|:---|
| `b1`, `b2` | nonspinning beta inspiral shape, at 1PN and 2PN |
| `a2` | alpha inspiral shape at 2PN (its 1PN coefficient is FIXED at `PN_SLOPE`) |
| `p00`, `p01` | the switch location, as `p_loss = p00 + p01*nu` |
| `w0` | the switch width, in `p_loss` |
| `r0` | scale on the merger-ringdown beta branch, multiplying `beta_r_phys` |
| `m0`, `m1` | the merger-ringdown alpha branch, as `base*(m0 + m1*nu)` |
| `sSO` | **1.5PN spin-orbit**, multiplying `chi1 * xc**1.5` |
| `sSS` | **2PN spin-spin**, multiplying `chi1**2 * xc**2` |

### Constants and their sources — additions (2026-09-26)

Code references are to `spin_dep/fit_pn_anc_spin_physical.py` unless another file is named
(line numbers as of 2026-09-26; the anchors, with current line numbers, are in "Anchors (derived, not
fitted)" after the Model section).

* **`base = X1**(6/5)`**: leading-order beta and alpha from matching the Newtonian chirp
  rate dx/dt = (64/5) eta x^5 and the Newtonian (2,2) amplitude between ppBHPT (mass ratio
  eta_pp = m2/m1 = 1/q) and the binary (nu), under tau = beta*t: beta_LO =
  (nu/eta_pp)^(3/5) = X1^(6/5).  `../alpha_beta_pn_scaling_note_revised.pdf` Eqs. (7)-(16).
* **`beta_r_phys`, why derived and not fitted.**  Late in the ringdown both waveforms ring
  at their own QNM frequency, so the time map must stretch ppBHPT time by exactly the ratio
  of the two: omega_pp / omega_NR = w220(chi1)/m1 over w220(chi_f)/M_f, and with t in m1 units
  and tau in M units that is w220(chi1) * M_f / w220(chi_f).  Everything in it is either
  exact Kerr perturbation theory or the NR remnant fit; `r0` (fitted) only scales it.
  [History: until 2026-09-26 the pp side was 0.3737*w220_BCW(chi1)/w220_BCW(0) (`W_SCHW`,
  `RESCALE_QNM`, now unused), i.e. the BCW spin dependence with its chi1 = 0 level pinned to
  the exact Schwarzschild value, while the NR side used the raw fit -- two different sources
  on the two sides of one ratio (audit C10).]
* **Fitted values.**  Pre-correction (the body of this file):
  `pn_anc_spin_physical_results/coeffs_physical.json` `theta`, in `NAMES` order:
  b1 = 0.4837, b2 = 7.7538, a2 = -22.6965, p00 = 3.8939, p01 = 5.8380, w0 = 2.0679,
  r0 = 0.6179, m0 = 1.3606, m1 = 0.3281, sSO = 2.8670, sSS = 2.2593.
  After the 2026-09-24 correction: `redo_properly_results/physical_partial.json`
  `rows[0].theta`: b1 = 1.0746, b2 = 6.0772, a2 = -22.6465, p00 = 3.3937, p01 = 11.0447,
  w0 = 2.2136, r0 = 0.6399, m0 = 1.2001, m1 = 1.2042, sSO = 2.8105, sSS = 2.4651.
  Current (2026-09-27 converged, `coeffs_physical.json` `theta`): see the 2026-09-27 section at the end.
* **t0 search** (`mismatch`, lines ~263-267): first an 81-point scan over [-250, +150] M
  (NR time offset), then a bounded 1-D minimisation within +- 60 M of the best point.
  This model scores every case at FULL resolution (no strides).
* **Penalties:** 50.0 if beta <= 0 anywhere or coverage < 0.88 (`MIN_COVERAGE`,
  `../fit_scaling_PN_opt_creative_q_dep.py:34`); 50 + 1000*(backwards step) if tau is
  non-monotone.
* **Window `[-5000.1, +100.0]`** (NR time, M units): `../fit_scaling_PN_opt_creative.py:21-22`.
* **Optimiser:** Powell (line ~589), xtol 1e-4, ftol 1e-6, `--maxiter` default 600,
  `--budget` default 100 minutes (lines 436-437).
* **Seed** (`seed_theta`, line ~271): the shipped nonspinning 9 plus sSO, sSS projected
  from the predecessor's s11..s22 at the reference x = 0.12 (a chosen mid-inspiral value):
  sSO = (s11 x + s21 x^2)/x^1.5, sSS = (s12 x + s22 x^2)/x^2.
* **nu range [0.0988, 0.1875]**: nu = q/(1+q)^2 at q = 8 and q = 3; q = 2 gives 0.2222,
  which is 18.5% above 0.1875.
* **The x values 0.02 ... 0.26 in the PN-ordering table** are sample points printed by
  `pn_structure()` (line ~294); 0.26 is `XCLIP`.  The predecessor column is
  s11*x + s21*x^2 with the shipped s11 = -0.0856, s21 = +13.3217
  (`pn_anchored_spin_results/coeffs_spin.json`); "eff. PN" is d log f / d log x.
  **"2.05"** = |s21|*0.8 / (|s22|*0.64) = 13.3217*0.8/(8.1237*0.64), i.e. the ratio of the
  chi1-linear to the chi1^2 x^2-terms at chi1 = +0.8 (0.8 = chi1, 0.64 = chi1^2).
  **"2.2x more at x = 0.02, 2.3x less at x = 0.26"** compare this model's sSO*x^1.5
  (sSO = 2.867) with the predecessor column: 0.00811/0.00362 = 2.2 at x = 0.02, and
  0.878/0.380 = 2.3 (predecessor larger) at x = 0.26.
* **Result-table sources.**  All values are the relative-L2 waveform mismatch on the 96
  cases; "mean" (the minimised objective), "median", "max" over the 96.  Predecessor row:
  `test_pn_spin_joint2d.py` evaluation 1 on the shipped coefficients.  Joint-2d row:
  `test_joint2d_results/` (logs `pnj2d.3512450/1.log`).  This model:
  `coeffs_physical.json` (`mean`, `median`, `max`, `mismatches`).  fluxanchored and
  energy_anchored rows: their own jsons (see those mds).  The chi1 = +0.8 per-q values and
  the q = 3 row come from the same jsons' per-case entries.  "8.7e-05 to 1.2e-04": the
  measured mismatch between BHPTNRSur1dq1e4 and BHPTNRSur2dq1e3 at chi1 = 0
  (`../history.md` line 170).
* **Correction-section sources:** clock-fix-only numbers from `redo.3535550.log`
  (2 starts); clock + constant from `redo_properly_results/physical_partial.json`.
* **Figure settings** (`plot_pn_anc_spin_physical_overlay.py`): figsize (12, 4), lw 2.2
  for q = 2, xlim (-1000, 80) — chosen plot settings.  The curves are drawn against NR
  time (`t_nr`), so the xlim is in M units.

---

## Model

Every symbol below is defined in the section above.

```python
Sw = sigmoid( (p_loss - (p00 + p01*nu)) / w0 )          # merger switch, in [0, 1]

beta  = (1 - Sw) * base * ( 1 + nu*( b1*xc + b2*xc**2           # nonspinning
                                     + sSO * chi1    * xc**1.5  # 1.5PN spin-orbit
                                     + sSS * chi1**2 * xc**2 )) # 2PN   spin-spin
      +      Sw  * beta_r_phys(q, chi1) * r0

alpha = (1 - Sw) * base * ( 1 + (55/42)*nu*xc + nu*a2*xc**2 )
      +      Sw  * base * ( m0 + m1*nu )

tau(t) = t0 + cumtrapz(beta, t) - interp(T_ANCHOR, t, cumtrapz(beta, t))
```

`alpha` carries **no** spin term: alpha's spin dependence measured out consistent with
zero (`spin_extension_plan.md` sec 7), so adding one would be fitting noise.

`chi1` therefore enters in exactly three places — twice in beta at a derivable PN order,
and once through `beta_r_phys` with **no fitted parameter at all**.

`xc**1.5` is safe: `x = x_pp * X1^(-4/5) > 0` everywhere by construction (x_pp = (m1*omega_22/2)^(2/3) from the ppBHPT phase; see Anchors A4).

### At chi1 = 0 this is NOT nonspinning `pn_anchored` — read this before comparing

The functional FORM reduces correctly: both spin terms carry a factor of `chi1`, and
`beta_r_phys(q, 0)` is the nonspinning expression. But the COEFFICIENT VALUES do not
match, because all 11 are fitted on the 2d grid and **`chi1 = 0` is one of its eight
columns**. So:

* the `chi1 = 0` column is **fitted data, not a null check**;
* the 9 nonspinning coefficients differ substantially from the shipped nonspinning
  fit — `m1` by 12x (3.9838 -> 0.3281), `p01` by 2.6x (2.2404 -> 5.8380), `m0` by
  1.8x, `b1` by 1.4x;
* at `chi1 = 0` the model scores **5.2128e-04** against nonspinning `pn_anchored`'s
  **5.5425e-04** — 6% BETTER, not identical.

That gain is not free accuracy, it is the source-mismatch effect: these 9 were fitted
against BHPTNRSur2dq1e3, the surrogate actually being calibrated, whereas the shipped
nonspinning 9 came from BHPTNRSur1dq1e4. The two differ from each other at `chi1 = 0` by
a mismatch of 8.7e-05 to 1.2e-04, so fitting the source in use wins by construction.

**Design consequence.** This model is therefore not a strict spin EXTENSION of
nonspinning `pn_anchored` — it is a separate calibration sharing its form. The
predecessor `fit_pn_anchored_spin.py` IS a strict extension (its 9 are frozen, so it is
bit-identical at `chi1 = 0`) [superseded 2026-09-27: its p00, p01, w0, r0, m0, m1 are now fitted on
the spin grid, so it is no longer a strict extension either]. If a strict extension is wanted, the 9 must be re-frozen
and only `sSO, sSS` fitted; that would cost the 6% at `chi1 = 0` and has not been tried.
Freeing all 11 was a deliberate choice, to make the three-way spin comparison against
`fluxanchored + spin` (9 fitted on 2d) and `energy_anchored + spin` (8 fitted on 2d)
like-for-like.

---

## Anchors (derived, not fitted)

*Written 2026-09-27 against the current code: `spin_dep/fit_pn_anc_spin_physical.py` (below "P";
`model_ab`, P:228-248), which imports its anchors from `spin_dep/fit_pn_anchored_spin.py` (below "F")
and `physics_common.py`. Equation numbers are those of `../alpha_beta_pn_scaling_note_revised.pdf`
("the note").*

**The two clocks.** The ppBHPT input (BHPTNRSur2dq1e3, `calibrated=False`) has time `t` in units of
the primary mass m1; the NR target (NRHybSur3dq8) has time `τ` in units of the total mass M. The
time map is `dτ/dt = β(t)`, so every anchor for β is an NR-clock rate over a ppBHPT-clock rate. The
bare change of mass unit would be `β = m1/M = X1`.

**Coefficient status.** All 11 coefficients (`NAMES`, P:219) are fitted together on the 96-point
(q, χ1) grid (`coeffs_physical.json`, `all_fitted_on_2d: true`); none is taken from the nonspinning
fit. No anchor below carries a fitted parameter; the table at the end says which coefficient
multiplies which anchor.

### A1. Inspiral anchor: β_LO = α₂,LO = X1^(6/5)

`base = X1^(6/5) = (q/(1+q))^(6/5) = (m1/M)^(6/5)`, P:239 (`base = X1 ** 1.2`).

*What it is.* The leading-order (Newtonian) time map between the ppBHPT and NR clocks and the
leading-order (2,2) amplitude ratio; the base of both β and α.

*Derivation (note Sec. 3).* At Newtonian order `dx_pp/dt = (64/5) η_pp x_pp^5` with `η_pp = m2/m1 = 1/q`,
and `dx_NR/dτ = (64/5) ν x_NR^5` (Eq. 7). With `τ = βt` the orbital phase is shared, so
`Ω_NR = Ω_pp/β`, `x_NR = β^(−2/3) x_pp` (Eq. 8), and the mapped chirp rate is
`(64/5) η_pp β^(5/3) x_NR^5` (Eq. 9). Equality with the binary's requires `η_pp β^(5/3) = ν` (Eq. 10);
with `ν/η_pp = X1²` (Eq. 11), `β_LO = (ν/η_pp)^(3/5) = X1^(6/5)` (Eq. 12). Matching the Newtonian (2,2)
amplitudes `η_pp x_pp` and `ν x_NR` (Eqs. 13-14) gives `α₂,LO = X1² β^(−2/3) = X1^(6/5) = β_LO`
(Eqs. 15-16).

*Why derived.* The exponent follows from the linearity of the Newtonian chirp rate in the mass ratio
and the `β^(−2/3)` scaling of x, so the leading q-dependence is known and not fitted; it tends to 1 as
q → ∞. It is spin-independent: spin enters the chirp rate first at 1.5PN (A5).

*Use in this model* (P:242-247):
- `β_insp = X1^(6/5) [1 + ν(b1 xc + b2 xc² + sSO χ1 xc^1.5 + sSS χ1² xc²)]` — the note's Eq. 58 with a
  ν-suppressed residual in the PN-like basis of Eq. 62 (x, x^(3/2), x² at order ν), so
  `β_insp → X1^(6/5)` as ν → 0 (Eq. 61).
- `α_insp = X1^(6/5) [1 + (55/42) ν xc + a2 ν xc²]` — note Eq. 57 (A3); no spin term.
- `α_mr = X1^(6/5) (m0 + m1 ν)` — `X1^(6/5)` is only a normalisation carried over from the inspiral;
  `m0, m1` set the merger level. The note derives `X1^(6/5)` for the inspiral and says merger–ringdown
  needs its own anchor (Sec. 8, after Eq. 62).

### A2. Ringdown anchor: β_r(q, χ1) = w220(χ1) · M_f / w220(χ_f)

Code: `S.beta_r_phys(q, chi1)` (P:245), defined at F:208-227 as `w_pp_qnm(chi1) * mf / w220(chif)`, with
`w_pp_qnm = w220 = physics_common.w220_exact` (F:90-95, 203-205).
- `w220(χ)` = `M ω_220`, real part of the Kerr l = m = 2, n = 0 quasinormal-mode frequency in units of
  the ringing hole's own mass, **exact**: Leaver's continued fraction via the `qnm` package
  (L. C. Stein 2019, JOSS 4, 1683), `physics_common.py:54-72`. For χ < 0 the (2,2) mode is the
  counter-rotating branch `|ω_{2,−2}(|χ|)|` (`physics_common.py:59`) — used on the ppBHPT side for
  χ1 < 0 and on the NR side where χ_f < 0 (large q with χ1 = −0.6; χ_f = −0.125 at q = 8).
- ppBHPT side: the ringing hole is the primary, mass m1 and spin χ1 (the secondary's contribution is
  neglected), so `ω_pp = w220(χ1)/m1`, i.e. `w220(χ1)` per unit of the ppBHPT clock — no mass conversion.
- NR side: the remnant, `ω_NR = w220(χ_f)/(M_f M)`, i.e. `w220(χ_f)/M_f` per unit of the NR clock;
  `M_f` (units of M) and `χ_f` from surfinBH `NRSur3dq8Remnant` (Varma et al. 2019, PRL 122, 011101) at
  `(q, [0,0,χ1], [0,0,0])` (F:60, 225-226). This is the model's only NR-informed input.

*Why it is the frequency ratio.* The mapped ppBHPT waveform has frequency `(dφ_pp/dt)/(dτ/dt) = ω_pp/β`;
for the ringdowns to stay in phase it must equal `ω_NR`, so at ringdown
`β = dτ_NR/dt_pp = ω_pp/ω_NR = w220(χ1) M_f / w220(χ_f)`. Its spin dependence costs no parameter.

*No free parameter; `r0` multiplies it.* `β_MR = r0 · β_r(q, χ1)` (P:245), `r0` fitted (0.5556 in the
current `coeffs_physical.json`). At χ1 = 0 it reduces to `0.373672 · M_f / w220(χ_f)` (w220(0) = 0.373672).
Values (evaluated 2026-09-27 with the functions above): q = 3: 0.748 / 0.763 / 0.859 at
χ1 = −0.6 / 0 / +0.8; q = 8: 0.876 / 0.879 / 0.906.

*What it replaces (2026-09-26).* Until 2026-09-26 w220 was the Berti–Cardoso–Will prograde fit
`f1 + f2(1−χ)^f3` (1.5251, −1.1568, 0.1292; PRD 73, 064030, Table VIII), which (a) is 1.5% low at χ = 0
(0.3683 vs 0.37367), (b) was used on the ppBHPT side rescaled to 0.3737 (F `W_SCHW`, `RESCALE_QNM`,
F:86-87, now unused) but raw on the NR side, and (c) was extrapolated to χ < 0, where it is 3% low at
χ = −0.2 and 7% low at χ = −0.6. The exact values agree with the measured ppBHPT ringdown frequency at
q = 5 to +0.09 / +0.26 / −0.67 / +0.18 % for χ1 = −0.2 / 0 / 0.2 / 0.6 (`physics_common.py:6-15`). Before
2026-09-24 the ppBHPT side also carried a spurious `(1+q)/q` (`UNITS_AUDIT.md` A1).

### A3. Fixed 1PN amplitude slope 55/42

`S.PN_SLOPE = 55/42` (F:97; used at P:246), the ν x coefficient of `α_insp`. The 1PN (2,2) amplitude is
`h22 = h22^N [1 + (−107/42 + (55/42)ν) x + 2π x^(3/2) + …]` (note Eq. 55); the `−107/42` and `2π x^(3/2)`
terms are present in the test-particle limit and so already in the ppBHPT waveform, leaving
`1 + (55/42) ν x` in the ratio at fixed x (Eq. 56), the ν x term of Eq. 57. Not fitted; defined for the
note's x (A4).

### A4. The note's x: x = x_pp · X1^(−4/5), and XCLIP

`S.get_x` (F:195-200) returns `x_pp = (ω_22/2)^(2/3) = (m1 Ω)^(2/3)` on the m1 clock (phase
Savitzky–Golay smoothed over 80 m1 of time via `physics_common.savgol_window`, polyorder 3; clipped to
[1e-8, 0.6]). The note's PN variable is `x = (M Ω)^(2/3)` (Eq. 6); Eq. 8 with `β_LO = X1^(6/5)` (Eq. 12)
gives `x = X1^(−4/5) x_pp`, `physics_common.x_scale` (`physics_common.py:88-90`), applied at P:240 as
`xc = X1^(−4/5) · min(x_pp, XCLIP)`. This is the binary's x at the same phase under the leading-order map,
not the bare unit change `X1^(−2/3)` (they differ by `X1^(−2/15)`, 3.9% at q = 3, 1.6% at q = 8). The PN
orders of A5 are orders in this x. `XCLIP = 0.26` (F:97) is a chosen cap on x_pp, not an anchor (0.327 to
0.286 on the note's x for q = 3 to 8).

### A5. Spin terms at their PN orders: χ1 x^(3/2) (spin-orbit) and χ1² x² (spin-spin)

`sSO · χ1 · xc^1.5` and `sSS · χ1² · xc²` inside the ν-bracket of `β_insp` (P:243-244). What is fixed
here is the **power of x attached to each power of χ1**, not the coefficient: in the PN expansion of an
aligned-spin inspiral the leading spin-orbit coupling (linear in the spin) enters at 1.5PN, relative
order x^(3/2), and the leading spin-spin coupling (quadratic in the spin) at 2PN, relative order x²
(Kidder 1995, PRD 52, 821; Blanchet 2014, Living Rev. Relativ. 17, 2). These orders are not in the note,
which treats non-spinning binaries; they fill the x^(3/2) slot of its PN-like basis (Eq. 62) with the
spin-orbit term. They sit inside the ν-bracket because the test-particle spin effects are already in the
ppBHPT waveform (computed on a Kerr background of spin χ1), so only the finite-mass corrections to them
are calibrated and they vanish as ν → 0 (Eq. 61). `sSO`, `sSS` are fitted (2.7485, 1.4577 in the current
`coeffs_physical.json`). The predecessor's four-term form placed χ1 and χ1² at x and x² and had no
x^(3/2) term ("PN ordering" below).

### A6. T_ANCHOR = −100 (a convention)

`S.T_ANCHOR = −100.0` (F:97), ppBHPT time in m1 units. `mismatch` sets `ts = ∫β dt − [∫β dt]_{t = T_ANCHOR}`
(P:270-271) and `τ = t0 + ts` (F `_err_at_t0`, F:246), so `τ(T_ANCHOR) = t0`, the per-case nuisance offset
optimised with the mismatch (scan over [−250, +150] M, then ±60 M, P:276-280). It fixes only what t0
means; it is not a physical anchor.

### Which fitted coefficient multiplies which anchor

| anchor | enters as | coefficients attached (all fitted on the spin grid) |
|:--|:--|:--|
| A1 `X1^(6/5)` | base of `β_insp` | `b1`, `b2` (ν xc, ν xc²); `sSO`, `sSS` (ν χ1 xc^1.5, ν χ1² xc²) |
| A1 `X1^(6/5)` | base of `α_insp` | `a2` (ν xc²) |
| A1 `X1^(6/5)` | base of `α_mr` | `m0`, `m1` |
| A2 `β_r(q, χ1)` | `β_MR = r0 · β_r` | `r0` |
| A3 `55/42`, A4 `X1^(−4/5)`, A5 powers of x | fixed | none |
| switch (not an anchor) | `Sw = sigmoid((p_loss − (p00 + p01 ν))/w0)` | `p00`, `p01`, `w0` |

---

## PN ordering — why this replaced the polynomial spin form

This is the reason the model exists, and it is a property of the form, independent of
the fitted values.

The predecessor spent four coefficients on

    nu * [ (s11*chi1 + s12*chi1**2) * x  +  (s21*chi1 + s22*chi1**2) * x**2 ]

which places `chi1` at 1PN and 2PN, and `chi1**2` at 1PN and 2PN:

| term | order it sits at | verdict |
|:---|:---|:---|
| `s11*chi1*x` | chi1 at **1PN** | WRONG — spin-orbit enters at 1.5PN |
| `s12*chi1**2*x` | chi1^2 at **1PN** | WRONG — spin-spin enters at 2PN |
| `s21*chi1*x**2` | chi1 at **2PN** | legitimate, but *subleading* spin-orbit |
| `s22*chi1**2*x**2` | chi1^2 at **2PN** | correct |

**There is no `x**1.5` term anywhere in it, so the leading spin-orbit effect had nowhere
to live.**

### Measured consequences in the predecessor

With its shipped values `s11 = -0.0856, s12 = -0.2769, s21 = +13.3217, s22 = +8.1237`,
taking the local effective PN order as `d log f / d log x` on the linear-in-chi1 piece of
beta's nu-bracket:

| x | predecessor `s11*x + s21*x**2` | eff. PN | this model `sSO*x**1.5` | eff. PN |
|---:|---:|---:|---:|---:|
| 0.02 | +0.00362 | **2.31** | — | 1.50 |
| 0.06 | +0.04282 | **2.12** | — | 1.50 |
| 0.12 | +0.18156 | **2.06** | — | 1.50 |
| 0.20 | +0.51574 | **2.03** | — | 1.50 |
| 0.26 | +0.87828 | **2.03** | — | 1.50 |

Never below 2.03. Because `s11` is negligible against `s21`, the x term contributes
nothing and the piece is pure `x**2` — leading spin-orbit was carried at 2PN, one half
order too high, across the entire band.

**The knock-on effect: the predecessor cannot separate the two spin orders at all.** PN
requires spin-orbit / spin-spin to grow as `x**-0.5` toward low frequency. Instead the
predecessor's ratio flattens to the constant `abs(s21)*0.8 / abs(s22)*0.64 = 2.05` — it
treats two different PN orders as the same order and splits them by a fixed factor.

The live numbers for both models are printed by

    python fit_pn_anc_spin_physical.py --report

which calls `pn_structure()` and needs no waveform evaluation.

### Where the spin correction moves

Relative to the predecessor, the PN-ordered form puts roughly **2.2x more** spin
correction at `x = 0.02` and **2.3x less** at `x = 0.26` — a factor ~5 redistribution
across the band, out of the merger plateau (where the `x**2` basis has the most leverage)
and back into the inspiral. That is the substantive change, and it is what the
`..._vs_predecessor_diff.pdf` figure shows directly.

### Secondary benefit: the comparison becomes fair

The spin sector is now **parameter-matched** to `fit_fluxanchored_spin.py` and
`fit_energy_anchored_spin.py`, which carry 2 spin coefficients each. The predecessor's 4
made the three-way spin ranking unfair.

---

## What is NOT changed from the predecessor

The drive (`x`, `p_loss`), the switch, `beta_r_phys`, alpha's spin-blindness, the
96-point grid, and the objective. Only the two spin terms in beta differ, plus the fact
that all 11 coefficients are fitted on the 2d grid rather than 9 being transplanted.

---

## Objective and grid

The objective is the **MEAN** over the grid of the per-case waveform mismatch

    (n1 + n2 - 2*abs(Z)) / (2*n1)

on the window `t in [-5000.1, +100.0] M` with minimum coverage 0.88, maximised
analytically over `phi0` and numerically over `t0`.

**Median and max are reported as diagnostics and are NOT minimised.** A run whose median
rises while its mean falls is behaving exactly as specified; this has caused
misreadings before and is worth stating in the model's own documentation.

Grid: `TRAIN_Q` (12 values in [3, 8]) x `CHIS` (8 values in [-0.6, +0.8]) = 96 points.

**q = 2 is NOT in the grid**, and reading it is a *double* extrapolation: outside the
fitted range `nu in [0.0988, 0.1875]` (from `TRAIN_Q` in [3, 8]) — q = 2 is
`nu = 0.2222`, **18.5% above its top** — AND outside BHPTNRSur2dq1e3's own `q_min = 3`
(the surrogate only warns and evaluates). The q = 2 curve appears on the overlays
because the nonspinning overlays carry it, and its legend entry says so. See
`history.md`, open items.

---

## RESULT

*Numbers below are from `pn_anc_spin_physical_results/coeffs_physical.json` as written
by job 3512764 (`--promote`) from pass 2. A 6h continuation (job 3512777) is running;
see CONVERGENCE below for why it is not expected to move them.*

Comparison on the identical 96-point grid. The predecessor's row is not a cross-run
number: it is `test_pn_spin_joint2d.py`'s eval 1, i.e. the shipped coefficients scored
by the same code on the same grid.

| model | coef | **mean** (objective) | median | max |
|:---|---:|---:|---:|---:|
| `pn_anchored + spin` (predecessor, 9 of 13 transplanted from 1d) | 13 | 1.2808e-03 | 7.6858e-04 | 9.6416e-03 |
| same, all 13 refitted on 2d (`test_pn_spin_joint2d.py`, pass 1) | 13 | 1.0739e-03 | 7.3963e-04 | 6.6085e-03 |
| **`pn_anc_spin_physical`** (pass 2, 3217 evals total) | **11** | **1.1408e-03** | 8.4850e-04 | **5.6069e-03** |
| `fluxanchored + spin` | 9 | — | 1.4110e-03 | — |
| `energy_anchored + spin` | 8 | — | 1.7311e-03 | — |

Against the predecessor: **−11% on the objective and −42% on the max, with 2 fewer
coefficients, at the cost of +10% on the median.**

The `chi1 = +0.8` column, the one the predecessor handled worst, across q = 3/5/8:

| | q=3 | q=5 | q=8 | spread |
|:---|---:|---:|---:|---:|
| predecessor, 13 coef | 9.6416e-03 | 2.5301e-03 | 7.0317e-03 | 3.81x |
| same, all 13 refitted on 2d | 6.61e-03 | 2.22e-03 | 4.45e-03 | 2.98x |
| **this model, 11 coef** | **5.61e-03** | 2.26e-03 | **5.38e-03** | **2.42x** |

So the PN re-ordering flattens the q-U-shape more than freeing two extra coefficients
does, which is the result the missing-order hypothesis predicted.

Per-column medians: 6 of 8 improve (0.88x–0.95x); two regress by 3–6% (`chi1 = -0.4` at
1.03x, `chi1 = +0.2` at 1.06x). The regression is concentrated in the **q = 3 row**,
which sits at 1.04e-03 to 1.34e-03 for `chi1 <= +0.4` against the joint-2d refit's
6.97e-04 to 9.78e-04. q = 3 is also the bottom edge of the training grid, so "the PN
form is worse at q = 3" and "q = 3 is the grid edge" are not yet separated.

## CONVERGENCE

**The fit is converged.** Pass 2 resumed from pass 1 and spent 1605 further evaluations
to move the objective by **0.21%**:

| pass | evals | mean |
|:---|---:|---:|
| 1 | 1612 | 1.1432e-03 |
| 2 | +1605 | 1.1408e-03 |

The two spin coefficients are stable across that: `sSO` 2.883 -> 2.867 (0.6%),
`sSS` 1.917 -> 2.259. `sSS` is the looser of the two, which is expected — it multiplies
`chi1**2 * xc**2`, the smallest term in the model.

An earlier reading of the pass-1 trace as "still descending" (1.2166e-03 at 35 min ->
1.1499e-03 at 61 min) did not survive pass 2 and should not be repeated.

## Provenance caveat on the seed

The seed — the predecessor's coefficients projected
onto the PN basis at `x = 0.12`, implemented in `seed_theta()` — evaluates at mean
**6.06e-03**, 4.7x *worse* than the predecessor itself. So that projection is a poor
approximation, which is independent evidence that `x**2` was never a clean stand-in for
`x**1.5`; but it also means pass 1 spent much of its budget climbing back down, and had
965 evals against the joint-2d run's 1620. The pass-1 comparison above is therefore not
yet a fair structural one.

Because the seed is poor, `--refit` **auto-resumes** from stored coefficients whenever
they exist; `--fresh` is the explicit opt-out. (The predecessor's pass-2 script resumed
only if `$RES` happened to be set at submit time, which would have silently thrown away
pass 1.)

---

## Figures

`Agentic_spin_plots/pn_anc_spin_physical/`, from
`plot_pn_anc_spin_physical_overlay.py`:

| file | content |
|:---|:---|
| `pn_anc_spin_physical_alpha_beta_overlay_chi{m0.6,m0.2,p0.2,p0.6}.pdf/png` | one 1x2 figure per chi1 (alpha left, beta right), one curve per q in {2,3,5,8} |
| `pn_anc_spin_physical_alpha_beta_overlay_all.pdf/png` | 4x2 contact sheet, rows = chi1, so the spin trend reads down the column |
| `pn_anc_spin_physical_vs_predecessor_diff.pdf/png` | 4x2 sheet of `Delta alpha/alpha` and `Delta beta/beta` in percent against the 13-coefficient predecessor — **where the PN re-ordering actually shows up** |

Format is verbatim that of the nonspinning overlays and of
`spin_alpha_beta_overlay.py`: figsize (12, 4) per row, q coloured
tab:red/orange/green/blue, lw 2.2 for the q = 2 extrapolation, `xlim (-1000, 80)`, grid
on, default rcParams, mathtext labels, PDF + PNG. So these overlay directly onto the
predecessor's figures.

The `E` quoted in each legend entry is the model's **actual** grid mismatch at that
(q, chi1), obtained from the same two-stage t0 search the fit uses — not a replotted
number.

---

## Open

* **The q-U-shape at fixed chi1 is still unexplained.** Drive quarantine (0.97x), a
  chi1 < 0 / chi1 >= 0 domain split (8% for 2x the coefficients), chi_eff (k = 1.94,
  wrong direction) and eps = 1/q weighting (loses in all three models) have each been
  tested and rejected. The joint-2d refit flattens the chi1 = +0.8 column from a 4.5x
  spread across q to 3.0x but does not close it. Whether the PN re-ordering does better
  is the first thing to read off this model's own +0.8 column.
* **The median regression DID survive convergence** (+10% at pass 2, median 8.4850e-04
  against 7.6858e-04) while the mean is −11% and the max −42%. It is a real trade, not
  an optimisation artefact. Whether it is acceptable is a judgement about which
  statistic matters, and it is the user's to make.
* **Is the q = 3 cost a property of the PN form, or of the grid edge?** The regression
  is concentrated in the q = 3 row (1.04e-03 to 1.34e-03 for `chi1 <= +0.4`, against
  the joint-2d refit's 6.97e-04 to 9.78e-04), and q = 3 is simultaneously the bottom of
  `TRAIN_Q` and BHPTNRSur2dq1e3's own `q_min`. The two explanations are not separated.
  A fit on a grid extended to q = 3.0 only as an interior point, or a per-row residual
  comparison against the surrogate's own error floor there, would separate them. **This
  is the highest-value next test, and worth more than further optimisation** — the
  objective moves 0.2% per 1600 evaluations now.
* **`beta_r` frozen at chi1 = 0 has NOT been tested.** That is the mechanism test for
  whether the remnant anchor's spin dependence is what the pn-anchored line's advantage
  rests on. It is *not* what `test_pn_spin_joint2d.py` did — that run keeps `beta_r`'s
  full spin dependence and only removes the 1d->2d transplant confound.
* **A 2.5PN spin-orbit tail term** (`chi1 * xc**2.5`) is the natural next order if a
  third spin coefficient is ever justified. Not tried.

---

## 2026-09-24 units correction

**Root cause (all models):** the ppBHPT waveform's time axis is in units of the PRIMARY
mass (`calibrated=False` returns the raw training grid); the NR waveform's is in TOTAL
mass.  Anything crossing between them needs an `X1 = q/(1+q)` conversion.
Mechanical check: `python units_selftest.py --q 3 5 8` (all pass).

**What affected this model:** the `beta_r` `(1+q)/q` error (see scaling_pn_anchored.md).

**Baseline reproduced exactly** (mean 1.1408e-03, median 8.4850e-04, ratio 1.00002).

**Corrected fit** (`redo_properly.py`, all 11 coefficients refit):
- clock fix only: mean 1.2754e-03, median 1.0563e-03 (2 starts, within 0.3%)
- clock + constant (the version to use): **mean 1.2723e-03 (+11.5%), median 1.0467e-03
  (+23.4%)**, max 5.8134e-03.  **ONE start only** -- the job hit its 2 h limit.
Beats published in 26 of 96 cells.  The q=3 row is the worst (row median 1.99e-03, about
twice the others); from q=3.5 up the rows are 0.80-1.13e-03.  chi1=+0.8 column median
2.84e-03 -- much better than pn_anchored_spin's 6.64e-03, consistent with this model
fitting its switch on spinning data rather than freezing it.  q=2 row: 1.6-4.7e-02 at every
spin, so the low-q problem here is not a spin problem.

The constant fix (0.3683 -> 0.3737) moves the median by <1% here -- its real size.
[Superseded 2026-09-26: the 0.3737-rescaled BCW pp side / raw BCW NR side was replaced by exact
Kerr w220 on both sides; see "Anchors (derived, not fitted)" A2.]

**Status: open, not converged.**  Coefficients: `redo_properly_results/physical_partial.json`.
Plots REDRAWN: `Agentic_spin_plots/pn_anc_spin_physical/` (30 figures); pre-correction in
`pn_anc_spin_physical_PREFIX_BACKUP_20260924/`.


## 2026-09-26 fix round -- coefficients of that date (superseded 2026-09-27 by the converged refit below)

Refit on the fixed code (x on the note's clock, exact Kerr QNMs including the counter-rotating branch for chi1 < 0 -- the BCW fit used before was 3-7% low there -- and the new nonspinning 9).  All 11 coefficients free, started from the new nonspinning 9 plus
projected spin terms (`--fresh`); stopped on its time budget while still improving (mean
1.49e-3 -> 1.22e-3 over the last 30 minutes).  Stored in `coeffs_physical.json`, stamped `model_form`.
Full-resolution mathcalE over the 96-point grid:

| | published | 2026-09-26 (not converged) |
|:--|--:|--:|
| mean | 1.1408e-03 | 1.2214e-03 |
| median | 8.4850e-04 | 9.6134e-04 |
| max | 5.6069e-03 | 7.0730e-03 |

**Within 7% (mean) / 13% (median) of published and still falling when the budget ran out.**
Next step (needs compute): `fit_pn_anc_spin_physical.py --refit --resume` for another budget.
Plots in `Agentic_spin_plots/pn_anc_spin_physical/` predate this refit.


## 2026-09-27 full refit on Lonestar6 -- CONVERGED (current coefficients)

Job 3473739: `--refit` resumed from the 09-26 coefficients (start reproduced, mean 1.2214e-03),
scoring now parallel over the 96 cases (identical to serial, checked), 5138 evals in 9.3 min.
Written to `coeffs_physical.json` (stamped model_form); previous files in
`PHYSICAL_RESULTS_pre_full_3473739.tar.gz`.  Single start (no multistart yet).

| | published (09-18) | 09-26 (budget stop) | 2026-09-27 converged |
|:--|--:|--:|--:|
| mean | 1.1408e-03 | 1.2214e-03 | 1.1430e-03 |
| median | 8.4850e-04 | 9.6134e-04 | 9.6416e-04 |
| max | 5.6069e-03 | 7.0730e-03 | 6.2482e-03 |

Mean back to published (+0.2%); median +13.6%, max +11%.  Coefficients: b1 0.906, b2 3.98,
a2 -15.24, p00 5.32, p01 7.34, w0 2.44, r0 0.556, m0 1.120, m1 1.570, sSO 2.749, sSS 1.458.
The q = 3 row is the weak row (1.07e-3 to 3.72e-3 across chi1).  Against pn_anchored_spin with
switch + merger freed (test, 1.1722e-03 / 8.65e-04 / 6.58e-03): mean -2.5%, median +11.5%, max
-5%; better in 33 of 96 cases (wins at chi1 = +0.8, loses at low/mid spin).

**Below the grid, q = 2 and 2.5 (2026-09-27, `score_lowq_grid.py`, job 3474032; extrapolation --
below TRAIN_Q and below BHPTNRSur2dq1e3's q_min = 3; the surrogate warns "outside bounds").**
q = 3 rescored to ratio 1.00000 for both models.  mathcalE x 1e-3, chi1 = -0.6 ... +0.8:

| model | q | -0.6 | -0.4 | -0.2 | 0 | +0.2 | +0.4 | +0.6 | +0.8 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pn_anc_spin_physical | 2.0 | 14.76 | 26.35 | 29.38 | 36.29 | 29.90 | 26.37 | 13.85 | 20.72 |
| pn_anc_spin_physical | 2.5 | 2.98 | 6.10 | 6.59 | 7.54 | 6.89 | 5.96 | 4.16 | 7.19 |
| pn_anchored_spin switch+merger (test) | 2.0 | 7.78 | 14.84 | 16.42 | 22.44 | 17.69 | 13.69 | 9.05 | 25.97 |
| pn_anchored_spin switch+merger (test) | 2.5 | 1.88 | 3.74 | 4.08 | 4.86 | 4.64 | 3.82 | 4.10 | 11.27 |

Worst at chi1 = 0 for both, where nonspinning pn_anchored (on BHPTNRSur1dq1e4, valid there)
scores 1.16e-3 at q = 2 -- so part of this is likely the 2d surrogate's own extrapolation, not
the model.  Not yet separated.  Values: `test_free_switch_results/lowq_grid.json`.

**Full q-chi1 grid, current coefficients** -- mathcalE x 1e-3 (full resolution, vs NRHybSur3dq8).
q = 2 and 2.5 (bold) are extrapolation: below TRAIN_Q and below BHPTNRSur2dq1e3's q_min = 3.

| q \ chi1 | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **2.000** | 14.76 | 26.35 | 29.38 | 36.29 | 29.90 | 26.37 | 13.85 | 20.72 |
| **2.500** | 2.98 | 6.10 | 6.59 | 7.54 | 6.89 | 5.96 | 4.16 | 7.19 |
| 3.000 | 1.07 | 1.44 | 1.58 | 1.45 | 1.57 | 1.47 | 1.72 | 3.72 |
| 3.256 | 0.98 | 0.81 | 0.81 | 0.74 | 0.88 | 0.97 | 1.38 | 3.16 |
| 3.513 | 1.06 | 0.58 | 0.52 | 0.56 | 0.66 | 0.86 | 1.28 | 2.87 |
| 3.897 | 1.21 | 0.55 | 0.42 | 0.60 | 0.65 | 0.96 | 1.27 | 2.64 |
| 4.200 | 1.27 | 0.61 | 0.44 | 0.66 | 0.71 | 1.08 | 1.31 | 2.50 |
| 4.600 | 1.28 | 0.65 | 0.46 | 0.70 | 0.76 | 1.17 | 1.34 | 2.31 |
| 5.000 | 1.24 | 0.66 | 0.46 | 0.66 | 0.77 | 1.21 | 1.33 | 2.12 |
| 5.436 | 1.17 | 0.64 | 0.45 | 0.56 | 0.73 | 1.16 | 1.27 | 1.96 |
| 5.949 | 1.14 | 0.61 | 0.43 | 0.42 | 0.63 | 1.05 | 1.16 | 1.93 |
| 6.590 | 1.24 | 0.63 | 0.45 | 0.29 | 0.51 | 0.88 | 1.04 | 2.28 |
| 7.231 | 1.51 | 0.74 | 0.55 | 0.23 | 0.41 | 0.73 | 0.98 | 3.37 |
| 8.000 | 2.08 | 1.04 | 0.77 | 0.27 | 0.33 | 0.59 | 1.11 | 6.25 |

Ratio to pn_anchored_spin with switch + merger freed (below 1 = physical better):

| q \ chi1 | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **2.000** | 1.90 | 1.78 | 1.79 | 1.62 | 1.69 | 1.93 | 1.53 | 0.80 |
| **2.500** | 1.58 | 1.63 | 1.62 | 1.55 | 1.49 | 1.56 | 1.01 | 0.64 |
| 3.000 | 1.15 | 1.37 | 1.35 | 1.39 | 1.26 | 1.25 | 0.83 | 0.61 |
| 3.256 | 1.09 | 1.20 | 1.18 | 1.24 | 1.16 | 1.20 | 0.87 | 0.66 |
| 3.513 | 1.11 | 1.10 | 1.04 | 1.14 | 1.13 | 1.25 | 0.97 | 0.74 |
| 3.897 | 1.17 | 1.10 | 1.01 | 1.16 | 1.18 | 1.32 | 1.13 | 0.86 |
| 4.200 | 1.19 | 1.14 | 1.05 | 1.19 | 1.21 | 1.32 | 1.20 | 0.94 |
| 4.600 | 1.18 | 1.17 | 1.08 | 1.21 | 1.22 | 1.28 | 1.22 | 0.99 |
| 5.000 | 1.13 | 1.15 | 1.08 | 1.22 | 1.20 | 1.21 | 1.17 | 0.98 |
| 5.436 | 1.04 | 1.10 | 1.04 | 1.21 | 1.18 | 1.13 | 1.09 | 0.91 |
| 5.949 | 0.94 | 1.00 | 0.98 | 1.19 | 1.15 | 1.04 | 0.98 | 0.84 |
| 6.590 | 0.85 | 0.90 | 0.92 | 1.17 | 1.13 | 0.93 | 0.88 | 0.80 |
| 7.231 | 0.81 | 0.86 | 0.92 | 1.13 | 1.14 | 0.83 | 0.83 | 0.84 |
| 8.000 | 0.82 | 0.88 | 0.96 | 1.20 | 1.14 | 0.77 | 0.91 | 0.95 |

Physical wins chi1 = +0.8 at every q and most spins from q ~ 6 up; it loses chi1 = 0 and +0.2
at every q and the whole q = 3 row except the high-spin end.  Its gains fall on the largest
cells (lower mean and max), its losses on the many small ones (higher median).

**Figures redrawn 2026-09-27 on the current coefficients** (job 3474058): chi_const / q_const
alpha-beta overlays, coarse + dense mismatch heatmaps, in `Agentic_spin_plots/pn_anc_spin_physical/`; previous
versions in `Agentic_spin_plots_pn_pre20260927.tar.gz`.  **New paper-style figures** (after Islam+
arXiv:2407.18319v2 Figs 5, 7, 9a; `plot_spin_paper_figs.py`) in `Agentic_spin_plots/pn_anc_spin_physical/paper/`:
* `pn_anc_spin_physical_hist_train_vs_heldout` -- log10 mathcalE, training (96) vs held-out (270 in-range points
  off the training grid: 3 < q < 8, chi1 in (-0.6, 0.8), 15 x 18).  Held-out scores in
  `paper/heldout_mismatch.json`.
* `pn_anc_spin_physical_calibration_chi+0.6` -- Re h22 before / after calibration, q = 3, 5, 8 at chi1 = +0.6.
* `pn_anc_spin_physical_mismatch_vs_chi1` -- mathcalE vs chi1 for q = 3, 5, 8.

Held-out (270 in-range points): mean 8.51e-4 / median 7.62e-4 / max 2.39e-3 -- below the training numbers because the held-out set is interior (no q = 3 or 8 row, no chi1 = -0.6 or +0.8 column), where this model's errors are largest; it is interpolation, not a harder test.
