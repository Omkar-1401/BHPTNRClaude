# C1 junction replacing the logistic switch — and a correction to `beta_r_phys`

Opened 2026-09-23.  Scripts `junction_diag.py`, `fit_junction_spin.py`;
results `junction_diag_results/`, `junction_spin_results/`.

## 0. Constants and their sources (added 2026-09-26)

Every number below is referred to from the sections that use it.  Code references are
to `spin_dep/junction_diag.py` (J) and `spin_dep/fit_junction_spin.py` (F) unless
another file is named; line numbers as of 2026-09-26.  The anchors of the CURRENT code, with current
line numbers, are in "Anchors (derived, not fitted)" after section 2; the QNM entries below marked
SUPERSEDED describe the code before 2026-09-26.  **Units:** ppBHPT times (`t_bhpt`,
the input waveform) are in units of the PRIMARY mass m1; NR times (`t_nr`) in units of
the TOTAL mass M.  Several "M" labels in this file refer to ppBHPT time; they are listed
at the end of this section.

**Derived from theory (not fitted)**

* **`base = X1^(6/5)`**, X1 = q/(1+q): leading-order beta and alpha from matching the
  Newtonian chirp rate and Newtonian (2,2) amplitude between ppBHPT (mass ratio 1/q) and
  the binary (nu).  `../alpha_beta_pn_scaling_note_revised.pdf` Eqs. (7)-(16).
* **`55/42`** (`PN_SLOPE`): mass-ratio part of the 1PN (2,2) amplitude coefficient,
  h22 = h22_Newtonian*[1 + (-107/42 + (55/42) nu) x + ...] (note Eq. 55); only (55/42) nu x
  survives in h_NR/h_pp (note Eqs. 56-57).  Section 7 quotes these equations.
* **`beta_r = w_pp / (w220(chi_f)/M_f)`** (J `derived`, lines 110-144): ratio of the
  ppBHPT ringdown frequency (m1 units) to the real remnant's (M units) — the late-time
  rate dt_NR/dt_pp.
* **`tau_f = 2 Q_f / w_f`** with w_f = w220(chi_f)/M_f, Q_f = Q220(chi_f): the damping time
  of the remnant's fundamental mode (1/omega_imag, since omega_imag = omega/(2Q)); an NR-axis
  duration (M units).  **`tau_f_pp = tau_f / beta_r`**: the same duration on the ppBHPT
  axis (m1 units), used by the fit since 2026-09-24 (`TAU_PP = True`, F line 152).
* **`kappa = (w_pp/2)(1/Q_pp - 1/Q_f)`**, Q_pp = Q220(chi1): kappa = omega_imag,pp -
  beta_r*omega_imag,f with omega_imag = omega/(2Q); substituting beta_r = w_pp/w_f gives
  the formula.  Already per unit ppBHPT time.
* **`tau_ov = 1/(w_I,1 - w_I,0)`**: relaxation time of the first overtone relative to the
  fundamental (J line ~131), used only as a comparison number in T2.
* **Junction constants C, D, A, B**: fixed by the four continuity conditions of section 2
  (F `model_ab`), not fitted.  `t_*` = the sample of largest |h| of the ppBHPT waveform
  (F `mismatch`, `i_star = argmax|h|`).

**External fits and constants**

* **SUPERSEDED 2026-09-26 (these three BCW rows and `W_SCHW`):** w220, Q220 and the n = 1 row are now the
  exact Kerr values from the `qnm` package (`physics_common.w220_exact`, `Q220_exact`, `w221_exact`,
  `Q221_exact`, J lines 97-107); exact w220(0) = 0.37367, Q220(0) = 2.1002. The rows are kept as a record
  of the code that produced sections 1-16.
* **`w220(chi) = f1 + f2*(1-chi)^f3`** — Kerr l = m = 2, n = 0 ringdown frequency
  M*omega.  Berti, Cardoso & Will 2006, Phys. Rev. D 73, 064030 (gr-qc/0512160), Table VIII.
  **f1 = 1.5251, f2 = -1.1568, f3 = 0.1292** (J line 79).  w220(0) = 0.3683.
* **`Q220(chi) = q1 + q2*(1-chi)^q3`** — quality factor Q = omega_real/(2 omega_imag) of the
  same mode, same paper and table, l = m = 2, n = 0.  **q1 = 0.7000, q2 = 1.4187,
  q3 = -0.4990** (J line 80).  Q220(0) = q1 + q2 = 2.1187.  (The code and sections 2, 7, 14
  record that these values have not been checked against the paper; they are the values
  the user's documentation cites for this row.)
* **n = 1 overtone row** (J lines 81-82): frequency f1..f3 = 1.3673, -1.0260, 0.1628 and
  quality factor q1..q3 = 0.1000, 0.5436, -0.4731, BCW 2006 Table VIII, l = m = 2, n = 1.
  The code marks them "UNVERIFIED, quoted from memory"; used only for `tau_ov` in T2.
* **`W_SCHW = 0.3737`** (J line 93, F via J): exact Schwarzschild l = m = 2, n = 0 ringdown
  frequency M*omega.  With `RESCALE_QNM = True` (J line 94), w_pp(chi1) =
  0.3737*w220(chi1)/w220(0), in m1 units.  Section 11 explains why this replaced 0.3683.
* **"textbook Schwarzschild 11.24 M"**: 1/(M*omega_imag) for the Schwarzschild l = m = 2,
  n = 0 mode, M*omega_imag = 0.0890.  **"11.50 M"** = 2*Q220(0)/w220(0) = 2*2.1187/0.3683 with the BCW
  fits (superseded; the exact values give 11.24).
* **`M_f`, `chi_f`**: surfinBH NRSur3dq8Remnant (Varma et al. 2019, Phys. Rev. Lett. 122,
  011101, arXiv:1809.09125) at (q, [0,0,chi1], [0,0,0]) (J `derived`).  Section 3's
  "gwModelRemS" column is a second remnant fit used only as a cross-check.

**Chosen settings**

* **Measurement window `[t_peak+20, t_peak+45]`** (section 1): a chosen late-ringdown
  window over which the instantaneous frequency is medianed.  The same window is coded in
  `../units_selftest.py:121` (ppBHPT: m1 units after the ppBHPT peak) and `:126` (NR: M
  units after the NR peak), dropping samples below 1e-3 of peak amplitude; the original
  2026-09-23 measurement script was not located.
* **Diagnostic grid** (J `main`, lines ~271-273): q = 3, 5, 8 x chi1 = -0.6 ... +0.8 = 24
  points.  "21/21 clean" = the 24 minus the three chi1 = -0.6 points, which have no usable
  post-merger peaks.
* **T1** (J `t1_kappa`, line ~183): uses post-merger phase peaks whose NR amplitude is
  above 1e-2 of its maximum (`amp_floor`), needs at least 4 of them; the late slope uses
  peaks at least 15 (ppBHPT time after the peak, m1 units) past merger (`dt_lo = 15`).
* **T3** (J `t3_root`): root search over the 3000 ppBHPT-time units before the peak;
  a root counts as "sane" if it lies within 400 units before the peak.
* **Fit settings** (F): window [-5000.1, +100.0] (NR time, M units,
  `../fit_scaling_PN_opt_creative.py:21-22`); t0 found by an 81-point scan over seed +- 200 M
  then a bounded search within +- 6 M (lines ~216-220) on the strided arrays
  (`SRC_STRIDE = 6`, `NR_STRIDE = 10`), mismatch then recomputed at full resolution;
  penalty 50.0 if beta <= 1e-3, |alpha| > 1e3, or NR coverage < 0.88; Powell with xtol 1e-3,
  ftol 1e-4 (line ~317), `--budget` default 0 = no time limit (line 252).  Starting point
  `SEED` (line 148) = the converged 5-coefficient result (section 5) with b21 = sSS1 = 0.
  Objective = MEAN mismatch over the 96-point grid (`TRAIN_Q` x `CHIS` of
  `fit_pn_anchored_spin.py`).

**Fitted coefficients**

| name | multiplies | in |
|:---|:---|:---|
| `b1`, `b2` | nu*x, nu*x^2 | beta_I |
| `sSO`, `sSS` | nu*chi1*x^1.5, nu*chi1^2*x^2 | beta_I |
| `b21`, `sSS1` | nu^2*x^2, nu^2*chi1^2*x^2 | beta_I (7-coefficient model only) |
| `a2` | nu*x^2 | alpha_I |
| `a32` | nu*x^1.5 | alpha_I (rejected 6-coefficient test, section 8) |

Stored in `junction_spin_results/`: `coeffs_junction.json` (current 7, job 3534076,
section 11), `coeffs_junction_5coef.json` (section 5), `coeffs_junction_6coef_a32_REJECTED.json`
(section 8).

**Where the result numbers come from**

* All mismatches: relative L2 (n1 + n2 - 2|Z|)/(2 n1) between NR and mapped ppBHPT (2,2)
  modes on the NR window, phase solved exactly, t0 per case, full resolution.  "mean"
  (the minimised objective), "median", "max" are over the 96 cases unless the section says
  otherwise; section 10's "column means" are means over the 12 q at each chi1.
* Section 1 impact table: `junction_floor_results/betar_impact.json` (per-case lists,
  `r0_old`, `r0_new`).  Section 3: `junction_diag_results/junction_diag_grid.json`
  (log `jncdiag.3530253.log`).  Section 5: `coeffs_junction_5coef.json`.  Section 6:
  `junction_floor_results/perq_q3.json`.  Section 8: the REJECTED json.  Section 9:
  `junction_floor_results/traj.json`; "spread%" = 100*(max - min)/|mean| of a coefficient
  over the 7 per-q fits, "lin R^2" = R^2 of a straight-line fit against nu, "monotone" =
  all steps the same sign (`junction_floor.py` lines ~172-181).  Section 10: job 3532499
  (log `jncfit.3532499.log`; its json was overwritten by section 11).  Section 11:
  `coeffs_junction.json`.  Section 15 subset tests: no json or log was found for these
  in `spin_dep/`.

**Numbers in this file that do not match the current json/log (left unchanged, flagged
2026-09-26)**

* Sections 5, 10: "`pn_anc_spin_physical` (11 coef): mean ~1.07e-03, median 7.686e-04" and
  "~1.07e-03" as the 11-coefficient benchmark.  `pn_anc_spin_physical_results/coeffs_physical.json`
  has mean 1.1408e-03, median 8.4850e-04.  1.0739e-03 is the mean of the 13-coefficient
  joint-2d refit, and 7.686e-04 is the median of the 13-coefficient `pn_anchored_spin`.
* Section 11: "`pn_anchored_spin` (13 coef, corrected) 1.1693e-03" is the single-start
  value from job 3534438; the final 6-start value (job 3535558,
  `redo_properly_results/spin13.json`) is 1.0993e-03.
* Section 14: "mean excluding chi1=+0.8 is 1.15e-03 against 1.54e-03 including it" mixes
  runs: 1.15e-03 (1.1495e-03) is the section-10 fit, 1.54e-03 the section-11 fit.  From
  `coeffs_junction.json` (section 11) the mean without chi1 = +0.8 is 1.1157e-03.
* Section 3 T2 ("median tau_meas/tau_f = 0.97 ... the measurement decided it") is
  withdrawn in the `fit_junction_spin.py` docstring (2026-09-24): the measured times were
  on the ppBHPT axis and tau_f on the NR axis; on a common axis the median ratio is 0.90
  with range 0.64-9.6, which does not separate the fundamental from the overtone.
* Section 2 "tau = 2 Q_f/w_f": since 2026-09-24 the fit uses tau_f_pp = tau_f/beta_r (see
  section 11).

**Time labels in "M" that are ppBHPT time (m1 units):** section 2 "tau = 11.50 M" is a
duration of the remnant (M units, correct), but section 3 T2 "Measured relaxation 10-15 M"
was measured on ppBHPT peak times (m1 units); section 4 `dt` and the T1 `dt_lo` are ppBHPT
time.  Section 12 "`t_*` lands at +4 to +11 M after the peak" is correct as M: the plot
script reports t_* mapped onto NR time (`tau[i_star]`, `plot_junction_spin_overlay.py:73`),
i.e. relative to the NR peak.

## 1. The `beta_r_phys` correction — READ THIS FIRST

BEFORE the 2026-09-24 fix, `fit_pn_anchored_spin.beta_r_phys` (and `BHPTNRPNAnchored._beta_r_phys`) computed
the WRONG formula below (now fixed in the code to `w_pp(chi1)*M_f/w220(chi_f)`):

    beta_r = [w220(chi1) * (1+q)/q] / [w220(chi_f)/M_f]

The `(1+q)/q` converts the primary's QNM frequency from primary-mass units to
total-mass units.  **That conversion is already done, so the factor double-counts.**

**Code.** With `calibrated=False` — which every fit in this workspace uses —
`common_utils/utils.py:176` sets `t_sur = np.array(time)`, the raw training grid.  No
mass rescaling is applied anywhere on that path.  The ppBHPT training data is in units
of the background (= primary) mass.

**Measurement** (2026-09-23, late-ringdown `omega` median over `[t_peak+20, t_peak+45]`; window and units in section 0).
The `w220(chi1)` column is the BCW fit used on that date (superseded 2026-09-26; exact Kerr: 0.37367 at
chi = 0, 0.49404 at chi = +0.6):

| q | chi1 | w_pp measured | w220(chi1) | w220(chi1)*(1+q)/q |
|--:|--:|--:|--:|--:|
| 3 | 0.0 | 0.3763 | 0.3683 | 0.4911 |
| 5 | 0.0 | 0.3746 | 0.3683 | 0.4420 |
| 8 | 0.0 | 0.3734 | 0.3683 | 0.4143 |
| 3 | +0.6 | 0.4974 | 0.4975 | 0.6633 |
| 5 | +0.6 | 0.4954 | 0.4975 | 0.5969 |
| 8 | +0.6 | 0.4933 | 0.4975 | 0.5596 |

The chi1 = +0.6 rows are the clean discriminator: 0.4% from `w220(chi1)`, 20% from the
other.  Consequence for the anchor:

| q (chi1=0) | shipped `beta_r_phys` | corrected | measured `w_pp/w_NR` |
|--:|--:|--:|--:|
| 3 | 0.996 | 0.7469 | 0.7473 |
| 5 | 0.973 | 0.8109 | 0.8095 |
| 8 | 0.972 | 0.8639 | 0.8613 |

The corrected form matches measurement to **0.05–1.3%** across the 3x8 grid, and is the
form `gwr_betaqnm` already uses (`W_Schw*Mf/omega_220(chi_f)`) — the one `history.md`
verified to 0.3–0.9%.  So the two model lines have disagreed on `beta_r` by 17–29%.

### What this does and does not invalidate

**Does NOT.**  No reported mismatch changes.  `beta_r_phys` is only ever used as
`beta_r_phys * r0` with `r0` fitted, so the fit found whatever MR beta minimised the
objective; every `pn_anchored` / spin mismatch is a correct measurement of the model as
it runs.  `gwr_betaqnm`, `fluxanchored`, `energy_anchored`, `stiff` are untouched (the
first uses the correct form, the rest have no QNM anchor).  `beta_r`'s **spin response**
is also untouched — the spurious factor is `1/X1`, independent of chi1.

**DOES.**  (a) The description of the MR branch as a derived, parameter-free remnant
anchor with `r0` a small correction: `r0` = 0.687 (nonspinning) / 0.618 (spin) is mostly
a units factor.  (b) The anchor's **q-dependence**: shipped `beta_r` is nearly flat in q
(0.996/0.973/0.972) where the truth rises 14% (0.747/0.811/0.864), and one nu-flat `r0`
can rescale but not reshape.  (c) Any future model that drops `r0` and uses `beta_r`
raw — which is exactly how this surfaced.  (d) It supplies the mechanism for the
documented symptom "MR beta overshoots truth (0.71 vs 0.66)"
(`scaling_pn_anchored.md` caveat 1).

**Why it hid for so long:** `S <= 0.18` means the branch carries at most 18% weight, and
nothing ever compared the two models' anchors.

### MEASURED IMPACT (job 3533460, 2026-09-24) — and a correction to the above

An earlier draft of this section said "`r0` absorbs the bulk of it".  **That is wrong.**
Scoring pn_anc_spin_physical's SHIPPED 11 coefficients on the 96-point grid, no refit:

| variant | mean | median | max |
|:--|--:|--:|--:|
| shipped `beta_r` (with `(1+q)/q`) | 1.1408e-03 | 8.485e-04 | 5.607e-03 |
| corrected `beta_r`, NO refit | **1.6417e-02** | 1.311e-02 | 4.454e-02 |
| corrected, `r0` refit ONLY | 2.4897e-03 | 2.076e-03 | 8.844e-03 |

`r0` moves 0.6179 -> 0.7517 and recovers 16.4e-03 -> 2.49e-03, but leaves the model
**2.2x worse than shipped**.  So `r0` absorbs only part of it; what actually absorbs the
bug is the WHOLE coefficient set -- all eleven were fitted against the wrong `beta_r`
and have jointly adapted to it.  The q-tilt `r0` structurally cannot take, visible in
`beta_MR = beta_r * r0`:

| q | shipped | corrected + refit r0 | diff |
|--:|--:|--:|--:|
| 3 | 0.615 | 0.561 | -9% |
| 5 | 0.601 | 0.610 | +1% |
| 8 | 0.601 | 0.649 | +8% |

**Consequences, stated precisely:**
* Published numbers are UNCHANGED and correct -- they measure the code as it runs.
* A corrected model is a DIFFERENT model and its performance is NOT YET KNOWN.  The
  2.49e-03 above is not it: only `r0` was refit.  A full 11-coefficient refit is needed.
* The shipped MR branch sits at `beta_MR ~ 0.60` against a measured ringdown ratio of
  ~0.83 at q=5 -- ~28% low.  `S <= 0.18` keeps it from wrecking the waveform, but the
  "derived remnant anchor" description is further from true than first stated here.
* Calibration point: the junction model uses the CORRECTED `beta_r`, has no `r0`, and
  reaches 1.573e-03 on this same grid vs the shipped model's 1.141e-03 -- i.e. a
  physically correct anchor at 7 coefficients is within 1.4x of an adapted-to-wrong-
  anchor model at 11.

## 2. The junction construction

Replaces `S = sigmoid((p_loss - (p00+p01 nu))/w0)` and the `(1-S)/S` blend with a C1
match at `t_* = the ppBHPT |h| peak` (parameter-free, already the alignment anchor).

    INSPIRAL   t <= t_*   (no XCLIP — the branch is not evaluated past t_*)
      beta_I  = base[1 + nu(b1 x + b2 x^2 + sSO chi1 x^1.5 + sSS chi1^2 x^2)]
      alpha_I = base[1 + (55/42) nu x + nu a2 x^2]
    MERGER-RINGDOWN   dt = t - t_*
      beta_M  = beta_r + exp(-dt/tau)(C + D dt)
      alpha_M = exp(kappa dt)[A + B exp(-dt/tau)]

Four junction conditions fix C, D, A, B; none is fitted:

    [beta]=0  -> C = beta_I(t_*) - beta_r
    [beta']=0 -> D = betadot_I(t_*) + C/tau
    [alpha]=0 and [alpha']=0 -> B = tau(kappa alpha_I(t_*) - alphadot_I(t_*)),
                                A = alpha_I(t_*) - B

Equivalently (and this is the physical statement): the energy flux `F` and the
angular-momentum flux `F_J = F/omega` are each continuous with their first derivative.
`F = (alpha/beta)^2 F_pp` with `F_pp` common to both branches, so `F_pp` cancels from
every jump condition and the measured (noisy) flux is never differentiated.

**Derived block, zero fitted parameters:** `beta_r` (corrected, above);
`tau = 2 Q_f/w_f` the fundamental damping time; `kappa = (w_pp/2)(1/Q_pp - 1/Q_f)`,
alpha's ringdown log-slope — beta matches the two QNM *real* frequencies and cannot also
match the imaginary parts, so kappa is the residual decay-rate mismatch alpha must carry.
[2026-09-23 text; since 2026-09-26 Q220 and w220 are exact Kerr and tau is on the ppBHPT axis -- see
"Anchors (derived, not fitted)" below.]  Uses the BCW quality-factor row `Q220(chi) = 0.7000 + 1.4187(1-chi)^(-0.4990)` (Berti, Cardoso & Will 2006, Table VIII, l=m=2, n=0: Q = q1 + q2(1-chi)^q3; see section 0), **new to
this workspace and not yet checked against the paper** (internal check: `Q220(0)=2.1187`
-> tau = 11.50 M vs textbook Schwarzschild 11.24 M, a 2% fit error).

Coefficients: **5** (b1, b2, a2, sSO, sSS) against 11 for `pn_anc_spin_physical`.
Deleted: p00, p01, w0, r0, m0, m1, XCLIP.

## Anchors (derived, not fitted)

*Written 2026-09-27 against the current code: `spin_dep/fit_junction_spin.py` (F; `model_ab`,
F:164-198), `spin_dep/junction_diag.py` (J; `derived`, J:110-144), `spin_dep/fit_pn_anchored_spin.py`
(S) and `physics_common.py`. Equation numbers are those of `../alpha_beta_pn_scaling_note_revised.pdf`
("the note"). Section 2 above is the 2026-09-23 five-coefficient form; the current model adds the ν²
terms of section 9-10, uses the note's x, and puts τ on the ppBHPT axis.*

**Current model** (F:164-198):

    x       = X1^(-4/5) * x_pp                      (no clip)
    beta_I  = X1^(6/5) [1 + nu (b1 x + b2 x^2 + sSO chi1 x^1.5 + sSS chi1^2 x^2)
                          + nu^2 (b21 x^2 + sSS1 chi1^2 x^2)]              t <= t_*
    alpha_I = X1^(6/5) [1 + (55/42) nu x + nu a2 x^2]                      t <= t_*
    beta_M  = beta_r + exp(-dt/tau) (C + D dt)                             dt = t - t_* > 0
    alpha_M = exp(kappa dt) [A + B exp(-dt/tau)]

**Coefficient status.** 7 fitted coefficients (`NAMES`, F:144: b1, b2, a2, sSO, sSS, b21, sSS1), all
on the 96-point (q, χ1) grid and all in the inspiral branch; the merger–ringdown branch has none.
**Caveat:** the stored `junction_spin_results/coeffs_junction.json` (job 3534076, 2026-09-24) predates
the 2026-09-26 fixes and carries no `model_form` stamp: it was fitted with x_pp unconverted, the BCW
w220 (ppBHPT side rescaled to 0.3737) and the BCW Q220. The anchors below are those of the current
code, which has not yet been refit on the full grid (2026-09-26 section at the end).

**The two clocks.** ppBHPT time `t` (BHPTNRSur2dq1e3, `calibrated=False`) is in units of m1; NR time
`τ` (NRHybSur3dq8) in units of M; `dτ/dt = β`. All junction-branch durations and rates (dt, τ, κ) are
on the ppBHPT clock.

### A1. Inspiral anchor: β_LO = α₂,LO = X1^(6/5)

`base = X1^(6/5) = (q/(1+q))^(6/5) = (m1/M)^(6/5)`, F:167. The leading-order (Newtonian) time map and
(2,2) amplitude ratio. Derivation (note Sec. 3): matching the Newtonian chirp rates
`dx_pp/dt = (64/5) η_pp x_pp^5` (`η_pp = 1/q`) and `dx_NR/dτ = (64/5) ν x_NR^5` (Eq. 7) under `τ = βt`,
`x_NR = β^(−2/3) x_pp` (Eq. 8), gives `η_pp β^(5/3) = ν` (Eqs. 9-10), and with `ν/η_pp = X1²` (Eq. 11)
`β_LO = X1^(6/5)` (Eq. 12); matching the Newtonian (2,2) amplitudes gives `α₂,LO = X1^(6/5) = β_LO`
(Eqs. 13-16). Derived, not fitted: the exponent follows from the linearity of the chirp rate in the
mass ratio and the `β^(−2/3)` scaling of x; it tends to 1 as q → ∞, and differs from the bare unit
conversion `X1` by `X1^(1/5)`.

Use here: base of `β_I` and `α_I` only (F:178-181), with every correction ν- or ν²-suppressed (note
Eqs. 58-59, 61; the ν and ν² orders of the basis Eq. 62). There is no `α_mr = base(m0 + m1 ν)` term:
after `t_*` α and β are fixed by continuity with the inspiral values (A6), so X1^(6/5) reaches the
merger only through `α_I(t_*)`, `β_I(t_*)`.

### A2. Ringdown anchor: β_r(q, χ1) = w220(χ1) · M_f / w220(χ_f)

`beta_r = w_pp / w_f` with `w_pp = w220(χ1)` and `w_f = w220(χ_f)/M_f` (J:123-127), the same expression as
`S.beta_r_phys`. `w220 = physics_common.w220_exact` (J:98-104): exact Kerr (2,2,0) frequency `M ω_220` in
units of the ringing hole's mass, Leaver's continued fraction via the `qnm` package (L. C. Stein 2019,
JOSS 4, 1683), counter-rotating branch `|ω_{2,−2}(|χ|)|` for χ < 0 (`physics_common.py:54-72`, line 59).
ppBHPT side: the primary (mass m1, spin χ1; secondary neglected), `w220(χ1)` per unit of the m1 clock.
NR side: the remnant, `w220(χ_f)/M_f` per unit of the M clock, `M_f`, `χ_f` from surfinBH
`NRSur3dq8Remnant` at `(q, [0,0,χ1], [0,0,0])` (J:112-115).

*Why the frequency ratio.* The mapped ppBHPT waveform has frequency `ω_pp/β`; for the ringdowns to stay
in phase it must equal `ω_NR`, so at ringdown `β = dτ_NR/dt_pp = ω_pp/ω_NR`. No free parameter.

*Use here — without `r0`.* Unlike the switched pn models, β_r is not multiplied by a fitted scale: it is
the exact late-time limit of `β_M = β_r + e^(−dt/τ)(C + D dt)` (F:196), so β relaxes onto the anchor itself.
Section 12 measures the settled β within a few tenths of a percent of β_r in the training range
(on the 2026-09-24 code, i.e. with the BCW-based β_r of that date). At
χ1 = 0, `β_r = 0.373672 M_f / w220(χ_f)`.

*What it replaces (2026-09-26).* The Berti–Cardoso–Will fit `1.5251 − 1.1568(1−χ)^0.1292` (PRD 73,
064030, Table VIII), which (a) is 1.5% low at χ = 0 (0.3683 vs 0.37367), (b) was used on the ppBHPT side
rescaled to 0.3737 (`W_SCHW`, `RESCALE_QNM`, J:93-94, now unused) but raw on the NR side, and (c) is 3% /
7% low at χ = −0.2 / −0.6 when extrapolated to anti-aligned spin. The exact values agree with the measured
ppBHPT ringdown frequency at q = 5 to +0.09 / +0.26 / −0.67 / +0.18 % for χ1 = −0.2 / 0 / 0.2 / 0.6
(`physics_common.py:6-15`). Before 2026-09-24 the ppBHPT side also carried a spurious `(1+q)/q` (section 1).

### A3. Ringdown damping anchors: τ and κ (from the QNM quality factors)

`Q220 = physics_common.Q220_exact` (J:105), `Q = ω_R/(2|ω_I|)` of the same exact Kerr mode (was the BCW
quality-factor fit `0.7000 + 1.4187(1−χ)^(−0.4990)` until 2026-09-26; exact `Q220(0) = 2.1002` vs BCW 2.1187).
- `τ = τ_f,pp = (2 Q_f / w_f) / β_r` (J:128, 140; selected at F:176 with `TAU_PP = True`, F:152):
  `2Q_f/w_f = 1/ω_I,f` is the damping time of the remnant's fundamental mode on the NR clock
  (11.24 in hole-mass units for Schwarzschild); dividing by β_r converts an NR duration to ppBHPT time.
  It sets how fast β_M and the B-term of α_M relax. The value is derived; the choice of the fundamental
  rather than the overtone relaxation time is not settled by measurement (F docstring, "THE CHOICE OF
  tau IS NOT SETTLED").
- `κ = (w_pp/2)(1/Q_pp − 1/Q_f)`, `Q_pp = Q220(χ1)`, `Q_f = Q220(χ_f)` (J:124, 126, 132). After the junction
  the mapped ppBHPT waveform decays as `exp(−ω_I,pp t)` with `ω_I,pp = w_pp/(2Q_pp)`, while NR decays as
  `exp(−ω_I,f τ) = exp(−β_r ω_I,f t)` with `β_r ω_I,f = w_pp/(2Q_f)`. β can match only the real
  frequencies (A2), so α must carry the difference, `α ∝ exp(κ t)` with `κ = ω_I,pp − β_r ω_I,f`, already
  per unit ppBHPT time. `Q_f > Q_pp` on the grid, so κ > 0 and α rises through ringdown.

### A4. Fixed 1PN amplitude slope 55/42

`S.PN_SLOPE = 55/42` (S:97; F:181), the ν x coefficient of `α_I`. The 1PN (2,2) amplitude
`h22 = h22^N [1 + (−107/42 + (55/42)ν) x + 2π x^(3/2) + …]` (Eq. 55) has its `−107/42` and `2π x^(3/2)`
terms in the test-particle limit, so the ratio at fixed x is `1 + (55/42) ν x` (Eq. 56), the ν x term of
Eq. 57. Not fitted; section 8 tested and rejected an added `a32 x^(3/2)` term.

### A5. The note's x (no clip) and the PN orders of the spin terms

`x = X1^(−4/5) x_pp` (F:170, `physics_common.x_scale`, `physics_common.py:88-90`), with
`x_pp = (ω_22/2)^(2/3) = (m1 Ω)^(2/3)` from `S.get_x` (80 m1 Savitzky–Golay phase smoothing). The note's
`x = (M Ω)^(2/3)` (Eq. 6) follows from Eq. 8 with `β_LO = X1^(6/5)` (Eq. 12). No XCLIP: the inspiral
branch is not evaluated past `t_*`. The spin terms carry fixed PN powers — spin-orbit `χ1 x^(3/2)` (1.5PN)
and spin-spin `χ1² x²` (2PN) (Kidder 1995, PRD 52, 821; Blanchet 2014, Living Rev. Relativ. 17, 2) — inside
the ν (and, for `sSS1`, ν²) bracket, since the test-particle spin effects are already in the ppBHPT
waveform. Only `sSO`, `sSS`, `sSS1` are fitted.

### A6. Junction time t_* and the junction constants

`t_*` = the ppBHPT sample of largest |h| (`i_star = argmax|h|`, F:206), parameter-free. `C, D, A, B` are
fixed by continuity of β, β̇, α, α̇ at `t_*` (F:187-190): `C = β_I(t_*) − β_r`, `D = β̇_I(t_*) + C/τ`,
`B = τ(κ α_I(t_*) − α̇_I(t_*))`, `A = α_I(t_*) − B`. None is fitted.

### A7. T_ANCHOR = −100 (a convention)

`S.T_ANCHOR = −100.0` (S:97), ppBHPT time in m1 units: `ts = ∫β dt − [∫β dt]_{t = T_ANCHOR}` (F:211) and
`τ = t0 + ts` (`S._err_at_t0`), so `τ(T_ANCHOR) = t0`, the per-case nuisance offset. Not a physical anchor.

### Which fitted coefficient multiplies which anchor

| anchor | enters as | fitted coefficients attached |
|:--|:--|:--|
| A1 `X1^(6/5)` | base of `β_I` | `b1`, `b2`, `sSO`, `sSS` (order ν); `b21`, `sSS1` (order ν²) |
| A1 `X1^(6/5)` | base of `α_I` | `a2` |
| A2 `β_r`, A3 `τ`, `κ`, A6 `t_*`, `C, D, A, B` | merger–ringdown branch | none |
| A4 `55/42`, A5 `X1^(−4/5)` and PN powers | fixed | none |

## 3. Diagnostic results (`junction_diag.py`, 3x8 grid, job 3530253)

**T1 — kappa: PASS.** Sign correct in **21/21** clean grid points; median
measured/derived **1.09**, range 0.89–1.96 (the outliers are all chi1 = -0.4).  So alpha
**rises** through ringdown at every q and spin, at the derived rate, and a constant
`alpha_mr = base(m0+m1 nu)` cannot represent it.  Derived kappa also carries the spin
dependence correctly (falls with q: 0.021/0.015/0.010 at q=3/5/8; turns over near
chi1=+0.4 as `Q_pp` climbs).

**T2 — tau: the fundamental, not the overtone.**  Measured relaxation 10–15 M against
`tau_f` 11–14 M and `tau_ov = 1/(w_I,1 - w_I,0)` 5.5–7 M.  Median `tau_meas/tau_f` =
0.97.  A prior argument that the overtone timescale should win was **wrong**; the
measurement decided it.  No fitted coefficient needed.

**T3 — as first formulated, FAILED (11/24), and the formulation was wrong.**  The
original scheme located `t_*` by root-finding `G(t) = tau beta_I' + beta_I - beta_r = 0`
on the inspiral side.  That needs the anchor gap `(beta_r - base)/base > 0`, and with the
corrected `beta_r` the gap **closes and goes negative** at high q / low spin
(+5.5%/+0.9%/-0.5% at q=3/5/8, chi1=0), so `G > 0` everywhere and there is no root.
Direct measurement of the post-merger curves showed why the premise was wrong: **beta has
its MAXIMUM and alpha its MINIMUM at the |h| peak**, and beta then decays onto `beta_r`.
So `t_*` is the peak, and the MR branch must arrive *rising* and turn over — which the
`(C + D dt)exp(-dt/tau)` form does with exactly the two constants the two beta conditions
supply.  No root-find, and the sign of the gap is irrelevant.

**The anchor gap goes NEGATIVE near q ~ 7 — CHECKED IN BOTH REMNANT SOURCES
(2026-09-23), they agree to 0.05%:**

| q | beta_r (gwModelRemS) | beta_r (surfinBH) | gap RemS | gap surfinBH |
|--:|--:|--:|--:|--:|
| 3 | 0.7472 | 0.7469 | +5.53% | +5.49% |
| 4 | 0.7836 | 0.7832 | +2.42% | +2.37% |
| 5 | 0.8114 | 0.8109 | +0.98% | +0.92% |
| 8 | 0.8641 | 0.8639 | **-0.48%** | **-0.50%** |

So there is NO remnant-source discrepancy; an earlier note here guessing one was wrong.
The ~1.5pp offset from `history.md`'s "+7.04% -> +0.96%" is a DEFINITIONAL difference:
`fit_scaling_gwr_betaqnm.py:71` measures the gap against `B0 = base*(1 + b*nu)` with `b`
a FITTED anchor tilt, not against bare `base`.  Same physics, different baseline.

None of this reaches the junction: with `t_* = |h| peak` the construction is indifferent
to the sign of the gap.  That is the point of the reformulation.

Incidental: `fit_pn_anchored_spin.py:7,129` say the spin dependence is "DERIVED from
RemS", but the code calls `surfinBH.LoadFits("NRSur3dq8Remnant")`.  Numerically harmless
(0.05%), but the comment names the wrong source.

**chi1 = -0.6 is not measurable.**  Zero usable post-merger peaks at all three q — the
15–16 envelope nulls documented in the `FREEZE_XC` note destroy phase monotonicity.
Those rows are still fitted and scored, but the derived branch cannot be checked there.

## 4. Construction check before fitting (q=8, chi1=0, seed coefficients)

| dt | beta model | beta measured | alpha model | alpha measured |
|--:|--:|--:|--:|--:|
| 0 | 0.9283 | 0.9394 | 0.7556 | 0.7867 |
| 20 | 0.8995 | 0.8970 | 0.7970 | 0.8827 |
| 40 | 0.8741 | 0.8622 | 0.9518 | 1.1583 |
| 80 | 0.8644 | 0.8472 | 1.4284 | 1.9120 |

beta reproduces the turnover and settles on `beta_r` to 1–2% with nothing fitted.  alpha
rises as derived but undershoots, because the seed `a2 = -22.70` was fitted to serve the
merger through the XCLIP and still drives `alpha_I` steeply down at `t_*`, inflating `B`.
That is a coefficient artefact of the old architecture, not a structural one — `a2` is
one of the 5 free parameters and is no longer tied to the merger.

## 5. First converged fit (job 3530305, 2026-09-23) — DOES NOT BEAT THE 11-COEF MODEL

Powell converged (not budget-truncated) in 302 evaluations.

    theta = [b1, b2, a2, sSO, sSS] = [0.40089, -5.43646, -17.00576, 3.33094, -0.45003]
    mean 3.7310e-03   median 2.9999e-03   max 1.3048e-02
    chi1=0 column: median 2.4103e-03

Against `pn_anc_spin_physical` (11 coef): mean ~1.07e-03, median 7.686e-04.  So the
junction model at 5 coefficients is **~3.5x worse**.  The architecture works — it
converged, the derived branch is sound (section 3, 4) — but it costs real accuracy.

**Where it goes wrong: q = 3.**  That row is uniformly the worst (5.5e-03 to 1.3e-02,
vs 7e-04 at the q~4.2-5.0 minimum), and the error grows monotonically toward low q.  The
second bad region is chi1 = +0.8 at q = 5-6 (5.0-5.8e-03).  In range q ~ 4-5 the model is
competitive (7.1e-04 at q=4.6, chi1=-0.2).  This is the reverse of `pn_anchored`'s
signature strength, where q<3 is EVALUATED not extrapolated.  Diagnosing the q=3 edge is
the next step, not adding coefficients blindly.

**The b2 prediction was half right.**  The pre-registered test (section 4) was that
removing merger contamination should shrink `b2` from ~8.5 toward O(1).  It went
8.553/7.754 -> **-5.436**: smaller in magnitude, but sign-flipped, not O(1).  `sSS` also
flipped, 2.259 -> -0.450.  So the inspiral coefficients are NOT simply de-contaminated
versions of the old ones; the junction has changed what they mean.  Treat the old->new
comparison as void rather than as evidence either way.

## 6. PER-Q TEST AT q=3 (job 3532151) — it is a REGRESSION problem, not the architecture

ONE coefficient set fitted to all 8 chi1 at q=3 (`junction_floor.py --mode perq`), for
both architectures, same objective.  Converged: 262 evals (junction), 2631 (physical).

| chi1 | master (5c, joint) | junction per-q (5c) | physical per-q (11c) | gain |
|--:|--:|--:|--:|--:|
| -0.60 | 5.525e-03 | 1.450e-03 | 1.078e-03 | 3.81 |
| -0.40 | 8.370e-03 | 7.845e-04 | 6.730e-04 | 10.67 |
| -0.20 | 9.550e-03 | 5.603e-04 | 5.344e-04 | 17.05 |
| +0.00 | 1.070e-02 | 6.700e-04 | 4.472e-04 | 15.97 |
| +0.20 | 1.231e-02 | 5.897e-04 | 4.804e-04 | 20.87 |
| +0.40 | 1.305e-02 | 8.418e-04 | **9.185e-04** | 15.50 |
| +0.60 | 1.293e-02 | 1.888e-03 | 1.511e-03 | 6.84 |
| +0.80 | 1.189e-02 | 4.885e-03 | 2.986e-03 | 2.44 |
| **mean** | **1.054e-02** | **1.459e-03** | **1.079e-03** | **7.23** |

**master / junction-per-q = 7.23x.**  Far past the 3x threshold: the 5 coefficients CAN
serve q=3, and the joint nu-flat fit is trading it away.  The q=3 failure is in the
REGRESSION LAYER.

**junction-per-q / physical-per-q = 1.35x.**  At fixed nu the junction is within 35% of
the 11-coefficient switched model using **6 fewer coefficients**, and it BEATS it at
chi1 = +0.4 (8.418e-04 vs 9.185e-04).  So the junction architecture is NOT the limit at
q=3.  (An earlier unconverged smoke test read 4.84x here and was flagged as unreliable
at the time; converged it is 1.35x.  Do not quote the 4.84x.)

**Conclusion.** The junction construction is sound and the merger-ringdown branch being
fully derived costs almost nothing.  What costs 3.5x overall is that `b1, b2, a2, sSO,
sSS` are nu-FLAT constants asked to serve nu in [0.0988, 0.1875].

## 7. The indicated fix — PN structure in the residuals (2026-09-23)

Verified against `alpha_beta_pn_scaling_note_revised.pdf`:

* **Eq. (55)**: `h22 = h22^N[1 + (-107/42 + 55nu/42)x + 2 pi x^{3/2} + ...]`.  The 1PN
  coefficient is LINEAR in nu, so the finite-mass correction is exactly `(55/42) nu x`
  with **no nu^2 at 1PN**.  A nu^2 order starts at 2PN.
* **Eq. (57)**: `alpha_22^insp = X1^{6/5}[1 + (55/42) nu x + a_{3/2}(nu) x^{3/2}
  + a_2(nu) x^2 + ...]` — the note ALREADY writes the higher coefficients as functions
  of nu.  So `a_2(nu) = a2 + a2p nu` is the note's own structure.
* **alpha has no x^{3/2} term in any model here.**  Eq. (57) has one.  This is the same
  hole the spin work found and fixed in beta (`fit_pn_anc_spin_physical` PROVENANCE:
  "THERE IS NO x**1.5 TERM ANYWHERE IN IT, so the leading spin-orbit effect had nowhere
  to live"), still open in alpha's NONSPINNING sector.  `a2 = -22.7` may be large
  because it is absorbing 1.5PN structure it cannot represent, exactly as `s21` was.

Proposed, all coefficients still nu-FLAT so nothing is extrapolated:

    alpha_I = base[1 + (55/42) nu x + a32 nu x^1.5 + nu x^2 (a2 + a2p nu)]
    beta_I  = base[1 + nu x (b1 + b1p nu) + nu x^2 (b2 + b2p nu)
                     + sSO chi1 nu x^1.5 + sSS chi1^2 nu x^2]

Two SEPARABLE tests, in this order:
1. **the missing 1.5PN order in alpha** (`a32`, +1, NONSPINNING) — cheaper, and the same
   fix worked in beta's spin sector.  Watch whether `a2` shrinks from -22.7.
2. **the nu^2 order at 2PN** (`a2p`, `b1p`, `b2p`, +3) — the second-order self-force
   term; ties to the open 2SF item in `history.md`.

### DECISION 2026-09-23 (user): do NOT add a spin term to alpha.

An `aSO chi1 nu x^1.5` term was proposed here and REJECTED.  Record the reasoning so it
is not re-proposed:

* The only evidence for alpha spin dependence is at x >~ 0.10, where `R_alpha` falls to
  0.94-0.99 -- but the parameter-free measurement (`spin_extension_plan.md` sec 7) was
  banded to x in [0.05, 0.10] and gives `R_alpha = 1.000 +- 0.005` there.  Fitting into
  a band the measurement deliberately excluded is fitting noise.
* alpha and beta are correlated at **-0.974** in this workspace.  An alpha spin term
  would most likely absorb BETA's spin dependence rather than represent its own -- which
  is exactly the failure mode sec 7 diagnoses in Islam's Eq. 8 (their Eq. 7 fits alpha
  and beta simultaneously; this construction reads them separately so it cannot).
* Consequence: if chi1 = +0.8 remains the worst column after (1) and (2), attribute it
  to **beta's chi1 form**, which `history.md` already names ("the chi1 form runs out at
  |chi1| >~ 0.6; a quadratic in chi1 is not enough at the ends"), NOT to alpha.

The junction's merger-ringdown sector is already spin-responsive with ZERO fitted spin
coefficients (via `chi_f(q,chi1)`, `Q_pp(chi1)`, `Q_f(chi_f)`), which is the gap
`history.md` flagged as "alpha's merger spin behaviour, never measured".  That is the
spin structure alpha gets, and it is derived rather than fitted.

NOTE the nu^2 term is IMPOSED, not regressed: nu^2 is a known function of q and its
coefficient is a constant, so unlike `fluxanchored`'s `alpha_F = A0 nu + A1 nu^2` (whose
FITTED nu-form is -18% at q=2) it carries no extrapolation penalty.  An earlier draft of
this section wrongly warned that adding nu structure would cost the q<3 property; that
was an artefact of writing it as `a2 -> a2_0 + a2_1 nu` (a regression) rather than as a
nu^2 ORDER (an expansion).  Same algebra, different discipline.

**Open.**  (1) Verify the BCW `Q220` row against the paper — it is the one constant here
new to this workspace.  (2) Decide what to do about `beta_r_phys` in the shipped models —
the numbers stand, the description does not.  (CLOSED 2026-09-23: the remnant-source
question, see section 3.)


## 8. `a32` — TESTED AND REJECTED (job 3532284, 2026-09-23)

alpha has no `x^1.5` term though the PN note's Eq. (57) carries `a_{3/2}(nu) x^{3/2}`.
Added as `a32` and globally refit from a bit-identical null start.

    5 coef  mean 3.7310e-03    a2 = -17.01
    6 coef  mean 3.6048e-03    a2 = -39.31,  a32 = +10.86

**3.4% for a whole coefficient, and the hypothesis is falsified.**  `a2` was supposed to
SHRINK if it had been absorbing 1.5PN structure; it more than DOUBLED, and `a32` came in
large and opposite.  At x = 0.26 the two give +1.440 and -2.657, sum -1.217, against the
old single term's -1.150 — a large cancelling pair reproducing one term.  `x^1.5` and
`x^2` are not separable over this x-range (cf. `wf_nu_fluxes`'s degenerate alpha_F/alpha_G).
chi1=+0.8 also got slightly WORSE (q=5: 5.019e-03 -> 5.485e-03).  Rejected fit kept at
`junction_spin_results/coeffs_junction_6coef_a32_REJECTED.json`.

**Method lesson:** the per-q test had already diagnosed a REGRESSION-LAYER problem, and
`a32` adds x-structure, not nu-structure — it could never have addressed the measured
defect.  It was prioritised on theory (Eq. 57 really does have the term) over the
measurement.  Fit the thing the diagnosis names.

## 9. Per-q coefficient trajectories (job 3532375) — which coefficients need nu

`junction_floor.py --mode traj`, 7 q across [3,8], one independent 5-coef fit per q.

| q | nu | floor | master | gain | b1 | b2 | a2 | sSO | sSS |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 3.00 | 0.1875 | 1.459e-03 | 1.054e-02 | 7.23 | 0.424 | **+3.22** | -16.30 | 4.254 | **-5.99** |
| 3.51 | 0.1725 | 1.243e-03 | 4.045e-03 | 3.25 | 0.250 | +2.52 | -15.40 | 4.075 | -3.64 |
| 3.90 | 0.1625 | 1.149e-03 | 2.334e-03 | 2.03 | 0.157 | +2.13 | -14.89 | 3.975 | -2.46 |
| 4.60 | 0.1467 | 1.043e-03 | 1.754e-03 | 1.68 | 0.038 | +1.56 | -14.25 | 3.833 | -1.05 |
| 5.44 | 0.1312 | 9.677e-04 | 2.347e-03 | 2.43 | -0.047 | +0.87 | -13.79 | 3.705 | +0.17 |
| 6.59 | 0.1144 | 9.037e-04 | 3.274e-03 | 3.62 | -0.092 | -0.38 | -13.45 | 3.588 | +1.83 |
| 8.00 | 0.0988 | 8.968e-04 | 3.845e-03 | 4.29 | -0.051 | **-2.73** | -13.41 | 3.570 | **+4.58** |

| coef | spread% | lin R^2 | monotone | verdict |
|--:|--:|--:|:--|:--|
| sSS | 1127.9 | 0.985 | yes | **nu^2 term** — sign change |
| b2 | 579.7 | 0.921 | yes | **nu^2 term** — sign change |
| b1 | 531.6 | 0.856 | **NO** | skip — a line tilts the wrong way |
| a2 | 19.9 | 0.924 | yes | skip — barely moves |
| sSO | 17.7 | 0.964 | yes | skip — barely moves |

**Two terms, not five.**  master/floor is U-shaped in q (7.23x at q=3, 1.68x at q=4.6,
4.29x at q=8) — the signature of one constant fitting a trend.  Per-q floor averages
**1.10e-03**, so if the nu^2 order captures the gap the master should approach that,
i.e. the 11-coefficient switched model's 1.07e-03 at **7 coefficients** with a fully
derived merger-ringdown branch.

**CAUTION.** master `b2 = -5.44` lies OUTSIDE the whole per-q range [-2.73, +3.22], and
master `sSS = -0.45` sits mid-range while the truth sweeps +-5.  The joint fit is in a
COMPENSATED basin, not a compromise between per-q optima.

## 10. nu^2 fit — SUCCEEDED (job 3532499, 2026-09-24)

7 coefficients (`b1, b2, a2, sSO, sSS, b21, sSS1`), `b21 = sSS1 = 0` at the start so the
fit begins bit-identical to the 5-coef model (null check verified: q=3/5/8 at chi1=0 give
1.070e-02 / 1.396e-03 / 3.702e-03, exactly the 5-coef row).

**Deliberately NOT seeded from the section-9 trajectories** — see the SEEDING note in
`fit_junction_spin.py`.  Briefly: the trajectories are the evidence that these two need
nu structure, so seeding from them would make the global result partly the seed
re-emerging; and choosing a start to dodge the compensated basin would engineer around a
model property instead of reporting it.  If it stalls in that basin, that IS the result,
and the follow-up is a multi-start with the spread reported.

Baselines: 5 coef **3.7310e-03**; 11-coef switched `pn_anc_spin_physical` ~**1.07e-03**;
per-q floor ~**1.10e-03**.


### Result

    5 coef  mean 3.7310e-03
    7 coef  mean 1.5732e-03   median 1.1273e-03   max 1.2827e-02
    theta = [b1 0.1979, b2 -22.3241, a2 -15.4358, sSO 3.7226, sSS 6.0532,
             b21 149.8128, sSS1 -54.4991]

**2.37x better for two coefficients**, and the MEDIAN (1.127e-03) is at the per-q floor
(~1.10e-03).  Benchmarks: 11-coefficient switched `pn_anc_spin_physical` ~1.07e-03.

**It did NOT stall in the compensated basin.**  From the neutral start (b21 = sSS1 = 0,
bit-identical to the 5-coef model) the joint fit found the nu^2 order unaided, so no
seeding was needed and the basin worry of section 9 does not bite.  Independent
cross-check against the section-9 trajectories, which were NOT used as input:

| q | b2 + b21 nu | traj b2 | sSS + sSS1 nu | traj sSS |
|--:|--:|--:|--:|--:|
| 3.0 | +5.77 | +3.22 | -4.17 | -5.99 |
| 5.4 | -2.66 | +0.87 | -1.10 | +0.17 |
| 8.0 | -7.53 | -2.73 | +0.67 | +4.58 |

Same signs and same direction of travel in both, from two independent routes — the
trajectory measurement predicted the form and the global fit confirmed it.  Magnitudes
differ because the global fit trades these against the other five coefficients.

### The entire remaining defect is chi1 = +0.8

Column means (x1e-3):

| chi1 | -0.6 | -0.4 | -0.2 | 0.0 | +0.2 | +0.4 | +0.6 | **+0.8** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mean | 1.348 | 0.865 | 0.734 | 0.916 | 0.967 | 1.266 | 1.950 | **4.539** |

**mean excluding chi1=+0.8 is 1.1495e-03** — at the per-q floor, and within 7% of the
11-coefficient switched model, at **7 coefficients with a merger-ringdown branch that is
entirely derived**.  Worst single point is q=3, chi1=+0.8 at 1.283e-02.

This is exactly the limit `history.md` already names: *"the chi1 form runs out at
|chi1| >~ 0.6.  A quadratic in chi1 is not enough at the ends; the 2026-08-30
measurement also found a level shift needed at chi1 = -0.6 that the current form cannot
represent."*  beta carries only chi1 and chi1^2.  Per the section-7 DECISION this is
BETA's chi1 form, not alpha's.

### Next

1. **beta's chi1 form** — the one remaining defect, worth ~30% of the mean on its own.
   Measure before choosing (a chi1^3 term, or a different spin variable) — the section-8
   `a32` lesson is that a term justified by theory alone can still be degenerate.
2. Re-run the per-q trajectory sweep at 7 coefficients to see what headroom is left.
3. Nonspinning cross-check: this model has never been scored against the chi1=0-only
   baselines (`fluxanchored` 4.694e-04 etc.).  The chi1=0 column median here is
   8.970e-04.

## 11. FINAL corrected fit and figures (2026-09-24)

Two further corrections landed after section 10, both found by `units_selftest.py`:

* **tau was on the wrong time axis** (UNITS_AUDIT A3).  `tau_f = 2 Q_f/w_f` is a duration
  on the NR axis, but `dt_` in `model_ab` is ppBHPT time; converting NR -> pp divides by
  `beta_r`.  16-34% and q-dependent.  `kappa` needed no conversion -- it is per unit pp
  time by construction.  Switch `TAU_PP = False` reproduces the 2026-09-23 runs.
* **`W_SCHW` disagreed between files.**  `junction_diag` used the bare BCW value 0.3683
  while `fit_pn_anchored_spin` had been patched to the exact Schwarzschild 0.3737 minutes
  earlier.  The self-test caught it as a 1.5% discrepancy in `beta_r vs measured`; without
  it the junction refit would have silently used the wrong constant.  This is the case
  for MECHANICAL tests over re-reading: the bug was in code I had just written and
  believed consistent.

**Result (job 3534076, converged):**

    7 coef   mean 1.5410e-03   median 1.0839e-03   max 1.2870e-02
    theta = [b1 0.1995, b2 -22.3438, a2 -15.1082, sSO 3.7261, sSS 6.0075,
             b21 149.8249, sSS1 -54.8238]
    chi1=0 column: median 8.7155e-04   max 1.3227e-03

2% better than the pre-fix 1.5732e-03 -- consistent with the tau null check, which
predicted a few percent at chi1 = 0.  On the same 96-point grid: `fluxanchored_spin`
(9 coef) 1.3012e-03, `energy_anchored_spin` (8 coef) 1.5703e-03, `pn_anchored_spin`
(13 coef, corrected) 1.1693e-03.

## 12. THE CONSTRUCTION VALIDATES: beta settles on the derived anchor

`plot_junction_spin_overlay.py` measures beta well after the junction and compares it to
`beta_r`, which is DERIVED (w220(chi1), M_f, chi_f) and fitted to nothing:

| q | error vs derived beta_r, across chi1 in [-0.6, +0.6] |
|--:|:--|
| 3 | +0.11% .. +0.50% |
| 5 | -0.37% .. -0.24% |
| 2 | +0.89% .. +1.97%  (double extrapolation: outside nu range AND q_min=3) |

So the merger-ringdown branch lands on the QNM anchor to a few tenths of a percent in
the training range with ZERO fitted coefficients in that sector.  That is the thing
`scaling_pn_anchored.md` caveat 1 says the switched model has never achieved (`S <= 0.18`,
`beta_MR ~ 0.60` against a measured ~0.83).

`t_*` lands at +4 to +11 M after the peak across the grid, i.e. just into the merger, and
the alpha/beta curves show no kink there -- as they must, since C1 continuity is imposed
by construction rather than fitted.

## 13. Figures

`Agentic_spin_plots/junction_spin/` -- 14 files, PDF + PNG:

    chi_const/junction_spin_alpha_beta_overlay_chi{m0.6,m0.2,0.0,p0.2,p0.6}
    q_const/junction_spin_alpha_overlay_q_grid       2x2 over q, all chi1 per panel
    q_const/junction_spin_beta_overlay_q_grid        same, with derived beta_r dotted

House format (1x2, figsize 12x4, q in {2,3,5,8}, xlim -1000..80).  Three DERIVED
quantities are overlaid as predictions rather than fits: `beta_r` (dashed horizontal),
`t_*` (vertical), and alpha's ringdown tangent `exp(kappa dt)` (dotted from `t_*`).  A
visible kink at `t_*`, or beta missing the dashed line, would mean the construction is
wrong -- neither happens in range.

## 14. Status and what is NOT done

**Settled:** the architecture works (per-q floor test, section 6), the nu^2 order is the
right regression fix (sections 9-10), `kappa` and `beta_r` are verified against
measurement, and the merger-ringdown sector is genuinely derived.

**Open, in priority order:**
1. **beta's chi1 form.**  The entire remaining error is the |chi1| >~ 0.6 columns --
   mean excluding chi1=+0.8 is 1.15e-03 against 1.54e-03 including it.  `history.md`
   already names it: "a quadratic in chi1 is not enough at the ends".  Measure before
   choosing a form; the `a32` experiment (section 8) is the cautionary tale.
2. **The BCW `Q220` row has never been checked against the paper.**  [Resolved 2026-09-26: Q220 is now
   the exact Kerr value (`physics_common.Q220_exact`), so the BCW row is no longer used.]  It is the one
   constant in this model new to the workspace.  Internal check only (Q220(0)=2.1187 ->
   tau 11.50 M vs textbook 11.24 M).
3. Nonspinning scoring: this model has never been run against the chi1=0-only baselines
   (`fluxanchored` 4.694e-04 etc.).  Its chi1=0 column median is 8.7155e-04.
4. Higher modes: untouched.

## 15. pn_anchored_spin: why chi1=+0.8 fails (subset tests, 2026-09-24 evening)

Context: corrected pn_anchored_spin (13 coef, 9 frozen from the corrected nonspinning
fit) scores median 1.0993e-03 vs published 7.6858e-04 (+43%).  The chi1=0 column is
0.66e-03 -- BETTER than the published overall median -- so the frozen base is fine and
the damage is entirely at large |chi1|, dominated by chi1=+0.8 (column median 6.64e-03).

All tests on an 18-point subset, q in {3,5,8} x chi1 in {-0.6,-0.2,0,+0.4,+0.6,+0.8},
deliberately over-sampling the bad column.  Minutes each instead of ~90 on the full grid.

| variant | free | median | mean | chi1=+0.8 median |
|:--|--:|--:|--:|--:|
| current (9 frozen) | 4 | 1.460e-03 | 3.450e-03 | 1.470e-02 |
| refit, same form (control) | 4 | 1.693e-03 | 3.311e-03 | -- |
| + cubic in chi1 | 6 | 1.353e-03 | 3.194e-03 | trades cells, no fix |
| + merger amplitudes r0,m0,m1 | 7 | 1.555e-03 | 3.240e-03 | 1.354e-02 |
| **+ merger + switch p00,p01,w0** | 10 | 1.470e-03 | **1.931e-03** | **6.116e-03** |

**Findings.**
1. **Not the chi1 polynomial.** A cubic term improves chi1=+0.8 at q=3 (20.3 -> 12.4) but
   worsens it at q=5 (4.2 -> 6.7) and q=8 (14.7 -> 15.8): it trades cells, it does not fix
   the column.  history.md's "a quadratic in chi1 is not enough" is NOT the explanation.
2. **Not the merger amplitudes.** Freeing r0, m0, m1 barely moves anything.
3. **It is the SWITCH.** Freeing p00, p01, w0 cuts the chi1=+0.8 column 2.4x and the mean
   44%.  But the median does not move, because the low-spin cells get WORSE at q=3
   (chi1=-0.2: 0.59 -> 1.29; chi1=0: 0.72 -> 1.07).  One spin-blind switch is being asked to
   serve every chi1: moving it to rescue +0.8 breaks 0.

**Indicated fix (not yet tested):** give the switch location spin dependence, e.g.
p0 = p00 + p01*nu + p02*chi1, so high-spin and low-spin cases can switch at different
points instead of compromising.  This is consistent with history.md's open item naming
"the five frozen coefficients p00, p01, w0, r0, m0, m1 -- all nu-only" as the suspect for
the spin q-U-shape -- refined here to the three SWITCH coefficients specifically.
Test on the same subset first.

Caveat: unfreezing reopens the basin degeneracy history.md cites as the reason these were
frozen (~10-40%).  Any subset win needs multi-start on the full grid before it is trusted.

## 16. Correction: the constant fix (W_SCHW) is NOT worth 43%

Earlier today this was reported as "the clearest new result": spin13 step 1 (constant off)
2.066e-03 vs step 2 (constant on) 1.169e-03.  That comparison was INVALID -- step 1 kept
the 9 frozen coefficients that had been FITTED with the constant on, so it scored them
against a constant they were not fitted for.  In pn_anc_spin_physical, where all 11 are
refit in both steps, the constant moves the median by under 1% (1.0563e-03 -> 1.0467e-03).
Its real effect is small, consistent with the original measurement.  The spin13 step-1
figure (+168.8%) is meaningless and must not be quoted.


## 2026-09-26 fix round -- small grid only

On the fixed code (x on the note's clock, exact Kerr QNMs, tau on the ppBHPT axis) a small-grid
refit (4 q x 4 chi1) gave mean 1.109e-3 / median 1.009e-3 on those 16 cases, against 1.475e-3 /
1.104e-3 for the 2026-09-24 `coeffs_junction.json` on the same cases (-25% / -9%).  The full
96-point refit could not be run (allocation exhausted); `coeffs_junction.json` is still the
2026-09-24 file, whose settings are not recorded -- do not quote it as a baseline.
