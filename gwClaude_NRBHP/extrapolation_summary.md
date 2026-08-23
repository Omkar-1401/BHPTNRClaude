# Low-q extrapolation summary (mismatch at q = 2.5, 2.0, 1.5)

"Efficiency" here = the mismatch `mathcalE` of each q-dependent calibration when
**extrapolated below its training range** to q = 2.5, 2.0, 1.5. Every model listed
was trained/calibrated at q ≥ 2.5, so all three columns are extrapolation.

`mathcalE = (‖h₁‖² + ‖h₂‖² − 2·Re⟨h₁,h₂⟩) / (2‖h₁‖²)`.

## How these numbers were produced (one consistent harness for ALL models)

- **Reference:** NRHybSur3dq8 (2,2) mode, nonspinning.
- **Window:** the full common time support (down to ≈ −5000 M), *not* a near-merger
  window.
- **Free parameters at evaluation:** only the two alignment nuisances (time shift
  `t0`, phase `phi0`), which carry no q-information. **Amplitude is NOT re-optimised**
  — it is whatever the model predicts. (This matters: see the correction note below.)
- **`invalid`** = the extrapolated coefficients give a non-monotone time map or fail
  the coverage guard — the model breaks (treat as ≫ 1), not a finite large mismatch.
- The gwAgentic models and the native BHPTNRSurrogate calibration were re-evaluated
  here on this harness and **validated in-range** (q=5) against reported errors.

## Table

| model | regression form | calib q | in-range mathcalE† | q = 2.5 | q = 2.0 | q = 1.5 |
|:---|:---|:---:|---:|---:|---:|---:|
| **PN_opt_nu_Pade / poly_nu** | deg-4 poly in nu, PP-anchored | [3,8] | 7.05e-5 | **3.9e-4** | 6.46e-2 | invalid |
| PN_opt_nu_Pade / cheb_1q | deg-4 Cheb in 1/q, PP-anchored | [3,8] | 7.42e-5 | 2.14e-3 | invalid | invalid |
| PN_opt_nu_Pade / pade_nu | [1/1] Padé in nu, PP-anchored | [3,8] | 1.37e-3 | 0.146 | 0.369 | 0.934 |
| PN_opt_remnant_partial | cubic in chi_f + physical β_r | [3,8] | 6.21e-5 | 2.27e-3 | 5.84e-2 | 0.402 |
| PN_opt_creative_q_dep | cubic in 1/q | [3,8] | 7.66e-5 | 1.23e-2 | 0.214 | invalid |
| **const α-β — BHPTNRSurrogate native** | quartic in 1/q, const α,β (→SXS) | [2.5,1e4] | 1.07e-3 | 3.90e-3 | **8.87e-3** | **0.353** |
| const α-β — gwAgentic `scaling_qdep` | quartic in 1/q, const α,β (→NRHyb) | [3,10] | 1.07e-3 | 4.05e-3 | 2.44e-2 | 0.704 |
| gwAgentic q-&-t-dep | linear-in-t α,β; quartic in 1/q | [3,8] | 9.39e-4 | 1.14e-2 | 0.337 | invalid |

Bold = best in that column. The last three rows are the same constant-α-β /
linear-in-time model class (the native BHPTNRSurrogate calibration and gwAgentic's
`scaling_qdep` share the identical quartic-in-1/q form, differing only in fitted
coefficients).

† In-range column: **median over the q∈[3,8] training grid** for the five
PN_opt_* rows; **single-point q=5 value** for the const-α-β and q-&-t-dep rows
(which were validated at q=5). The three extrapolation columns (q=2.5/2/1.5) are
single-point values for every row, all on the identical full-window, t0/phi-aligned,
amplitude-fixed harness vs NRHybSur3dq8.

## Correction to an earlier number

An earlier version of this comparison reported the native BHPTNRSurrogate at
**3.3e-2 (q=1.5)** and 6.5e-3 / 1.1e-2 at q=2.5 / 2.0. Those were computed on a
*different, more generous* harness than the other models — a near-merger window
`[-1000, 90]` (which hides the inspiral phase error that dominates at low q) and an
extra overall-amplitude re-optimisation (a DOF the fixed calibration does not have).
On the **consistent** harness used for every row above (full window, amplitude fixed
by the model), the native model gives **0.353 at q=1.5**. The corrected conclusion:
the constant α-β model does **not** stay accurate at q=1.5 — it fails there like
everything else.

## The picture in one paragraph

**q=2.5:** our high-accuracy time-dependent models win — `poly_nu` reaches 3.9e-4,
~10× better than the constant α-β model. **q=2:** the constant α-β model (native
coefficients, 8.9e-3) is the best — its 2-parameter q-dependence extrapolates more
stably than our 6+ flexible curves, so it beats our time-dependent models (~6e-2)
here. **q=1.5:** everything fails — 35–70% for the finite models, invalid for the
rest; `remnant_partial` (40%) is comparable to the native constant α-β (35%). There
is no usable model at q=1.5.

## Per-model caveats

**PN_opt_nu_Pade / poly_nu** — best model at q=2.5 by a wide margin; PP anchor keeps
α/β sane. But `p0`, `w` are unanchored, so their nu-polynomial rolls over below
q≈2.2 and the time map goes non-monotone at q=1.5 → invalid. Good to ~q=2.25.

**PN_opt_nu_Pade / cheb_1q** — included to prove the coordinate point: at equal
degree it matches poly_nu in-range but is already invalid at q=2, because 1/q is the
wrong extrapolation variable. Do not use for q<3.

**PN_opt_nu_Pade / pade_nu** — [1/1] rational too stiff to fit in-range (1.4e-3) and
poor everywhere in extrapolation. Listed for bake-off completeness.

**PN_opt_remnant_partial** — most graceful of *our* models at deep extrapolation; the
only NRHybSur3dq8-fit model of ours that stays finite at q=1.5 (40%), because chi_f
is bounded/monotone and β_r has an exact surfinBH QNM anchor. Best in-range median
(6.21e-5).

**PN_opt_creative_q_dep** — the original 1/q-cubic model; catastrophic extrapolation
(21% at q=2, invalid at q=1.5) from ill-conditioned 1/q coefficients. This is the
model whose failure motivated the whole nu/anchoring investigation.

**const α-β (native BHPTNRSurrogate & gwAgentic `scaling_qdep`)** — one model, two
coefficient sets. Wins at q=2 on extrapolation stability (few, stiff
DOF). Caveats: (1) native is calibrated to *SXS* NR but scored here against
*NRHybSur3dq8*, so its in-range 1.07e-3 is dominated by the two NR surrogates
disagreeing, not model error — its accuracy vs its own target is better; (2) native's
stated validity floor is **q=2.5** (`X_min=log10(2.5)`), so q=2 and 1.5 are out of
bounds and it emits warnings there; (3) both coefficient sets fail at q=1.5.

**gwAgentic q-&-t-dep** — α,β linear in normalised time, quartic in 1/q. Validated
in-range (9.4e-4 at q=5). More flexible than the constant model and correspondingly
worse in extrapolation (34% at q=2, invalid at q=1.5). (Two further gwAgentic models,
`PN_opt_q_dep` 12-param Hermite and `physical_smooth_qdep`, were not recomputed; both
are 1/q-cubic in q and expected to track `PN_opt_creative_q_dep`.)

## Cross-cutting caveats

- **All three columns are extrapolation for every model** (none calibrated below
  q=2.5, most below q=3).
- **A BHPT floor sits under everything.** Even a *direct* per-q fit of our model
  (using the q=2 data itself) bottoms out at 3.4e-3 at q=2, 12% at q=1.75, 36% at
  q=1.5 — a limitation of raw BHPT at comparable mass, independent of the q-regression.
- **The regression coordinate, not model complexity, governs extrapolation.** The
  *simplest* model (constant α-β) still fails at q=1.5 (35–70%) because its
  q-dependence is quartic in 1/q. Bounded coordinates (nu, chi_f) + endpoint anchoring
  help; unbounded 1/q polynomials do not.
- **NRHybSur3dq8 is itself only validated to q=8** with ~1e-4 intrinsic error and is
  an extrapolated reference below q=3, so absolute low-q numbers should not be
  over-interpreted.

## Bottom line

- **q ≥ 2.5:** `PN_opt_nu_Pade / poly_nu` is the best model.
- **q = 2:** the constant α-β model (native BHPTNRSurrogate coefficients) is best,
  by extrapolation stability — but only ~0.9%, and it is the same model as gwAgentic's
  simplest fit.
- **q ≤ 1.5:** no model is usable (≥ 35% or invalid); raw BHPT itself has given out.

Closing the q=2 gap for our models needs low-q calibration anchor points
(interpolation, useful down to ~q=2.25 before the BHPT floor dominates), or a
BHPTNRSur-style low-dimensional bounded-coordinate calibration in the extrapolation
regime — not more flexibility.
