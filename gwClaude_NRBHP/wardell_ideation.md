# Comparing the alpha-beta calibration against second-order self-force (Wardell et al.)

**Paper:** Wardell, Pound, Warburton, Miller, Durkan, Le Tiec,
"Gravitational waveforms for compact binaries from second-order self-force theory,"
arXiv:2112.12265 (PRL; v2 2023-03-30).  1PA waveforms from a two-timescale expansion of
the Einstein equations through second order in the mass ratio, nonspinning quasicircular.

**Goal (user, 2026-08-26):** eventually compare this workspace's calibration results
against that paper — specifically the expansion below their Eq. (3),

```
F      = nu^2 F1(x) + nu^3 F2(x) + O(nu^4)          flux at fixed x
E_bind = nu M [ E0(x) + nu E_SF(x) + O(nu^2) ]      binding energy at fixed x
x      = (M Omega)^(2/3)
```

This file collects the ideas, the mapping, and the plan.  Running log; newest at the
bottom of each section.

---

## 0. Definitions

Every symbol used below, so this file is self-contained.

**Masses and mass ratios**

```
m1, m2       component masses, m1 >= m2;  M = m1 + m2
q            = m1/m2  >= 1                (workspace convention; the PAPER uses q = 1/eps)
eps          = m2/m1  <= 1                the paper's small expansion parameter
nu           = m1 m2 / M^2 = q/(1+q)^2    symmetric mass ratio; nu -> 0 as q -> inf
X1           = q/(1+q) = m1/M             larger-mass fraction
X1^(6/5)     the Newtonian chirp factor: the leading-order conversion between the
             ppBHPT's mass/time units (m1-based, its own mass-ratio parameter eta_pp)
             and NR's (M-based, nu).  Derived, not fitted; equals (nu/eta_pp)^(3/5).
             Verified as alpha's and beta's early-inspiral level to 0.1-0.5% throughout
             the workspace.
```

**Waveforms and calibration (this workspace)**

```
h_pp(t)      the ppBHPT waveform: point-particle black hole perturbation theory,
             BHPTNRSur1dq1e4 surrogate with calibrated=False.  Adiabatic inspiral driven
             by the FIRST-ORDER flux; time in units of m1.
h_NR(t)      the "true" waveform: NRHybSur3dq8 surrogate (NR hybridised with PN at early
             times); time in units of M.
alpha(t)     amplitude calibration:  h_NR(tau) ~ alpha(t) * e^{i phi0} * h_pp(t)
beta(t)      time-map rate:          tau(t) = t0_nr + integral beta dt'
             so beta = d tau/dt; via phase matching, beta = omega_pp(t)/omega_NR(tau).
tau(t)       the BHPT->NR time map
t0_nr, phi0  per-q nuisances: time offset (fitted) and constant phase (analytic)
```

**Frequency coordinate**

```
omega_GW     GW angular frequency = d/dt unwrap(arg h_22)
Omega        orbital angular frequency = omega_GW/2 for the (2,2) mode
x            = (M Omega)^(2/3)    dimensionless inverse-separation / PN velocity-squared
             coordinate.  x_pp uses the ppBHPT's own frequency (computable without the
             answer); x_NR = beta^(-2/3) x_pp.  In this workspace x is measured by
             savgol-differentiating the unwrapped phase, window set in physical M.
```

**Fluxes and energies**

```
F  (paper: script-F)   total GW energy flux to infinity, dE/dt radiated
F1(x), F2(x)           its expansion at FIXED x:  F = nu^2 F1(x) + nu^3 F2(x) + O(nu^4).
                       F1 = first-order (test-particle-scaled) flux — what drives the
                       ppBHPT.  F2 = the second-order self-force correction — what the
                       ppBHPT is missing.  (Paper notation: F^(1), F^(2); F^(1) also
                       splits into infinity + horizon pieces there.)
E_bind                 binding energy;  E_bind = nu M [E0(x) + nu E_SF(x) + O(nu^2)]
E0(x)                  test-mass (geodesic) binding energy
E_SF(x)                first-order-self-force (conservative) correction to it
E(t), Eoft             THIS WORKSPACE's energy drive: cumulative radiated energy of the
                       ppBHPT from the window start (gw_remnant), units of M.  NOT the
                       same object as E_bind; E(t) = integral of the ppBHPT's F.
Ehat                   E(t)/E_tot in [0,1];  Edot = dE/dt = the ppBHPT's instantaneous F
```

**Paper machinery**

```
0PA / 1PA    post-adiabatic orders.  0PA = adiabatic, first-order dissipation only
             (phase ~ eps^-1 phi_0).  1PA adds eps^0 phi_1: second-order dissipative +
             first-order conservative effects.  The ppBHPT surrogate is closest to 0PA.
1PAT1/1PAT2/1PAF1   their three model variants (time-domain x-fixed / time-domain
             t-fixed / frequency-domain).  1PAT1 is the accurate one.
h1_lm, h2_lm  first- and second-order waveform mode amplitudes at fixed x (their Eq. 7):
             h_lm = [nu h1_lm + nu^2 (h2_lm + h1_lm - (2x/3) dh1_lm/dx)] e^{-im phi_p}
```

**Terminology warning: two different "anchored"s**

```
PP-anchored   (point-particle) — a nu-REGRESSION form: shape coefficients vanish at
              nu -> 0, a_k(nu) = A_k1 nu + A_k2 nu^2, no constant.  REQUIRED by the 1PA
              expansion (it IS the statement drift = nu*a(x) at leading order).
pn_anchored   (post-Newtonian) — a MODEL (BHPTNRPNAnchored.py): q-dependence imposed
              analytically from the PN note.  ALSO consistent with 1PA: PN and GSF are
              complementary expansions (PN in x at all nu; GSF in nu at all x), and the
              note's 55/42*nu*x term is the small-x Taylor limit of Wardell's
              second-order functions.  Corollary: the measured c1(x) of section 4c must
              reproduce the note's PN coefficients as x -> 0 — a free cross-check.
free-shape    the x_drive variant with a CONSTANT term in the shape coefficients.  The
              only one of the three that is INCOMPATIBLE with the 1PA expansion.
```

**Workspace-specific quantities referenced below**

```
b_E          the fitted coefficient of beta's drift term, beta = b_PP (1 + b_E * drive)
C_lm(nu)     the hm backbone's inter-mode mass factor |X2^(l+eps_lm-1) + (-1)^m X1^(...)|
             (X2 = 1/(1+q));  C_33 = (q-1)/(q+1) = sqrt(1-4nu)
R(x)         the flux ratio diagnostic defined in section 5, step 1
```

---

## 1. The identification

`F1(x)` **is** the first-order (BHPT) flux — the flux driving the ppBHPT surrogate, up to
mass-unit conversion.  It is not that F1 "contains incomplete-physics terms"; truncating
at F1 is the incomplete physics, and `nu^3 F2(x)` is exactly what the ppBHPT is missing.
Therefore, at fixed x and after unit conversion:

```
F_NR(x) / F_pp(x) - 1  ~  nu * F2(x)/F1(x)  +  O(nu^2)
```

The quantity below their Eq. (3) is directly measurable from the two waveform families
this workspace already has cached.

## 2. Why beta at fixed x IS the flux comparison

Adiabatically `xdot = F / (-dE_bind/dx)`, and the gauge-free measurement
(`beta_drift_tests/alpha_beta_of_x.py`) already defines beta through the chirp-rate
ratio, `beta^(5/3)(x_NR) = xdot_pp(beta^(2/3) x_NR) / xdot_NR(x_NR)`.  Expanding:

```
beta(x,q) / X1^(6/5) - 1  ~  -(3/5) nu [ F2/F1 - (binding-energy SF term) ](x)
                             + PN argument-shift terms + O(nu^2)
```

* The X1^(6/5) prefactor is the Newtonian mass-unit conversion (already derived and
  verified at the 0.1-0.5% level throughout the workspace).
* The PN argument-shift part is what `alpha_beta_pn_scaling_note_revised.pdf` derives
  (the 55/42 * nu * x slope in alpha); it is O(nu) too and must be separated before
  attributing the residual to F2.
* The exact combination (signs, the E_SF' term, the argument shift) has NOT been derived
  carefully yet — do that before any quantitative claim.  Only the structure is fixed:
  **the drift at fixed x, divided by nu, should be a q-universal function of x.**

Same for alpha against their Eq. (7): the relative amplitude correction is
`nu * h2_lm/h1_lm` plus a `-(2x/3) dh1/dx` frequency-shift term — the latter is
structurally the same object as the hm backbone's `beta^(-2/3)` frequency rescaling.

## 3. Do we need a model with alpha AND beta driven by Edot?  NO

(User's question, 2026-08-26.)  The paper's expansion is in nu **at fixed x** — Edot is
the thing being expanded, not the expansion variable.  An Edot-driven calibration would
hang the model on a drive whose own shape is q-dependent, scrambling the nu-ordering the
comparison needs (same failure mode the `Ehat(x)` universality measurement exposed for
the E-drive: ~2x spread across q at mid-inspiral, `Agentic_plots/alpha_beta_vs_x/`).

The right family is the one already built: `fit_scaling_x_drive.py` — alpha(x,q),
beta(x,q) as functions of x with nu-regressed coefficients.  No new model is needed for
the comparison; what is needed is the PREDICTION to overlay.

## 4. Supporting evidence already on disk

* Measured `|alpha excursion| ~ nu^0.99` (`alpha_beta_of_omega_v2.log`) — exactly the
  linear-in-nu scaling their Eq. (7) predicts for the relative amplitude correction.
* Measured `|beta excursion| ~ nu^2.04` — but this is MERGER-ALIGNED over the full
  window, not the fixed-x inspiral quantity, so it is not directly the 1PA prediction.
  The discrepancy (nu^2 vs the naive nu^1) is itself worth resolving; do not quote it as
  a contradiction until the fixed-x version is measured.
* The paper re-expands in nu at fixed M to restore the m1 <-> m2 symmetry — a
  first-principles justification for this workspace's empirical finding that nu (not 1/q)
  is the coordinate that extrapolates (`extrapolation_summary.md`, 2026-07-09).
* Their odd-m resummation factor `sqrt(1-4nu) = (m1-m2)/M` is the (3,3) case of the hm
  backbone's C_lm(nu) — the backbone carries the odd-m antisymmetry by construction that
  their perturbative amplitudes needed restored by hand.

## 4b. The calibration-implied F2 (user's proposal, 2026-08-26)

**Idea (user):** write the comparison as
`F_NR = nu^2 F_BHPT + nu^3 * (a term calculated from the time-dependent model, from
alpha and beta changing with time)` — i.e. derive the F2-analogue FROM the calibration
and compare it to the paper's F2.

**This works, via a closed-form identity.**  From the ansatz
h_NR(tau) = alpha(t) e^{i phi0} h_pp(t), dtau/dt = beta:

```
dh_NR/dtau = (1/beta) [ alpha * hdot_pp + alphadot * h_pp ]
```

In the adiabatic band hdot_pp ~ i omega_pp h_pp dominates and
alphadot/(omega alpha) ~ 2e-4 << nu F2/F1 ~ 1e-2..1e-1, so

```
F_NR(tau(t)) = [ alpha(t)^2 / beta(t)^2 ] * F_pp(t)      (adiabatic; exact given ansatz)
```

Master identity.  All unit conversion lives inside alpha^2, beta^2; it is a pure
waveform statement, INDEPENDENT of the binding energy.

**Expansion.**  With alpha = X1^(6/5)[1 + nu a(x)], beta = X1^(6/5)[1 + nu b(x)]
(a, b = the q-universal drift shapes, readable off the x_drive nu-regression
coefficients / nu), the X1^(12/5) cancels in the ratio, and matching at equal x
(undoing beta's frequency shift x_pp = beta^(2/3) x_NR in F1's argument):

```
F_NR = nu^2 F1(x) + nu^3 F2_calib(x) + O(nu^4)

F2_calib / F1 = 2 [ a(x) - b(x) ]                          amplitude-ratio piece
              + (d ln F1 / d ln x) * (2/3) * b(x) * (+-?)  frequency-shift piece
```

The second piece's exact coefficient/sign needs the careful derivation (same TODO as
section 2); the two-piece structure is fixed.

**Two consequences:**

1. **Internal consistency check, no external data.**  F2_calib (from the fitted
   alpha, beta) must agree with the DIRECTLY measured flux ratio (section 5, step 1) if
   the calibration is faithful.  Disagreement = the fit distorting the flux to buy
   phase — a diagnostic of the calibration itself, prior to any physics claim.
2. **Separating F2 from E_SF.**  The flux ratio alpha^2/beta^2 tests F2 ALONE (no
   binding energy in the identity).  beta's drift comes from the chirp rate
   xdot = F/(-E_bind'), so it carries a COMBINATION of F2 and E_SF'.  Measuring both and
   subtracting isolates the conservative self-force piece E_SF empirically — a
   decomposition neither measurement gives alone.

**Validity:** only where hdot_pp ~ i omega h_pp, i.e. the adiabatic inspiral.  Near
merger the alphadot and amplitude-derivative terms revive and the flux interpretation
degrades — consistent with the paper stopping at ISCO.

## 4c. F2_calib must be nu-INDEPENDENT (user's requirement, 2026-08-26)

In the paper's expansion F2(x) is a pure function of x by construction.  If the
calibration-implied F2_calib comes out depending on nu, the identification with
Wardell's expansion fails.  This requirement has consequences:

**It fixes the nu-form of the calibration.**  F2_calib nu-independent requires

```
alpha/X1^(6/5) - 1 = nu * a(x) + nu^2 * (...)        and likewise beta
```

i.e. the drift must vanish LINEARLY in nu as nu -> 0 with a q-independent leading
shape — which is exactly the PP-ANCHORED shape-coefficient form
(a_k(nu) = A_k1 nu + A_k2 nu^2, no constant).  **The `--free-shape` x_drive variant —
the one that won every mismatch comparison — is structurally incompatible with the 1PA
expansion**: its constant term says the drift survives at nu -> 0, where the ppBHPT is
exact.  The free-shape empirical win therefore needs explaining, not adopting.
Candidates: (i) merger contamination — the fits score beyond the adiabatic band where
the 1PA ordering does not apply (most likely); (ii) the fit absorbing surrogate error,
which does not organise in nu; (iii) genuine failure of the mapping.  The clean test
must use inspiral-band-only data.

Prior evidence for the required linearity: measured |alpha excursion| ~ nu^0.99
(alpha_beta_of_omega_v2.log).  The beta ~ nu^2.04 puzzle (section 7) is the open case —
but that is the merger-aligned full-window quantity, not the fixed-x one.

**The falsifiable protocol.**  At each fixed x on a grid in the adiabatic band
(x ~ [0.06, 0.13]), over the 64-q grid, fit

```
D(x; nu) = F_NR/F_pp - 1 = c0(x) + c1(x)*nu + c2(x)*nu^2
```

* c0(x) must be ZERO at noise level.  A significant nu^0 term falsifies the
  identification — it can only be surrogate error or calibration artefact, never 1PA.
* c1(x) = the measured F2/F1 plus PN-conversion terms (those are also linear in nu —
  the note's 55/42 nu x — so they shift the VALUE vs the paper's F2 but not the
  nu-ordering).
* c2(x) = a bound on / measurement of the NEXT order (nu^4 F3) — the 64-point grid is
  dense enough in nu to separate it.

**Spurious nu-dependence to guard against:** the x_drive shapes are functions of
s = x - x_start(q), and x_start varies with q (0.033..0.046).  Convert everything to
ABSOLUTE x before the per-x fits, or the window origin masquerades as nu-dependence.

## 4d. Use pn_anchored to compute both sides (user's choice, 2026-08-26)

**Decision (user): compute the two sides of F = nu^2 F1(x) + nu^3 F2(x) with the
pn_anchored model, and see what F2 turns out to be.**

Why this is the right model: pn_anchored is the only model in the workspace whose drift
is nu * (function of x) BY CONSTRUCTION — so its implied F2 is nu-independent
automatically, satisfying 4c before any test.  Its inspiral branch is closed-form
(BHPTNRPNAnchored.py):

```
alpha_insp = X1^(6/5) * [ 1 + (55/42) nu x + nu a2 x^2 ]     # 55/42 FIXED at 1PN
beta_insp  = X1^(6/5) * [ 1 + nu (b1 x + b2 x^2) ]
a2, b1, b2  from pn_anchored_results/coeffs.json  (3 of its 9 fitted constants)
```

Through the master identity (4b), the calibration-implied second-order flux is then
ANALYTIC and quadratic in x:

```
a(x) = (55/42) x + a2 x^2          b(x) = b1 x + b2 x^2

F2_calib(x)/F1(x) = 2[a(x) - b(x)] + (freq-shift term ~ b(x) dlnF1/dlnx)
                  = 2[(55/42 - b1) x + (a2 - b2) x^2] + shift term
```

Leading slope 2(55/42 - b1): one derived coefficient, one fitted.  "What F2 turns out
to be" = evaluate this expression and overlay on (i) the measured flux ratio (step 1)
and (ii) eventually Wardell/Warburton's F2.

**Caveats specific to pn_anchored:**
* The switch: alpha = (1-S) alpha_insp + S alpha_mr.  Closed form holds only where
  S ~ 0.  S <= 0.18 even at merger, so negligible in the adiabatic band — but CHECK at
  the band edge, don't assume.
* xc = min(x, x_clip=0.26): irrelevant in the band; remember if pushed toward ISCO.
* a2, b1, b2 were fitted FULL-WINDOW, so merger leakage into them is possible.  If the
  comparison shows a discrepancy, the first control is refitting those three on the
  inspiral band alone.
* The MR branch is under-engaged (S <= 0.18) and alpha_mr(q=2) ~ 1.0 unphysical — but
  the comparison never uses the MR branch, so this does not matter here.

## 4e. Clarification (user, 2026-08-26): the core comparison needs NO runs

The 4d comparison is already-fitted coefficients (a2, b1, b2 + the analytic 55/42)
against Wardell's published F2(x).  Evaluate a quadratic, overlay, one plot script.
The measurement runs in section 5 are VALIDATION, not the comparison — they only become
necessary if the coefficient comparison disagrees and one must separate model error
from physics.

What actually remains, none of it compute:
1. **Derive the frequency-shift term** — NOT optional: dlnF1/dlnx ~ 5 (Newtonian
   F1 ~ x^5), so the shift piece ~ (10/3) b(x) is the SAME SIZE as the main 2[a-b]
   term, sign-critical.  Pencil work.
2. **Derive the conversion layer** — which pieces of a(x), b(x) are mass-scale
   conversion beyond X1^(6/5) (the 55/42 slope is partly this, per the note) and which
   are their F2/E_SF physics.  Same derivation as (1).
3. **Acquire their F2(x)** — Warburton+ arXiv:2107.01298 data, likely via the Black
   Hole Perturbation Toolkit or Zenodo.  A download, not a run.

## 4f. Staleness check of pn_anchored, and its EXACT definitions (2026-08-26)

**User: rerun pn_anchored (or at least pin down proper definitions) before building the
comparison on month-old coefficients.**  Job 3448537: (1) evaluates the SHIPPED
coefficients and checks the documented mismatches reproduce on the current env
(median 6.16e-4, max 7.67e-4, q2 2.55e-3); (2) fresh `--refit` from scratch, written to
`pn_anchored_results/coeffs_refit_20260826.json` with the shipped `coeffs.json`
preserved as canonical (backup taken first); (3) prints the theta comparison, flagging
any parameter moving >5% — b1, b2, a2 are the three that matter here.

**Exact definitions inside pn_anchored (verified from source, 2026-08-26):**

```
x     (get_x / _get_x): unwrap arg h_22^pp; savgol smooth the PHASE with a FIXED
      401-SAMPLE window (order 3, mode="interp"); omega = |gradient(ph, t)|;
      x = (0.5*omega)^(2/3), clipped to [1e-8, 0.6].
      NOTE: 401 samples on the ppBHPT dt=0.2 M grid = 80 M of smoothing, and
      gradient-of-smoothed rather than savgol's analytic derivative — DIFFERENT from
      the 20 M analytic-derivative x used by x_drive/alpha_beta_of_omega_v2.  One-sided
      (ppBHPT only), so the v1 two-sided asymmetry bug does NOT apply; but a2, b1, b2
      are defined with respect to THIS x.  Negligible in the adiabatic band; state it.
xc    = min(x, 0.26)
S     = sigmoid((p_loss - (p0_0 + p0_1*nu))/w0), p_loss = 0.5*(Ehat+Jhat) wf-flux
alpha = (1-S) * X1^(6/5) * [1 + (55/42) nu xc + nu a2 xc^2] + S * alpha_mr
beta  = (1-S) * X1^(6/5) * [1 + nu (b1 xc + b2 xc^2)]       + S * beta_r_phys(q) * r0
fit   : global joint Powell, 9 params, even-nu 12 q in [3,8], full window
shipped theta: b1=0.352, b2=8.553, a2=-22.9193 (+ 6 MR/switch params)
```

## 4g. THE DERIVATION — shift term and conversion layer resolved (2026-08-29)

Setup: h_NR(tau) = alpha h_pp(t), dtau/dt = beta, with pn_anchored's
alpha = X1^(6/5)[1+nu a(x)], beta = X1^(6/5)[1+nu b(x)].  The ppBHPT is exactly
first-order with amplitude parameter eps = m2/m1 = 1/q in m1 units, so
F_pp = eps^2 F1(x_pp).  Exact bookkeeping: nu^2 = eps^2 X1^4 and x_pp = beta^(2/3) x_NR.
The adiabatic identity F_NR(tau) = (alpha^2/beta^2) F_pp(t) at fixed x_NR = x:

```
nu^2 F1(x)[1 + nu R2(x)] = eps^2 [1+nu a]^2/[1+nu b]^2 * F1( X1^(4/5) (1+nu b)^(2/3) x )
```

Collecting O(nu), with L(x) := dlnF1/dlnx:

```
R2(x) = F2/F1 = 2 a(x) - 2 b(x) + (2/3) L(x) b(x) + (4/5)[5 - L(x)]
        --------  amplitude ratio | shift term      | conversion layer
```

* **kappa resolved: +10/3 at Newtonian order (L=5), POSITIVE sign.**  The conversion
  term vanishes identically at Newtonian order (sanity check passes).
* Newtonian-order quadratic, shipped theta (b1=0.352, b2=8.553, a2=-22.9193):
  `C1 = 55/21 + (4/3) b1 = 3.088`,  `C2 = 2 a2 + (4/3) b2 = -34.43`.
* **The conversion layer is NOT negligible beyond Newtonian**: with the test-mass PN
  flux, 5 - L(x) ~ 3.711 x - 18.85 x^(3/2) (1PN + tail), so
  c_conv = (4/5)(5-L) ~ -0.18 at x = 0.1 — about 60% of C1*x, opposite sign, and it
  SIGN-FLIPS inside the band at x ~ 0.039 (tail term overtakes 1PN).
* **L(x) needs no external input**: F1 IS the ppBHPT's own flux, so L is measurable from
  the cached waveforms.  First attempt (20 M smoothing) was too noisy — Edot oscillates
  on the ORBITAL timescale (documented in RESUME for F), so L(x) needs ORBIT-AVERAGED
  flux (smooth over ~1-2 orbital periods, 60-200 M).  Small refinement, not yet done;
  the PN expression above is the interim evaluation.

**Refit degeneracy caveat (job 3448537, 2026-08-29).**  The shipped theta reproduces its
documented mismatches to 3-4 digits (not stale), but an independent Powell refit landed
in a DIFFERENT, WORSE basin (in-range median 7.632e-4 vs 6.148e-4; q2 4.52e-3 vs
2.55e-3) with b1 -42%, b2 -32%, a2 -15%.  Individual parameters are loosely pinned by
the objective.  HOWEVER the combinations entering R2 are much stabler:
C1 = 3.088 (shipped) vs 2.890 (refit basin), C2 = -34.43 vs -31.35 — 7-9%.  Quote
R2's coefficients with ~10% uncertainty from this source; the shipped theta stays
canonical (verified restored to coeffs.json; refit preserved as
coeffs_refit_20260826.json).

**Step 3 RESOLVED (2026-08-29): the data path is WaSABI.**
WaSABI (Waveform Simulations of Asymmetric Binary Inspirals), Wardell, Mathews & Honet,
Zenodo DOI 10.5281/zenodo.17405583 — the public package implementing the 1PA models.
It carries the forcing functions F0(x), F1(x) of Wardell Eq. (4) (and the underlying
flux data), from which F2 is reconstructable.  Note F1 is itself directly comparable:
it is the chirp-rate correction, i.e. the combination beta's drift measures — so BOTH
of our observables get a counterpart (alpha^2/beta^2 <-> F2 flux; beta-drift <-> F1).
Also: arXiv 2303.18026 (SEOBNRv5 + 2SF fluxes) ingested these fluxes and documents them.

**Findings from reading Warburton+ 2107.01298 itself (2026-08-29):**
* Their expansion IS our target: F^SF_lm(nu,x) = nu^2 F1_lm(x) + nu^3 F2_lm(x) + O(nu^4),
  Newtonian-normalised by F^N_22 = 32 x^5 nu^2 / 5.
* **Their frequency variable is xbar = (M*varpi)^(2/3), varpi = Phidot_22/2 — the
  WAVEFORM half-frequency.  Ours is the same definition.  Frames match; no conversion.**
* **THE LOCK: their Fig. 7 names the leading O(nu^3) PN term of the (2,2) flux as
  55x/21.  Our derived leading term is 2*(55/42)x = 55x/21 — identical.**  The note's
  1PN amplitude slope, doubled by flux = amplitude^2, reproduces their PN coefficient
  exactly.  The x->0 cross-check closes by construction; the nontrivial comparison
  content is everything BEYOND 55/21: the fitted (4/3)b1 (=+0.47 shipped / +0.27 refit
  — note the refit basin is CLOSER to pure PN), the x^2 terms, tails.
* Their Fig. 4 already runs our section-4c protocol (subtract nu^2*1SF from NR flux at
  fixed xbar, residual ~ nu^3; subtract 2SF, residual ~ nu^4) — methodological precedent.
* Their Fig. 7 curve spans x in [0.02, ~0.13] — exactly our adiabatic band.
* Mode note: for (2,2), O(nu^4) corrections enter at relative 2PN; for (3,3) at
  relative 1PN — relevant if the comparison is extended to higher modes.

**Data ACQUIRED (2026-08-29).**  WaSABI cloned to `../wasabi/` (MIT licence, cite
Zenodo 10.5281/zenodo.17405583 + "This work makes use of the Black Hole Perturbation
Toolkit").  The 2SF flux: `InspiralModels/sf_data/2SF_Flux/Schwarz_Circ/`
`2SFCircShwarzDotEInf.m` — 22 points, stored as {ln r0, ln F2} (complex log; decode
`F2 = Re[Exp[.]]`, the i*pi imaginary part meaning F2 < 0), x = 1/r0 (M=1), covering
**x in [0.020, 0.160] — exactly our band**.  F2_EI is NEGATIVE throughout.  Parsed to
`wardell_data/F2_EI_schwarz_circ.json` (+ the 241-point 1SF file).

The consuming code `InspiralModels/1PAT1.m` is the convention Rosetta stone: it builds
`F1[r0] = (3(1-3/r0)^{3/2} sqrt(1/r0))/(1-6/r0) * ( F2_EI[r0]
          + 2(1-3/r0)^{3/2}/(1-6/r0) * F1SF[r0] * EFLx'[1/r0] )`
with `EFLx` from the redshift invariant z (first law) — i.e. the F2/E_SF SPLIT of
section 4b is written out explicitly there, and Omega' = (1/M)[nu F0 + nu^2 F1] is
Wardell Eq. (4) verbatim.

**REMAINING CONVERSION, not yet done — the last step before the overlay.**  The data
file is the eps-based F2 at fixed ORBITAL r0; Fig. 7's F-hat^SF,2 (leading term
+55x/21, positive) is the nu-based, waveform-x-bar, Newtonian-normalised version.  The
raw file is NEGATIVE (like the fixed-orbital-x PN term -35x/12), so the eps->nu
re-expansion (+4*F1-type terms) and the orbital->waveform frequency shift TOGETHER flip
the sign and produce 55/21.  Our derivation's conversion layer is the same algebra from
our side.  Do this conversion carefully against 1PAT1.m before overlaying — comparing
the raw file to our F2_calib directly would be comparing different conventions.


## 5. The plan, in cost order — now OPTIONAL VALIDATION for the core comparison

1. **Direct flux-ratio diagnostic — no model, no external data.**  From cached waveforms
   per q: `F_22 ~ |hdot_22|^2` for both sides, ratio at equal x, double-normalised at a
   reference x to kill constant unit factors.  Then the COLLAPSE TEST: is `R(x)/nu` one
   universal curve across q?  If yes, that collapsed curve is an empirical measurement of
   `F2/F1` (plus PN conversion terms) extracted from the calibration data.  Mode-by-mode
   ((2,2) first) to avoid mode-content mismatch.  This is essentially their Fig. 4
   diagnostic (`omega^2/omegadot` at fixed omega), i.e. gauge-free.  Minutes on skx-dev.
2. **Same collapse test on the measured alpha(x), beta(x)**
   (`beta_drift_tests/alpha_beta_of_omega_v2.json`), restricted to the inspiral band
   where adiabaticity holds.  Free — data exists at 7 q.
3. **Quantitative overlay — the only step needing external data.**  `F2(x)` from
   Warburton+ PRL 127, 151102 (arXiv:2107.01298); `E_SF(x)` from Pound+ PRL 124, 021101
   (arXiv:1908.07419); likely via the Black Hole Perturbation Toolkit or paper data
   releases.  Albertini+ arXiv:2208.01049 is the practical GSF-vs-NR/EOB comparison paper
   to calibrate expectations against.  Before this step, derive the exact
   beta-drift <-> (F2, E_SF) relation carefully (signs, E_SF', argument shift).

## 6. Caveats to carry into any writeup

* The comparison is only meaningful in the ADIABATIC INSPIRAL band.  Their model stops at
  the ISCO ("we must transition to a plunge model followed by a quasinormal mode
  ringdown" — their future work).  Nothing here says anything about merger-ringdown,
  where most of this workspace's difficulty lives.
* Their waveforms are 1PA: 0PA (first-order dissipative) + 1PA (second-order dissipative
  + first-order conservative).  The ppBHPT surrogate is adiabatic first-order — closest
  to their 0PA, so the calibration drift maps onto (1PA - 0PA) physics plus mass-scale
  conversion.  Verify exactly what flux the ppBHPT surrogate uses before quoting.
* Both "NR" and "BHPT" here are surrogates (NRHybSur3dq8, BHPTNRSur1dq1e4); surrogate
  error enters the ratio directly.  NRHybSur3dq8 is hybridised with PN at early times —
  check where the hybridisation ends before trusting the low-x end of the ratio.
* q < 2.5 has the BHPT surrogate out of its own domain (X_min = log10(2.5)) — exclude
  from the collapse test or flag.

## 7. Open questions

* Why does the merger-aligned beta excursion scale as nu^2 when the naive 1PA drift at
  fixed x should be nu^1?  (Gauge? Merger dominance? Cancellation?)  The fixed-x
  measurement in step 2 decides.
* Does the inspiral-only result (b_E > 0, rising beta at all 64 q) have the SIGN that
  F2/F1 predicts?  Requires the careful derivation first; do not assert.
* Their Fig. 5: 1PAT2 only useful for q >~ 50000 near ISCO — irrelevant here, but their
  1PAT1-vs-NR agreement at q as low as 1 is the benchmark this comparison should be
  read against.

## 4h. THE COMPARISON PLOT — first result (2026-08-29)

`Agentic_plots/wardell_comparison/F2_comparison.{pdf,png}`
(`wardell_comparison_plot.py`, job 3449326).  Curves: calibration-implied
F2_22/F1_22 (shipped + refit thetas), the analytic PN (2,2) nu-part through 2PN
(leading 55x/21, x^2 coefficient 2*A0*A1 + 2*B1 = -16.57 Newtonian-normalised, from the
PN amplitude Hhat_22 — 2PN coefficients per Blanchet LRR/Faye+ 1204.1043, TO BE
independently verified), and the WaSABI TOTAL-flux data (different observable, drawn
for orientation; its PN limit -(35/12)x verified to 5% at x=0.05, which pins the
file's convention as nu-scheme total flux at fixed ORBITAL x).

**Readings:**
1. **Low-x agreement.**  For x <~ 0.05 the calibration-implied curve and the PN (2,2)
   prediction agree (0.09-0.10 vs 0.07-0.10, crossing near x ~ 0.045).  The leading
   55x/21 lock is visible; the fitted (4/3)b1 excess is the small offset.
2. **High-x divergence, factor ~2 in the x^2 content.**  Ours turns negative above
   x ~ 0.065 (-0.24 at x = 0.11) while PN22 stays ~ +0.1.  Our x^2 coefficient
   (2 a2 + (4/3) b2 ~ -34) is ~2x the PN value (~ -17 after normalisation).  Since
   Warburton Fig. 7 shows their 2SF (2,2) result tracking PN closely through the band,
   the blue curve approximates the truth there — so the divergence is OURS to explain.
   Prime suspect (pre-registered in 4d): a2, b1, b2 were fitted FULL-WINDOW; merger
   leakage steepens the x^2 term.  Secondary: our L(x) is PN-through-2PN only.
3. **Basin robustness confirmed.**  Shipped vs refit curves are nearly indistinguishable
   — the R2 combination is far better determined than the individual parameters.

**Next controls, in order:**
* Refit (a2, b1, b2) on the INSPIRAL WINDOW only (machinery exists: the x_drive
  per-q caches at t_cut = -200 are x-polynomials on that window; or refit pn_anchored
  with the truncated objective).  If the x^2 coefficient halves toward -17, the
  divergence was merger leakage and the calibration AGREES with second-order
  self-force across the band — the headline result.
* Verify the 2PN amplitude coefficients independently before quoting the blue curve.
* DONE (see UPDATE below): the (2,2)-mode 2SF curve is on the plot, from the WaSABI
  amplitude data.

**UPDATE (plot v2, job 3449463): the TRUE (2,2) 2SF curve replaces the total-flux one.**
Built from the WaSABI mode-resolved Teukolsky amplitudes
(`AmplitudeModels/sf_amp_data/{1SF,2SF}/Schwarz_Circ/*TeukampSchwarzCirc22.m`) via the
assembly in `AmplitudeModels/1PAT1.m` (which IS Wardell Eq. 7 in r0 coordinates):
`R2_22 = 2 Re[h2_eff/h0]`, `h2_eff = h0 + h2 + (2/3) r0 (1-(1/r0)/sqrt(1-3/r0)) h0'`.
Parsed to `wardell_data/R2_22_from_amplitudes.json` (23 points, x in [0.033, 0.16]).
VALIDATED: R2_22/x = 2.33 at x=0.05 bending toward 55/21 = 2.62 at low x, as PN
predicts.  The total-flux curve is off the plot (it answers a different question —
its nu-part is NEGATIVE, -35x/12, because the higher modes' 1-4nu factors overwhelm
the positive (2,2) contribution; keep `F2_EI_schwarz_circ.json` for any future
total-flux comparison).

**Readings against the real (2,2) data:**
* The 2SF curve stays POSITIVE across the whole band, peaking ~+0.17 at x ~ 0.11, and
  sits ABOVE the 2PN truncation by 15-60% (the 2PN proxy under-predicts; its x^2 term
  overshoots without the 2.5PN tail).
* Calibration vs 2SF: agreement holds to x ~ 0.05-0.06 (calib +0.091 vs data +0.116 at
  x=0.05), then the calibration turns down and negative while the data stays positive —
  the divergence is now confirmed against the REAL second-order data, not a PN proxy.
  Crossing zero at x ~ 0.075, calib is -0.24 vs data +0.17 at x = 0.11.
* The prime suspect is unchanged and now better motivated: the calibration's x^2
  content (2a2 + (4/3)b2 ~ -34) is fitted FULL-WINDOW; the decisive control remains the
  inspiral-window refit of (a2, b1, b2).

## 4i. TOTAL-flux comparison via the hm backbone (user's idea, 2026-08-29)

`Agentic_plots/wardell_comparison/F2_total_comparison.{pdf,png}`
(`wardell_total_flux.py`, job 3449581).  The per-mode generalisation of the 4g
derivation, using the hm backbone alpha_lm = alpha_22 * C_lm(nu) * beta^(-(l+eps-2)/3):

```
R2_lm(x) = 2[a + c_lm + e_lm b] - 2b + (2/3) L_lm b + (4/5)(p_lm - L_lm)
c_lm = -(l+eps-1) (0 for 22, exact from C_lm)   e_lm = -(l+eps-2)/3
p_lm = l+eps+3;  L_lm analytic (exact PN 22; p_lm + (L22-5) for subdominant)
R2_tot = sum w_lm(x) R2_lm(x),  w_lm measured from the q=8 ppBHPT hm cache
```

Newtonian consistency verified analytically: the backbone's beta-power makes the
per-mode conversion constants cancel iff p_lm = l+eps+3, which holds — the backbone's
mass factors are exactly Newtonian-consistent.

**Readings:**
1. **The sign flip emerges from the backbone.**  The total is negative throughout
   (the C_lm nu-slopes overwhelm the (2,2)'s +55x/21), tracking the PN -35x/12 line at
   the low-x end — the (2,2)/total sign structure of the 2SF picture is reproduced by
   the note's mass factors with nothing fitted.
2. **Quantitative agreement at low x: ~11% at x = 0.05** (backbone -0.166 vs data
   -0.149).  Same quality as the (2,2) comparison at the same x.
3. **Same high-x over-steepening as the (2,2): factor ~2 by x = 0.11** (backbone -0.85
   vs data -0.41).  The total is (2,2)-weighted at ~97%+, so the SAME suspect — the
   full-window-fitted a2, b2 — explains BOTH plots.  One cause, two observables,
   coherent divergence pattern.
4. Robustness: weights from the q=3 vs q=8 caches indistinguishable; the measured-rho
   variant shifts things by <5%.
5. Caveats: measured-L(x) was tried and spikes at envelope notches (v1) — analytic L
   used instead; the low-x printed check (-4.0/x at x=0.035 vs -2.92 PN) reflects the
   x^2 contamination from a2, b2 reaching down even there, NOT a failure of the C_lm
   content (which is exact); their file is at orbital x vs our waveform x-bar.

**The composite picture after 4h + 4i:** the calibration + backbone reproduce
second-order self-force at the ~10-25% level for x <~ 0.05-0.06 in BOTH the (2,2) and
the total flux, and over-steepen by ~2x toward the band edge in both — with the
full-window x^2 coefficients (a2, b2) as the single shared suspect.  The
inspiral-window refit of (a2, b1, b2) remains THE decisive next run: it would either
collapse both divergences at once or falsify the leakage explanation.

**Refinement of the divergence interpretation (user, 2026-08-29).**  The comparison band
(x <= 0.16) never touches the MR region (ISCO x ~ 0.167, merger x ~ 0.27-0.33): the
self-force ratio is valid essentially everywhere compared — their data stops at
r0 = 6.25 precisely where two-timescale validity ends.  So the divergence is not "we
compared where 1PA is invalid."  Rather: OUR x^2 coefficients import MR information into
the band — pn_anchored's inspiral polynomial serves the model out to xc = 0.26 (past
ISCO), and the post-merger region carries 32-38% of the full-window mismatch numerator
that chose a2, b2.  Their curve cannot have this disease; ours is built to.  This makes
the leakage explanation strongly favoured a priori, and the inspiral-window refit is now
a CONFIRMATION run with a falsifiable prediction: the divergence should largely collapse.
If it does not, the drift contains genuinely beyond-1PA content.

---

## NEXT SESSION — agreed agenda (2026-08-29)

Two runs, in this order:

1. **The model-free flux-ratio measurement** (section 5 step 1, promoted from
   "optional validation" to ARBITER of the high-x divergence).  No calibration anywhere:
   F vs x measured from each waveform independently (envelope form F ~ m^2 Omega^2
   |h|^2 to dodge the orbital-timescale oscillation), ratio at equal x, per q over the
   64-q grid, nu-decomposition per 4c (c0 = 0 check, c1(x) = the empirical F2/F1).
   Decision rule: measurement agrees with Warburton -> truth is in our data, red curve's
   divergence is pn_anchored's fault (form/window); measurement disagrees -> the
   discrepancy is in the surrogates and no refit fixes it.
2. **The inspiral-window refit of (a2, b1, b2)** — the confirmation run for the leakage
   mechanism (fit currently spends 32-38% of its objective past ISCO where the flux
   identity is meaningless).  Prediction: the x^2 coefficient ~halves and both plots'
   divergences collapse.

Where everything is: this file (the whole programme), Agentic_plots/wardell_comparison/
(both plots), wardell_data/ (their parsed data), ../wasabi/ (the clone),
scaling_pn_anchored.md (verification + refit-basin caveats).  ALL UNCOMMITTED.

## 4j. THE ARBITER RULED — model-free measurement sides with 2SF theory (2026-08-30)

`Agentic_plots/wardell_comparison/F2_measured_{22,total}.{pdf,png}`
(`wardell_flux_ratio_measured.py`, job 3450343).  The model-free flux ratio,
R2_meas(x) = (F_NR/F_pp / X1^4 - 1)/nu, measured pointwise from the two waveforms with
NO calibration anywhere — pulled forward from the next-session agenda at user request,
as two separate plots ((2,2) from the q_dep cache at q = 3,4,5,6,8; total from the hm
cache at q = 3,4,5,8).

**Conversion validation passed:** F_NR/F_pp/X1^4 sits at 1.01-1.05 for the (2,2)
(0.87-0.98 for the total, correctly below 1) — the derived unit bookkeeping is right.

**THE VERDICT: the measured curves track the WARBURTON/2SF theory curves, not the
calibration-implied ones.**
* (2,2): measured family positive throughout, clustering around the 2SF theory curve
  (~+0.1..+0.2 through mid-band) while the calibration-implied curve dives to -0.6.
* Total: measured family straddles the Warburton total curve; the backbone-implied
  curve falls well below it.
* The q-ordering of the measured family (q=3 highest, q=8 lowest, both plots) is
  consistent with a POSITIVE next-order term (R2_meas = R2 + nu*R3 + ..., R3 > 0), i.e.
  the spread is finite-nu physics, not noise.

**Consequence: the high-x divergence of 4h/4i is UNAMBIGUOUSLY pn_anchored's fault**
(its full-window-fitted x^2 coefficients), NOT the surrogates and NOT missing physics —
the flux content matching 2SF theory was sitting in the calibration's own input data
all along.  The leakage explanation is confirmed by measurement; the inspiral-window
refit of (a2, b1, b2) is now expected to succeed and remains worth running as the
constructive fix (a calibration model whose implied F2 matches 2SF across the band).

Measurement caveats: oscillations in the measured curves are envelope beating /
residual eccentricity between the two surrogates (worst 0.08 < x < 0.11); an isolated
q=8 glitch at x ~ 0.104; the band starts at x ~ 0.066 (NR cache length) where NRHybSur
is partly PN-hybridised; q = 2, 2.5 excluded (BHPT domain).

**Why the calibrated curve is so far off the measured one (user question, 2026-08-30):**
(1) the fit's objective is INTEGRATED PHASE (L2 mismatch); the plot is a LOCAL DERIVATIVE
(flux slope at fixed x) — a 6e-4-mismatch model does not certify local slopes (the
workspace's "small mismatch does not certify the ingredients", now at flux level);
(2) the x^2 coefficients were bought by the merger — 5-6x more basis leverage at
xc = 0.26 than at x = 0.11, and 32-38% of the mismatch numerator lives past merger;
(3) the plot amplifies by ~2/nu: the R2 gap of ~0.4 at x = 0.11 corresponds to only a
~3% alpha error in the band — cheap in mismatch, O(1) in F2/F1;
(4) the flux identity assumes the ansatz h_NR = alpha h_pp exactly; at finite mismatch
the least-squares compromise leaks into the implied flux.

**New idea from this: use the measured R2(x) as a CALIBRATION TARGET.**  Constrain
2a - 2b + (2/3)Lb + conv to match the measured flux ratio (a pointwise, derivative-level
constraint) alongside — or instead of — the mismatch.  That would make the calibration
1PA-consistent by construction, fix the band-flux content the L2 objective is
indifferent to, and is the natural endgame beyond the inspiral-window refit.

**Error budget of the comparison (user question, 2026-08-30).**  Warburton side:
Fig. 7 carries an explicit gray error band on F-hat^SF,2 — ~1e-4..1e-3 Newtonian-
normalised against a signal of ~0.1-0.2, i.e. 0.1-1% of signal mid-band, few % near the
edge; their NR cross-validation (total-flux agreement 1.9e-3 at q=10 / 2.5e-3 at q=1
relative to the FULL flux) certifies the nu^3 piece at the ~5% level.  The calibrated-
vs-theory gap at x=0.11 is ~0.41 = ~240% of signal -> exceeds their errors by ~50-200x.
Comparison-side systematics dominate over theirs but are still small: our amplitude-file
construction of the (2,2) curve is validated to few % (55/21 limit); nu-truncation is
bounded by the measured family's spread (~+-0.15 at high x) — and the calibrated curve
lies outside the ENTIRE measured family, so no error source covers it.  If a band is
wanted on the plot, use the measured-family envelope as the finite-nu band (free) or
digitise Fig. 7's gray band (their numerics).
