# Extending the calibration to aligned spin (chi1)

**Reference:** Islam, Rink et al., "BHPTNRSur2dq1e3", arXiv:2407.18319 — the aligned-spin
ppBHPT surrogate, calibrated to NR by the alpha-beta procedure.  Their Eqs. (6)-(9) are
the thing to replicate/extend.  Opened 2026-08-30.

---

## 1. What they actually do

**Eq. (6) — the scaling.**  `h^{lm}_{S,alpha,beta}(t; q,chi1) = alpha_l * h^{lm}_S(beta t; q,chi1)`
with a SINGLE time-independent `beta(q,chi1)` and a set of time-independent
`{alpha_l(q,chi1)}`.  Note `alpha_l`, not `alpha_lm`: they fit on l=m modes and reuse
`alpha_l` for l != m (they state `alpha_lm ~ alpha_ll` is not the accuracy limiter).

**Eq. (7) — the objective.**  Relative L2 in the time domain (their Eq. 5), per mode,
minimised over `alpha_l, beta` at discrete (q,chi1) with NRHybSur3dq8 as the NR proxy.

**Eqs. (8)-(9) — the parameterisation.  SEPARABLE and MULTIPLICATIVE:**

```
alpha_l(q,chi1) = 1 + ( sum_{n=1..4} A^n_{alpha_l} q^-n ) ( 1 + sum_{n=1..2} B^n_{alpha_l} chi1^n )
beta   (q,chi1) = 1 + ( sum_{n=1..4} A^n_beta      q^-n ) ( 1 + sum_{n=1..2} B^n_beta      chi1^n )
```

Training: q in [3,10], chi1 in [-0.6, 0.8], 90 points.  B^n set to ZERO for alpha_3 and
alpha_4 ("little changes as we change spin").  Coefficients in their Tables I/II — and
also shipped inside the surrogate h5 at `nr_calib_params/(l,m)/{alpha,beta}` as
6-vectors [A1,A2,A3,A4,B1,B2] (verified identical to the paper, 2026-08-30).

**Two structural points that matter for us:**

1. **This is the Islam+22 "classic" model class** — constant alpha, beta, PP-anchored
   polynomial in 1/q.  This workspace already has the nonspinning version as
   `q_dep_classic` (~1e-3 in-range, ~1e-2 at q=2).  Every time-dependent model here
   beats it by ~10x.  So we are not replicating a better model; we are borrowing their
   SPIN PARAMETERISATION and grafting it onto better alpha(t), beta(t) machinery.
2. **Their form is PP-anchored by construction, in exactly our sense.**  The spin factor
   MULTIPLIES the 1/q polynomial, so alpha, beta -> 1 as q -> inf for ANY chi1.  The spin
   correction modulates the finite-mass correction rather than adding to it — physically
   right (at q -> inf the correction itself vanishes, so spin cannot matter) and
   compatible with every PP-anchored form in this workspace.

## 2. How weak is "weak"?  (measured from their coefficients)

| q | alpha_2(chi=0) | swing over chi1 in [-0.6,0.8] | beta(chi=0) | beta swing |
|--:|--:|--:|--:|--:|
| 3 | 0.7102 | 2.26% | 0.7110 | 1.35% |
| 5 | 0.8088 | 1.31% | 0.8036 | 0.81% |
| 8 | 0.8735 | 0.80% | 0.8675 | 0.51% |
| 10 | 0.8966 | 0.64% | 0.8912 | 0.40% |

Against a q-dependence of 0.710 -> 0.897 (26%) over q in [3,10].  So yes: spin is a
~1-2% effect on top of a ~26% mass-ratio effect, and Fig. 8's curves look nearly flat.

**BUT — "weak" is NOT "negligible", for beta.**  RESUME_gwremnant.md records that a
**0.2% constant error in beta accumulates ~60 M of drift that t0 cannot absorb**.
Scaling that: beta's 1.35% spin swing at q=3 corresponds to **~400 M of accumulated
drift** (243 M at q=5, 152 M at q=8).  Beta's spin dependence must be modelled even
though it is numerically tiny.  Alpha's 1-2% is a genuine amplitude effect but does not
accumulate.  Do not read Fig. 8(b)'s flatness as licence to drop the spin term in beta.

## 3. The proposed extension: time-independent chi1 modulation (user, 2026-08-30)

Keep every model's existing time dependence; make the SPIN dependence a separable,
time-independent multiplicative modulation:

```
alpha(t; q,chi1) = alpha(t; q) * [ 1 + G_alpha(nu) * sum_n B^n_alpha chi1^n ]
beta (t; q,chi1) = beta (t; q) * [ 1 + G_beta (nu) * sum_n B^n_beta  chi1^n ]
```

with `G(nu) -> 0` as nu -> 0 to preserve PP-anchoring (the analogue of Islam's
"multiply the 1/q polynomial").  Simplest choice `G = nu`, or reuse each model's own
finite-mass correction factor.

This is the right FIRST model, and cheap: 2-4 new coefficients on top of any existing
model, nothing else touched.

**The caveat that makes it an experiment rather than an assumption.**  Islam's alpha,
beta are time-independent BY CONSTRUCTION (Eq. 6) — they never allow time dependence at
all.  So "alpha and beta depend weakly on chi1" is a statement about window-averaged
scalars.  It says NOTHING about whether the TIME-DEPENDENCE of alpha(t), beta(t) is
spin-dependent — and we should expect it to be: chi1 moves the ISCO, the QNM
frequencies, and the merger morphology.  Their own Sec. A warns that for chi1 <~ -0.5
retrograde QNMs are significantly excited, which is a statement about waveform SHAPE.

So the separable ansatz is a hypothesis with a measurable test (section 5, step 1).

## 4. Where spin enters pn_anchored — mostly DERIVED, not fitted

pn_anchored's inspiral/MR structure (BHPTNRPNAnchored.py):

```
beta_insp  = X1^(6/5) * [1 + nu (b1 xc + b2 xc^2)]
alpha_insp = X1^(6/5) * [1 + (55/42) nu xc + nu a2 xc^2]      55/42 FIXED (1PN)
beta_r     = beta_r_physical(q) * r0                           QNM remnant time-stretch
alpha_mr   = X1^(6/5) * (m0 + m1 nu)
S          = sigmoid((p_loss - (p00 + p01 nu))/w0)
```

Spin entry points, in order of how much is free:

1. **beta_r — the QNM remnant anchor.  FULLY DERIVED, ZERO new parameters.**
   `beta_r_phys` already calls `NRSur3dq8Remnant.mf/chif(q, [0,0,0], [0,0,0])`.
   Changing to `[0,0,chi1]` gives the spin dependence for free, and it is the STRONGEST
   spin dependence in the problem (chi_f depends strongly on chi1).  Same for
   `gwr_betaqnm`'s B1 = W_Schw*Mf/omega_220(chi_f) via gwModelRemS.
   **This is pn_anchored's structural advantage over Islam's empirical B^n: their
   spin dependence is fitted; ours can be derived.**
2. **X1^(6/5)** — a mass-scale conversion.  Spin-independent.  Unchanged.
3. **The 55/42 1PN amplitude slope** — spin-orbit enters the PN amplitude at 1.5PN, so
   there is a KNOWN analytic spin term to impose in the same spirit as 55/42.  Needs the
   PN reference (Blanchet LRR / the note); not yet looked up.
4. **p_loss (the switch coordinate)** — a waveform flux coordinate, so its spin
   dependence comes free from the spinning waveform.  The switch's fitted constants
   (p00, p01, w0) may need chi1 terms.
5. **Residual constants (b1, b2, a2, m0, m1, r0)** — the empirical part.  These are where
   Islam-style `1 + sum B^n chi1^n` factors would go, if the measurements say they are
   needed.

## 5. Plan

**Step 0 — infrastructure (DONE 2026-08-30).**
* `BHPTNRSur2dq1e3.py` was already present in `../BHPTNRSurrogate/surrogates/`.
* Downloaded the 729 MB h5 from Zenodo 13340319 into `../BHPTNRSurrogate/data/`.
* Verified (job 3451822) `generate_surrogate(q=6, spin1=+0.6/-0.4, calibrated=False)`
  returns 136251-sample waveforms, t in [-13500, 125] m1, with the expected
  "NOT NR calibrated - 0PA contribution only" warning — i.e. the RAW ppBHPT, which is
  what we calibrate.  Modes available: (2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4).
* Verified NRHybSur3dq8 accepts `chi1 = [0,0,0.6]` (job 3451813).
* NOTE the secondary is a NON-SPINNING point particle in ppBHPT, so NR must be called
  with chi2 = 0 to match.

**Step 1 — MEASURE before modelling** (the workspace's own repeatedly-learned lesson).
On a (q, chi1) grid, extract per-point optimal alpha(t), beta(t) and ask two questions
that decide the whole ansatz, BEFORE fitting anything:
* **Separability**: is `alpha(t; q,chi1) / alpha(t; q,0)` independent of q?
  (Islam's Eq. 8 assumes yes.)
* **Time-independence**: is that same ratio independent of t?
  (The user's proposal assumes yes.  Islam CANNOT have tested this.)
Plot both ratios; if either fails, the failure mode tells us the right form.

**Step 2 — the minimal model.**  Whichever of pn_anchored / gwr_energy_fluxanchored /
x_drive is the base, add the separable factor of section 3 and refit only the new
coefficients, with beta_r's spin dependence DERIVED (section 4.1) rather than fitted.

**Step 3 — validate against their model.**  Their calibrated surrogate is available
(`calibrated=True`) and their coefficients are in the h5, so we can reproduce their
alpha_l, beta exactly and compare our spin dependence against theirs point by point.

**Grid and cost.**  Calibration overlap is q in [3,8] (NRHybSur3dq8's training range;
Islam pushed to 10) and chi1 in [-0.6, 0.8] — avoid chi1 < -0.6, where their Sec. A
documents retrograde-QNM excitation and complicated morphology.  Every cache in this
workspace is nonspinning, so a spin axis multiplies the grid: a 64-q x N_chi grid is
expensive.  Start with a coarse grid (say q in {3,4,5,6,8} x chi1 in {-0.6,-0.3,0,0.3,0.6,0.8}
= 30 points, matching Islam's 90-point density loosely) and refine only if step 1 says
the structure is q-dependent.

## 6. Open questions

* Does the spin-dependence of alpha(t), beta(t) factorise in time?  (Step 1.)
* Is the 1.5PN spin-orbit amplitude term worth imposing analytically (the 55/42 analogue)?
* Does `gw_remnant`'s energy drive E(t) behave for spinning waveforms?  (It is computed
  from h, so mechanically yes — but E_tot's nu-scaling was measured nonspinning.)
* chi1 < -0.5: retrograde QNM excitation changes ringdown morphology.  Our MR branches
  (beta_r QNM plateau, alpha_mr) assume a prograde-ish remnant.  Likely needs its own
  treatment — Islam avoid it by cutting at chi1 = -0.6.

---

## 7. STEP 1 RESULTS — the measurement (2026-08-30, jobs 3451839 / 3451846)

`spin_measure_alpha_beta.py` (30-point grid, q in {3,4,5,6,8} x chi1 in
{-0.6,-0.3,0,0.3,0.6,0.8}) + `spin_measure_plots.py`.  Parameter-free: alpha, beta read
off by merger-aligned equal-accumulated-phase pairing, no fit anywhere.  Curves cached in
`spin_measure_results/curves_win20.json`.  Plot:
`Agentic_plots/spin_extension/spin_ratio_tests.{pdf,png}`.
Clean band x in [0.05, 0.10] (away from the NR start and from merger).

### ALPHA: essentially NO spin dependence in the inspiral

Band-mean `R_alpha` is **1.000 +- 0.005 at every (q, chi1)** — flat in x (Q1 passes
trivially) and flat in chi1 (Q2 vacuous).  Islam's Eq. 8 predicts up to **+2.2%** at
q=3, chi=+0.8; we measure **-0.7%**, i.e. *opposite sign*.  Alpha's spin dependence only
appears at x >~ 0.10 (visible in the top-left panel, dropping to 0.94-0.99 near merger).

**Reading:** in the inspiral, alpha's spin dependence is consistent with zero.  Their
non-zero B_alpha is most plausibly (a) merger-region behaviour folded into a single
window-averaged scalar, and/or (b) the alpha-beta degeneracy this workspace has measured
at -0.974 -- their Eq. 7 fits alpha and beta SIMULTANEOUSLY, so alpha can absorb beta's
spin dependence.  Our construction separates them (alpha from the amplitude ratio, beta
from the frequency ratio, at the same paired points), so it cannot.
Note they already set B = 0 for alpha_3, alpha_4 on "little change with spin" grounds;
our measurement says the same is true of alpha_2 in the inspiral.

### BETA: real, monotone, and 2-3x LARGER than their fit on the positive-spin side

| q | chi1 | measured mean R_beta | x-spread | Islam Eq. 9 | (meas-1)/(Islam-1) |
|--:|--:|--:|--:|--:|--:|
| 3 | -0.60 | 0.99442 | 1.38% | 0.99358 | 0.87 |
| 3 | +0.30 | 1.00436 | 0.89% | 1.00285 | 1.53 |
| 3 | +0.80 | 1.02349 | 3.35% | 1.00706 | **3.33** |
| 8 | -0.60 | 0.99726 | 0.47% | 0.99759 | 1.14 |
| 8 | +0.80 | 1.00978 | 1.09% | 1.00265 | **3.69** |

* **Negative spin: we agree with them** (ratio 0.83-1.29 across the grid).
* **Positive spin: we measure 1.5-3.7x their effect**, growing with chi1.  The
  discrepancy is systematic in chi1, not scattered -- their quadratic
  `1 + B1 chi + B2 chi^2` under-predicts the positive-spin response.
* **Q2 (separability) essentially HOLDS in shape**: the measured curves have the same
  q-ordering and the same monotone chi1 shape as Eq. 9 predicts (their form's own
  q-dependence, since the spin factor multiplies the 1/q polynomial).  Only the
  amplitude of the positive-spin response differs.
* **Q1 (time-independence) FAILS at high spin.**  `R_beta(x)` rises monotonically with x
  -- beta's spin dependence GROWS toward merger.  x-spread over the clean band:
  0.17-0.89% for |chi1| <= 0.3, but **1.1-3.4% at chi1 = +0.8**.

### What this means for the proposed ansatz

The time-independent separable modulation is:
* **fine for alpha** -- in the inspiral it can simply be omitted (B_alpha = 0), which is
  simpler than Islam;
* **fine for beta at |chi1| <~ 0.3** (sub-1% x-variation);
* **NOT sufficient for beta at high positive spin.**  Recall the drift arithmetic
  (RESUME: 0.2% constant beta error ~ 60 M of drift): the 3.4% x-variation at chi=+0.8
  is ~1000 M of differential drift across the band -- far too large to absorb into t0.

**Indicated form for our time-dependent models.**  Rather than a constant modulation,
let spin enter the DRIFT, which is strictly more general and costs the same:

```
beta(t; q,chi1) = b_PP(q,chi1) * (1 + b_E(q,chi1) * drive(t))
```

i.e. give BOTH beta's level and its drift coefficient a chi1 dependence.  The measured
R_beta(x) rising roughly linearly in x is exactly what a spin-dependent drift
coefficient produces, while a constant modulation cannot.  Alpha keeps its nonspinning
form in the inspiral.

For pn_anchored specifically this maps onto: `b1, b2 -> b1(chi1), b2(chi1)` (the x and
x^2 drift coefficients), with `beta_r`'s spin dependence DERIVED from RemS (section 4.1)
rather than fitted -- and no spin terms needed on the alpha side of the inspiral branch.

### Caveats on the measurement
* Merger-aligned gauge (the same one used throughout this workspace); the x > 0.10
  behaviour is where pairing and smoothing degrade, so the alpha drop near merger and
  the R_beta upturn past 0.11 should not be over-read.
* 20 M smoothing window only -- the 10/40 M robustness check of v2 has NOT been repeated
  here.
* chi1 >= -0.6 by construction (retrograde-QNM regime avoided, per their Sec. A).
* NRHybSur3dq8 is the NR proxy and is PN-hybridised at early times; the low-x end of the
  band inherits that.

## 8. The FORM of the spin correction, measured (2026-08-30)

`spin_drift_form.py`, `spin_drift_form2.py`, `spin_drift_fit.py` — all cheap (they read
the cached step-1 curves; no waveforms, no mismatch).

**Fitting strategy: fit the spin terms to the MEASURED RATIO, not to the waveform
mismatch.**  The Wardell work established that the L2 mismatch is nearly blind to
local-in-x structure and that full-window fits leak merger information into band
coefficients.  The ratio R_beta(x;q,chi) is pointwise and derivative-level, so fitting
against it pins exactly what the mismatch cannot.  Use the mismatch only as a final
check, or as a refinement seeded here.

**Two conditioning warnings, learned the hard way in this session:**
* `x` and `x^2` over x in [0.05,0.10] have condition number 74 — the db1/db2 split is
  unstable and individually meaningless (db1 spreads 330% across q while db2 is stable
  to 12-25%).  Do NOT report or fit them separately over a narrow band.
* Adding a constant to (R-1)/nu barely improves the fit EXCEPT at chi1 = -0.6, where it
  halves the residual (c ~ +0.28).  So at strong negative spin beta's LEVEL shifts too,
  not only its drift.  Everywhere else the correction is pure drift, which vindicates
  the b_E(q,chi1) framing.

**The measured form** (5 parameters, fitted to 1000 points across the full grid):

```
R_beta(x; q, chi1) - 1 = nu^p [ (a1 + a2 x) chi1 + (a3 + a4 x) chi1^2 ]

p = 1.60,  a1 = -0.0389,  a2 = +3.518,  a3 = -0.4954,  a4 = +8.962
rms 1.7e-3 = 4.0% of the largest correction
```

What each ingredient buys (rms, as % of max |D|):

| variant | rms | note |
|:---|--:|:---|
| full, p free | 4.0% | |
| p fixed at 1.5 | 4.0% | p is only loosely determined; 1.5 is as good as 1.60 |
| p fixed at 1 | 4.6% | pn_anchored's explicit `nu*(...)` is only slightly worse |
| no x-dependence (constant modulation) | 8.1% | **the user's original ansatz — 2x worse** |
| no x-dep AND p=1 | 8.2% | |

**Readings:**
1. **The spin term does NOT ride nu^1.**  p ~ 1.5-1.6, i.e. the correction is suppressed
   more strongly toward large q than pn_anchored's explicit `nu` factor would give.
   But p = 1 costs only 4.6% vs 4.0%, so KEEPING pn_anchored's `nu` and absorbing the
   difference into the coefficients is defensible and simpler.
2. **The x-dependence is the ingredient that matters** — dropping it doubles the error.
   Both the chi and chi^2 coefficients are dominated by their x-slopes (a2 = +3.5,
   a4 = +9.0 against intercepts -0.04, -0.50), i.e. the spin correction is close to
   PROPORTIONAL TO x over this band.  This is the quantitative version of the Q1 failure
   in section 7: a constant modulation cannot represent it.
3. **chi1^2 dominates at high spin** (a4 x chi^2 ~ 0.5 at chi=0.8, x=0.09, vs
   a2 x chi ~ 0.25) — consistent with the strongly asymmetric response measured in
   section 7.

**Recommended pn_anchored form** (keep the explicit nu, absorb p into coefficients):

```
beta_insp = X1^(6/5) [ 1 + nu ( b1(chi1) x + b2(chi1) x^2 ) ]
b1(chi1) = b1 + s11 chi1 + s12 chi1^2
b2(chi1) = b2 + s21 chi1 + s22 chi1^2
```

4 new coefficients (s11, s12, s21, s22), PP-anchored automatically by the outer `nu`,
and reducing to the shipped nonspinning model at chi1 = 0 by construction.  Seed them
from section 8's fit (converting nu^1.60 -> nu at the grid's mean nu costs ~10%, within
the residual scatter).  Alpha's inspiral branch needs NO spin terms (section 7).
`beta_r` takes its spin dependence from RemS, derived, zero parameters (section 4.1).

**Residual honesty:** even the best 5-parameter form leaves ~20-30% q-scatter in
g(chi) = D/nu^p (section 7 tables), and the negative-spin branch is poorly described by
a pure power law in nu (fit r = 0.09-0.86 vs 0.99 for positive spin).  So this form is a
good SEED and a good structural statement, not a final calibration.  The chi1 = -0.6
level shift (above) is not in it at all.
