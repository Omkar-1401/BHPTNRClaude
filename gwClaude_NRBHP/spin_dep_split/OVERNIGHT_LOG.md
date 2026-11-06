# Overnight log — 2026-10-03/04 (autonomous; user detached, back tomorrow)

## SUMMARY (read this first)
**Result: a split model that beats the current leader on held-out points** -- fluxanchored_spin_split (VARIANT, not installed;
18 coefficients: 9 inspiral + 9 merger-ringdown).  Full-window error, full-window alignment (comparable to published numbers):

| | T96 median | T96 mean | T96 max | H270 median | H270 mean | H270 max |
|:--|--:|--:|--:|--:|--:|--:|
| installed pn_anchored_spin (leader, 13c) | 8.65e-4 | 1.17e-3 | 6.6e-3 | 7.10e-4 | 8.27e-4 | 2.9e-3 |
| **flux split, T96-fitted (18c)** | **5.17e-4** | **9.69e-4** | 6.7e-3 | **4.59e-4** | **5.94e-4** | **2.8e-3** |
| flux split, S18-fitted (18c) | 6.43e-4 | 1.10e-3 | 5.8e-3 | 6.05e-4 | 7.58e-4 | 2.9e-3 |

Held-out: -35% median, -28% mean, -3% max vs the leader; H270/T96 median 0.89 (generalises).  Split errors, insp-aligned, T96:
E_insp median 1.5e-4 (leader 2.0e-4), E_MR 7.7e-3 (leader 2.2e-2, i.e. 3x better).  Not uniform yet: worse than the leader at
the low-q retrograde corner (q <= 3.5, chi1 <= -0.4; up to 3.4x at q = 3, chi1 = -0.6) and at chi1 = +0.8, q 4.2-7.2 (1.2-1.5x);
T96 max +2% (6.7e-3 vs 6.6e-3), so EXPLORATION's G1 misses narrowly on the max only.  Figure:
Agentic_spin_split_plots/fluxanchored_spin_split/T96_heatmaps.png.

What made it work (each step tested, numbers below):
1. **Derived PN spin-orbit term in the inspiral beta** (zero new parameters):
   delta ln beta = (3/2) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]   (from Blanchet LRR energy + flux; see step 1 + CORRECTION).
   It supplies the chi1-dependence of the early-inspiral beta that every clean inspiral form lacked; per-case and equal-phase
   measurements confirm its sign and q-scaling, size within ~1.4x (next-order SO terms likely).  This also explains the
   installed pn_anchored_spin's never-off switch: its beta_r admixture was smuggling in exactly this chi1-dependence.
2. **Flux/energy-driven inspiral** beats the x-polynomial (the pn polynomial does not benefit from the SO term).
3. **MR branch = a second copy of the flux form** (own coefficients), blended over the user's Ori–Thorne window (T = -1 to the
   ppBHPT peak, eta = 1/q), starting equal to the inspiral copy.  Two-stage fit (E_insp then E_MR, inspiral alignment).

Also settled tonight: eta = nu vs 1/q makes no difference (< 1%); window starting at the ISCO is worse than T = -1;
first pn-form split failed (rigid MR branch + weak inspiral form).  One bug of mine found and fixed (t0 seed in pool workers,
results quarantined in results/invalid_seedbug/); one derivation error found and corrected (3/5 -> 3/2 for x-dependent terms).
Not done: step 4 (per-case free alpha(t), beta(t) collapse tests).  Compute tonight: ~2 SU.

**For the user to decide:** adopt this as the spin_dep_split model line?  If yes, next: fix the two hot corners (chi1 = +0.8,
low-q retrograde), alpha's SO term (by amplitude matching alpha gets -(2/3) delta_SO + amplitude SO terms, not derived yet),
then mds/plots/wishlist figures.

## Rules for tonight
* Everything in `spin_dep_split/`; nothing in `spin_dep/` edited, nothing installed.
* Budget cap 20 SU (development queue, small grids first).  Each number states grid, coefficient count, job.
* Model-definition choices are tested as VARIANTS and reported, never adopted.
* Derivations from the LaTeX sources (read-arxiv-source rule), sign and size checked against data.

## Plan (in order)
1. Derive the PN spin-orbit correction to beta (and alpha): binary vs test-mass-on-Kerr, at order nu,
   in the ppBHPT (m1) units.  Sources: Kidder 1995 / Blanchet LRR.  Check sign + size vs the per-case
   b_pp(chi1) at q = 4 (Fig. 10 study: +0.42% per unit chi1).
2. Per-case inspiral fits on S18 (q = 3, 5, 8): how does the per-case beta spin dependence scale with q?
3. Inspiral-only diagnostic with the SO term: derived coefficient (0 params) vs fitted, pn + flux forms.
   Target: installed pn_anchored_spin median E_insp 2.1e-4 (S18, insp-aligned).
4. Per-case free alpha(t), beta(t) (splines) on S18 -> collapse tests vs x, OT T, E(t).
5. If 3 works: split model with the improved inspiral + MR fixes (blend from the ISCO; MR shape from 4),
   S18, scored both alignments.
6. eta = nu vs eta = 1/q for the OT window (user TODO).

## Log
### Step 1 — PN spin-orbit term in beta (derived; sources in refs/blanchet_lrr/LRR_2024_v3.tex)
* LRR Eq. ESO: E_SO = -(nu m x/2) x^(3/2) (14/3 s + 2 Delta sig);  Eq. fluxSO: F_SO = F_N x^(3/2) (-4 s - 5/4 Delta sig),
  s = S_l/(G m^2), sig = Sigma_l/(G m^2), S = S1 + S2, Sigma = S2/X2 - S1/X1 (Eq. defSSigma).
* dx/dt = F / (-dE/dx)  =>  SO coefficient -47/3 s - 25/4 Delta sig.  chi2 = 0: s = X1^2 chi1, sig = -X1 chi1
  => A_NR^SO = -chi1 (19/6 X1^2 + 25/4 X1) x^(3/2);  test mass on Kerr (X1 = 1, m1 units): A_pp^SO = -113/12 chi1 x_pp^(3/2)
  (= Kidder's 113/12; checks).
* Repo note's matching, eta_pp beta^(5/3) (1 + A_pp(x_pp)) = nu (1 + A_NR(x_NR)), x_NR = beta^(-2/3) x_pp,
  x_pp^(3/2) = X1^(6/5) x^(3/2) (x = note's x):
      delta ln beta_SO = (3/5) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]   (-> (77/60) X2 at leading order; 0 as q -> inf)
* q = 4: coefficient 0.1068; over x = 0.07..0.10 -> +0.20..0.34% in beta per unit chi1.  Per-case fits (Fig. 10 study,
  flux form, q = 4): +0.42% per unit chi1.  Sign agrees, size within ~1.5x.  Zero fitted parameters.
* Side note (not pursued tonight): the same matching at 1PN predicts a nonspinning b1 ~ -3 at q = 4, while fitted b1 is
  O(+0.5..1); the full-window x^1/x^2 residuals are not 1PN-clean (known leakage, wardell_comparison.md).
### Step 1b — derived SO term vs the model-free measured beta (spin_dep/spin_measure_results/curves_win20.json; no compute)
d ln(beta)/d chi1 at fixed x_pp (linear fit over chi1 -0.6..+0.6), measured / derived, % per unit chi1:
| q | x_pp 0.06 | 0.08 | 0.10 | 0.12 | 0.14 |
|--:|:--|:--|:--|:--|:--|
| 3 | +0.93 / +0.25 | +1.69 / +0.38 | +1.29 / +0.53 | -0.13 / +0.70 | -3.07 / +0.88 |
| 4 | +0.66 / +0.20 | +1.24 / +0.32 | +0.93 / +0.44 | -0.85 / +0.58 | -3.39 / +0.73 |
| 5 | +0.57 / +0.17 | +1.00 / +0.27 | +0.77 / +0.38 | -0.78 / +0.49 | -3.30 / +0.62 |
| 6 | +0.49 / +0.15 | +0.80 / +0.23 | +0.72 / +0.33 | -0.85 / +0.43 | -3.24 / +0.54 |
| 8 | +0.38 / +0.12 | +0.65 / +0.18 | +0.66 / +0.26 | -0.64 / +0.34 | -3.39 / +0.43 |
* Early inspiral (x_pp <= 0.08): sign right, q-scaling right (ratio 3.2-3.7 at x 0.06, 3.5-4.4 at 0.08, flat in q),
  magnitude ~3.4x the 1.5PN term.  Late (x_pp >= 0.12): sign flips -- strong field, outside PN.
* Caveat: equal-phase-from-merger beta is not the mismatch-optimal beta.  Step 3's fitted k cross-checks the factor.
### Steps 2-3 results, and a CORRECTION to step 1
* Step 2 (per-case inspiral fits, S18, diag_percase.py, job 3487196, 25 s): per-case floors E_insp median 7.7e-5 (flux form, 4/case),
  1.0e-4 (pn form, 3/case).  Per-case b_pp/X1^(6/5) spin slope: +0.417/+0.318/+0.295 % per unit chi1 at q = 3/5/8.
* Step 3 (global inspiral fits, S18, diag_so_term.py, job 3487192, 8 min), median / mean / max E_insp:
  flux 9c 6.6e-4/1.0e-3/3.3e-3 -> + SO (step-1 coef) 5.0e-4/6.9e-4/2.4e-3 -> + SO with k fitted (10c) 2.8e-4/3.5e-4/1.1e-3, k = 3.57.
  installed pn_anchored_spin (13c): 2.1e-4/7.9e-4/4.7e-3.  pn x-polynomial: SO does not help (k -> -22.6, fights s11..s22);
  pn3 (b1 b2 a2 + SO): 1.1e-3, k = 3.0.  flux with s1 = s2 = 0 + SO: 1.5e-3 (the energy-coupling spin terms are still needed).
* **CORRECTION (step 1 used 3/5; correct is 3/2).**  Step 1 copied the note's constant-beta matching.  With beta = beta(x):
  equal phase => omega_NR = omega_pp / beta; x_NR = beta^(-2/3) x_pp; dx_NR/dtau = beta^(-5/3) xdot_pp [1 - (2/3) x dln(beta)/dx].
  For delta = dln(beta) ∝ x^p:  delta = 3 (A_NR - A_pp) / (5 - 2p).  p = 0 -> 3/5 (the note; unchanged);
  p = 3/2 -> 3/2.  So the SO coefficient is 2.5x the step-1 value:
      delta ln beta_SO = (3/2) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]
  With it: fitted k = 3.57/2.5 = 1.43; equal-phase measured/derived ~3.4/2.5 = 1.36 (early x).  p = 5/2 gives 5 - 2p = 0:
  an x^(5/2) beta term is unconstrained by chirp matching (degenerate with the phase constant, like 2.5PN phasing).
* Note for later: by amplitude matching alpha ∝ beta^(-2/3), so the SO beta term implies alpha_SO = -(2/3) delta_SO plus the
  amplitude's own SO terms (not derived tonight).
* Step 3b (corrected coefficient, zero added parameters; diag_so_term.py --corrected, job 3487204):
  **flux form + derived SO (9c): E_insp median 3.0e-4, mean 4.1e-4, max 1.4e-3** (bands 1.5e-4 / 4.2e-5 / 3.3e-5 / 5.9e-5).
  vs fitted k (10c) 2.8e-4/3.5e-4/1.1e-3, vs installed pn_anchored_spin (13c) 2.1e-4/7.9e-4/4.7e-3 (mean 2x, max 3.3x better;
  median 1.4x worse).  pn3 + corrected SO: 1.1e-3 (the pn polynomial does not benefit).
### Step 5 — flux split variant: first runs INVALID (seed bug), rerun
* fit_flux_spin_split.py jobs 3487212 (window T-1) / 3487213 (isco): the fit's t0 seed was read inside the pool workers from a
  fork-time copy (always -50 M), so every stage evaluation aligned within +-10 M of -50 -> wrong objective (stage 1 'started'
  at E_insp 0.12 for coefficients that score 3e-4).  Final scores used a wide t0 search, so they are correct for the badly
  fitted coefficients but say nothing about the method.  Results moved to results/invalid_seedbug/.  Fixed (seed passed
  with the task, as the diagnostics' harness already did); rerun.
* Useful from those runs (start row = inspiral copy extended through merger, no fit of the MR side): insp-aligned
  E_insp 2.95e-4, E_MR 7.8e-2, E_full 5.8e-3 (median) vs installed pn_anchored_spin 2.1e-4 / 3.0e-2 / 2.1e-3 and installed
  fluxanchored_spin 7.4e-4 / 7.1e-2 / 4.9e-3.
* **Rerun (jobs 3487237 T-1, 15 min; 3487238 isco, 7 min) -- fluxanchored_spin_split, 18c (9 insp + 9 MR), S18:**
  full-aligned E_full median/mean/max **7.0e-4 / 1.42e-3 / 5.8e-3** vs installed pn_anchored_spin 1.12e-3 / 1.72e-3 / 6.6e-3
  (-37% / -17% / -12%).  insp-aligned: E_insp 2.7e-4/3.9e-4/1.4e-3 (pn13 2.1e-4/7.9e-4/4.7e-3); E_MR 1.0e-2/3.3e-2/0.25
  (pn13 3.0e-2/3.9e-2/0.15).  Gate P ok.  Stage 2 moved E_insp 3.8% (one repeat pass).  ISCO-start window variant worse
  (E_full median 1.07e-3), so the user's T = -1 start is better.  Caveats: 18c vs 13c; S18 only; E_MR max from few cases.
### Step 5b — T96 (job 3487260, 13 min) and step 6 — eta = nu (job 3487261, 15 min)
| T96, full-aligned E_full | coef | median | mean | max |
|:--|--:|--:|--:|--:|
| installed pn_anchored_spin (leader) | 13 | 8.65e-4 | 1.17e-3 | 6.6e-3 |
| installed fluxanchored_spin | 9 | 1.30e-3 | 2.11e-3 | 1.4e-2 |
| flux split, S18-fitted, scored on T96 (78/96 unseen) | 18 | 6.43e-4 | 1.10e-3 | 5.8e-3 |
| **flux split, T96-fitted** | 18 | **5.17e-4** | **9.69e-4** | 6.7e-3 |
* T96 insp-aligned: E_insp 1.53e-4/2.40e-4/1.7e-3 (leader 1.97e-4/4.69e-4/4.7e-3); E_MR 7.7e-3/2.2e-2/0.31 (leader 2.2e-2/2.7e-2/0.15).
* vs EXPLORATION gate G1 (beat leader on T96 mean AND median, max not worse, H270 median < 7.1e-4): mean -17%, median -40%,
  max +2% (6.7e-3 vs 6.58e-3) -> G1 misses narrowly on max; H270 not yet scored.  The S18-fitted version passes all three T96
  statistics.  Stage 2 moved E_insp 0.3% on T96.  Gate P ok.
* eta = nu (S18): every statistic within 1% of eta = 1/q (E_full 6.97e-4/1.40e-3/5.8e-3 vs 7.04e-4/1.42e-3/5.8e-3) -> irrelevant at
  this level; keep 1/q (user's choice).
* Worst cells: chi1 = +0.8 (all q; flux family's known hot corner) and q = 3, chi1 = -0.6 (retrograde).
### H270 (job 3487290, 2.4 min; reproduction: installed pn_anchored_spin H270 median 7.096e-4 vs EXPLORATION 7.10e-4)
flux split T96-fitted: full-aligned E_full 4.59e-4/5.94e-4/2.8e-3 (median/mean/max); insp-aligned E_insp 1.44e-4, E_MR 6.99e-3.
flux split S18-fitted: 6.05e-4/7.58e-4/2.9e-3.  installed pn13: 7.10e-4/8.27e-4/2.9e-3; installed flux9: 1.00e-3/1.44e-3/6.4e-3.
