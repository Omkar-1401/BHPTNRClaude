# split B: inspiral b = 0, slim merger branch, Ori–Thorne window from T = -3 (13 coefficients + window start) — VARIANT, not installed

Spin model on the ISCO-split line (`spin_dep_split/`).  Differs from the slim split (`scaling_fluxanchored_spin_split.md`)
in two places: the inspiral b_pp has no nu correction (b = 0, so b_pp = X1^(6/5) exactly in the inspiral), and the merger blend
starts at Ori–Thorne time T = -3.  Chosen 2026-10-15 by 4-fold cross-validation on T96 (`CV_LOG.md`); scored once on F100.
Code: `cv_split.py` (model "tline", fit, window rule), `score_f100.py`, `plot_split_AB.py`, `diag_split_AB.py`;
the form itself is `explore_split.py` (MR form `slim`).  Coefficients: `results/cv/tline__foldall.json`.
Figures: `Agentic_spin_split_plots/split_B_b0_T-3/`.

## Model

All quantities on the ppBHPT clock t (m1 units); X1 = q/(1+q), nu = q/(1+q)^2.

    h_model(tau) = alpha(t) e^{i phi0} h_pp(t),      tau(t) = t0 + int beta dt        (t0, phi0: per-case alignment)

Inputs from the ppBHPT waveform: E(t) cumulative radiated energy (per total mass, despiked), F(t) = Edot / max Edot,
x = X1^(-4/5) min(x_pp, 0.26), x_pp = (omega_22 / 2)^(2/3).

**Inspiral copy (8 fitted: c0, c1, A0, A1, P0, P1, s1, s2)**

    alpha_i = a_pp [1 + a_F F],      beta_i = [X1^(6/5) + P E] [1 + kappa_SO(q) chi1 x^(3/2)]
    a_pp = X1^(6/5)(1 + c0 nu + c1 nu^2),   a_F = A0 nu + A1 nu^2,   P = P0 + P1 nu + nu (s1 chi1 + s2 chi1^2)

**Merger-ringdown copy (5 fitted: A0', A1', b', s1', s2'; c0, c1, P0, P1 shared with the inspiral copy; no SO factor)**

    alpha_m = a_pp [1 + a_F' F],     a_F' = A0' nu + A1' nu^2
    beta_m  = b_pp' + P' E,          b_pp' = X1^(6/5)(1 + b' nu),   P' = P0 + P1 nu + nu (s1' chi1 + s2' chi1^2)

**Blend (window start tuned on training data, otherwise no fitted coefficients)**

    alpha = (1 - S) alpha_i + S alpha_m,    beta = (1 - S) beta_i + S beta_m
    S = u^3 (10 - 15u + 6u^2),  u = clip[(T - T_start)/(T_peak - T_start), 0, 1],   T_start = min(T0 + T1 chi1, T_peak - 0.55)
    T(t) = T* + (dtau/dt)_isco (t - t_isco) / (eta^(-1/5) tau_o(chi1)),   eta = 1/q,  T* = 0.7154
Chosen: **T0 = -3, T1 = 0**, so T_start = -3 for every case (on T96 the guard never acts: every start is -3).

**Coefficients** (fit on all of T96)

| c0 | c1 | A0 | A1 | b | P0 | P1 | s1 | s2 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.18256 | -0.80625 | 3.90427 | -36.17056 | 0 (pinned) | -4.84212 | 43.57566 | 14.15030 | -10.24908 |

| A0' | A1' | b' | s1' | s2' |
|--:|--:|--:|--:|--:|
| -2.1262 | 4.6000 | 0.2175 | 40.6126 | -53.8184 |

Count: 13 fitted coefficients + the window start T0 (tuned on a 4-value grid).  The slope T1 was allowed (14th) and came
out 0, so it is not part of the model.

Fit (split protocol, `README.md`), all on T96: (1) the 8 inspiral coefficients on mean E_insp (NR [-5000.1, t_isco));
(2) for each (T0, T1) on the grid {-1, -2, -3, -4} x {0, 1.5, 3, 4.5}, the 5 merger coefficients on mean E_MR (inspiral
frozen, one shared t0, phi0 set by the inspiral); (3) keep the grid point with the lowest mean full-window E on T96.
Training mean full-window E (e-4) over the grid, T1 = 0 row: T0 = -1 9.48, -2 8.60, **-3 8.49**, -4 8.82; any T1 > 0
raised the mean (it lowered the median: prograde and retrograde want different starts, see below).

## How it was chosen

The idea came from the H270 exploration (a chi1-dependent start: retrograde wanted T <= -4, prograde -0.5 to -1); its
coefficients were never read off H270.  The (T0, T1) choice was repeated inside each of the 4 CV folds on the fold's
training points only: all 4 folds and the full fit chose T0 = -3, T1 = 0.  Pooled CV held-out: best **mean** (8.55e-4)
and **max** (5.85e-3) of 23 split variants, median 6.24e-4 (model A has the best median, 4.76e-4).  Held-out medians per
chi1 (e-4), -0.6 .. +0.8: 7.2 2.8 1.8 3.4 4.1 4.8 7.4 18.1 — vs slim T = -2: 11.5 5.1 3.3 2.4 2.6 4.2 8.2 27.3.  Better at both
spin edges (-37% at chi1 = -0.6, -34% at +0.8), worse for 0 <= chi1 <= 0.4.

## Results (full-window E, full-window alignment; median / mean / max)

| set | this model | model A (slim, T = -2) |
|:--|:--|:--|
| T96 (training) | 6.21e-4 / 8.49e-4 / 5.6e-3 | 4.64e-4 / 8.54e-4 / 6.5e-3 |
| T96, 4-fold CV held out (96) | 6.24e-4 / 8.55e-4 / 5.85e-3 | 4.76e-4 / 8.87e-4 / 7.0e-3 |
| H270 (validation; spent by the exploration, so not untouched) | 4.08e-4 / 5.36e-4 / 2.2e-3 | 3.98e-4 / 5.41e-4 / 2.7e-3 |
| **F100 (test, untouched; one look 2026-10-15)** | **4.85e-4 / 6.82e-4 / 1.94e-3** | 4.99e-4 / 7.15e-4 / 2.74e-3 |
| q = 2.5 / q = 2 (median over 8 spins; ppBHPT itself extrapolated) | 6.5e-3 / 2.7e-2 | 4.8e-3 / 2.1e-2 |

Best of the split line on the test set (all three numbers), and 29% lower worst case than A.  Below q = 3 it extrapolates
worse than A and than the one-form models (flux_anc_spin_with_orbit 4.4e-3, pn_anchored_spin 4.1e-3 at q = 2.5).
For scale (H270, never tuned on it): flux_anc_spin_with_orbit (9) 7.20e-4 / 7.90e-4 / 2.7e-3; pn_anchored_spin (13)
7.10e-4 / 8.27e-4 / 2.9e-3.

## Diagnostics and known defects

Figures: train/validation/test histogram, before/after calibration (q = 3, 5, 8 at chi1 = +0.6), alpha/beta vs time,
E vs chi1, (q, chi1) map, dense 20 x 20 heatmap, error by time band and residual vs time (q = 4), peak errors on T96,
alpha/beta overlays (chi_const/, q_const/), Fig. 10 t_end study (q = 4; two stuck points re-fitted from the t_end = -50 solution, `fix_tend_B.py`).
* **Retrograde ringdown**: at q = 4, chi1 = -0.6, 80% of E comes from NR [0, +100] M (error_by_band); alpha has narrow dips
  just after the peak for chi1 < 0.  Not yet traced (flux-drive F(t) after the peak is the suspect).
* **Step in E at chi1 ~ 0** across all q, also in the published models: likely the ppBHPT input; untested.
* chi1 = +0.8 is the worst row (held-out median 1.8e-3).

## Anchors

* **X1^(6/5)** in a_pp, b_pp: leading-order chirp-rate and (2,2)-amplitude matching (`alpha_beta_pn_scaling_note_revised.pdf`,
  Eqs. 7-16): eta_pp beta^(5/3) = nu with eta_pp = 1/q gives beta = X1^(6/5); the amplitude match gives alpha = beta.
* **Kerr ISCO cut and Ori–Thorne window**: t_isco where x_pp reaches Omega_isco(chi1)^(2/3) (Bardeen–Press–Teukolsky);
  tau_o, (dtau/dt)_isco from Ori & Thorne (gr-qc/0003032) closed forms with Finn's Edot_isco (their Table I, reproduced to
  3.4e-4); T* = 0.7154 from their universal transition solution; eta = 1/q (particle m2 on a Kerr hole m1).  Details: README.
* **Derived spin-orbit term kappa_SO** (no fitted parameter):

      kappa_SO(q) = (3/2) [ 113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1 ]

  1. Spin-orbit parts of the binding energy and the flux, from Blanchet's Living Review (arXiv:1310.1528, v3 source in
     `refs/blanchet_lrr/`; Eqs. ESO, fluxSO, defSSigma), spins in G m^2 units, S = S1 + S2, Sigma = S2/X2 - S1/X1:
         E = -(nu m x / 2) [1 + e2 x + e_SO x^(3/2) + ...],     e_SO = 14/3 S_l + 2 Delta Sigma_l
         F = F_N [1 + f2 x + (4 pi + f_SO) x^(3/2) + ...],      f_SO = -4 S_l - 5/4 Delta Sigma_l
  2. Chirp rate dx/dt = F / (-dE/dx).  Differentiating x * x^(3/2) = x^(5/2) gives -dE/dx ∝ 1 + 2 e2 x + (5/2) e_SO x^(3/2);
     with no x^(1/2) terms, nothing else reaches x^(3/2), so the spin part of the x^(3/2) coefficient is
         A_SO = f_SO - (5/2) e_SO = -47/3 S_l - 25/4 Delta Sigma_l
     (the standard TaylorT4 SO coefficient; combined here from the two LRR equations, not printed in that form there).
     The 4 pi tail term is spin-independent and belongs to the nonspinning beta.
  3. chi2 = 0: S_l = X1^2 chi1, Sigma_l = -X1 chi1, Delta = X1 - X2  =>  A_NR = -chi1 (19/6 X1^2 + 25/4 X1) x^(3/2).
     Test mass on Kerr (the ppBHPT; X1 -> 1, m1 units): A_pp = -113/12 chi1 x_pp^(3/2).  Checks: Kidder's 113/12;
     energy -> 8/3 chi and flux -> -11/4 chi, the Kerr circular-orbit values.
  4. Matching (as the note): eta_pp beta^(5/3) (1 + A_pp(x_pp)) = nu (1 + A_NR(x_NR)), x_NR = beta^(-2/3) x_pp.  Because beta
     varies with x, equal phase gives omega_NR = omega_pp / beta and dx_NR/dtau = beta^(-5/3) xdot_pp [1 - (2/3) x dln(beta)/dx];
     for a term in ln(beta) ∝ x^p this gives delta = 3 (A_NR - A_pp) / (5 - 2p): 3/5 for a constant (the note's case),
     **3/2 for p = 3/2**.  With x_pp^(3/2) = X1^(6/5) x^(3/2) (note's x) the result is kappa_SO above.  It vanishes as q -> inf,
     the bracket ≈ (77/60) X2 at leading order, so kappa_SO ≈ (77/40) X2 (exact value 0.267 at q = 4).
  Only the binary's departure from the test-mass value enters, because ppBHPT already contains the spin exactly for a
  test mass.  Data checks: sign and q-scaling match the model-free measured d ln(beta)/d chi1 at q = 3-8; a free multiplier
  fits k ≈ 1.4 (~40% missing, plausibly next-order SO).  First version used the note's 3/5 (2.5x too small); corrected.
* **Not anchored**: the merger-copy coefficients (fitted); alpha has no spin-orbit term yet (by amplitude matching
  alpha would get -(2/3) delta_SO plus the amplitude's own SO terms — not derived).
* **b = 0**: the inspiral clock rate is the bare anchor X1^(6/5) (plus the energy coupling P E and the SO factor), with no
  fitted nu correction.  The slim fit's b was tiny (0.008, a 0.1% correction at q = 4); with the T = -3 window the b = 0 form has the best CV mean.
* **Window start T_start = -3: tuned, not anchored.**  Ori & Thorne's transition starts near T ~ -1; -3 was chosen on the
  training data (nested in the CV folds), so it counts as a tuned setting, not a derived anchor.
