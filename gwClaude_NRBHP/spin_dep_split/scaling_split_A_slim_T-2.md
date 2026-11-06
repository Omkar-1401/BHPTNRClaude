# split A: slim split, Ori–Thorne window from T = -2 (14 coefficients) — VARIANT, not installed

Spin model on the ISCO-split line (`spin_dep_split/`).  Same form and fit as `scaling_fluxanchored_spin_split.md` (slim, 14
coefficients); the only change is that the merger blend starts earlier, at Ori–Thorne time T = -2 instead of -1.
Chosen 2026-10-15 by 4-fold cross-validation on T96 (`CV_LOG.md`); scored once on the fresh test set F100.
Code: `explore_split.py` (variant `slim__win-2.0`; model, fit), `cv_split.py` (CV), `score_f100.py`, `plot_split_AB.py`,
`diag_split_AB.py`.  Coefficients: `results/explore/slim__win-2.0.json`.  Figures: `Agentic_spin_split_plots/split_A_slim_T-2/`.

## Model

All quantities on the ppBHPT clock t (m1 units); X1 = q/(1+q), nu = q/(1+q)^2.

    h_model(tau) = alpha(t) e^{i phi0} h_pp(t),      tau(t) = t0 + int beta dt        (t0, phi0: per-case alignment)

Inputs from the ppBHPT waveform: E(t) cumulative radiated energy (per total mass, despiked), F(t) = Edot / max Edot,
x = X1^(-4/5) min(x_pp, 0.26), x_pp = (omega_22 / 2)^(2/3).

**Inspiral copy (9 fitted: c0, c1, A0, A1, b, P0, P1, s1, s2)**

    alpha_i = a_pp [1 + a_F F],      beta_i = [b_pp + P E] [1 + kappa_SO(q) chi1 x^(3/2)]
    a_pp = X1^(6/5)(1 + c0 nu + c1 nu^2),   a_F = A0 nu + A1 nu^2,
    b_pp = X1^(6/5)(1 + b nu),              P   = P0 + P1 nu + nu (s1 chi1 + s2 chi1^2)

**Merger-ringdown copy (5 fitted: A0', A1', b', s1', s2'; c0, c1, P0, P1 shared with the inspiral copy; no SO factor)**

    alpha_m = a_pp [1 + a_F' F],     a_F' = A0' nu + A1' nu^2
    beta_m  = b_pp' + P' E,          b_pp' = X1^(6/5)(1 + b' nu),   P' = P0 + P1 nu + nu (s1' chi1 + s2' chi1^2)

**Blend (no fitted coefficients)**

    alpha = (1 - S) alpha_i + S alpha_m,    beta = (1 - S) beta_i + S beta_m
    S = u^3 (10 - 15u + 6u^2),  u = clip[(T - T_start)/(T_peak - T_start), 0, 1],   T_start = -2
    T(t) = T* + (dtau/dt)_isco (t - t_isco) / (eta^(-1/5) tau_o(chi1)),   eta = 1/q,  T* = 0.7154
T_peak = T at the ppBHPT (2,2) peak.  S = 0 before T_start, 1 from the peak on.

**Coefficients** (fit on all of T96)

| c0 | c1 | A0 | A1 | b | P0 | P1 | s1 | s2 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.16292 | -0.65455 | 6.54632 | -45.84031 | 0.00779 | -5.87310 | 45.12084 | 16.27536 | -13.72318 |

| A0' | A1' | b' | s1' | s2' |
|--:|--:|--:|--:|--:|
| -2.5467 | 6.4923 | 0.2162 | 48.3354 | -64.5922 |

Fit (split protocol, `README.md`): two stages, mean over T96.  (1) inspiral coefficients on E_insp (NR [-5000.1, t_isco));
(2) merger coefficients on E_MR (NR [t_isco, +100]), inspiral frozen, one shared t0, phi0 set by the inspiral.  The inspiral
copy is the T = -1 slim fit's (stage 1 does not see the window); only stage 2 was redone at T = -2.

## How it was chosen

The T_start = -2 window, and the slim merger branch, were first picked on H270 (EXPLORE_LOG rounds 1-4) — a mistake, since
H270 is held out.  Redone without H270 (`CV_LOG.md`): 22 split variants, 4-fold CV on T96 (fold = (iq + ic) mod 4, refit
from scratch on 72, scored on 24).  This model has the lowest pooled held-out **median** (4.76e-4); picking the window
start inside each fold by training median gives T = -2 in 3 of 4 folds.  Ranked by mean, model B
(`scaling_split_B_b0_T-3.md`) is ahead.  The user took both to the test set.

## Results (full-window E, full-window alignment; median / mean / max)

| set | this model | slim, T = -1 (parent) |
|:--|:--|:--|
| T96 (training) | 4.64e-4 / 8.54e-4 / 6.5e-3 | 5.42e-4 / 9.44e-4 / 6.9e-3 |
| T96, 4-fold CV held out (96) | 4.76e-4 / 8.87e-4 / 7.0e-3 | 5.44e-4 / 9.53e-4 / 7.3e-3 |
| H270 (validation; used for selection, so optimistic) | 3.98e-4 / 5.41e-4 / 2.7e-3 | 4.88e-4 / 5.86e-4 / 2.8e-3 |
| **F100 (test, untouched; one look 2026-10-15)** | **4.99e-4 / 7.15e-4 / 2.74e-3** | not scored |
| q = 2.5 / q = 2 (median over 8 spins; ppBHPT itself extrapolated) | 4.8e-3 / 2.1e-2 | 5.0e-3 / 2.1e-2 |

For scale (H270, never tuned on it): flux_anc_spin_with_orbit (9) 7.20e-4 / 7.90e-4 / 2.7e-3; pn_anchored_spin (13)
7.10e-4 / 8.27e-4 / 2.9e-3.  F100 = 100 points uniform in 3 < q < 8, -0.6 < chi1 < 0.8 (`fresh_set.py`, seed 20261013).

## Diagnostics and known defects

Figures: train/validation/test histogram, before/after calibration (q = 3, 5, 8 at chi1 = +0.6), alpha/beta vs time,
E vs chi1, (q, chi1) map, dense 20 x 20 heatmap, error by time band and residual vs time (q = 4), peak errors on T96,
alpha/beta overlays (chi_const/, q_const/), Fig. 10 t_end study (q = 4).
* **Retrograde ringdown**: at q = 4, chi1 = -0.6 the phase residual oscillates after +20 M and runs to +45 rad by +100 M,
  with +-20% amplitude wiggles; alpha has narrow dips just after the peak for chi1 < 0.  Not yet traced (flux-drive F(t)
  after the peak is the suspect).
* **Step in E at chi1 ~ 0** across all q (dense heatmap), also in the published models: likely the ppBHPT input; untested.
* Worst cells: chi1 = +0.8 and chi1 = -0.6 at q ~ 3.

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
* **Window start T_start = -2: tuned, not anchored.**  Ori & Thorne's transition starts near T ~ -1 (where the adiabatic
  inspiral fails), which was the original, physically motivated choice.  Starting the blend at -2 was chosen by the data
  (T96 CV), so it counts as a tuned setting, not a derived anchor.
