# fluxanchored_spin_split (slim, 14 coefficients) — VARIANT, not installed

Spin model on the ISCO-split line (`spin_dep_split/`).  Parent: `spin_dep/fluxanchored_spin` (same functional family).
Code: `fit_flux_spin_split.py` (model, two-stage fit), `slim_split.py` (merger-branch selection + T96 refit).
Coefficients: `results/slim_confirm_aF_bPP_S.json`.  Design decisions: `README.md`.

## Model

All quantities on the ppBHPT clock t (m1 units); X1 = q/(1+q), nu = q/(1+q)^2.

    h_model(tau) = alpha(t) e^{i phi0} h_pp(t),      tau(t) = t0 + int beta dt        (t0, phi0: per-case alignment)

Inputs from the ppBHPT waveform: E(t) cumulative radiated energy (per total mass, despiked), F(t) = Edot / max Edot,
x = X1^(-4/5) min(x_pp, 0.26), x_pp = (omega_22 / 2)^(2/3).

**Inspiral copy (9 fitted: c0, c1, A0, A1, b, P0, P1, s1, s2)**

    alpha_i = a_pp [1 + a_F F],      beta_i = [b_pp + P E] [1 + kappa_SO(q) chi1 x^(3/2)]
    a_pp = X1^(6/5)(1 + c0 nu + c1 nu^2),   a_F = A0 nu + A1 nu^2,
    b_pp = X1^(6/5)(1 + b nu),              P   = P0 + P1 nu + nu (s1 chi1 + s2 chi1^2)

**Merger-ringdown copy (5 fitted: A0', A1', b', s1', s2'; the rest shared with the inspiral copy)**

    alpha_m = a_pp [1 + a_F' F],     a_F' = A0' nu + A1' nu^2
    beta_m  = b_pp' + P' E,          b_pp' = X1^(6/5)(1 + b' nu),   P' = P0 + P1 nu + nu (s1' chi1 + s2' chi1^2)

**Blend (no fitted coefficients)**

    alpha = (1 - S) alpha_i + S alpha_m,    beta = (1 - S) beta_i + S beta_m
    S = u^3 (10 - 15u + 6u^2),  u = clip[(T + 1)/(T_peak + 1), 0, 1]
    T(t) = T* + (dtau/dt)_isco (t - t_isco) / (eta^(-1/5) tau_o(chi1)),   eta = 1/q,  T* = 0.7154

Fit: two stages, mean over T96.  (1) inspiral coefficients on E_insp (NR [-5000.1, t_isco)); (2) merger coefficients on
E_MR (NR [t_isco, +100]), inspiral frozen, alignment set by the inspiral.  Merger groups chosen by greedy forward
selection on S18 (kept A0'A1', b', s1's2' at -19/-20/-24%; rejected c0'c1' (-1%) and P0'P1').

## Results (full-window E, full-window alignment; median / mean / max)

| model | coef | T96 | H270 (held out) | q = 2.5 | q = 2 |
|:--|--:|:--|:--|--:|--:|
| Rink et al. 2024 (published calibrated BHPTNRSur2dq1e3, constant alpha, beta; 90 pts, q 3-10) | 12 | 2.35e-3 / 3.47e-3 / 1.4e-2 | 1.90e-3 / 2.52e-3 / 9.8e-3 | 1.3e-2 | 1.1e-1 |
| fluxanchored_spin (installed parent) | 9 | 1.30e-3 / 2.11e-3 / 1.4e-2 | 1.00e-3 / 1.44e-3 / 6.4e-3 | 8.3e-3 | 2.8e-2 |
| parent + derived SO term, one form | 9 | 8.9e-4 / 1.09e-3 / 6.5e-3 | 7.2e-4 / 7.9e-4 / 2.7e-3 | 4.4e-3 | 1.9e-2 |
| pn_anchored_spin (installed; 10 fitted on the spin grid + 3 from the nonspinning fit) | 13 | 8.65e-4 / 1.17e-3 / 6.6e-3 | 7.10e-4 / 8.27e-4 / 2.9e-3 | 4.1e-3 | 1.6e-2 |
| **this model** | **14** | **5.42e-4 / 9.44e-4 / 6.9e-3** | **4.88e-4 / 5.86e-4 / 2.8e-3** | 5.0e-3 | 2.1e-2 |

q = 2, 2.5: medians over 8 spins; the ppBHPT surrogate is itself extrapolated there (trained q >= 3), so only the ranking
is meaningful.  Best in range; below q = 3 the 9-coefficient one-form model and pn_anchored_spin extrapolate better
(the 5 merger coefficients' free nu, nu^2 terms).  Hot cells: chi1 = +0.8, low-q retrograde.

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
