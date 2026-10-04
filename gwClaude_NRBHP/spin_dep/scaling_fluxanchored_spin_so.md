# fluxanchored_spin_so (9 coefficients) — fluxanchored_spin + derived PN spin-orbit term; VARIANT, not installed

One form for the whole inspiral–merger–ringdown (no split, no switch).  It is the installed `fluxanchored_spin` with one
change: beta carries the derived 1.5PN spin-orbit factor (no fitted parameter).  Found 2026-10-04 while diagnosing the
split line (`spin_dep_split/`): the ISCO-split error exposed that the early-inspiral beta of every global form lacked the
chi1-dependence the data want (full-window fits had hidden it; see "How it was found").

Code: `fit_fluxanchored_spin_so.py` (fit + scoring; reuses the split line's case builder and scorer with the blend off);
coefficients and per-case scores: `fluxanchored_spin_so_results/coeffs.json` (key theta_best); job script
`ls6/spin_dep/run_fluxanchored_spin_so.slurm`.  Not installed (no generator / selftest entry yet).

## Model

ppBHPT clock t (m1 units); X1 = q/(1+q), nu = q/(1+q)^2; E(t) cumulative radiated energy (per total mass, the model
family's despiked drive), F(t) = Edot / max Edot, x = X1^(-4/5) min(x_pp, 0.26), x_pp = (omega_22/2)^(2/3).

    h_model(tau) = alpha(t) e^{i phi0} h_pp(t),     tau(t) = t0 + int beta dt                (t0, phi0 per case)
    alpha = a_pp [1 + a_F F(t)]
    beta  = [b_pp + P E(t)] [1 + kappa_SO(q) chi1 x^(3/2)]
    a_pp = X1^(6/5)(1 + c0 nu + c1 nu^2),   a_F = A0 nu + A1 nu^2,
    b_pp = X1^(6/5)(1 + b nu),              P   = P0 + P1 nu + nu (s1 chi1 + s2 chi1^2)
    kappa_SO(q) = (3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]          (derived; see Anchors)

(a_pp, a_F, b_pp, b_e = P/b_pp) are `fit_fluxanchored_spin.params_at(q, chi1, theta)` with WEIGHT = "nu", exactly as the
parent; the only addition is the factor [1 + kappa_SO chi1 x^(3/2)] on beta.

**Fit**: mean over T96 of the full-window mathcalE (NR [-5000.1, +100] M, t0 and phi0 optimised) -- the parent's own
objective.  Two starts (the installed parent's coefficients; the split model's inspiral copy) converge to the same
optimum, mean 1.0898e-3 on the fit grid.

    c0 = 0.224287   c1 = -1.026587   A0 = -1.739395   A1 = 1.674651   b = 0.007907
    P0 = -6.453695  P1 = 43.393337   s1 = 15.620569   s2 = 1.714523

## Results (full-window mathcalE, full-window alignment; median / mean / max)

| model | coef | T96 | H270 (held out) | q = 2.5 | q = 2 |
|:--|--:|:--|:--|--:|--:|
| Rink et al. 2024 (published calibrated BHPTNRSur2dq1e3, constant alpha, beta; 90 pts, q 3-10) | 12 | 2.35e-3 / 3.47e-3 / 1.4e-2 | 1.90e-3 / 2.52e-3 / 9.8e-3 | 1.3e-2 | 1.1e-1 |
| fluxanchored_spin (installed parent) | 9 | 1.30e-3 / 2.11e-3 / 1.4e-2 | 1.00e-3 / 1.44e-3 / 6.4e-3 | 8.3e-3 | 2.8e-2 |
| **this model** | **9** | **8.86e-4 / 1.09e-3 / 6.5e-3** | **7.20e-4 / 7.90e-4 / 2.7e-3** | 4.4e-3 | 1.9e-2 |
| pn_anchored_spin (installed; 10 fitted on the spin grid + 3 from the nonspinning fit) | 13 | 8.65e-4 / 1.17e-3 / 6.6e-3 | 7.10e-4 / 8.27e-4 / 2.9e-3 | 4.1e-3 | 1.6e-2 |
| fluxanchored_spin_split, slim (`spin_dep_split/`) | 14 | 5.42e-4 / 9.44e-4 / 6.9e-3 | 4.88e-4 / 5.86e-4 / 2.8e-3 | 5.0e-3 | 2.1e-2 |

* vs the parent, same 9 coefficients: -32% / -48% / -54% on T96, -28% / -45% / -58% on H270, -47% at q = 2.5, -31% at q = 2.
* vs Rink et al.: 2.6x / 3.2x / 3.6x better on H270 with fewer coefficients.
* Level with pn_anchored_spin (13 coefficients) in range; better than the 14-coefficient split below q = 3.
* q = 2, 2.5: medians over 8 spins (chi1 = -0.6 .. +0.8); the ppBHPT surrogate is itself extrapolated there (trained q >= 3),
  so only the ranking is meaningful.  The parent's numbers reproduce its stored coefficient file (T96 median 1.30e-3); the
  installed pn_anchored_spin reproduces its recorded H270 median (7.10e-4).
* Insp-aligned split errors (T96 median): E_insp 2.8e-4, E_MR 1.9e-2 (parent 5.2e-4, 3.7e-2).

## How it was found

The Kerr-ISCO split (`spin_dep_split/README.md`) separated the inspiral error.  Per-case inspiral fits reach ~1e-6 (Fig. 10
study) to ~8e-5, but every global inspiral form stalled at 6e-4 - 1e-3, with the excess in the early inspiral (< -1000 M).
The installed pn_anchored_spin did better only through its never-off switch: a 5-10% admixture of r0 beta_r(q, chi1) whose
fitted weight was negative (r0 = -0.06) -- i.e. a chi1-shaped correction, not ringdown physics.  Per-case fits then showed
the early-inspiral beta prefactor rising ~0.3-0.4% per unit chi1 at q = 3-8, which no global form provided.  The PN
spin-orbit term supplies exactly that.  In full-window fits the merger dominates the error (about half of it from 7% of
the signal), which is why the gap had not shown.

## Anchors

* **X1^(6/5)** in a_pp and b_pp: leading-order chirp-rate and (2,2)-amplitude matching (`alpha_beta_pn_scaling_note_revised.pdf`,
  Eqs. 7-16): eta_pp beta^(5/3) = nu with eta_pp = 1/q gives beta_LO = X1^(6/5); the amplitude match gives alpha_LO = beta_LO.
* **PP anchoring**: a_pp, b_pp -> X1^(6/5) -> 1 and a_F -> 0 as nu -> 0 (all corrections carry powers of nu), as in the parent.
* **Energy drive**: E per unit TOTAL mass, despiked in the 2d surrogate's join window (`physics_common.ENERGY_DRIVE_2D`),
  as in the parent (`scaling_fluxanchored_spin.md`).
* **Derived spin-orbit term kappa_SO** (no fitted parameter):

      kappa_SO(q) = (3/2) [ 113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1 ]

  1. Spin-orbit parts of the binding energy and the flux, from Blanchet's Living Review (arXiv:1310.1528, v3 source in
     `spin_dep_split/refs/blanchet_lrr/`; Eqs. ESO, fluxSO, defSSigma), spins in G m^2 units, S = S1 + S2, Sigma = S2/X2 - S1/X1:
         E = -(nu m x / 2) [1 + e2 x + e_SO x^(3/2) + ...],     e_SO = 14/3 S_l + 2 Delta Sigma_l
         F = F_N [1 + f2 x + (4 pi + f_SO) x^(3/2) + ...],      f_SO = -4 S_l - 5/4 Delta Sigma_l
  2. Chirp rate dx/dt = F / (-dE/dx).  Differentiating x * x^(3/2) = x^(5/2) gives -dE/dx ∝ 1 + 2 e2 x + (5/2) e_SO x^(3/2);
     with no x^(1/2) terms, nothing else reaches x^(3/2), so the spin part of the x^(3/2) coefficient is
         A_SO = f_SO - (5/2) e_SO = -47/3 S_l - 25/4 Delta Sigma_l
     (the standard TaylorT4 SO coefficient; combined here from the two LRR equations, not printed in that form there).
     The 4 pi tail term is spin-independent and belongs to the nonspinning beta (carried by the fitted coefficients).
  3. chi2 = 0: S_l = X1^2 chi1, Sigma_l = -X1 chi1, Delta = X1 - X2  =>  A_NR = -chi1 (19/6 X1^2 + 25/4 X1) x^(3/2).
     Test mass on Kerr (the ppBHPT; X1 -> 1, m1 units): A_pp = -113/12 chi1 x_pp^(3/2).  Checks: Kidder's 113/12;
     energy -> 8/3 chi and flux -> -11/4 chi, the Kerr circular-orbit values.
  4. Matching (as the note): eta_pp beta^(5/3) (1 + A_pp(x_pp)) = nu (1 + A_NR(x_NR)), x_NR = beta^(-2/3) x_pp.  Because beta
     varies with x, equal phase gives omega_NR = omega_pp / beta and dx_NR/dtau = beta^(-5/3) xdot_pp [1 - (2/3) x dln(beta)/dx];
     for a term in ln(beta) ∝ x^p this gives delta = 3 (A_NR - A_pp) / (5 - 2p): 3/5 for a constant (the note's case),
     **3/2 for p = 3/2**.  With x_pp^(3/2) = X1^(6/5) x^(3/2) (note's x) the result is kappa_SO above.  It vanishes as q -> inf,
     ≈ (77/60) X2 at leading order.
  Only the binary's departure from the test-mass value enters, because ppBHPT already contains the spin exactly for a
  test mass.  Data checks: sign and q-scaling match the model-free measured d ln(beta)/d chi1 at q = 3-8; a free multiplier
  fits k ≈ 1.4 (~40% missing, plausibly next-order SO).  A first version used the note's 3/5 (2.5x too small); corrected.
  Here the factor multiplies the whole beta, including the merger, where the PN expansion has no meaning.  With x clipped
  at x_pp = 0.26 it saturates at kappa_SO chi1 x_clip^(3/2) ≈ 4% per unit chi1 at q = 4 (not negligible); the fitted
  merger-side behaviour (P, s1, s2 through E(t)) absorbs whatever it gets wrong there.  It is justified as an inspiral term.
* **Not anchored**: c0, c1, A0, A1, b, P0, P1, s1, s2 (fitted, as in the parent); alpha has no spin-orbit term yet (by
  amplitude matching alpha would get -(2/3) delta_SO plus the amplitude's own SO terms -- not derived).
