# Pass 48: the fully closed-form spin calibration (pass 40 and its lean variant)

**What it is.** The pass-40 model (`scaling_p40.md`) with every numerical input replaced by a closed-form expression. The calibration coefficients are unchanged: pass 40's 18, and the lean variant's 16. The only inputs left are:
- the ppBHPT waveform (BHPTNRSur2dq1e3, `calibrated=False`, (2,2) mode);
- its two measured drives, E(t) (radiated energy) and x_pp(t) (the PN parameter from its frequency).

Everything else is an explicit formula.

- **Model equations:** `scaling_p40.md` sections 1–4. Only the inputs listed in section 1 below differ.
- **Lean variant:** blend onset s1 = s_isco + (T_start - T*) u_OT, with T_start = -1 (pass 42 K1); `scaling_p40.md` and EXPLORATION.md pass 42.
- **Stand-alone generator:** `exploration/bhpt_spin_calib_p48.py`. It needs numpy only, with every constant in the file, and reproduces the exploration chain to 2.2e-9 relative in alpha, beta.

**Rule for every constant below:** each one was fitted OFFLINE, to theory or reference data (exact Kerr QNMs, NRSur3dq8Remnant, the ppBHPT input, the spline shapes), before and independent of the calibration. No NR waveform of the calibration enters. None of them counts as a calibration coefficient.

## 1. The closed forms

**1.1 R2(x), 2SF / 1SF (2,2) amplitude ratio (pass 40, option O2).** x is clipped to [0.005, 1/6].

    R2 = (55/21) x (1 - 6x)(1 + 4.768 x + 21.68 x^2)

**1.2 Kerr (2,2,0) QNM (pass 46).** These replace the `qnm` package. The fit covers chi in [-0.85, 0.95]; for chi < 0 it follows the counter-rotating branch.

    w220(chi)     = sum_{k=0..5} a_k (1 - chi)^(k/2),  a = +0.991804, -1.35691, +1.33461, -0.875476, +0.335733, -0.0561001
    gamma220(chi) = sum_{k=0..5} b_k (1 - chi)^(k/2),  b = +0.00278198, +0.319987, -0.512191, +0.442597, -0.201742, +0.0375373

- **Accuracy on [-0.65, 0.92]:** 8.7e-5 in w220 and 3.1e-4 in gamma220 (relative).
- **Extremal limits:** w220 -> 0.99 and gamma220 -> 0.003 as chi -> 1.
- **Q220** = w220 / (2 gamma220).

**1.3 Remnant mass and spin (pass 46).** These replace NRSur3dq8Remnant. The fit covers q in [1.5, 10], chi1 in [-0.8, 0.8], chi2 = 0.

    chi_f   = X1^2 chi1 + nu sum_{i,j=0..3} c_ij nu^i chi1^j
    1 - M_f = nu (1 - E_isco(X1^2 chi1)) sum_{i,j=0..3} m_ij nu^i chi1^j
    E_isco(a) = (1 - 2v^2 + a v^3) / sqrt(1 - 3v^2 + 2a v^3),  v = r_isco(a)^(-1/2)   (Kerr ISCO energy, Bardeen r_isco)

- **Accuracy for q in [2.5, 8.5]:** 7e-4 in chi_f and 4.6e-4 in M_f (absolute).
- **Coefficients** (rows i = nu power 0..3, columns j = chi1 power 0..3):

| c_ij | j=0 | j=1 | j=2 | j=3 |
|:--|--:|--:|--:|--:|
| i=0 | 3.40368 | -1.00918 | -0.29764 | -0.20977 |
| i=1 | -3.29533 | 3.87943 | 1.52577 | 1.81545 |
| i=2 | 3.15170 | -22.22368 | -5.00700 | -6.95544 |
| i=3 | -1.94152 | 62.42726 | 11.81863 | 11.65857 |

| m_ij | j=0 | j=1 | j=2 | j=3 |
|:--|--:|--:|--:|--:|
| i=0 | 0.99171 | 1.35048 | 1.97125 | -2.39922 |
| i=1 | 10.05348 | -23.85014 | -34.15642 | 48.75351 |
| i=2 | -17.28829 | 169.72016 | 230.24725 | -264.89735 |
| i=3 | 61.75778 | -347.04782 | -489.64177 | 440.78514 |

From these the model takes:
- beta_r = w220(chi1) M_f / w220(chi_f);
- gamma_NR = gamma220(chi_f) / M_f.

**1.4 Measured ppBHPT ringdown decay (pass 46).** This replaces the per-waveform slope fit over s in [15, 40] m1.

    gamma_meas = gamma220(chi1) [1 + d0 + d1 chi1 + d2 chi1^2 + d3 chi1^3 + d4 chi1^4 + nu (e0 + e1 chi1 + e2 chi1^2)]
    d = -0.0427554, -0.105900, +0.0827743, +0.463087, -0.309926;   e = +0.0332619, +0.589434, -0.0652758

- **Accuracy:** 1.1% max, 0.4% rms, against the per-case slopes of the 96 training waveforms.

**1.5 The four shape functions (pass 47b).** s is clipped to the given range, exactly as the splines were. Two forms are used:
- **alpha shapes**, sech form, 4 bumps, fitted pointwise:

      f(s) = A0 exp(min(s,0)/tau) [= A0 for s >= 0] + sum_{k=1..4} A_k sech((s - mu_k)/w_k)

- **beta shapes**, sech^2 form, 6 bumps. sech^2 = d tanh / ds. They are fitted to the value AND its cumulative integral, because beta is integrated into the time map:

      f(s) = A0 exp(min(s,0)/tau) [= A0 for s >= 0] + sum_{k=1..6} A_k sech^2((s - mu_k)/w_k)

| shape | form | s range | parameters [A0, tau, (A_k, mu_k, w_k) ...] | fit |
|:--|:--|:--|:--|:--|
| phi_B | sech^2, 6 | [-2000, 60] | +0.0604494, +1173.77, +83.1059, +14.5978, -12.6293, +1.83466, -42.2998, +56.9947, -150.006, -16.8878, +50.8677, +149.006, -16.617, +50.5639, -83.6113, +14.5727, +12.6551, +23.5673, +1189.12, +335.38 | max 0.005, cumulative 0.042 m1 |
| psi_B | sech^2, 6 | [-2000, 60] | +0.12311, +847.237, -1307.61, +1255.96, +508.821, -0.238702, -82.813, +44.3646, -1.20531, +2.37442, +25.0353, +1.38982, -4.24923, +14.8196, +1.26273, +51.915, +146.593, +406.493, +950.855, +505.951 | max 0.010, cumulative 0.028 m1 |
| phi_A | sech, 4 | [-2000, 8] | +10.6591, +268.661, -1.13138, +23.6316, +23.71, -0.673908, +7.98088, +7.53269, -0.401438, -53.4225, +37.4666, -13.4049, +225.536, +256.416 | rms 0.009, max 0.068 |
| psi_A | sech, 4 | [-2000, 60] | +0.48097, +201.105, -2211.89, +0.0655289, +12.5934, -2.41189, -7.2994, +34.2126, +4424.78, -0.301667, +12.5967, -2212.18, -0.666791, +12.5832 | rms 0.016, max 0.048 |

**Caveat on the shape forms.** They reproduce the splines faithfully, but they are fits, not physics, and two of them are poorly conditioned:
- psi_A has three nearly cancelling bumps at mu = 12.6 (amplitudes -2212, +4425, -2212);
- phi_B and psi_B each carry one very wide term that acts as a background.

A regularised refit (bounded amplitudes and widths) would give a cleaner, more readable form at the same accuracy. That is the open item for the shapes.

## 2. Scores (full-window mathcalE; median / mean / max)

| set | pass 40 (numerical inputs) | **pass 48** (closed form, 18c) | pass 40 lean | **pass 48 lean** (closed form, 16c) |
|:--|:--|:--|:--|:--|
| T96 (training) | 1.381 / 1.995 / 8.54 e-4 | 1.367 / 2.016 / 7.92 e-4 | 1.457 / 2.142 / 10.07 e-4 | 1.418 / 2.154 / 9.04 e-4 |
| H270 (validation) | 1.359 / 1.684 / 5.72 e-4 | 1.370 / 1.699 / 5.89 e-4 | 1.370 / 1.712 / 5.01 e-4 | 1.342 / 1.719 / 5.15 e-4 |
| **R100 (sealed test)** | 1.410 / 1.947 / 9.50 e-4 | **1.435 / 1.963 / 9.55 e-4** | 1.475 / 2.046 / 9.11 e-4 | **1.487 / 2.054 / 9.15 e-4** |
| **L20 (sealed test, 2.5 < q < 3)** | 7.85 / 9.93 / 28.3 e-4 | **7.75 / 9.58 / 27.0 e-4** | 7.20 / 9.20 / 26.5 e-4 | **6.73 / 8.84 / 25.1 e-4** |
| q = 2.5 max / q = 2 max | 4.61e-3 / 2.29e-2 | 4.42e-3 / 2.24e-2 | 4.37e-3 / 2.15e-2 | 4.18e-3 / 2.10e-2 |
| rows q = 3 / 5 / 8 (e-4) | 2.90 / 1.17 / 1.70 | 2.74 / 1.14 / 1.76 | 3.17 / 1.12 / 2.18 | 3.01 / 1.11 / 2.21 |
| columns chi1 = -0.6 / +0.8 (e-4) | 3.01 / 4.25 | 3.18 / 4.48 | 3.73 / 4.05 | 3.88 / 4.56 |

- **Cost of going closed form:**
  - Pass 48 vs pass 40 on R100: +1.7% median, +0.8% mean, +0.5% max.
  - Lean: +0.8% median, +0.4% mean, +0.4% max.
  - Nowhere above 2%.
  - Below q = 3, and on the training max, the closed forms are slightly better.
- **Pass 48 vs pass 48 lean:** the same trade as before. Lean is about 4% worse in range (R100 median) and 13% better below q = 3 (L20 median).
- **Gates:** unchanged from pass 40. G1, G2, G3, V, P, K and L pass; U fails at the spin edges (chi1 = -0.6 and +0.8 columns, q = 3 row).
- **Test-mass limit (gate P):** holds by construction, as in pass 39. Every surviving term carries g(nu); M_f -> 1 and chi_f -> chi1 as nu -> 0, so beta_r -> 1.

## 3. Reproduce

| step | command (in `exploration/`) | output |
|:--|:--|:--|
| closed-form constants (QNM, remnant, gamma_meas) | the fit snippets logged under EXPLORATION.md pass 46; constants in `closedform_coeffs.json`, module `closedform.py` | |
| alpha shapes | `python fit_shapes_closedform.py` | `shapes_closedform.json` |
| beta shapes (clock-preserving) | `python fit_shapes_cf2.py` (compute node) | `shapes_closedform_v2.json` |
| scores, all sets + sealed test | `python p48_closedmodel.py` | `p48/p48.json`, `t100/m48.json`, `t100/lean48.json` |
| stand-alone check | the chain comparison in EXPLORATION.md pass 48 (5 cases x {fit, full}, max 2.2e-9) | |

**Evaluating the closed-form model:**

```python
import bhpt_spin_calib_p48 as G
alpha, beta = G.alpha_beta(q, chi1, t, h, x_pp, E)        # ppBHPT grid t (m1); h, x_pp, E on the same grid
# lean variant: replace the onset s1 = t1a + t1b chi1 by s1 = s_isco + (-1 - 0.7154) u_OT (see pass 42; not yet in this file)
```
