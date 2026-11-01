# Pass 36 D1: time-dependent alpha(t), beta(t) for the spinning ppBHPT (2,2) mode

**What it is.** A map from the uncalibrated point-particle BHPT waveform (BHPTNRSur2dq1e3,
`calibrated=False`) to NR (NRHybSur3dq8) for aligned spin on the primary.
- It has 18 fitted coefficients.
- Its inspiral uses the flux-anchored form with the derived 1.5PN spin-orbit factor and the 2SF amplitude correction.
- Its merger-ringdown branch runs on the exact Kerr QNM clock, with the ringdown decay rate measured from each ppBHPT waveform.
- The two are blended in ln with a spin-dependent window.
- Four fixed shape functions add the residual late-inspiral structure.

**Status.** It is an exploration model, not installed.
- Coefficients: `exploration/p36/p36_D1.json`.
- History: EXPLORATION.md, Log rows 31–38 and the 2026-10-13/15 entries.

**Headline (full-window mathcalE; median / mean / max).**

| set | points | median | mean | max |
|:--|--:|--:|--:|--:|
| T96 (training) | 96 | 1.36e-4 | 1.99e-4 | 8.49e-4 |
| H270 (validation; used for model selection) | 270 | 1.34e-4 | 1.67e-4 | 5.69e-4 |
| **R100 (sealed test, never used for any choice)** | 100 | **1.40e-4** | **1.93e-4** | **9.4e-4** |
| L20 (sealed test, 2.5 < q < 3, extrapolated) | 20 | 7.8e-4 | 9.9e-4 | 2.8e-3 |
| q = 2.5 (8 chi1, extrapolated) | 8 | 1.6e-3 | | 4.6e-3 |
| q = 2 (8 chi1, extrapolated, BHPT input also out of domain) | 8 | 1.2e-2 | | 2.3e-2 |
| dense 20 x 20 grid (q 2..8, chi1 -0.6..0.8) | 400 | 1.96e-4 | | 2.29e-2 (q = 2) |

The comparison with Rink et al., the installed models and the split model is at the end (section 7).

---

## 1. Symbols

Units: **m1** = primary mass (the ppBHPT clock), **M** = total mass m1 + m2 (the NR clock). Every quantity is
dimensionless unless a unit is given.

| symbol | definition |
|:--|:--|
| q | mass ratio m1/m2 >= 1 |
| chi1 | dimensionless aligned spin of the primary (the secondary is non-spinning) |
| nu | symmetric mass ratio q/(1+q)^2 |
| nu5 | nu at q = 5, = 5/36 = 0.138889 (reference point of the nu weights) |
| w | nu/nu5 (= 1 at q = 5, -> 0 as q -> infinity) |
| X1 | m1/M = q/(1+q) |
| t | ppBHPT time, in m1: the native time grid of BHPTNRSur2dq1e3(q, spin1 = chi1, calibrated = False), (2,2) mode |
| h_pp(t) | that ppBHPT (2,2) mode (complex) |
| t_pk | the sample of t at which abs(h_pp) is largest (no interpolation) |
| s | t - t_pk, in m1 (time from the ppBHPT peak) |
| t_NR, h_NR | NR time (M) and (2,2) mode of NRHybSur3dq8(q, [0,0,chi1], [0,0,0]), dt = 0.1 M, f_low = 5e-3, kept on [-5000.1, +100] M (NR peak near 0) |
| x_pp(t) | ppBHPT PN parameter in m1 units: x_pp = (abs(dphi/dt)/2)^(2/3), phi = unwrapped phase of h_pp smoothed by a cubic Savitzky–Golay filter of 80 m1 span; clipped to [1e-8, 0.6] (`fit_pn_anchored_spin.get_x`) |
| x | X1^(-4/5) * min(x_pp, 0.26): the note's x (`physics_common.x_scale`; 0.26 = XCLIP) |
| E(t) | radiated energy of the ppBHPT (2,2) + (2,-2) modes per unit **total** mass, E = integral of E_dot from the first sample (E = 0 there); E_dot = (1/16 pi) sum over the two modes of abs(dh/dt)^2 with t, h first rescaled to M (factor X1), despiked only in the surrogate join window (`physics_common.energy_drive`, drive tag ENERGY_DRIVE_2D) |
| R2(x) | the 2SF / 1SF (2,2) amplitude ratio for circular Schwarzschild orbits: R2 = 2 Re[h2eff/h0], h2eff = h0 + h2 + (2/3) r0 (1 - (1/r0)/sqrt(1 - 3/r0)) dh0/dr0, x = 1/r0 (WaSABI 1PAT1, Wardell Eq. 7; data Zenodo 10.5281/zenodo.17405583). 22-point table on x in [0.04, 0.16] (`wardell_data/R2_22_from_amplitudes.json`), linear interpolation, held constant outside (R2 = 0.096 below, 0.0376 above) |
| kappa_SO(q) | derived 1.5PN spin-orbit coefficient: 1.5 (113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1); it vanishes at X1 = 1 (`fit_flux_anc_spin_with_orbit.kappa_so`) |
| omega220(chi) | complex Kerr (l, m, n) = (2, 2, 0) QNM frequency for spin chi, in units of that hole's mass (`qnm` package; for chi < 0 the counter-rotating branch, i.e. the (2,-2,0) mode at -chi) |
| w220(chi) | abs(Re omega220(chi)) |
| Q220(chi) | abs(Re omega220) / (2 abs(Im omega220)); so w220/(2 Q220) = abs(Im omega220) = the decay rate |
| M_f, chi_f | remnant mass (units of M) and spin from NRSur3dq8Remnant (surfinBH) at (q, chi1, chi2 = 0) |
| beta_r | ringdown clock ratio = w220(chi1) M_f / w220(chi_f). The ppBHPT ringdown is that of the primary (spin chi1, frequency per m1); NR's is the remnant's (per M) (`fit_pn_anchored_spin.beta_r_phys`) |
| gamma_NR | NR (2,2,0) decay rate per M: w220(chi_f) / (2 Q220(chi_f) M_f) |
| gamma_meas | **measured** ppBHPT ringdown decay rate per m1: minus the least-squares slope of ln abs(h_pp) against s over the samples with 15 <= s <= 40 m1 (per case, from the input waveform; pass 36). Means over q: 0.0765 / 0.0851 / 0.0834 per m1 at chi1 = -0.6 / 0 / +0.8 (spread over q <= 0.0011), vs Kerr 0.0889 / 0.0890 / 0.0756 |
| kappa_RD | gamma_meas / beta_r - gamma_NR |
| S_RING | 15 m1, start of the ringdown slope term |
| smooth5(u) | u^3 (10 - 15 u + 6 u^2) with u clipped to [0, 1] (C2 smoothstep) |
| phi_B(s), phi_A(s) | fixed shape functions from pass 12 (section 3) |
| psi_B(s), psi_A(s) | fixed shape functions from pass 17 (section 3) |
| T_ANCHOR | -100 m1 on the ppBHPT clock t: the integration constant of the time map |
| t0 | per-case nuisance: the NR time that ppBHPT time T_ANCHOR maps to (fitted for each waveform, not a model coefficient) |
| phi0 | per-case nuisance: constant phase, solved analytically |

## 2. The model

### 2.1 Waveform map

    tau(t)        = t0 + integral from T_ANCHOR to t of beta(t') dt'          (ppBHPT time in m1 -> NR time in M)
    h_model(tau(t)) = exp(i phi0) alpha(t) h_pp(t)

On the NR grid this is evaluated as h_model(t_NR) = exp(i phi0) alpha(tau^-1(t_NR)) h_pp(tau^-1(t_NR)), with linear
interpolation of alpha and h_pp against tau.

### 2.2 Inspiral branch (subscript I)

    beta_I  = [ X1^(6/5) (1 + b nu + b2 nu^2) + P E(t) ] * (1 + kappa_SO chi1 x^(3/2))
    P       = P0 + P1 nu + s2 nu5 chi1^2 + s3 (nu5/nu) chi1
    alpha_I = X1^(6/5) (1 + nu R2(x)/2)

- **Spin terms in P.** The original form is nu (s2 chi1^2) + s3 chi1. Each spin term carries an extra factor nu5/nu (passes 31–32; the fitted exponent was 1.15 and was rounded to 1).
- **alpha_I** has no fitted coefficient. The 2SF amplitude replaced the fitted (1 + c0 nu) of the earlier forms (pass 23).

### 2.3 Merger-ringdown branch (subscript M)

    ln beta_M  = ln beta_r + r1 chi1 + r3 chi1^2
    ln alpha_M = (6/5) ln X1 + w l0 + kappa_RD beta_r max(s - S_RING, 0)

- **r1, r3.** In the form w (r1 chi1 + r3 chi1^2) the nu5/nu factor cancels w, so these offsets are independent of q.
- **Ringdown slope.** It gives d ln alpha_M / ds = gamma_meas - gamma_NR beta_r for s > 15 m1. alpha turns the ppBHPT decay (per m1) into NR's decay mapped onto the ppBHPT clock.

### 2.4 Blend (in ln, on the ppBHPT clock)

    s1 = t1a + t1b chi1,   s2 = 0
    W(s) = smooth5( (s - s1) / (s2 - s1) )
    ln beta  = (1 - W) ln beta_I  + W ln beta_M
    ln alpha = (1 - W) ln alpha_I + W ln alpha_M

**Validity.** The form requires -150 <= s1 <= 30 and s1 + 3 <= s2 <= 150, otherwise the case is invalid. With the fitted t1a, t1b this means
**chi1 >= -0.6000** (s1 = -150.000 exactly at chi1 = -0.6). The model is undefined for chi1 < -0.6.

### 2.5 Shape terms (added after the blend)

    ln beta  += w (pa0 + pa2 chi1^2) phi_B(s)  +  eb1 chi1 psi_B(s)
    ln alpha += (pb0 + pb1 chi1 + pb2 chi1^2) phi_A(s)  +  w ea0 psi_A(s)

The eb1 and pb terms carry the factor w (nu5/nu) = 1, so they are independent of q. The pa and ea0 terms keep w.

## 3. Fixed shape functions

All four are natural cubic splines in s (m1), with s clipped to the knot range given (held constant outside it).

**phi_B** (pass 12, first principal shape of the per-case ln beta corrections on the V3 model; s clipped to [-2000, 60]):

| s | -2000 | -800 | -400 | -200 | -120 | -80 | -50 | -30 | -18 | -9 | 0 | 8 | 18 | 35 | 60 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| phi_B | 0.01108 | 0.03140 | 0.05002 | 0.07823 | 0.11454 | 0.12928 | 0.18307 | 0.43840 | 0.59132 | 0.54936 | 0.27530 | -0.05663 | -0.06540 | 0 | 0 |

**phi_A** (pass 12, same for ln alpha; s clipped to [-2000, 8], so phi_A = -0.85615 for s > 8):

| s | -2000 | -800 | -400 | -200 | -120 | -80 | -50 | -30 | -18 | -9 | 0 | 8 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| phi_A | -0.00120 | 0.05375 | 0.07829 | 0.12763 | 0.15042 | 0.11743 | 0.14415 | 0.24697 | 0.26004 | 0.14680 | -0.18531 | -0.85615 |

**psi_B** (pass 17, leading PCA component of the per-case ln beta corrections on the V6r model; zeros appended at 35, 60;
s clipped to [-2000, 60]):

| s | -2000 | -800 | -400 | -200 | -120 | -80 | -50 | -30 | -18 | -9 | 0 | 8 | 18 | 35 | 60 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| psi_B | 0.01076 | 0.03270 | 0.04217 | 0.08816 | 0.08739 | 0.06704 | 0.18721 | 0.20788 | 0.37499 | 0.67969 | 0.50241 | -0.00737 | -0.20931 | 0 | 0 |

**psi_A** (pass 17, same for ln alpha; zeros appended at 18, 35, 60; s clipped to [-2000, 60]):

| s | -2000 | -800 | -400 | -200 | -120 | -80 | -50 | -30 | -18 | -9 | 0 | 8 | 18 | 35 | 60 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| psi_A | -0.03162 | 0.05322 | 0.08233 | 0.15297 | 0.10484 | -0.19162 | -0.46780 | -0.38200 | -0.16551 | 0.15610 | 0.69209 | 0.15087 | 0 | 0 | 0 |

Sources: `exploration/p12/p12_shapes.json` (phi), `exploration/p17/p17_p15.json` pca dlnB / dlnA comps[0] (psi).

## 4. Coefficients (18 fitted)

| group | coefficient | value | enters |
|:--|:--|--:|:--|
| inspiral beta level | b | -0.023608 | beta_I |
| | b2 | 0.215600 | beta_I |
| energy coupling | P0 | -4.204361 | P |
| | P1 | 30.61615 | P |
| | s2 | -18.455319 | P (as s2 nu5 chi1^2) |
| | s3 | 2.545433 | P (as s3 (nu5/nu) chi1) |
| blend onset | t1a | -102.52678 m1 | s1 |
| | t1b | 79.122034 m1 | s1 |
| merger offsets | r1 | 0.081670 | ln beta_M |
| | r3 | -0.111895 | ln beta_M |
| | l0 | -0.114666 | ln alpha_M (x w) |
| beta shapes | pa0 | -0.097473 | phi_B (x w) |
| | pa2 | -0.368324 | phi_B (x w) |
| | eb1 | 0.097705 | psi_B |
| alpha shapes | pb0 | 0.169675 | phi_A |
| | pb1 | -0.273000 | phi_A |
| | pb2 | 0.382267 | phi_A |
| | ea0 | -0.066257 | psi_A (x w) |

**Fixed or derived (not counted).**
- s2 (blend end) = 0 and S_RING = 15 m1.
- XCLIP = 0.26 and T_ANCHOR = -100 m1.
- The fit window [15, 40] m1 for gamma_meas.
- All nu exponents = 1 (pass 32 rounded the fitted 1.15 / 1.28 / 0.57 to 1 at < 1% cost).
- X1^(6/5), R2, kappa_SO, beta_r, gamma_NR and gamma_meas.

**Zero in this form.** These were pruned, or never needed:
- c0, c1 (alpha_I level), A0, A1 (flux coupling of alpha_I: the instantaneous flux F does not enter);
- the nu chi1 term and the chi1^2 term of P, and b3 (nu chi1^2 in the beta_I level);
- the blend-end slope (s2 = 0 for every chi1);
- r0, r2, l1, l2, l3 (merger offsets);
- pa1, pa3 (phi_B coefficients);
- eb0, eb2 (psi_B coefficients);
- ea1, ea2 (psi_A coefficients).

The json also stores kab = -0.93 with tie = 0. kab is inactive.

## 5. How it is scored and fitted

**Error.**

    mathcalE = (n1 + n2 - 2 abs(Z)) / (2 n1),   n1 = sum abs(h_NR)^2,  n2 = sum abs(g)^2,  Z = sum h_NR conj(g)

- g = alpha(tau^-1) h_pp(tau^-1) on the NR samples inside both the NR window [-5000.1, +100] M and the mapped ppBHPT span.
- It equals || h_NR - exp(i phi0) g ||^2 / (2 ||h_NR||^2) at the optimal phi0.
- A case is invalid (error 50) if that overlap covers less than 88% of the NR samples.
- A case scores 1 if tau is not monotone, if alpha_I or beta_I <= 0 anywhere, or if s1 is outside its range (section 2.4). The blended alpha and beta are exponentials, so they are always positive.
- Code: `spin_dep/fit_pn_anchored_spin._err_at_t0`, imported, not re-implemented.

**t0 search.**
1. On the fit-stride arrays, scan t0 over 126 values from -300 to 200 M.
2. Refine with a bounded search over +-6 M.
3. Repeat at full resolution, +-6 M around that point.

**Fit stride.**
- Every 6th ppBHPT sample (SRC_STRIDE) and every 10th NR sample (NR_STRIDE; 1 M).
- Every number in this md is at full resolution.

**Training.**
- Powell minimisation of the mean fit-stride mathcalE over T96, on the free coefficients only.
- T96 = q in {3, 3.256, 3.513, 3.897, 4.2, 4.6, 5, 5.436, 5.949, 6.590, 7.231, 8} x chi1 in {-0.6, -0.4, ..., +0.8}.
- Started from the pass-33 A solution, with a 38-min budget (job 3498853).

**Point sets.**
- **H270:** q in linspace(2, 8, 20) with 3 < q < 8, x chi1 in linspace(-0.6, 0.8, 20) without the two ends. It was used to choose between models, so it is validation, not test.
- **R100 / L20:** drawn once at random and sealed (`exploration/t100_set.py`; `test_guard.py` refuses them in any fit). They were scored once, after the model was frozen.

## 6. Gates (EXPLORATION.md section 1.5)

| gate | condition | value | verdict |
|:--|:--|:--|:--|
| G1 | beats the installed models | mean 1.99e-4 < 1.09e-3, median 1.36e-4 < 8.65e-4, max 8.5e-4 < 6.5e-3, H270 1.34e-4 < 7.1e-4 | pass |
| G2 | median <= 5e-4, max <= 2.5e-3, H270 <= 4e-4 | | pass |
| G3 | median <= 2.5e-4, max <= 1e-3 | | pass |
| U | every row / column median <= 2 x 1.36e-4 = 2.72e-4; log10 spread (std of log10 E) <= 0.20 | columns chi1 = -0.6 3.00e-4, +0.8 4.33e-4; row q = 3 2.87e-4; spread 0.28 | **fail** |
| V | H270 median / T96 median <= 1.1 | 0.98 (R100 / T96 = 1.03) | pass |
| P | positivity and monotone tau | every scored case valid | pass |
| P | spin terms vanish at chi1 = 0 | s2, s3, t1b, r1, r3, pb1, pb2, eb1 all multiply chi1 or chi1^2 | pass |
| P | **alpha, beta -> 1 as nu -> 0** | not satisfied (section 8.3) | **fail** |
| K | <= 20 coefficients, each worth >= 5% | 18; cheapest single drop costs 11.6% | pass |
| L | q = 2.5 below 1e-2 at every chi1 | max 4.6e-3 | pass |

**Correction to check-in 4.** EXPLORATION.md check-in 4 listed P as passing. The nu -> 0 clause has failed since pass 32, which made the merger spin terms q-flat. Pass 36 added a second failure: the measured ringdown slope does not vanish as nu -> 0.

Row medians (q = 3 ... 8, 1e-4): 2.87, 1.83, 1.63, 1.45, 1.37, 1.25, 1.15, 1.09, 1.17, 1.28, 1.39, 1.70.

Column medians (chi1 = -0.6 ... +0.8, 1e-4): 3.00, 1.01, 1.12, 1.43, 1.22, 1.22, 0.75, 4.33.

## 7. Comparison (same scorer, same windows)

| model | coef | T96 median / mean / max | H270 median | R100 (test) median / mean / max | L20 (test) median | q = 2.5 / q = 2 median |
|:--|--:|:--|--:|:--|--:|:--|
| Rink et al. 2024 (published calibrated BHPTNRSur2dq1e3) | 12 | 2.35e-3 / 3.47e-3 / 1.37e-2 | 1.90e-3 | 1.87e-3 / 2.59e-3 / 9.7e-3 | 7.8e-3 | 1.3e-2 / 1.1e-1 |
| pn_anchored_spin (installed) | 13 | 8.65e-4 / 1.17e-3 / 6.58e-3 | 7.10e-4 | 6.94e-4 / 8.93e-4 / 2.8e-3 | 2.3e-3 | 4.1e-3 / 1.6e-2 |
| flux_anc_spin_with_orbit (installed) | 9 | 8.86e-4 / 1.09e-3 / 6.53e-3 | 7.20e-4 | 7.88e-4 / 8.61e-4 / 2.3e-3 | 2.2e-3 | 4.4e-3 / 1.9e-2 |
| split A (slim, window from T = -2; `spin_dep_split/scaling_split_A_slim_T-2.md`) | 14 | 4.64e-4 / 8.54e-4 / 6.51e-3 | 3.98e-4 | 3.80e-4 / 5.72e-4 / 2.3e-3 | 1.8e-3 | 4.8e-3 / 2.1e-2 |
| split B (b = 0, window from T = -3; `spin_dep_split/scaling_split_B_b0_T-3.md`) | 13 + T0 | 6.21e-4 / 8.49e-4 / 5.60e-3 | 4.08e-4 | 4.39e-4 / 5.82e-4 / 2.3e-3 | 3.3e-3 | 6.5e-3 / 2.7e-2 |
| **pass 36 D1** | **18** | **1.36e-4 / 1.99e-4 / 8.49e-4** | **1.34e-4** | **1.40e-4 / 1.93e-4 / 9.4e-4** | **7.8e-4** | **1.6e-3 / 1.2e-2** |

Before each model touched the test set, it reproduced its stored T96 scores at ratio 1.0000 (`exploration/t100/`).

## 8. Anchors (the physics account, EXPLORATION.md section 8)

The columns are:
- **argument:** where the piece comes from;
- **cost of removing it:** the measured change in the T96 mean when the piece is removed and the rest refitted;
- **base:** the model the removal cost was measured on.

### 8.1 Structural pieces

| piece | argument | cost of removing it | base |
|:--|:--|:--|:--|
| X1^(6/5) prefactor on alpha_I, beta_I | derived: leading-order (chirp-mass) scaling between the m1 and M clocks (the pn_anchored note) | not re-measured. In the non-spinning line, imposing it took gwr_energy_stiff (9c) to gwr_energy_anchored (6c) and improved q = 2 by 2.5x | nonspinning |
| P E(t) in beta_I | heuristic, energy-loss drive: beta drifts with radiated energy (CLAUDE.md "key physics insight") | P0, P1 single drops +544% / +1690% | pass 34 (18c) |
| kappa_SO chi1 x^(3/2) factor on beta_I | derived: 1.5PN spin-orbit phasing that the test-mass-on-Kerr limit lacks; no free parameter | flux_anc_spin_with_orbit vs its parent, both 9c: -32% median, -54% max on T96 | installed line |
| (1 + nu R2/2) on alpha_I | derived: 2SF (1PA) amplitude correction, Schwarzschild, circular (Wardell / WaSABI) | +3% mean vs a free (1 + c0 nu); dropping both costs about 11% | pass 23 |
| merger branch on beta_r | derived: exact Kerr QNM clock ratio, primary vs remnant | not ablated on this base (the branch is the model's backbone since pass 6) | |
| kappa_RD slope, gamma_NR part | derived: QNM decay of the remnant | | |
| kappa_RD slope, gamma_meas part | measured from the input waveform; 0 coefficients | +4% mean, +10% on the chi1 = -0.6 column, if replaced by the Kerr gamma of the primary | pass 33 A -> 36 |
| spin-dependent blend onset t1b | heuristic: the plunge starts earlier for retrograde spin | +39% | pass 34 |
| onset from the Kerr ISCO instead of fitted (t1a, t1b) | derived alternative | +6% with t1b dropped (I1); +33% with no fitted onset (I0) | passes 26, 33 B |
| nu5/nu on the spin terms (s2, s3, r1, r3, pb0, pb1, pb2, eb1) | empirical: pass 30 per-row refits drift about (nu5/nu)^1; fitted exponents 1.15 / 1.28 | pass 25 (20c, without it) vs pass 32 (19c, with it): +16% mean, +26% median, q = 8 row +38% | pass 25 -> 32 |
| shape terms phi_B, phi_A, psi_B, psi_A | data-driven: principal shapes of per-case spline corrections | single drops: pa2 +47%, pb1 +31%, pb0 +30%, pa0 +22%, ea0 +14%, eb1 +13%, pb2 +12% | pass 34 |

### 8.2 Single-coefficient removal costs

Each coefficient is set to 0 and the other 17 are refit for 2.2 min; the cost is the change in the fit-stride T96 mean, on the pass-33 A base (identical form, Kerr gamma):

| coefficient | pb2 | eb1 | ea0 | pa0 | pb0 | pb1 | t1b | pa2 | s2 | r3 | r1 | s3 | l0 | b2 | b | P0 | P1 | t1a |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cost | +12% | +13% | +14% | +22% | +30% | +31% | +39% | +47% | +81% | +97% | +105% | +187% | +190% | +220% | +270% | +544% | +1693% | invalid |

No coefficient drops for less than 3% (pass 34), so the 18c form is pruned as far as it goes.

### 8.3 Predictions and limits

**nu -> 0 (test-mass limit)**

| term | limit as nu -> 0 | -> 1? |
|:--|:--|:--|
| X1^(6/5), beta_r, and the w-weighted terms (l0, pa0, pa2, ea0) | 1, or exp(0) | yes |
| kappa_SO | 1.5 (113/12 - 19/6 - 25/4) = 0 | yes |
| P E | E per M scales as nu, so P0 E and P1 nu E -> 0. But s3 (nu5/nu) chi1 E -> s3 nu5 chi1 (E/nu), which is finite: about 0.35 chi1 (E/nu) | **no** (chi1 ≠ 0) |
| beta_M | beta_r exp(r1 chi1 + r3 chi1^2) -> exp(r1 chi1 + r3 chi1^2): 0.915 at chi1 = -0.6, 0.994 at +0.8 | **no** (chi1 ≠ 0) |
| eb1 chi1 psi_B and (pb0 + pb1 chi1 + pb2 chi1^2) phi_A | independent of q | **no**: pb0 phi_A survives even at chi1 = 0 (ln alpha = 0.17 x (-0.856) = -0.145 for s > 8) |
| kappa_RD | -> gamma_meas - gamma_Kerr(chi1) ≠ 0 (at chi1 = 0: 0.0851 vs 0.0890 per m1) | **no** |

The nu5/nu scaling (pass 32) was chosen for accuracy over q in [3, 8], and it traded away the test-mass limit that pass 14 imposed.
- **Restoring it** needs the spin terms to go to zero as nu -> 0, for example (nu5/nu) -> (nu5/nu)(1 - exp(-nu/nu_c)), and gamma_meas compared against the primary's Kerr gamma. This is untested.
- **Practical reach:** none of this matters for q <= 8. It makes the model unsafe to extrapolate to large q.

**How large the violation is (2026-10-15).**
- Method: the limiting map (w = 0, X1 = 1, beta_r = 1, kappa_SO = 0, gamma_NR = gamma_Kerr(chi1)) applied to the cached q = 8 ppBHPT waveforms. E/nu is taken from q = 8 as the proxy for the energy per unit reduced mass. The result is scored against the unmapped waveform, which is the correct answer at nu -> 0, with the same t0 / phi0 optimisation on [-5000, +100].
- Leading term at chi1 > 0: the inspiral s3 term.

| chi1 | max abs(ln beta), s < -200 | ln alpha at s = 0 / +40 | clock drift at s = -4000 | mathcalE: alpha only / beta only / both |
|:--|--:|:--|--:|:--|
| -0.6 | 0.009 | -0.09 / -0.68 | +13 M | 1.5e-3 / 2.1e-3 / 2.3e-3 |
| 0 | 0 | -0.03 / -0.26 | 0 | 2.3e-4 / 0 / 2.3e-4 |
| +0.6 | 0.014 | -0.03 / -0.08 | -20 M | 1.5e-4 / 4.9e-3 / 5.3e-3 |
| +0.8 | 0.021 | -0.04 / -0.03 | -29 M | 2.6e-4 / 1.0e-2 / 1.1e-2 |

So in the limit the model is off by 2e-4 (chi1 = 0) to 1e-2 (chi1 = +0.8). That is 2 to 80 times its in-range median of 1.4e-4.

**q = 10.** Not evaluated.

**Edges of the training box.**
- chi1 = +0.8 is the worst column at every q. There the calibrated peak comes 4–8 M early (`plots_p36/p36_peak_errors_grid`), and the per-case floor is 0.7–1.0e-5, 20–90x below the model (pass 35). The deficit is in beta's spin dependence near merger, which acts as a whole (pass 37).
- chi1 = -0.6 is the second worst. 60–70% of its error is in the ringdown (0 to +100 M; `p36_error_by_band_q4`), and the ringdown phase runs away after +20 M (`p36_residual_vs_time_q4`). This is the known retrograde-ringdown issue, low priority (user).
- chi1 < -0.6 is outside the model (section 2.4).

**q < 3 (extrapolated coefficients, t0 refit only).**
- The error crosses 1e-3 near q ≈ 2.8 and 1e-2 near q ≈ 2.1 (dense grid).
- q = 2.5 passes gate L (max 4.6e-3).
- Below q = 2.5 the BHPT input is also outside its trained range.

**F2 against Wardell / Warburton at chi1 = 0.** Not done.

## 9. Reproduce

All runs use the `ut_claude` env on Lonestar6 (account PHY26029, development queue). Slurm scripts are in `ls6/exploration/`.

| step | command (in `exploration/`) | output |
|:--|:--|:--|
| fit (from `p33/p33_A.json`) | `python p36_rdslope.py D1 38` | `p36/p36_D1.json` (theta, free, T96 per case, H270 / low-q summaries, gamma_meas per case) |
| sealed test scores | `python t100_score.py p36` | `t100/p36.json` |
| plots (waveforms, alpha / beta, E vs chi1, histogram, map) | `python plot_p36.py` | `plots_p36/` |
| dense heatmap | `python p36_dense.py [--plot-only]` | `plots_p36/p36_mismatch_{grid,heatmap}_dense.*` |
| Fig. 10, bands, residual, peaks | `python p36_diag.py --compute --plot` | `plots_p36/p36_{fig10_tend_q4, error_by_band_q4, residual_vs_time_q4, peak_errors_grid}.*` |
| common-axis histograms | `python plot_hist_common.py` | the three hist pdf / png |

**Evaluating alpha, beta for one case (q, chi1):**

```python
import sys, json, numpy as np
sys.argv = ["x", "D1"]
import p36_rdslope as D                     # loads the chain p31 -> p25 -> p23 -> p18 -> p14 -> p6
P6 = D.P6; P6.get = D.get                   # measured ringdown decay
th = np.array(json.load(open("p36/p36_D1.json"))["theta"])
alpha, beta = P6.ab(th, (q, chi1), "full")  # arrays on P6.get((q, chi1))["case"]["t_bhpt"]
E = P6._full((list(th), (q, chi1)))         # full-window mathcalE, t0 and phi0 optimised
```

The theta vector has 44 entries. Their names are `p32/p32_G2.json["names"]`, and the 18 free indices are `p36_D1.json["free"]`.

The model is spread over the import chain listed above. There is no stand-alone generator yet. Writing one, and checking it against these numbers, is the step before installing.
