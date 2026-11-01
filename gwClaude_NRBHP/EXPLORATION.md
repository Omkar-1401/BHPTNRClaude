# EXPLORATION — autonomous search for spin-dependent alpha(t), beta(t)

> **STATUS (2026-10-22): leader pass 48 = pass 40 (18c) with EVERY numerical input in closed form** (R2, Kerr QNM, remnant, ringdown decay, shapes; constants fitted offline to theory / reference data). Sealed test R100 1.44e-4 / 1.96e-4 / 9.5e-4 (median / mean / max). Lean variant pass 48 lean (16c, OT onset) R100 1.49e-4, L20 median 6.7e-4. Stand-alone numpy-only generator `bhpt_spin_calib_p48.py`. Model docs: `scaling_p48.md` (closed forms), `scaling_p40.md` (equations).  Gates: all pass except U.  See the Log.
> meanwhile in other sessions, not part of this loop: `spin_dep_split/` (ISCO split + Ori–Thorne join;
> slim split variant, 14c) and `flux_anc_spin_with_orbit` (derived 1.5PN spin-orbit term in beta,
> installed 2026-10-05).  User: explore independently; compare with the split model if a similar
> form comes up.

**Goal.**  Find a map h_NR(tau) ~ alpha(t) e^{i phi0} h_pp(t), with tau = t0 + ∫ beta dt, from the
ppBHPT (2,2) mode (BHPTNRSur2dq1e3, uncalibrated) to NR (NRHybSur3dq8, chi2 = 0).  The map should
1. be clearly better than every existing spin model,
2. be **uniformly** accurate across q in [3, 8] and chi1 in [-0.6, +0.8],
3. use few coefficients and derived anchors wherever possible, and
4. admit a physics explanation of each structural piece afterwards.

**How.**  The search is a loop.  Each pass guesses a form, screens it cheaply, fits it, checks a few
chosen diagnostics, and bases the next guess on where it failed.  Every pass, including the failures,
goes in the **Log** at the end.  Nothing here changes an installed model.

---

## 1. Metrics

### 1.1 Error measure (the repo's, so numbers stay comparable)

    mathcalE = || h_NR - e^{i phi0} alpha h_pp(tau) ||^2 / (2 || h_NR ||^2)

- (2,2) mode only.
- NR window [-5000.1, +100] M.
- phi0 is analytic; t0 is optimised per case (grid ±200 M, then a bounded refine).
- Computed by `spin_dep/fit_pn_anchored_spin._err_at_t0`, imported, never re-implemented.
- Derivation and meaning: `regression_methodology.md`.

For orientation only (flat noise, mathcalE < 1/(2 SNR^2)):

| mathcalE | SNR |
|:--|:--|
| 1e-3 | about 22 |
| 5e-4 | about 32 |
| 2.5e-4 | about 45 |

### 1.2 Split error: inspiral and merger-ringdown (user, 2026-10-02)

Split the window at t_s (NR time).  The default is t_s = the ISCO crossing of the mapped ppBHPT
orbit (B1, §4); -100 M and -50 M are the fallbacks.

```
E_I  = |h - e^{i phi0} g|^2 on [-5000.1, t_s] / (2 |h|^2 on the same span)
E_MR = |h - e^{i phi0} g|^2 on [t_s, +100]    / (2 |h|^2 on the same span)
```

**Three alignments:**

```
(a) t0, phi0 from the full window   ->  E = w_I E_I + w_MR E_MR exactly (w = share of |h|^2); where E comes from
(b) t0, phi0 from E_I alone         ->  E_MR includes the phase carried into merger
(c) each part aligned separately    ->  (b) - (c) = the inspiral-to-merger hand-off error
```

**Uses:**
- reported every pass;
- a fitting direction: inspiral coefficients on E_I, then merger coefficients on E_MR with the
  inspiral ones held, or a weighted E_I + lambda E_MR scanned over lambda.

Gates stay on the full-window E.

### 1.3 Point sets

| name | points | use |
|:--|:--|:--|
| **S18** | q in {3, 5, 8} x chi1 in {-0.6, -0.2, 0, +0.4, +0.6, +0.8} | screen: new form and leader, both refitted on S18 |
| **T96** | 12 q (3..8) x 8 chi1 (-0.6..+0.8) | full fit, only after a screen passes |
| **H270** | 270 cached interior points | validation (interpolation only), for G1 candidates |
| edges | a few points at q = 8.5..10 and chi1 = ±0.8 | smoothness only, not scored |

### 1.4 Baselines (T96 numbers re-read from the coeffs JSONs 2026-10-06)

| model | coef | T96 mean | T96 median | T96 max | H270 median | log10 spread |
|:--|--:|--:|--:|--:|--:|--:|
| pn_anchored_spin (**leader**) | 13 | 1.172e-3 | 8.65e-4 | 6.58e-3 | 7.10e-4 | 0.30 |
| pn_anc_spin_physical | 11 | 1.143e-3 | 9.64e-4 | 6.25e-3 | 7.62e-4 | 0.26 |
| fluxanchored_spin | 9 | 2.110e-3 | 1.30e-3 | 1.40e-2 | 1.00e-3 | 0.35 |
| flux_anc_spin_with_orbit (installed 2026-10-05) | 9 | 1.092e-3 | 8.86e-4 | 6.53e-3 | 7.20e-4 | |
| *variant, not installed:* fluxanchored_spin_split slim (`spin_dep_split/`) | 14 | 9.44e-4 | 5.42e-4 | 6.9e-3 | 4.88e-4 | |
| *nonspinning reference (1d, chi1 = 0):* pn_anchored | 9 | | 4.88e-4 | | | |

The external competitor, the published calibrated BHPTNRSur2dq1e3 (`calibrated=True`), was scored in the
side line: T96 median 2.35e-3 / mean 3.47e-3, H270 median 1.90e-3 (`spin_dep/scaling_flux_anc_spin_with_orbit.md`).

### 1.5 Gates

| gate | condition |
|:--|:--|
| **G1** beats the installed models | T96 mean < 1.09e-3 and median < 8.65e-4 (best installed of each), max ≤ 6.5e-3, H270 median < 7.1e-4 |
| **G2** decent | T96 median ≤ 5e-4, max ≤ 2.5e-3, H270 median ≤ 4e-4 |
| **G3** stretch | T96 median ≤ 2.5e-4, max ≤ 1e-3 |
| **U** uniform | every fixed-q and fixed-chi1 median ≤ 2x the overall median; log10 spread ≤ 0.20 |
| **L** low q (user, 2026-10-07) | q = 2.5: mathcalE < 1e-2 at every chi1 in the grid (extrapolated coefficients, t0 refit only); q = 2 reported alongside |
| **V** generalises | H270 median / T96 median ≤ 1.1 (current models: 0.8) |
| **P** physical | beta > 0, tau monotone, alpha > 0; alpha, beta -> 1 as nu -> 0; spin terms vanish at chi1 = 0; nothing tied to the join window [-9, -5] m1 |
| **K** parsimony | each added coefficient buys ≥ 5% on the T96 mean without worsening H270; **the final model ≤ 20 coefficients**; exploration may go above and prune down at the end (user, 2026-10-07) |

- **Floor about 1e-4.**  The 1d and 2d surrogates differ by 0.9-1.2e-4 at chi1 = 0.
- **Done** = G2 + U + V + P + K + L; then a summary for review.

---

## 2. Already known (do not re-learn)

- **alpha:** its inspiral spin dependence is consistent with zero (`spin_extension_plan.md` §7).
- **beta_r:** the exact ringdown clock is beta_r = w220(chi1) M_f / w220(chi_f).  The flux and energy
  models lack it, and their worst cells are chi1 = +0.8 and the retrograde ringdown.
- **The ~10% merger bump in beta** (-50..+10 M) is not a remnant effect (`angular_momentum_flux_failure.md`).
- **J or Jdot as a beta drive** is worse than E(t).
- **Short projection windows mislead.**  Always confirm with the full-window mismatch.
- **Spin and the merger location.**  The merger location in a flux coordinate moves with spin
  (pn_anc_spin_physical: p00 0.89 -> 4.08).  The user asked for no chi1 term in pn_anchored_spin's switch.
- **Retrograde ringdown** (chi1 = -0.6): not one clean frequency; "measured beta" is meaningless there.
  The post-peak spike in the installed model there is low priority (user).
- **Inputs:** q = 2 is extrapolation, so it is never used to validate.  Edot spikes in the 2d join
  window and must be despiked.  The PN x is X1^(-4/5) x_pp.
- **2SF, nonspinning** (`wardell_ideation.md`, `wardell_comparison.md`):
  - the ansatz implies F_NR = (alpha^2/beta^2) F_pp in the adiabatic band;
  - pn_anchored's full-window x^2 term leaks (its implied F2 is 2x off by x ~ 0.11);
  - a drift that survives as nu -> 0 is incompatible with 1PA.

---

## 3. Working method

- **One script per idea:** `exploration/<pass>_<name>.py`.  It imports
  `fit_pn_anchored_spin.build_case` (and `flux_loss_cache` when E is needed) and `_err_at_t0`, and
  writes one small JSON.
- **New drives:** quantities computable from the cached h_pp (frequency, amplitude, phase, Kerr
  ISCO quantities) are computed on the fly.
- **Measure before fitting:** the phase-matched alpha(t), beta(t) (`spin_measure_results/`, 30
  points; extended to S18 or T96 when needed) show whether a coordinate collapses the (q, chi1)
  family before any mismatch fit is paid for.
- **Representation floor:** a free per-case fit, on S18 only and only when needed, separates "wrong
  form" from "wrong regression".  The fig10 study's per-case fits at q = 4 are a ready-made version.
- **Reproduction gate:** the first script rescores the installed pn_anchored_spin on S18 and must
  give a ratio of 1.0000.
- **Common tools:**
  - blend log alpha and log beta (positivity by construction);
  - C1 smoothstep windows with compact support;
  - monotone bases where needed.

---

## 4. Ideas

Sources were read from the arXiv LaTeX.  The short notes in §4.5 give the equations used.

### 4.1 Transition structure (Ori & Thorne 2000; Küchler et al. 2024, 2025) — the core of the plan

**Key fact (Küchler et al. 2024, §7.1).**  BHPTNRSur's transition is the Apte–Hughes "generalized
Ori–Thorne" model.  That is the leading transition order (0PLT), with sharp switches at t_i and t_f
smoothed ad hoc.  So through the transition our input is a 0PLT evolution with eps_pp = 1/q on the m1
clock, while NR carries all orders with nu on the M clock.  That fixes what alpha and beta must supply
there.  Still to check: that BHPTNRSur2dq1e3 uses the same transition in Kerr (Islam+ 2407.18319).

| id | idea | falsified if |
|:--|:--|:--|
| **B1** | **t_isco as the switch time** (user, 2026-10-02).  t_isco is when the ppBHPT orbital frequency omega_22/2 reaches the Kerr ISCO value M Omega_isco(chi1) on the m1 clock.  Only the ppBHPT ISCO is needed, and it is unambiguous.  It depends on q and chi1 with nothing fitted.  The switch is monotone in time, so it cannot fall back after the peak.  It also defines the split point t_s (§1.2). | Pass 0: t_isco for the 96 cases against where the fitted switch of the leader sits and where the measured beta turns over.  Falsified if neither sits at a consistent offset from t_isco |
| **B2** | **Scaled-frequency coordinate and windows.**  u_T = (omega_22/2 − Omega_isco(chi1)) / eps_pp^(2/5), with eps_pp = 1/q.  Equivalently, the transition clock T ∝ eps_pp^(1/5)(t − t_isco).  The windows are C1 smoothsteps in u_T (inspiral / transition / plunge-merger-ringdown), applied to log alpha and log beta.  Width ∝ eps^(2/5); the centre offset from the ISCO is one spin-independent constant, or follows the critical-frequency scaling Omega_* − c eps^(2/7).  Inside the transition, log alpha is linear in u_T (the transition amplitude is linear in Omega − Omega_*).  The outer pieces take the matched asymptotic shapes: a chirp ∝ 1/(Omega_* − Omega) before the ISCO and ∝ (Omega − Omega_*)^(3/2) after. | The measured alpha, beta are not smooth, low-order functions of u_T across q and chi1 |
| **B3** | **Transition beta from two Painlevé evolutions.**  Derive the phase-matched map between the ppBHPT's 0PLT evolution (lambda = q^(-1/5), m1 clock) and the binary's (nu^(1/5), M clock), plus the 2PLT correction, which is computable from 0PA fluxes.  This gives beta in the transition with no free parameter at leading order.  It needs a careful derivation first: beta is set by equal *phase*, not equal frequency. | The derived beta misses the measured one by more than the O(nu^(2/5)) corrections allow |
| **B4** | **ISCO anchor for beta.**  In the transition, beta ≈ Omega_isco,pp(chi1) / Omega_isco,binary, a third anchor between beta_LO and beta_r.  The binary ISCO is a genuine choice: the self-force corrected ISCO, the EOB ISCO, or Kerr at an effective spin.  The measured beta decides.  EOB is deferred (user: the old references are not state of the art). | The measured transition beta is near none of the candidates |
| **B5** | **Ringdown alpha from the damping mismatch** (zero parameters).  One real beta matches the QNM oscillation (beta_r) but not its damping.  The mapped ppBHPT ringdown decays at gamma_pp/beta_r on the NR clock against gamma_NR for NR, so for tau ≳ 10-15 M after the peak, alpha must carry exp[(gamma_pp/beta_r − gamma_NR)(tau − tau_pk)].  gamma = |Im w220|, known exactly. | The measured late-ringdown log alpha does not have this slope |

### 4.2 Self-force inspiral (Wardell 2112.12265; Warburton 2107.01298; data in `wardell_data/`)

| id | idea | falsified if |
|:--|:--|:--|
| **S1** | **Zero-parameter 1PA inspiral beta**: beta/X1^(6/5) − 1 = nu B(x) + O(nu^2).  B(x) comes from the 1PAT1 chirp: F0 = a F^(1), F1 = a F^(2)_inf − a^2 F^(1) dE_SF/dOmega_hat.  Only the spin part and the merger are fitted. | At chi1 = 0 on S18 it is clearly worse than the fitted b1, b2.  That would itself show that full-window fits distort the inspiral to buy merger phase |
| **S2** | **Tie alpha to beta through F_NR = (alpha^2/beta^2) F_pp**, with the flux ratio from 2SF (or its resummed form F^resum = [F/F^N re-expanded to O(nu)] F^N), so only one of the two is fitted in the inspiral. | The derived alpha is worse than the fitted one |
| **S3** | **1PA amplitude**: alpha/X1^(6/5) − 1 = nu [h2/h1 + 1 − (2x/3) d ln h1/dx], the all-orders analogue of 55/42 (`wardell_data/R2_22_from_amplitudes.json`). | Worse than 55/42 + a2 on S18 |

### 4.3 Spin

| id | idea | falsified if |
|:--|:--|:--|
| **P1** | **Spin variable**: chi1, X1 chi1 (= a1/M, Warburton's choice), X1^2 chi1, or chi_f for merger quantities, chosen by which collapses the measured residuals.  The wrong one fakes a q-chi1 cross term, a candidate cause of pn_anchored_spin's U shape. | No choice collapses them |
| **P2** | **Derived spin-orbit term** (the spin analogue of 55/42): the nu-dependent part of the 1.5PN spin-orbit phasing and amplitude that the test-mass-on-Kerr limit lacks, fixed analytically. | Fixing it is worse than letting it float |

### 4.4 Unconventional forms (when the structured ideas stall)

| id | idea | falsified if |
|:--|:--|:--|
| **D1** | **Fitted exponents**: nu^p, x^n, (u_T)^k with free exponents, snapped to rationals if close (this is how X1^(6/5) was found). | The exponents drift with the point set |
| **D2** | **Symbolic regression** on the measured alpha, beta or on per-case best parameters, over {nu, chi-variable, x, u_T, chi_f}. | Found forms do not beat the hand forms on S18 |
| **D3** | **Frequency-domain map**: Psi_NR(f) = Psi_pp(f/beta_LO) + dPsi(f), with dPsi a short power-law and log series, for the inspiral part. | No better than S1 over the inspiral |
| **E1** | **Input remapping**: evaluate ppBHPT at an effective spin chi1 + nu g(chi1).  This changes the framework, so it is **screen only until the user approves**. | — |

### 4.5 Reference notes (only what the ideas above use)

**Ori & Thorne 2000 (gr-qc/0003032).**

```
Kerr ISCO:   r_isco/M = 3 + Z2 - sign(a)[(3 - Z1)(3 + Z1 + 2 Z2)]^(1/2)
             Z1 = 1 + (1 - a^2)^(1/3)[(1 + a)^(1/3) + (1 - a)^(1/3)],   Z2 = (3a^2 + Z1^2)^(1/2)
             M Omega_isco = 1 / (r_isco^(3/2) + a)
transition:  R = eta^(2/5) R0 X,   tau = eta^(-1/5) tau0 T,   d^2X/dT^2 = -X^2 - T   (Painlevé I)
             X = sqrt(-T) for T << -1;  X = -6/(3.412 - T)^2 at plunge
             a = 0: R0 = 2.925, tau0 = 21.05 (proper time; Table I has the spin dependence)
scalings:    duration ∝ eta^(-1/5) (ΔT ≈ 3.3),  frequency width ∝ eta^(2/5),  E, L deficits ∝ eta^(4/5)
```

- **Name clash:** their alpha, beta are potential constants, unrelated to ours.
- **Validity:** asymptotic in eta << 1.  At our nu = 0.10-0.19, eta^(2/5) ≈ 0.4-0.5, so the
  structure is a guide with fitted coefficients, not an anchor.

**Küchler et al. 2024 (2405.00170, Schwarzschild).**

```
Omega = Omega_* + lambda^2 DeltaOmega,   lambda = eps^(1/5),   Omega_* = 1/(6 sqrt6 M)
dOmega/dt = lambda^3 F_[0](DeltaOmega) + lambda^5 F_[2](DeltaOmega) + ...    F_[0] from Painlevé I
buffer region:  -lambda^0 << M(Omega - Omega_*) << -lambda^2
composite (0PA-2PLT) = inspiral + transition - common terms;  trustworthy only for eps ≲ 1/180
critical frequency (transition beats inspiral):  Omega_* - c eps^(2/7)  (0PA-2PLT);  eps^(2/9) (1PA-7PLT)
transition amplitude:  H_lm = 2 eps { C/(m Omega)^2 |_* + (Omega - Omega_*) d_Omega[C/(m Omega)^2]|_* }
NR comparison:  eps -> nu, M -> M_tot, aligned at omega_22/2 = Omega_*
```

- They correct Ori–Thorne: Omega − Omega_* also scales as eps^(2/5); it is not fixed at the ISCO.
- **Use for us:** the composite architecture, not its coefficients.

**Küchler et al. 2025 (2506.02189, Schwarzschild).**
- **Three regimes:** inspiral dOmega/ds = eps[F_(0) + eps F_(1)]; transition dDeltaOmega/ds =
  eps^(1/5)[F_[0] + eps^(2/5) F_[2] + ...]; plunge dOmega/ds = F_{0}(Omega) + eps F_{1}.
- **Matching:** p + n = 5i + j, m = j.  On the plunge side the near-ISCO expansions run in
  half-integer powers of (Omega − Omega_*), with F_{0} ~ (Omega − Omega_*)^(3/2).
- **Waveform:**
  - the stationary-phase form is < 1% until about 4 cycles before the peak;
  - the QNM sum is good from about 10-15 M after it;
  - neither covers the merger;
  - the phase stays tied to the orbit until about 0.7 rad after the peak.
- **Remnant spin:** the post-peak error against NR is dominated by the missing remnant spin.
- **Stitching without a transition:** stitching an inspiral to a plunge directly leaves no common
  frequency.

**Wardell 2023 (2112.12265) and Warburton 2021 (2107.01298).**

```
dOmega/dt = -(dE_bind/dOmega)^(-1) F
F      = nu^2 F^(1)(x) + nu^3 F^(2)_inf(x)
E_bind = nu M [E0(x) + nu E_SF(x)],   E0 = (1 - 2x)/sqrt(1 - 3x) - 1
1PAT1:   dOmega/dt = (nu/M^2)[F0 + nu F1]
         F0 = a F^(1),   F1 = a F^(2)_inf - a^2 F^(1) dE_SF/dOmega_hat,   a = -(dE0/dOmega_hat)^(-1)
```

- **Correction:** the "3" before a^2 in the PDF copy is not in the source.
- **Spin variable:** Warburton uses a_i/M = X_i chi_i.  A small primary spin in the 2SF expansion
  reproduces the first-order Kerr flux to 4e-5.

### 4.6 Starting order (then led by the diagnostics)

1. **Pass 0** (no fitting, about 1 SU):
   - reproduction gate;
   - calibrated BHPTNRSur2dq1e3 on S18 and T96;
   - E_I / E_MR for the installed models;
   - the B1 t_isco check;
   - collapse tests of the measured alpha, beta in u_T and in the P1 spin variables;
   - the frequency-domain comparisons of §5;
   - the B5 late-ringdown slope check.
2. **S1** at chi1 = 0 on S18 (zero-parameter 1PA inspiral against fitted b1, b2).
3. **B1 + B2** with beta_r and B5: the first full transition-structured form, on S18.
4. **B3, B4, P1, P2, S2, S3**, as the diagnostics point.  **D1, D2** after two passes without improvement.

---

## 5. Diagnostics (few, chosen)

**Every pass, as text in the Log:**
- the gates;
- E_I and E_MR under alignments (a) and (b);
- the fixed-q and fixed-chi1 medians;
- the automatic flags: hot cells, beta ≤ 0, join-window features, edge blow-ups;
- **frequency-domain comparisons** of model against NR, independent of alignment:
  - omega^2/omegadot against omega (Wardell Fig. 4);
  - omegadot(omega) through the transition;
  - |h|(omega).

  These say *where in frequency* a model fails.

**Figures:**
- **Every T96 model:** one q-chi1 heatmap of log10 mathcalE, each cell marked with whether E_I or
  E_MR dominates.
- **When a flag fires:** one figure of alpha, beta against the measured values for the 2-3 flagged
  cases.
- **Final model only:** the three wishlist figures (Islam+ Figs 5, 7, 9a) and one alpha/beta overlay
  at fixed chi1.

PNG at 150 dpi; PDF only for the final figures.  Every figure is looked at, and what is wrong goes in
the Log.

---

## 6. Compute and storage

- **Order of runs:**
  - screens first;
  - T96 only after a screen passes;
  - H270 only for G1 candidates;
  - 3 multistarts only for a new leader.
- **Jobs:** one node per job, development queue.
  - A pass costs about 1-3 SUs.
  - **Cap: 300 SUs before the first check-in** (19,952 available).
- **Storage:**
  - no new waveform caches;
  - one JSON and one log per pass;
  - logs of rejected passes are trimmed to their summary;
  - no per-case arrays;
  - `exploration/` stays under 50 MB.
- **Fewer time samples, only if runs get slow** (user, 2026-09-29).
  - **Why it works:** with h = A e^{-i phi}, the error needs only ∫A_r^2, ∫A_g^2 and
    |∫A_r A_g e^{i(phi_g - phi_r)}|, all slowly varying once aligned.
  - **Greedy nodes** (shared by all cases, NR time): add the time where (A, phi) interpolation is
    worst over the T96 waveforms and a spread of trial models, until 1e-3 rad and 1e-4 relative.
    Interpolate h_pp by amplitude and phase too.
  - **Accept only if**, for the leader and about 20 nearby coefficient sets, the reduced-node E
    matches full resolution within 1%, the ranking is unchanged, and t0 agrees within 0.05 M.
  - **Never on reduced nodes:** every reported or gated number is recomputed at full resolution.

## 7. The loop

1. **Hypothesis** in the Log: the form, what it answers, and what would falsify it.
2. **Script**, with the P limits asserted.
3. **Screen** (collapse test and/or S18, paired with the leader).  Promote if ≥ 10% better on the
   S18 mean, or if it fixes a flagged region without losing > 10% elsewhere.
4. **Full**: T96, then H270 if it is a G1 candidate, then the gates and the heatmap.
5. **Diagnosis** in the Log; then the next idea.

**Check in when:**
- Done is reached;
- 8 passes produce no new leader;
- spending reaches 300 SUs;
- a genuine model choice comes up (B4's binary ISCO, E1); or
- a derivation (B3, S1, P2) is uncertain enough that you should check it.

Clear bugs are fixed without asking.

**Fresh context for new directions** (user, 2026-10-03).  Long sessions anchor on the forms already
tried.  So:
- **Self-contained Log.**  The Log and this file are kept complete enough that a fresh session can
  resume from them alone: the current leader, open flags, and the next idea with its reasoning.
- **New idea family.**  Before starting a completely new family of ideas (for example moving from
  the §4.1 transition forms to §4.4), I write the state into the Log.  Then either a fresh agent
  gets only this file and the task, or I suggest you run `/clear` and restart from this file.
- **Fresh look when stalled.**  When stalled (two passes without improvement), one fresh agent reads
  only this file and the latest diagnostics and proposes forms without seeing my reasoning.  Its
  proposals are judged by the same gates.

## 8. After a model works: the physics account

For each structural piece:
- **(a) the argument:** a derivation where one exists (PN, test-mass limit, 1PA, transition
  expansion, QNM, remnant), otherwise a heuristic labelled as such;
- **(b) the ablation:** the mismatch cost of removing that piece;
- **(c) the predictions:** nu -> 0, q = 10, the edges, and the implied F2 against Wardell /
  Warburton at chi1 = 0.

It is written into the model's md with a full "Anchors" section, for the paper.

## 9. Rules

- Everything lives in `exploration/`.  No installed model, coefficient file, cache or md is edited.
- No long run on code that has not passed its limits and the reproduction gate.
- Every Log number states its point set, coefficient count and job id.
- Failures are written up like successes.
- Check-ins are short and plain.

---

## Log

| pass | date | idea | coef | screen (vs leader) | T96 mean / median / max | H270 median | gates | job | verdict / next |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| 24 | 2026-10-07 | 20c model refitted on a power-mean objective (mean E^p)^(1/p), p = 2, 4 (no new coefficients) | 20 | – | p2 2.47e-4 / 1.72e-4 / 1.35e-3; p4 2.77e-4 / 2.23e-4 / **1.04e-3** | p2 1.81e-4; p4 1.92e-4 | U fails for both | 3495412 | **negative: U is not an objective artefact.** Edge columns unchanged (+0.8 4.9-5.5e-4, -0.6 3.9-4.3e-4); the power mean only trims the max and raises the bulk. U is structural for this form |
| 25 | 2026-10-08 | V7-S3 with free nu exponents on the MR offsets and the shapes (wv = (nu/nu5)^p), then single-drop pruning and joint drop < 3% | 22 -> 20 | – | 2.35e-4 / 1.61e-4 / **1.18e-3** | 1.71e-4 (max 6.9e-4) | U fails (q8 row 2.63e-4 = 2.2x min; was 2.7x); L ok (q2.5 max 6.5e-3; q2 3.0e-2) | 3497832 | **partial.** p_sh = 0.57, p_mr = 0.98 (-> 1, dropped); t2a dropped (s2 = 0). Reproduce check 1.0000. Shapes want a weaker-than-nu weight; the rise past q = 5 is reduced, not removed. New 20c leader on mean / max (median +2%) |
| 26 | 2026-10-08 | blend onset s1 = ppBHPT Kerr-ISCO crossing (+ offset: I1; none: I0), on the pass-25 model | 19 / 18 | – | I1 2.49e-4 / 1.67e-4 / 1.29e-3; I0 3.12e-4 / 2.62e-4 / 1.16e-3 | I1 1.79e-4; I0 2.50e-4 | I1 rows like p25 (q8 2.77e-4); I0 rows FALL with q (4.65 -> 1.93 -> 2.51e-4 at q = 3 / 7.23 / 8); L ok both (q2.5 7.0e-3 / 5.8e-3) | 3498214 (first run 3497834 invalid: fork-order bug; its "I1" = constant onset, 2.92e-4, +24%) | **I1 = derived spin slope of the onset for +6% (19c option).** s_isco median -90 / -52 / -11 m1 at chi1 -0.6 / 0 / +0.8 vs fitted s1 -150 / -102 / -39. I0 (no fitted onset) costs +33% but removes the high-q rise -- the window onset and the q-trend are coupled |
| 27 | 2026-10-08 | per-case spline floor + PCA on the pass-25 model (q-structure of the remaining residual) | 0 | – | floor 3.0e-5 / 1.33e-5 / 1.8e-4 (flat in q) | | | 3498179 | Variance now spread: dlnB 0.43 / 0.35 / 0.13, dlnA 0.42 / 0.37 / 0.10. Comp 1 of both fields carries the q-trend, steep at chi1 = -0.6 and +0.8, mild mid-spin; chi x (1, nu) explains 87% (B) / 76% (A) of it, chi alone 50% / 47%. dlnA1 = peak amplitude (knot 0: -0.80), decays from q = 3 faster than nu. -> pass 28 (spin-dependent nu exponent) |
| 28 | 2026-10-08 | shape weight (nu/nu5)^(p_sh + g2 chi1^2) (X2, 21c); separate alpha exponent pA + gA2 chi1^2 (X3, 23c); on pass 25 | 21 / 23 | – | X2 2.31e-4 / 1.52e-4 / 1.20e-3; X3 2.30e-4 / 1.54e-4 / 1.18e-3 | X2 1.63e-4; X3 1.64e-4 | U fails (q8 row 2.42 / 2.45e-4, 2.1x the min); L ok (q2.5 5.7e-3) | 3498220 | **negative (fails K: -1.5% mean for +1c).** Exponent -0.11 + 1.36 chi1^2: shapes nu-flat at mid spin, ~nu^1 at the edges, but the rise stays. Reweighting the shapes is not the lever |
| 29 | 2026-10-09 | onset with a nu slope: J1 s1 = s_isco + t1a + t1n (nu/nu5 - 1) (20c), J2 s1 = t1a + t1b chi1 + t1n (nu/nu5 - 1) (21c); on pass 25 (`p26_iscowin.py J1|J2`) | 20 / 21 | – | J1 2.48e-4 / 1.76e-4 / 1.20e-3; J2 2.35e-4 / 1.62e-4 / 1.19e-3 | J1 1.82e-4; J2 1.71e-4 | U fails (q8 2.66 / 2.63e-4) | 3498331 | **negative.** A nu slope on the onset buys nothing (J1 = I1 + 1c at -0.5%; J2 = pass 25). The falling rows of I0 are a different compromise, not an onset q-slope |
| 30 | 2026-10-09 | per-q-row refit of the pass-25 model (17 free per row; b2, P1, p_sh at global values) | – | – | per-row medians 1.70 / 1.17 / 0.97 / 1.03 / 1.13 / 1.19 e-4 at q = 3 / 3.9 / 5 / 5.95 / 7.23 / 8 (global 3.03 / 1.36 / 1.18 / 1.57 / 1.93 / 2.63) | | | 3498589 | **the rise is in the q-dependence of the coefficients and is fixable:** the per-row floor is flat to 1.2x above q = 5. Drifting coefficients are mostly spin terms: s3 1.96 -> 3.44, r1 0.055 -> 0.132, eb1 0.067 -> 0.151, pb1 -0.23 -> -0.37, r3 -0.09 -> -0.15 (also pb0 0.13 -> 0.22, s2 -12.8 -> -26.8); roughly an extra (nu5/nu)^1 on the spin terms |
| 31 | 2026-10-09 | spin terms x (nu/nu5)^(-gamma): G1 one gamma on {s2, s3, r1, r3, pb1, eb1} (21c); G2 gamma_I on {s2, s3}, gamma_M on {r1, r3, pb0, pb1, pb2, eb1} (22c) | 21 / 22 | G1 2.11e-4 / 1.37e-4 / 9.96e-4; G2 2.01e-4 / 1.27e-4 / 9.60e-4 | rows (q = 3 / 5 / 8) G1 2.72 / 1.02 / 2.00, G2 2.64 / 0.98 / 1.86 e-4 (pass 25: 3.03 / 1.18 / 2.63); H270 median G1 1.50e-4, G2 1.40e-4 | q = 2.5 max G1 4.9e-3, G2 4.3e-3 (gate L passes); q = 2 max 2.3e-2 / 2.2e-2 | gamma G1 1.09; G2 1.15 / 1.28 | 3498659 | **gain: -14% mean / -21% median vs pass 25, q = 8 row -29%.** gamma ~ 1, as the pass-30 drift predicted. The rise is reduced but not gone (q = 8 / q = 5 = 1.9). Prune in pass 32 |
| 32 | 2026-10-09 | prune pass 31 G2 (single-drop costs, joint drop < 3%; p_sh and gammas drop to 1) | **19** | 2.02e-4 / 1.28e-4 / 9.31e-4 | rows (q = 3 / 5 / 8) 2.45 / 0.98 / 1.91 e-4; H270 median 1.42e-4 | q = 2.5 max 4.9e-3 (L passes); q = 2 max 2.4e-2 | zeroed p_sh, gamma_I, gamma_M (all exponents now exactly 1: spin terms x nu5/nu, so the shape spin terms are q-flat); next cheapest drop r0 3.9% | 3498733 | **new leader, 19c: -14% mean / -20% median vs pass 25 (20c).** G1 prune (3498734) cancelled as redundant. Remaining rise sits at chi1 <= 0 (q = 8 chi 0: 1.81 vs 0.71 e-4 at q = 5.44) and the (8, +0.8) corner (8.8e-4) |
| 33 | 2026-10-09 | on pass 32: A drop r0 (18c); B ISCO onset, t1b dropped (18c); C free r2 (20c, for the chi <= 0 rise) | 18 / 18 / 20 | A 2.08e-4 / 1.36e-4 / 8.77e-4; B 2.14e-4 / 1.40e-4 / 1.04e-3; C 2.01e-4 / 1.28e-4 / 9.17e-4 | rows (q = 3 / 5 / 8) A 2.87 / 1.17 / 1.73, B 2.68 / 1.03 / 1.87, C 2.53 / 1.01 / 1.70 e-4; H270 median A 1.38, B 1.46, C 1.37 e-4 | q = 2.5 max A 4.7e-3, B 5.4e-3, C 4.8e-3 (L passes) | C: r0 -0.006, r2 +0.078 | 3498777 | **A is the 18c option: +3% mean, +6% median, but max -6%, H270 median -3% and q = 8 / q = 5 = 1.5 (was 1.9).** B costs +6%; C gains < 1% for +1c (rejected). Next: prune A (pass 34) |
| 34 | 2026-10-09 | single-drop costs + joint prune of pass 33 A (18c) | 18 | 2.08e-4 / 1.37e-4 / 8.84e-4 (joint refit, nothing dropped) | rows (q = 3 / 5 / 8) 2.80 / 1.16 / 1.77 e-4; H270 median 1.37e-4 | q = 2.5 max 4.6e-3 | cheapest drops pb2 11.6%, eb1 12.6%, ea0 14.2% | 3498827 | **18c is pruned to the bone** (no drop < 3%). Gates: G2, V (H270/T96 median 1.00), K, L pass; **U fails** at the chi1 columns -0.6 (3.3e-4) and +0.8 (4.5e-4) > 2 x median 2.7e-4, and q = 3 row 2.80e-4 is at the edge. Next: residual at the spin extremes (pass 35) |
| 35 | 2026-10-09 | pass-27 per-case spline floor + PCA on the 18c model, read per chi1 column | 0 | floor 2.91e-5 / 1.28e-5 / 1.99e-4 | chi1 = +0.8 floor 0.07-0.10e-4 at every q (model 2.1-8.8e-4); chi1 = -0.6 floor 0.9-2.0e-4 | | | 3498851 | **diagnostic: the spin-extreme residual is a q-FLAT ringdown slope in ln alpha.** Spline dlnA at s = 18 / 35 m1: chi1 -0.6 -0.17 / -0.63, 0 -0.00 / -0.07, +0.6 +0.07 / +0.18, +0.8 +0.13 / +0.36 (std over q <= 0.04). q-flat in m1 means a ppBHPT-side property: the Kerr gamma_pp in kappa_RD is not the decay the ppBHPT (2,2) has over the scored ringdown. Next: measured decay (pass 36) |
| 36 | 2026-10-09 | kappa_RD with gamma_pp -> measured -dln|h_pp|/ds over s in [15, SE] m1 (per case, from the input; 0c). D1 SE = 40, D2 SE = 60, refit 18c | 18 | D1 1.99e-4 / 1.36e-4 / 8.49e-4; D2 2.05e-4 / 1.36e-4 / 8.60e-4 | D1 rows (q = 3 / 5 / 8) 2.87 / 1.15 / 1.70 e-4; cols -0.6 3.00, +0.8 4.33 e-4; H270 median 1.34e-4 | q = 2.5 max 4.6e-3 | measured gamma (per m1) vs Kerr: -0.6 0.0765 vs 0.0889, 0 0.0851 vs 0.0890, +0.8 0.0834 vs 0.0756 (q-flat to 1e-3) | 3498853 | **D1 small gain, 0c: -4% mean, chi -0.6 column -10%, +0.8 only -4%.** The ringdown slope is real but is not what carries the +0.8 error. D1 is the 18c leader. Next: knot ablation (pass 37) |
| 37 | 2026-10-09 | which knots of the pass-35 per-case splines carry the gain at chi1 = -0.6 / 0 / +0.8 (no refit) | 0 | | medians (1e-4) none / all / beta only / alpha only: -0.6 3.32 / 1.26 / 2.80 / 2.24; 0 1.42 / 0.12 / 0.25 / 1.04; +0.8 4.56 / 0.08 / 1.13 / 3.28 | | partial beta masks (early / merger / ringdown alone) are WORSE than none: beta acts through the integrated phase, so it must be corrected as a whole | 3498906 | **diagnostic: beta carries the remaining error at chi1 >= 0; at -0.6 both fields.** Leading beta residual (pass 35 dlnB comp 0, knots -30..0) is 64% explained by 0.055 chi1; dlnA comp 1 66% by 0.072 chi1 |
| 38 | 2026-10-09 | on pass 36 D1: N1 + c_B chi1 phiB35(s); N2 + c_A chi1 phiA35(s) too (shapes = pass-35 PCA vectors, q-flat) | 19 / 20 | N1 1.98e-4 / 1.37e-4 / 8.79e-4; N2 1.87e-4 / 1.38e-4 / 9.64e-4 | N2 cols -0.6 2.59, +0.8 3.52 e-4; H270 median 1.28e-4 | q = 2.5 max N2 4.3e-3 | c_B +0.030 (N1), +0.054 / +0.075 (N2) | 3498909 | **no leader.** N1 buys 0.3%; N2 -6% mean for +2c (fails K, 5% per coefficient) and +0.8 is still 2.6x the median |

**Check-in 4 (2026-10-09 ~10:45).**  Leader pass 36 D1 (**18c**; `exploration/p36/p36_D1.json`; model = p25 equations, spin terms x nu5/nu (pass 31-32, all fitted exponents snapped to 1), r0 dropped (33 A), kappa_RD with the measured ppBHPT ringdown decay over s = 15..40 m1 (36)): T96 1.99e-4 / 1.36e-4 / 8.49e-4, H270 median 1.34e-4 (max 5.7e-4), q = 2.5 max 4.6e-3, q = 2 max 2.3e-2.  vs check-in 3 (V7-S3, 20c): mean -18%, median -14%, max -39%, H270 median -21%, 2 coefficients fewer.  Rows q = 3 / 5 / 8: 2.87 / 1.15 / 1.70 e-4 (V7-S3 3.05 / 1.21 / 3.21): q = 8 / q = 5 = 1.5 (was 2.7).  Gates: G1, G2, **G3 (now: max 8.5e-4 < 1e-3)**, V (0.98), P, K, L pass; **U fails**: cols -0.6 3.00, +0.8 4.33 e-4 and row q = 3 2.87 e-4 vs 2 x median 2.72 e-4.  What remains (passes 35, 37): the per-case floor is 0.08e-4 at +0.8 and 0.9-2.0e-4 at -0.6; beta carries the +0.8 gap and acts as a whole; chi-linear residual shapes do not regress cheaply (38).  The +0.8 column is non-monotone in q (8.4 / 3.3 / 5.0 / 1.9 / 8.5 e-4 at q = 3 / 3.5 / 5 / 6.6 / 8), worst at both grid ends.  Open for the user: (1) relax U or count it on the interior, (2) B4 binary ISCO choice or E1 for the +0.8 edge, (3) go to §8 + final plots with D1, (4) the deferred like-for-like Rink per-point comparison.  ~105 SU used.
| 23 | 2026-10-07 | **S3**: inspiral alpha prefactor (1 + c0 nu) -> (1 + k nu R2(x)/2), R2 = 2SF (2,2) amplitude ratio (WaSABI, Schwarzschild). Z: k = 1 fixed; K: k free | **20** / 21 | – | **Z: 2.42e-4 / 1.58e-4 / 1.40e-3**; K: 2.33e-4 / 1.49e-4 / 1.40e-3, k = 0.617 | Z 1.69e-4 (max 7.04e-4); K 1.64e-4 | Z: G1, G2, V, P, L (q = 2.5 max 5.5e-3), **K (20c)** pass; U fails as V7 | 3495374 | **V7-S3 (20c, Z) = LEADER within the cap**: the zero-parameter 1PA amplitude costs 3% where dropping c0 cost 11%. The data want ~60% of the 1PA amplitude correction (K). -> pass 24: is U an objective artefact? |
| 22 | 2026-10-07 | prune the 27c leader: 19 single drops (2.5-min refits), joint drop of all < 2% | **21** | single costs: free (< 2%): pa1 0.4, eb0 1.1, eb2 1.2, ea1 1.5, flux_A0 1.7, ea2 1.9; then ea0 3.2, eb1 3.4, r0 4.9, pa0 6.9, flux_c0 10.9, pb2 11.6, pb0 25, t1b 26, l0 42, flux_s2 81, b / P0 / P1 > 245 | **2.35e-4 / 1.50e-4 / 1.40e-3** | **1.63e-4** (max 6.77e-4) | G1, G2, V (1.08), P, L (q = 2.5 max 5.6e-3) pass; G3 median yes, max no; U fails (cols -0.6 2.5x, +0.8 3.1x; rows q = 3 1.8x, q = 8 2.15x); K: one more drop to <= 20 | 3495273 | **V7 (21c) = LEADER**: same count as V6r, 13-15% better. The pass-18 shapes reduce to eb1 chi1 (beta, linear in spin) and ea0 (alpha, constant). The flux alpha coupling (A0) is gone: inspiral alpha is a constant X1^(6/5)(1 + c0 nu). -> pass 23: S3 replaces c0 by the 1PA amplitude (20c) |
| 21 | 2026-10-07 | 27c leader + PN-ordered inspiral terms x nu, before the blend. A: nu(g1 x + g2 chi1 x^1.5 + g3 chi1^2 x^2) in ln beta, nu(h1 + h2 chi1) x in ln alpha; B: A + nu^2 x in each | 32 / 34 | – | A 2.19e-4 / 1.38e-4 / 1.56e-3; B 2.13e-4 / 1.31e-4 / 1.58e-3 | A 1.45e-4; B 1.37e-4 | K fails (1-2%/coef); L passes; U unchanged | 3495167 | **negative for the edges**: +0.8 col 4.1e-4, rows q = 3 / 8 2.7-2.8e-4 / 3.2-3.3e-4 unchanged. The per-case inspiral fix (pass 20) does not regress onto PN-like terms. Edge errors grow smoothly toward the ends of the training ranges (q = 3 = ppBHPT surrogate's low end; q = 8 and chi1 = +0.8 = NRHybSur3dq8's training edges) -- regression edge effect and/or input quality, not separable here. -> consolidate: pass 22 prunes the 27c leader |
| 20 | 2026-10-07 | per-case inspiral terms on the 27c leader: ln beta += (d0 + d1 x^1.5 + d2 x^2)(1-W), ln alpha += (d3 + d4 x)(1-W) | 5 per case | – | **1.09e-4 / 6.8e-5 / 6.5e-4** (leader 2.24e-4 / 1.48e-4 / 1.44e-3) | – | – | 3494956 | inspiral regression is worth 2x; largest gains at +0.8 (x0.33) and q = 8 (x0.27); d's smooth in q at every chi1, d1/d2 degenerate. 20b (greedy term choice on the per-case functions, `p20b_select.py`): poor collapse, not a usable guide -> pass 21 fits PN-ordered terms globally |
| 15 (V5) result | 2026-10-07 | V5 ablation, joint drop of 13 | 19 | – | 3.43e-4 / 2.62e-4 / 1.43e-3 | 2.47e-4 (max 9.6e-4) | – | 3494200 | **V5 line closed**: at reduced size V6r (21c, 2.69e-4 / 1.77e-4, H270 1.88e-4) is better on every number; the test-mass limit is both more physical and more economical |
| 19 | 2026-10-07 | band decomposition of the 27c leader vs the pass-17 per-case floor | – | – | – | – | – | 3494950 | **edges located.** +0.8: 2.8e-4 of 5.0e-4 in the inspiral ([-5000, -150] M) where the floor has 4e-6 -> inspiral spin regression. q = 3 / 8 rows: inspiral bands 7e-5 + 7e-5 / 1.2e-4 + 8e-5 vs 2-4e-5 mid-range, floor flat in q -> inspiral nu regression. -0.6: late ringdown [15, 100] 2.4e-4 of 4.3e-4, floor 1.2e-4 there (framework-limited; the known -0.6 ringdown issue, low priority per user). -> pass 20: per-case inspiral terms |
| 18 | 2026-10-07 | V6r + leading pass-17 shapes (1+1 and 2+2), amplitude w (e0 + e1 chi1 + e2 chi1^2) | 27 / 33 | – | **27c: 2.24e-4 / 1.48e-4 / 1.44e-3**; 33c: 2.09e-4 / 1.36e-4 / 1.34e-3 | 27c 1.57e-4 (max 6.69e-4); 33c 1.47e-4 | L passes (q = 2.5 max 5.2e-3 / 5.1e-3); K: 27c 2.8%/coef, 33c 1.1%/coef (fail before pruning); U fails: cols -0.6 3.4e-4, +0.8 4.0e-4; **rows q = 3 2.7e-4, q = 8 3.3e-4** | 3494502 | 27c = leader pending pruning; the second shapes are not worth it. q edges now as bad as spin edges -> pass 19 locates them in time |
| 16 (V6) | 2026-10-07 | **B2**: window s_i = t_ia (L/L0)^p, L = q^(1/5) tau_o(chi1)/(dtau/dt), the ppBHPT's Ori-Thorne unit; replaces t1b (t2b already 0) | 21 | – | 2.72e-4 / 1.76e-4 / 1.71e-3 | 1.93e-4 (max 7.29e-4) | **L passes** (q = 2.5 max 4.2e-3); U as V6r | 3494389 | **neutral** (p = 0.744 fitted): a physical window at no cost, not a gain. Low q, max / median: q = 2.5 V6 32c 5.1e-3 / 2.0e-3, V6r 4.7e-3 / 2.0e-3, OT 4.2e-3 / 1.8e-3; q = 2: 2.7e-2 / 1.3e-2, 2.5e-2 / 1.4e-2, 2.3e-2 / 1.3e-2 (worst at +0.8, then -0.6) |
| 17 | 2026-10-07 | per-case free splines on V6r (21c) + PCA (D2, second round) | 30 per case | – | floor **3.0e-5 / 1.26e-5 / 1.79e-4** (base 2.69e-4 / 1.78e-4) | – | – | 3494389 (first step) | 10x left on the table, and structured: beta comp 0 (62%) peaks at s = -9..0, amplitude -0.06 (chi1 <= -0.2) -> +0.07 (+0.8), q-scatter small; alpha comp 0 (68%) dip at -50..-30, rise at 0, amplitude -0.14 at -0.6 -> -0.01..-0.05 elsewhere. -> pass 18 |
| 15b | 2026-10-07 | **P1 collapse test**: pass-12 per-case shape amplitudes (A beta, B alpha) against chi1, X1 chi1, chi_f, Omega_isco, r_isco, w220, ln beta_r, chi_f - chi1 | – | rms of fit residual | – | – | – | login (`p15b_spinvar_collapse.py`) | **negative: no variable beats chi1.** A: rms 0.099, every 1-2 coefficient form leaves 0.054-0.060 (case scatter, not a missed variable); chi1 alone 0.0558 vs the 4-coef form 0.0544 -> pa2, pa3 likely droppable. B needs chi1^2 (0.050 -> 0.027). Per-case corr(A, B) = -0.62 only (the -0.93 was the shape-level ratio), so the tie may cost |
| 15 (V6) | 2026-10-07 | ablation of V6: 24 single drops (3.5-min refits from the V6 optimum), then joint drop of all singles costing < 3% | **21** | single-drop costs on the T96 fit mean: free (< 1.2%): flux_s4, l3, pa3, l1, t2b, flux_b3, flux_s1, l2, flux_A1, flux_c1; r2 2.8%, flux_s2 3.2%; essential: flux_b2 +191%, r1 +43%, pa2 +39%, flux_s3 +32%, r3 +19%, pb1 +13%, t1b 8%, pb2 7%, pa1 5%; tie of the alpha shape to beta +7% (rejected) | **2.69e-4 / 1.77e-4 / 1.69e-3** | **1.88e-4** (mean 2.32e-4, max **7.08e-4**) | as V6 (U still fails: col -0.6 4.1e-4, +0.8 5.2e-4) | 3494199 | **V6r (21c) = new leader**: equal or better than 32c V6 on everything but the T96 max (+8%) |
| 15 (V5) | 2026-10-07 | ablation of V6 and V5: single-coefficient drops (short refits), then a joint drop of everything costing < 3% | 32 -> ? | – | running | – | – | 3494200 | – |
| 14 | 2026-10-06 | **V6** = V5 with the test-mass limit imposed (all merger-ringdown corrections x nu/nu5) | 32 | – | **2.74e-4 / 1.74e-4 / 1.57e-3** | **1.91e-4** (mean 2.39e-4, max 8.84e-4) | G1, G2, V (1.09), K, **P** pass; G3 median yes, max no; **U fails** (col 3.1x, spread 0.28, hot cell (8, +0.8)) | 3492168 | co-leader with V5: better bulk, worse spin edges -> **check-in** |
| 13 | 2026-10-06 | **V5** = V4 + pass-12 principal shapes as fixed basis functions | 32 | – | **2.89e-4 / 2.32e-4 / 1.19e-3** | **2.35e-4** (mean 2.72e-4, max 7.27e-4) | **G1, G2, V, K pass; G3 median yes, max +19%; U marginal (col 2.09x, spread 0.216); P fails (nu -> 0)** | 3492050 | **LEADER** |
| 12 | 2026-10-06 | free splines on top of V3, all T96, + PCA of the corrections (D2) | 30 per case | – | floor 3.5e-5 / 1.3e-5 / 2.3e-4 | – | – | 3491964 | one beta shape = 87% of variance, spin-driven amplitude |
| 11 | 2026-10-06 | **V4** = V3 + chi1^2 in rho, alpha level | 25 | – | **5.82e-4 / 3.67e-4 / 3.06e-3** | **3.68e-4** (mean 4.27e-4, max 1.19e-3) | G2 except T96 max; U rows 1.39x, columns no (+0.8 4.15x, -0.6 2.3x); K borderline (4.8%/coef) | 3491932 | **LEADER** |
| 8 | 2026-10-06 | **V3** = V1 + inspiral regression terms (b_pp: nu^2, nu chi1^2; P: chi1, chi1^2) | 23 | – | **6.44e-4 / 3.93e-4 / 3.44e-3** | **3.80e-4** (mean 4.65e-4, max 1.28e-3) | G2 except T96 max; U rows yes (1.38x), columns no (+0.8 4.1x, -0.6 2.6x); V 0.97; K yes (-24% for 2 beyond V1... vs V2 -24%) | 3491858 | **LEADER** |
| 7b | 2026-10-06 | V2: V1 + chi1^2 in rho and in the alpha level | 21 | – | **8.46e-4 / 5.29e-4 / 5.94e-3** | **4.74e-4** (mean 5.70e-4, max 2.32e-3) | G1 yes; K strictly no (-4.8% mean for 2) | 3491712 | **leader**; beats the split variant on every T96 / H270 number (21 vs 14 coef) |
| 7 | 2026-10-06 | V1: M2 + blend window linear in chi1 | 19 | – | 8.88e-4 / 5.91e-4 / **6.07e-3** | 5.07e-4 (mean 5.95e-4, max 2.27e-3) | **G1 passes in full**; K fails (-1.5% for 2) | 3491711 | small gain; the gap is in the inspiral |
| 6 | 2026-10-06 | **M2 global** (FLUX 9 + blend window + rho(nu, chi1) + alpha level(nu, chi1)) | 17 | S18-fit on T96: 9.45e-4 / 6.62e-4 / 5.49e-3 | **9.01e-4 / 6.01e-4 / 6.60e-3** | **5.08e-4** (mean 5.97e-4, max 2.28e-3) | G1 except T96 max (+1.5%); V ok (0.85) | 3491674 | **new leader**; spin edges and q = 3 row remain |
| 5b | 2026-10-06 | M2 floors on all T96, bounded blend | 8 per case | – | floors: FLUX 7.05e-04 / 6.66e-04; M2 4.00e-04 / 2.53e-04 / 1.82e-03 | – | – | 3491663 | M2 floor beats the installed T96 by 2.7x mean, 3.5x median |
| 5 | 2026-10-06 | per-case floors on S18: FLUX (installed form) vs M2 (FLUX + ppBHPT-clock ringdown blend) | 4 / 8 per case | floors: FLUX 7.8e-4 / 6.9e-4; **M2 4.4e-4 / 2.4e-4** | – | – | – | 3491645 | **M2 form reaches ~1-2.5e-4 except at the spin edges** |
| 4 | 2026-10-06 | measured-ppBHPT ringdown anchor; + merger bump in beta | 9 frozen + 2 (+4) | – | A 1.045e-3 / 7.98e-4 / 6.52e-3; B 1.045e-3 / 9.33e-4 / 4.79e-3 | – | no gain over pass 3 | 3491582 | negative |
| 3 | 2026-10-06 | beta_r ringdown anchor on frozen flux_anc_spin_with_orbit | 9 frozen + 2 | S18 1.46e-3 | 1.023e-3 / 8.30e-4 / 6.59e-3 | – | G1 mean+median yes, max +1% | 3491508 | -6%; B5 alpha adds nothing |
| 2 | 2026-10-06 | S1 with spin; full installed models on E_I | 0 | S1 3-8x worse than installed on E_I | – | – | – | smoke tests | **S1 falsified as a precision anchor**; errors live in merger-ringdown |
| 1 | 2026-10-06 | S1 zero-parameter 1PA inspiral beta, chi1 = 0 | 0 (+1 gauge) | E_I vs leader branch: better at 11/12 q | – | – | – | 3491485 | promising: S1c E_I median 5.9e-4; essential 1PA term confirmed |
| 0 | 2026-10-06 | measurements only (reproduction gate, landmarks, B5 slope, collapse) | – | gate ratio 1.000000 on S18 | – | – | – | 3491463 | B5 holds for chi1 ≤ 0.4; beta features at the peak, not t_isco; merger-gauge beta unusable for inspiral tests |

(Per-pass notes go below the table, newest last.)

**Pass 0 (2026-10-06, job 3491463, `exploration/p0_measure.py`, outputs `exploration/p0/`).**  No fitting.
- Reproduction gate: installed pn_anchored_spin rescored on S18 through the imported functions, ratio 1.000000 on all 18.
- Leader's switch: S never reaches 0.5 on T96 (S at the |h| peak 0.15-0.19).  Its merger branch is only an admixture.
- **B5 (ringdown alpha from the damping mismatch) holds** where it should: measured slope of ln alpha over
  NR [peak+15, peak+60] M against gamma_pp/beta_r - gamma_NR agrees to about 5% for chi1 in [-0.4, +0.4] at every
  q (e.g. q = 5, chi1 = 0: 0.0190 vs 0.0192; q = 8: 0.0132 vs 0.0126).  It fails at chi1 = +0.6, +0.8 (measured up
  to 2x steeper: q = 8, +0.8: 0.0215 vs 0.0116) and at retrograde -0.6 (mixed).  Measured ringdown beta / beta_r =
  1.005-1.016 except -0.6 (0.92-0.97).  Open: why high positive spin damps faster in the ratio.
- **B1 (t_isco) is weak as a merger landmark.**  t_isco - t_peak = -48 ± 28 m1 (range -103..-8, mostly with spin);
  the measured beta maximum is at t_peak - 2 ± 12 m1 and its steepest rise at t_peak - 12 ± 14 m1.  In OT time T the
  peak itself sits at T = 1.23-2.21, drifting with spin.  So the |h| peak, not t_isco, anchors the merger features.
  t_isco may still be the right inspiral cut.
- **The merger-aligned equal-phase "measured beta" is unusable for inspiral shape tests.**  Its phase constant adds an
  arbitrary gauge mode (∝ x^(5/2) in an equal-phase map) per case, so across-q collapse in the inspiral (spread 30%)
  and the spin-orbit ratio (-1.4 to 3.5) are dominated by gauge.  Inspiral forms must be tested gauge-free (chirps,
  or the mismatch).  Merger-window collapse of (beta/X1^(6/5) - 1)/nu in time from the peak is good at -100..-20 M
  (spread 3-17%).
- **Merger features sit at a fixed ppBHPT-clock offset from the ppBHPT peak, independent of q:** steepest beta rise at
  t_peak - 6 m1 (chi1 ≤ -0.4) to t_peak - 10..11 m1 (chi1 ≥ 0) for every q.  So a merger window in ppBHPT time from the
  ppBHPT peak (m1 units) is the natural coordinate; q enters only through the size of the bump.

**Pass 1 (S1, chi1 = 0, 12 q; `exploration/p1_s1_chi0.py`; job 3491485 running at 2026-10-06 ~11:00).**
- Built: `exploration/sf1pa.py`, a port of WaSABI 1PAT1 (F0, F1 from the 1SF/2SF fluxes and the first-law E_SF).
  Check: the ppBHPT's measured chirp equals the WaSABI 0PA chirp eps F0 (eps = 1/q) to 4 digits at q = 3.
- Method: equal-phase map ODE between the ppBHPT's own chirp and the binary's 1PAT1 chirp -> beta(Omega_pp), zero
  parameters (gauge constant from the local fixed point; fixed point at q = 3 is 0.703 vs X1^(6/5) = 0.708).
  Arms on the inspiral error E_I (NR up to peak - 150 M): leader inspiral branch, S1, 0PA-only, S1 + one gauge coefficient.
- **Result (job 3491485, 24 s).**  E_I = inspiral error on NR [-5000.1, peak - 150 M], own t0/phi0, chi1 = 0, 12 q,
  mean / median / max:
  - leader inspiral branch alone (S = 0): 7.93e-03 / 8.48e-03 / 9.72e-03
  - **S1, zero parameters: 2.90e-03 / 2.84e-03 / 4.93e-03** (better than the leader branch at 11 of 12 q; worst at q = 3)
  - 0PA-only map (F1 = 0): 2.51e-01 -- the 1PA (2SF flux + first-law) term is essential
  - **S1 + one global gauge coefficient (beta *= 1 + c nu x^(5/2), c = 55.3): 7.96e-04 / 5.87e-04 / 2.01e-03**
  - checks: measured NR chirp / 1PAT1 = 0.95-1.05 for x = 0.07-0.11 at every q; measured ppBHPT chirp / 0PA = 1.00 there.
- Caveats: the leader's branch alone is not the leader (its never-off switch mixes the merger branch into the inspiral),
  so the fair comparison is the full leader on the same E_I -- do that first in pass 2.  c = 55 is not small (about 5% of
  beta at x = 0.12, nu = 0.19), so the "gauge" term may be a real missing x^(5/2) piece; check whether c varies with q.
  E_S1c is worst at q = 3 (2.0e-3) and best near q = 6 (2.8e-4): a q-trend remains.
- NEXT (pass 2): (a) the full leader's E_I on the same window; (b) c per q, to see whether the gauge term is universal;
  (c) S1 with the derived SO factor at chi1 ≠ 0 (pp chirp is the Kerr one already; NR side 1PAT1 is Schwarzschild);
  then the first full form: S1 inspiral -> merger window in ppBHPT time from the ppBHPT peak -> beta_r + B5 alpha; S18.

**Pass 2 (2026-10-06, `exploration/p2_s1_spin.py`, smoke tests on the login node; the full grid was not run).**
- Spin version of S1: binary chirp = nu (ppBHPT Kerr chirp / eps) (1 + nu F1/F0)_Schw (1 - chi1 D x^(3/2)),
  D = [113 (X1^2 - 1) + 75 nu]/12.  Analytic check: at leading order this map gives exactly the installed kappa_SO,
  delta ln beta = (3/2) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1] (the X1^(6/5) comes from evaluating the
  ppBHPT chirp at Omega_pp = y w, y ≈ X1^(6/5)).  So the installed SO factor IS the leading term of this map.
- **The fair comparison kills S1.**  Full installed models on the same E_I (chi1 = 0): leader 3.5e-4 / 1.1e-4 / 7.9e-5,
  flux_anc_spin_with_orbit 5.7e-4 / 1.1e-4 / 8.1e-5 at q = 3 / 5 / 8, against S1 4.9e-3 / 2.7e-3 / 1.0e-3 and S1 + gauge
  2.0e-3 / 3.9e-4 / 6.5e-4.  At chi1 = +0.6, q = 5: installed 9e-5, S1 0.23 (constructed chirp off by 5-27%).
- Reading: 1PAT1's chirp is good to about ±5% at x = 0.07-0.11 here, which is too coarse for 1e-4 over thousands of
  cycles; at spin, a 1.5PN SO correction to the identified chirp is far too coarse.  S1 stays as the *explanation* of
  the leading inspiral structure (X1^(6/5), the SO factor, a nu-ordered drift) for §8, not as a model component.
- **Key consequence: the installed models' inspiral errors are ~1e-4, their totals ~1e-3.  The budget is in
  merger-ringdown.**  Pass 3 measures where exactly (band decomposition of E) before designing anything.

**Pass 3a (2026-10-06, job 3491496, `exploration/p3_where.py`): where the installed models lose.**  Full-window E,
full-window alignment, decomposed exactly by NR-time band (share of the T96-mean E / share of power):
inspiral [-5000, -150] 27-31% / 86%; [-150, -15] 26-28% / 9.5%; [-15, +15] 18-23% / 3.9%;
**late ringdown [+15, +100] 22-24% / 0.5%**, with rms amplitude error 19-26% and phase error 0.2-0.3 rad there.
Spin pattern (leader): the ringdown share is largest at chi1 = +0.4..+0.6 (31-39%) and -0.6 (29%); the [-150, -50]
band dominates at chi1 = 0, +0.2 (28-30%).
-> Pass 3: anchored ringdown (beta_r exact, alpha with the B5 slope) blended in after the ppBHPT peak on top of the
frozen flux_anc_spin_with_orbit; only the blend times s1, s2 fitted (S18).  Job 3491508.

**Pass 3 (job 3491508, `exploration/p3_rd.py`).**  ln beta blended to ln beta_r over s = t - t_peak,pp in [s1, s2].
- beta anchor only (RDb): s1 = -9.6, s2 = 33.0 m1; T96 mean 1.023e-3 / median 8.30e-4 / max 6.59e-3, against the frozen
  base's 1.092e-3 / 8.86e-4 / 6.53e-3 (-6% / -6% / +1%).  Column medians: -0.6 1.40e-3, -0.4 6.9e-4, -0.2 4.0e-4,
  0 4.3e-4, +0.2 6.1e-4, +0.4 8.3e-4, +0.6 8.9e-4, +0.8 1.80e-3.
- + B5 alpha (RD): no gain (1.028e-3); the fit collapses the blend to s1 ≈ s2 ≈ 11 m1.  The B5 slope is right
  (pass 0) but the ringdown amplitude LEVEL, not its slope, is what costs.  Parked.
- Uniformity is the problem: the chi1 = +0.8 and -0.6 columns are 3-4x the chi1 = 0 column.  At -0.6 the measured ringdown
  beta is 8% below beta_r (pass 0), so pass 4 takes the ppBHPT side of the anchor from the ppBHPT's own measured ringdown
  frequency (zero parameters, exact for the input).

**Pass 4 (job 3491582, `exploration/p4_ring_bump.py`; log in `exploration/p3/p4rb.3491582.log`).**  Negative.
- A, ppBHPT side of the anchor = the ppBHPT's own measured ringdown frequency: T96 1.045e-3 / 7.98e-4 / 6.52e-3.  The
  median improves but the mean does not, and the chi1 = -0.6 column gets worse (1.56e-3), so the retrograde problem is
  not the anchor value.
- B, + a merger bump nu (b0 + b1 chi1) G((s - sc)/w) on beta: mean unchanged (1.045e-3), max 4.79e-3 better, median
  9.33e-4 worse.  It moves error between columns.
- **Key fact (fig10 study, q = 4, per-case refits of each form):** flux_anc_spin_with_orbit is within 1.05-1.15x of its
  own form's floor; that floor is non-uniform (3.6e-4 at chi1 = -0.2, 1.0e-3 at +0.4).  The pn_anchored form's floor is
  2-3x lower at chi1 ≥ 0 (2.9e-4 at 0).  So no refit can rescue the flux form; tweaks on a frozen base cap at ~6%.
  -> Pass 5: measure floors of candidate forms first (cheap: a full-resolution error costs 0.2 s).

**Pass 5 (job 3491645, `exploration/p5_floor.py`): per-case floors on S18** (mean / median / max):
installed flux_anc_spin_with_orbit 1.61e-3 / 1.28e-3 / 6.53e-3; FLUX floor (same form, 4 per case) 7.84e-4 / 6.93e-4 /
2.53e-3; **M2 floor (FLUX + ln-blend on s = t - t_peak,pp to ln beta_r + rho and ln alpha_r with the B5 slope; 8 per case)
4.43e-4 / 2.37e-4 / 1.82e-3.**  M2 is 2.5-5x below the FLUX floor at chi1 = 0 .. +0.6 (e.g. q = 5, +0.6: 7.0e-4 -> 1.6e-4;
q = 8, 0: 1.6e-4 -> 9.3e-5).  The spin edges stay hard: +0.8 at 0.9-1.8e-3, -0.6 at 6.6-7.2e-4.  Typical M2 parameters:
s1 -50..+14, s2 13..54 m1, rho +0.02..+0.06 (beta's ringdown level 2-6% above beta_r, rising with spin).  Three fits
(chi1 = 0 at q = 3, 5; -0.6 at q = 8) escaped to degenerate whole-waveform blends (s2 > 800) -- pass 5b bounds them.

**Check-in 2 (2026-10-07).**  User: keep going; do both V5 and V6 and see what each buys; try every idea in §4 and
anything else; no crazy coefficient counts.  User follow-up: no cap while exploring, prune to ≤ 20 at the end; q = 2.5 must be < 1e-2 (gate L), q = 2 reported.
Budget 300 SU (~80 used), stop on the §7 conditions.
Plan, in order:
1. Pass 15: ablation of V5 and V6 (which of the 32 pay for themselves).
2. Spend the freed coefficients on the spin edges with physical variables instead of chi1 polynomials (P1: chi_f,
   X1 chi1, or the remnant's w220 ratio) for the merger-ringdown terms and the shape amplitude.
3. B2: blend window in the scaled transition coordinate (eps^(2/5) width) to replace the 4 window coefficients.
4. Second principal shape for the spin edges only if 1-3 leave room under the cap.
5. Then §8 (ablations already in hand from 1) and the final plots.

**Pass 15 (V6), job 3494199.**  V6r = V6 with flux_A1, flux_b3, flux_c1, flux_s1, flux_s4, l1, l2, l3, pa3, r2, t2b = 0:
alpha_PP = X1^(6/5)(1 + 0.056 nu), a_F = -2.03 nu; the merger alpha level is one constant (l0 = -0.109, times nu/nu5);
window s1 = -106.9 + 71.8 chi1, s2 = +36.0 m1.  21 coefficients.  The mean improved on dropping (2.74 -> 2.69e-4): the 32c
fit was not at its optimum (Powell budget), so drops partly act as a better-conditioned refit.  Next: pass 16 (B2 window,
queued) and the V5 ablation; then the spin edges (-0.6, +0.8 columns at 2.3-2.9x the median).

**Check-in 3 (2026-10-07 ~22:50).**  Leader V7-S3 (20c; `p23/p23_Z.json`; model = p23_s3.py Z on p18/p15 machinery):
T96 2.42e-4 / 1.58e-4 / 1.40e-3, H270 median 1.69e-4 (max 7.0e-4), q = 2.5 max 5.5e-3, q = 2 max 2.7e-2.
Gates: G1, G2, V, P, L, K pass; G3 median yes, max no (1.40e-3); **U fails** (cols -0.6 2.5x, +0.8 2.8x; rows q = 3 1.9x,
q = 8 2.0x).  Where the edges lose (pass 19): +0.8 and q = 3, 8 in the inspiral (regression: per-case inspiral terms halve
them but do not regress onto PN-like terms, pass 21); -0.6 in the late ringdown (framework floor 1.2e-4 there).  Not
fixed by: OT-scaled window (16), second shapes (18), PN inspiral terms (21), objective weighting (24).
Open for the user: (1) relax U, or count it on the interior; (2) approve E1 (ppBHPT at an effective spin, screen); (3) go
to §8 + final plots with V7-S3.  ~90 SU used.

**Check-in 3 answer and pass 25-26 plan (2026-10-08).**  User: the rise past q = 5 should go (compare Rink et al.); aim
below 20 coefficients; keep exploring.  Rink et al. 2024 (arXiv:2407.18319 source read 2026-10-08).  **Correction to a first claim made in chat:** their Fig. 11
(heatmap, per-point calibration error vs NRHybSur3dq8, each point its own constant alpha, beta; same E with the 1/2) does
NOT have the U: at chi1 >= 0 it falls monotonically with q (chi1 = +0.8: ~1.6e-2 at q = 3 to ~4e-3 at q = 8, 10), at
chi1 < 0 it is nearly flat with a shallow minimum near q = 5-6 (Fig. 9a the same).  The U appeared only when this repo
scored their PUBLISHED REGRESSED model (12c, 1/q polynomial, `spin_dep_split/results/rink_scores.json`): row medians
42 / 17.4 / 27.4 e-4 at q = 3 / 5 / 8.  Same point, two routes: q = 5, +0.8: Fig. 9a 7.6e-3, regressed 7.67e-3 (agree);
q = 8, +0.8: 4.8e-3 vs 1.37e-2; q = 8, 0: ~5e-4 vs 1.13e-3.  So their rise past q = 5 is a regression artefact of the
published fit, not physics -- the same conclusion as for V7-S3 below.  (Caveat: their per-point window may differ from
ours; q = 3, +0.8 regressed 1.32e-2 is below the per-point 1.6e-2.)  V7-S3: 3.05 / 1.21 / 3.21 e-4.  **The rise is not an input or reference floor**:
the pass-17 per-case spline floor is flat in q (0.17 / 0.13 / 0.12 e-4 at q = 3 / 5 / 8).  Diagnosis from the pass-17
PCA amplitudes (relative to V6r): the leading residual amplitude is q-flat at fixed chi1 while the model weights it by
nu (1.9x drop from q = 3 to 8), and the second component is monotonic in q with a zero near q = 4.5 (a missing
q-slope, which produces a U).  It is not a stretch or shift of the first component (<= 34% explained).  Hence pass 25
(free nu exponents) and, for coefficients, pass 26 (derived window onset).

**Test set (2026-10-13, user request: H270 was used for selection, so a final comparison needs untouched points).**  `exploration/t100_set.py`: R100 = 100 points uniform in 3 < q < 8, -0.6 < chi1 < 0.8; L20 = 20 points uniform in 2.5 < q < 3 (extrapolation); drawn once (seed kept only in the script); caches `exploration/t100/cache/`.  **SEALED: no fit, prune or model choice may use them; do not copy the points or the seed into notes; never merge them into T96 / H270.  A later final comparison draws a NEW set with a new seed after the models are frozen.**  Scored once per final candidate by `t100_score.py` (stored T96 reproduced on 12 points first); job 3507526.  Looks: (1) 2026-10-13 all 7 table models (p36, p32, v7s3, p38n2, flux_orbit, pn13, rink).
Look 1 result (job 3507526; all 7 reproduced stored T96 at ratio 1.0000), R100 / L20 median / mean / max: p36 D1 1.40e-4 / 1.93e-4 / 9.4e-4, 7.8e-4 / 9.9e-4 / 2.8e-3; p32 1.41e-4 / 2.00e-4 / 1.0e-3, 7.6e-4 / 1.01e-3 / 3.0e-3; V7-S3 1.76e-4 / 2.27e-4 / 9.9e-4, 8.2e-4 / 1.05e-3 / 3.2e-3; p38 N2 1.39e-4 / 1.89e-4 / 9.5e-4, 7.6e-4 / 9.5e-4 / 2.6e-3; flux_orbit 7.88e-4 / 8.61e-4 / 2.3e-3, 2.16e-3 / 2.79e-3 / 8.5e-3; pn13 6.94e-4 / 8.93e-4 / 2.8e-3, 2.26e-3 / 3.09e-3 / 7.6e-3; Rink 1.87e-3 / 2.59e-3 / 9.7e-3, 7.8e-3 / 8.4e-3 / 1.7e-2.  R100 / T96 median: exploration models 1.01-1.11 (no overfit), installed and Rink 0.80-0.89 (T96 weights the grid edges more than a uniform draw does).
**Sealed-set guard (2026-10-13).**  `exploration/test_guard.py` wraps both case builders (S / FS.build_case); imported by p3_rd (so every pass >= 3) and the passes that build cases directly (p0-p6).  Any key within 1e-3 (unit box q/5, chi1/1.4) of a sealed point raises SealedTestPoint; only t100_set.py opens it.  Checked: T96, H270, low-q pass; 360/360 perturbed sealed points refused.
Look 2 (2026-10-15, job 3511788; `t100_split.py`, read-only on spin_dep_split/): split slim__win-2.0 (14c) R100 3.80e-4 / 5.72e-4 / 2.3e-3, L20 1.8e-3 / 2.3e-3 / 4.6e-3; split tline (14c) R100 4.39e-4 / 5.82e-4 / 2.3e-3, L20 3.3e-3 / 3.7e-3 / 6.8e-3 (both reproduced stored T96 at 1.0000).  Pass-36 figures: `exploration/plot_p36.py` -> `exploration/plots_p36/` (calibration chi +0.6, alpha/beta, E vs chi1, train/val/test histogram, E map on T96 + H270).
Plots 2026-10-15: `exploration/plot_hist_common.py` redrew the pn_anchored_spin and flux_anc_spin_with_orbit paper hists (+ test R100 values) and the p36 hist on one x range [-4.5, -2.0], 0.1-dex bins (originals in `exploration/backup_hist_20261015/`); `exploration/p36_dense.py` (job 3511819) -> `plots_p36/p36_mismatch_heatmap_dense` on the 20 x 20 dense grid: median 1.96e-4, max 2.29e-2 (q = 2); interior reproduces H270 at 1.0000.
Diagnostics 2026-10-15 (job 3512975, `exploration/p36_diag.py`, protocols of plot_spin_diagnostics.py / fig10_ourmodels.py; q = 4 caches read from fig10_tend/): plots_p36/p36_{fig10_tend_q4, error_by_band_q4, residual_vs_time_q4, peak_errors_grid}.  T96 reproduced at 1.0000.  Fig. 10 per-case refit (18 free, Powell, <= 25 min): t_end = 75 M 3e-5..2.2e-4; t_end <= 0 drops to ~1e-5 for chi1 < 0 but stays 3-7e-5 for chi1 >= 0 down to -50 M (a step at chi1 = 0; the refit is local, so treat as an upper bound).  Bands: chi1 = -0.6, -0.5 are 60-70% ringdown.  Peaks: chi1 = +0.8 peak 4-8 M early at every q; amplitude +5.9% at (3, +0.8).
**Correction (2026-10-15, while writing `exploration/scaling_p36.md`):** gate P does NOT pass for pass 32 onward. Its "alpha, beta -> 1 as nu -> 0" clause fails: the nu5/nu spin terms (r1, r3, eb1, pb*, s3) do not vanish as nu -> 0, pb0 phi_A survives even at chi1 = 0, and pass 36 kappa_RD -> gamma_meas - gamma_Kerr(chi1) != 0. Check-in 4 listed P as passing; that was wrong. Positivity, monotone tau and "spin terms vanish at chi1 = 0" do hold. Irrelevant for q <= 8; it matters for any q > 8 extrapolation.
| 39 | 2026-10-15 | restore gate P's nu -> 0 limit on D1: every q-flat term (s2, s3, r1, r3, pb0, pb1, pb2, eb1) and the measured-decay shift of kappa_RD x g(nu) = 1 - exp(-(nu/nu_c)^2); L4 nu_c = 0.04 (g(q = 8) = 0.998), L6 nu_c = 0.06 (g(8) = 0.934); 18c refit from D1 (`p39_numlimit.py`) | 18 | – | L4 1.99e-4 / 1.37e-4 / 8.55e-4; L6 2.00e-4 / 1.37e-4 / 9.28e-4 | H270 median L4 1.35e-4, L6 1.34e-4; rows / cols as D1 to 2% | q = 2.5 max 4.6e-3 both (L passes); **P passes by construction** (every term that survived nu -> 0 now carries g ~ (nu/nu_c)^2) | g = 1 reproduces D1 at 1.0000; g(q) for L4: 0.998 at q = 8, 0.986 at 10, 0.88 at 15, 0.46 at 30, 0.06 at 100 | 3513375 | **L4 = D1 with gate P restored at no cost** (mean +0.0%, median +0.8%, max +0.7%). L6 costs max +9%. nu_c is not constrained by q in [3, 8]; the turnover scale is an assumption |

Test-set look 3 (2026-10-18, job 3518540): pass 39 L4 R100 1.400e-4 / 1.937e-4 / 9.49e-4, L20 7.82e-4 / 9.92e-4 / 2.84e-3 (T96 reproduced at 1.0000). **User adopted pass 39 L4 as the leader (2026-10-18).** Plots: `plot_p39.py`, `p39_dense.py`, `p39_diag.py` -> `plots_p39/` (job 3518542).
| 40 | 2026-10-18 | closed-form R2(x) in pass 39 L4 (user: interpretability), three forms fitted to the 22-point 2SF table, L4 coefficients FROZEN: O1 (55/21) x (1-6x)(1 + 3.691 x + 30.255 x^2) [2 fitted]; O2 same with the x term fixed by 2PN (4.768) [1 fitted, 21.68]; O3 R2_PN3.5 (1-6x)/(1 - 6.463 x + 7.656 x^2), R2_PN3.5 = 2 Re[N/H0] from Faye+ 2012 Eq. 6.5 (verified against the paper) [2 fitted]; each with x on the table range [0.04, 0.16] (t) and on [0.005, 1/6] (n, 0 at the ISCO) | 18 (+0) | – | natural range, T96 mean / median / max: O1n 1.988 / 1.375 / 8.54 e-4, O2n 1.995 / 1.381 / 8.54 e-4, O3n 1.989 / 1.376 / 8.54 e-4 (L4 table 1.987 / 1.368 / 8.55) | H270 median O1n 1.352, O2n 1.359, O3n 1.353 e-4 (L4 1.351) | q = 2.5 max 4.61e-3 all | table reproduces L4 at 1.0000; fit errors 0.0020 / 0.0039 / 0.0034; PN 3.5 alone matches the table to 0.5-1% at x = 0.04-0.05, 8% at 0.10 | 3518561 | **all three are free (< 1%, no refit needed).** O1 is the most accurate, O2 the most economical (1 number), O3 the most physical (full PN 3.5 + 2 strong-field numbers) |

O3' (2026-10-18): R2_PN3.5 (1 - 6x) / [(1 - 6x) + c x^3], c = 30.69: strong-field factor = 1 + O(x^3), so exact through 3.5PN; max error 0.0024 (0.8% for x < 0.14). The first O3 (denominator 1 - 6.463 x + 7.656 x^2) was NOT PN-exact beyond 1PN (its factor is 1 + 0.46 x + ...). **User chose O2 for the model (Occam: one line, one fitted number) and asked for O3' to be mentioned.** Pass 40 = L4 + O2 (natural range), no refit; doc `exploration/scaling_p40.md`.
Test-set look 4 (2026-10-18, job 3518626): pass 40 (L4 + O2) R100 1.410e-4 / 1.947e-4 / 9.50e-4, L20 7.85e-4 / 9.93e-4 / 2.83e-3 (pass 39: 1.400e-4 / 1.937e-4 / 9.49e-4). T96 reproduced at 1.0000.

**User rule (2026-10-18):** a piece replaced by a direct theory sub-model (derived, or fitted OFFLINE to theory data only, e.g. Wardell's R2, the 1.5PN kappa_SO) does not count against the coefficient budget. Candidates for future passes: (1) inspiral beta level + drift (b, b2, P0, P1) from the 1PA phasing (Wardell/Warburton F1, F2 + first-law binding energy; idea B3 / the open F2 check); (2) r1, r3 from the MEASURED ppBHPT ringdown frequency (as gamma_meas did for the decay, pass 36); (3) t1a, t1b from Ori–Thorne transition timing with the theory exponent (pass 16 fitted p = 0.74); (4) the shape terms from the universal OT / Küchler transition solution (section 4.1). Pass 41 (job 3519411): Hessian sigma / correlations of the pass-40 coefficients.
| 42a | 2026-10-18 | (diagnostic, login-node arithmetic) do r1, r3 come from the ppBHPT ringdown FREQUENCY? measured omega over s in [15, 40] m1 / Kerr w220(chi1), q = 3, 5, 8 | 0 | – | ratio 0.99 / 0.98 / 0.99 / 1.00 / 1.01 / 1.00 / 1.00 / 0.98 at chi1 = -0.6 ... +0.8 (q scatter up to 3% when retrograde) vs the model offsets exp(r1 chi + r3 chi^2) = 0.915 / 0.951 / 0.979 / 1 / 1.012 / 1.015 / 1.009 / 0.994 | | | – | **negative: candidate 2 dropped.** ppBHPT rings at the Kerr frequency to 1-2%; r1, r3 are a plunge-merger clock correction (they act across the blend, s1 .. 0), not a ringdown frequency effect |
| 41 | 2026-10-19 | Hessian sigma of the pass-40 coefficients (collaborators): finite differences of the T96 fit objective, steps for a 1% rise, repeated at h/2; cov = 2 f0 H^-1/(N - p) | 18 | – | H positive definite; Newton gain 0.19% (pass 40 is at its optimum) | | | 3519411 | **nothing removable: every |theta/sigma| >= 2.9** (pb2 2.9, eb1 3.1, ea0 3.3 weakest); predicted drop costs match pass 34 to 1-15% for the weakest eight. b-b2 -0.97, P0-P1 -0.94: the inspiral clock (b, b2, P0, P1) carries about 2 effective dof -> strongest case for the 1PA theory replacement (candidate 1). s3-r1 +0.79. Table in scaling_p40.md section 10 |
| 43a | 2026-10-19 | (diagnostic, login-node arithmetic; `exploration/sf1pa_clock.py`) the 1PA THEORY CLOCK for the inspiral beta, chi1 = 0: WaSABI 1PAT1 forcing (dOmega/dt = nu F0 + nu^2 F1 on M; ppBHPT = eps F0 on m1), pairing by equal accumulated phase, beta = Omega_pp / Omega_NR = (x_pp/x_NR)^(3/2); one constant c (alignment phase) not fixed by theory | 0 (+c) | – | vs pass-40's fitted beta_I at chi1 = 0, best c per q, fit on x_NR < 0.12: rms 1.6e-3 (q = 3), 9e-4 (5), 2.7e-4 (8); pointwise within 1e-3 for s < -1000 m1 at q = 3 and within 2e-4 for s < -600 at q = 8 | | | – | **the 1PA theory reproduces the fitted inspiral clock in the adiabatic regime with one constant** (c = 1.2 / 3.5 / 5.1 at q = 3 / 5 / 8, smooth). It departs in the last ~300-400 m1 before the peak (+2 to +12% at s = -250 .. -150), where x_NR -> ISCO and 1PA breaks down (nu F1/F0 -> -1) -- i.e. exactly the Ori–Thorne transition region (T ~ -1). Next (pass 43): beta_I = theory clock with c(nu) = c0 + c1 (nu5/nu - 1) (2c) replacing b, b2, P0, P1 (4c), blended into the merger branch from the OT onset (pass 42 K1); spin terms kept |
| 42 | 2026-10-19 | blend onset from Ori–Thorne theory on pass 40: s1 = s_isco + (T_start - T*) u_OT (ppBHPT's own ISCO crossing and OT unit); K1 T_start = -1 fixed (0 onset coefficients), K2 T_start fitted; onset guard widened to -1000 (reproduces pass 40 at 1.0000) | 16 / 17 | – | K1 2.14e-4 / 1.46e-4 / 1.01e-3; K2 2.10e-4 / 1.43e-4 / 1.05e-3 (pass 40: 2.00e-4 / 1.38e-4 / 8.54e-4) | H270 median K1 1.37e-4, K2 1.41e-4 (pass 40 1.36e-4); H270 max K1 5.0e-4 (pass 40 5.7e-4) | rows (q = 3 / 5 / 8) K1 3.17 / 1.12 / 2.18 e-4: the q = 8 row rises again (q8/q5 1.95 vs 1.45); q = 2.5 max 4.37e-3 (better) | K2 fitted T_start = -0.66 | 3519419 | **K1: -2 coefficients for +7.7% mean, +5.5% median, +18% max** (3.9% per coefficient, below the 5% K bar, so by gate K the two fitted onset coefficients did not pay their way; but G3's max <= 1e-3 is now marginal at 1.007e-3). K2 buys back 2% for 1c. Decision deferred to pass 43 (H2 = 1PA clock + K1 onset) |
| 43 | 2026-10-19 | the 1PA theory clock as beta_I (replacing b, b2, P0, P1 by c(nu) = c0 + c1 (nu/nu5 - 1)), frozen where abs(nu F1/F0) >= 0.25; spin terms kept; H1 fitted onset (16c), H2 OT onset T = -1 (14c) | 16 / 14 | – | H1 1.92e-3 / 1.83e-3 / 6.5e-3; H2 2.02e-3 / 2.05e-3 / 8.1e-3 | H270 median 1.6 / 1.75 e-3 | q = 2.5 max 2.4e-2 / 1.7e-2 | c0 +3.43, c1 -5.96 (as pass 43a) | 3519561 | **negative (10x worse).** The theory clock agrees with the fitted one to < 1e-3 only for s < -1000 m1; from there it rises (+1% by s ~ -600, +5% by -300 .. -150) while the fitted clock stays flat, and a 1% clock error over hundreds of M is a large phase error. Positive spins worst (2-3e-3): a Schwarzschild clock cannot follow the Kerr ppBHPT inspiral. The 1PA pairing breaks down well before the ISCO (nu F1/F0 ~ -0.1 at the departure). Kept as a physics result: the fitted inspiral clock IS the 1PA clock in the adiabatic regime (figure `plots_p39/p43_1pa_clock_vs_fit`). Not a coefficient saving |
| 42n | 2026-10-19 | pass 42 K1 with the OT small parameter eta = nu (u_OT ~ nu^(-1/5)) | 16 | – | 2.17e-4 / 1.49e-4 / 9.8e-4 | H270 median 1.39e-4 | rows q = 3 / 5 / 8 3.15 / 1.14 / 2.21 e-4 (q8/q5 1.94, as K1) | | 3519707 | **negative: the onset stays fitted, pass 40 stays the leader.** Any OT onset moves earlier at large q (u_OT grows with q), while the data want an onset fixed in s; the q-rise returns either way. By the same token the shape terms (fixed in s) should not be re-expressed in OT time: candidate 4 dropped. Next: stand-alone pass-40 generator |
Stand-alone generator (2026-10-19): `exploration/bhpt_spin_calib_p40.py` (one module: numpy, scipy, qnm, surfinBH + the p12 / p17 shape jsons) reproduces the pass-40 chain to 2.2e-9 relative in alpha, beta on 5 cases x {fit, full} (`verify()`). Documented in scaling_p40.md section 9.
| 44 | 2026-10-19 | pass 40 beyond the training range: q = 9, 10 x the 8 T96 spins (NRHybSur3dq8 itself extrapolated above q = 8; caches exploration/p44/cache) | 18 | – | q = 9 median 2.8e-4 max 4.0e-3; q = 10 median 3.3e-4 max 1.2e-2 (q = 8: 1.7e-4 / 8.5e-4) | | per chi1 at q = 8 / 9 / 10: -0.6 3.2 / 3.6 / 4.1 e-4; 0 1.5 / 1.7 / 1.8 e-4; +0.4 0.6 / 0.8 / 1.5 e-4; +0.6 1.0 / 3.5 / 11.6 e-4; +0.8 8.5 / 40 / 119 e-4 | T96 reproduced at 1.0000 | 3519741 | **graceful for chi1 <= +0.4 (<= 4e-4 at q = 10); blows up at chi1 >= +0.6.** Pass 44b (job 3519895) scores the installed models and Rink on the same points to separate model from NR-surrogate extrapolation |
| 44b | 2026-10-19 | the installed models and Rink on the pass-44 points | – | – | at (q, chi1) = (10, +0.8): pass 40 1.2e-2, flux_orbit 2.7e-2, pn13 2.5e-2, Rink 3.6e-2; (10, +0.6): 1.2e-3 / 4.4e-3 / 2.6e-3 / 5.6e-3; (10, 0): 1.8e-4 / 3.0e-4 / 5.9e-4 / 2.1e-3 | | | | 3519895 | **every model blows up at chi1 >= +0.6 above q = 8, Rink (calibrated on NR to q = 10) included**, so the growth is at least partly NRHybSur3dq8's own extrapolation; pass 40 is 2-7x better than the others at every q = 9, 10 point |

**Check-in 5 (2026-10-19).**  Leader pass 40 (18c; scaling_p40.md; stand-alone `bhpt_spin_calib_p40.py`, verified to 2e-9).  Since check-in 4: gate P restored (39), closed-form R2 (40), sealed test set + guard, Hessian sigma (41: nothing removable; inspiral clock b, b2, P0, P1 ~ 2 effective dof), theory replacements tried: ringdown frequency (42a, no), OT onset (42 / 42n: -2c for +8% and the q-rise back; not adopted), 1PA clock (43: 10x worse in the model; agrees with the fit for s < -1000 m1 -- kept as a physics result), q = 9, 10 (44 / 44b).  Open for the user: (1) the uniformity gate U (chi1 = -0.6 / +0.8 columns, q = 3 row); (2) whether to adopt the OT onset (16c, +8%) as the lean variant; (3) the remaining structural idea for the spin edges, if any.
| 45a | 2026-10-19 | (diagnostic, offline) are the spin-edge residuals a spin-dependent TIME STRETCH of the merger structure? pass-35 per-case residual splines (q-averaged) regressed on the stretch mode s d(ln beta, ln alpha)/ds of the pass-40 model at q = 5 | 0 | – | stretch explains 28% of dlnB at chi1 = -0.6, <= 10% elsewhere, 0.5% / 2% (dlnB / dlnA) at +0.8; a rank-1 collapse of the residuals in s / T_isco(chi1), s / tau_QNM, s / (1/w220) gains <= 4.5% over plain s | | | – | **negative: no rescaled clock (ISCO period, QNM time) fixes the edges** -- the edge residuals are not a stretch of the model. Not run as a pass |
Test-set look 5 (2026-10-19, job 3520131): lean variant k1 (pass 42 K1, 16c) R100 1.475e-4 / 2.046e-4 / 9.11e-4, L20 7.20e-4 / 9.20e-4 / 2.65e-3 (pass 40: R100 1.410e-4 / 1.947e-4 / 9.50e-4, L20 7.85e-4 / 9.93e-4 / 2.83e-3). T96 reproduced at 1.0000. User likes the lean variant; CV of both running (pass 45).
| 45 | 2026-10-19 | 4-fold CV of pass 40 on T96 (fold (iq + ic) mod 4; 18c refit on 72, scored on 24; start = full fit) | 18 | – | pooled held-out 1.40e-4 / 2.02e-4 / 9.6e-4 (median / mean / max) vs full fit 1.38e-4 / 2.00e-4 / 8.5e-4 | | | coefficient fold spread (jackknife-scaled) = 0.04-0.25 x the pass-41 Hessian sigma | 3520135, 3520136 | **pass 40 generalises: CV +1% median, +1% mean, +12% max; agrees with R100.** Lean-variant CV running (3520230, 3520369). scaling_p40.md section 11 |
| 45k | 2026-10-19 | 4-fold CV of the lean variant (pass 42 K1, 16c; same folds) | 16 | – | pooled held-out 1.45e-4 / 2.16e-4 / 1.10e-3 (pass 40 CV: 1.40e-4 / 2.02e-4 / 9.6e-4) | | | fold held-out medians 1.44 / 1.41 / 1.54 / 1.40 e-4 vs training 1.46 / 1.49 / 1.43 / 1.46 | 3520230, 3520369 | **lean vs pass 40, selection-free: +4% median, +7% mean, +15% max** (R100: +5% / +5% / -4%; L20: -8% / -7% / -6%). Both generalise; the lean variant costs ~5% in range for 2 fewer coefficients and is better below q = 3 |
| 46 | 2026-10-22 | closed-form inputs on pass 40 (`closedform.py`, constants fitted offline: `closedform_coeffs.json`): Q Kerr QNM = 6-term series in sqrt(1 - chi) (9e-5 / 3e-4); R remnant M_f, chi_f = ISCO-energy-weighted / 16-term polynomials fitted to NRSur3dq8Remnant (5e-4 / 7e-4 in range); G measured ringdown decay = 8-constant form in (chi1, nu) (1.1%); frozen, then refit | 18 | – | Cq 1.993 / 1.372 / 8.52; Cqr 1.997 / 1.379 / 8.45; Call 1.997 / 1.380 / 8.44; Call_r 1.996 / 1.380 / 8.54 e-4 (mean / median / max; pass 40 1.995 / 1.381 / 8.54) | H270 median 1.355 .. 1.362 e-4 | q2.5 max 4.58-4.61e-3 | | 3527225 | **all three closed forms are free (< 0.3%, no refit needed)**: the qnm package, NRSur3dq8Remnant and the per-waveform decay fit are replaced. Pass 47: closed-form shapes |
| 47 | 2026-10-22 | closed-form shapes (tail + 4 sech bumps, 14 constants each, fitted pointwise to the splines) on pass 46 Call | 18 | – | S0 frozen 2.17e-3 / 9.0e-4 / 1.15e-2 (10x worse); S0r refit 2.86e-4 / 1.93e-4 / 1.37e-3 (mean / median / max; pass 46: 2.00 / 1.38 / 8.4 e-4) | H270 median S0r 1.91e-4 | q2.5 max 5.5e-3 | | 3528258 | **negative as fitted, and the cause is understood:** the beta shapes act on the INTEGRATED clock, and the pointwise fits (tail weighted 0.3) left a 5-6 m1 cumulative error in int phi ds over the inspiral. Integral-weighted refits cut it to 0.3 m1 (sech) and are being pushed further with sech^2 (tanh-step derivative) forms; the alpha shapes are pointwise and less critical |
| 47b | 2026-10-22 | clock-preserving closed-form BETA shapes (tail + 6 sech^2 = tanh-step derivatives, fitted to the value AND its cumulative integral; `fit_shapes_cf2.py`, `shapes_closedform_v2.json`; cumulative clock error 0.042 / 0.028 m1 for phi_B / psi_B, was 6.0 / 5.1) + the pointwise alpha shapes, on pass 46 Call, field by field | 18 | – | frozen: alpha only 2.002 / 1.366 / 8.41; beta only 2.011 / 1.399 / 7.91; both 2.016 / 1.367 / 7.92; both refit 2.006 / 1.400 / 8.34 e-4 (mean / median / max; pass 40 1.995 / 1.381 / 8.54) | H270 median both 1.370e-4 (pass 40 1.359) | q2.5 max 4.42e-3 (4.61) | | 3528360 | **the whole model is now closed form at < 1% cost** (mean +1.1%, median -1.0%, max -7%), pass-40 coefficients unchanged (refit buys nothing). -> pass 48 = pass 40 coefficients with every numerical input closed form |
| 48 | 2026-10-22 | the FULLY CLOSED-FORM model (R2 O2, closed-form QNM / remnant / ringdown decay, closed-form shapes), coefficients frozen: m48 = pass 40's 18; lean48 = pass 42 K1's 16 + OT onset; stand-alone `bhpt_spin_calib_p48.py` (numpy only; = chain to 2.2e-9) | 18 / 16 | – | T96 (median / mean / max) m48 1.367 / 2.016 / 7.92 e-4; lean48 1.418 / 2.154 / 9.04 e-4 | H270 median m48 1.370e-4, lean48 1.342e-4 | q2.5 max 4.42e-3 / 4.18e-3; q2 max 2.24e-2 / 2.10e-2 | **sealed test (look 6, 7): m48 R100 1.435 / 1.963 / 9.55 e-4, L20 median 7.75e-4; lean48 R100 1.487 / 2.054 / 9.15 e-4, L20 6.73e-4** (pass 40: 1.410 / 1.947 / 9.50, L20 7.85; lean: 1.475 / 2.046 / 9.11, L20 7.20) | 3528452 | **closed form costs < 2% anywhere** (m48 vs 40 on R100: +1.7% / +0.8% / +0.5%; lean48 vs lean +0.8% / +0.4% / +0.4%, and lean48 is better than lean on T96 (-3% median, -10% max) and L20 (-7%)). Gate status unchanged (U fails at the spin edges) |
| 49 | 2026-10-26 | greedy backward pruning of pass 48 lean (16c) for the Pareto front: at each size try dropping the 5 Hessian-cheapest coefficients (2.5-min refit each, selection on T96 fit-mean only), keep the cheapest, refit 10 min (`p49_prune.py`, `p49/p49.json`). Dropped in order pb2, eb1, pb1, ea0 | 15 / 14 / 13 / 12 | – | T96 (median / mean / max) 15c 1.741 / 2.428 / 10.57; 14c 1.912 / 2.720 / 10.47; 13c 1.965 / 2.993 / 10.55; 12c 2.388 / 3.344 / 10.76 e-4 | H270 median 1.695 / 1.865 / 1.944 / 2.191 e-4 | q2.5 max 4.37 / 4.76 / 4.59 / 4.64 e-3 | **sealed test (one look each, `p49_test.py`): R100 median / mean / max 15c 1.780 / 2.299 / 8.61; 14c 2.066 / 2.602 / 8.44; 13c 2.098 / 2.757 / 9.04; 12c 2.295 / 3.005 / 9.34 e-4; L20 median 7.08 / 6.53 / 7.22 / 7.76 e-4.** Every drop costs 3-23% (no free coefficient left: the 16c model is tight); all four sit on the Pareto front and dominate the split models (14c, R100 3.8e-4) by ~2x; 12c is still 3.4x better than flux+orbit (9c). Fronts: `pareto/pareto_front.png` | 3535104, 3535339 |
| 49b | 2026-10-26 | the pass 49 pruning on pass 40 lean instead (qnm package, NRSur3dq8Remnant and the spline shapes called directly; `p49_prune.py ... p40lean`, `p49b/p49.json`) | 15 / 14 / 13 / 12 | – | same drop order (pb2, eb1, pb1, ea0), same trial costs to +-1%; T96 (median / mean / max) 15c 1.772 / 2.433 / 10.78; 14c 1.958 / 2.729 / 10.56; 13c 2.006 / 3.002 / 10.40; 12c 2.419 / 3.356 / 10.73 e-4 | H270 median 1.707 / 1.863 / 1.951 / 2.217 e-4 | q2.5 max 4.44 / 4.80 / 4.65 / 4.64 e-3 | **sealed test: R100 median / mean / max 15c 1.788 / 2.298 / 8.62; 14c 2.088 / 2.604 / 8.44; 13c 2.154 / 2.758 / 8.89; 12c 2.313 / 3.019 / 9.31 e-4; L20 median 7.19 / 6.53 / 7.28 / 7.75 e-4.** Within 0.5-2.6% of pass 49 at every size: the closed forms neither help nor hurt the pruned models | 3535562 |
| 50a | 2026-10-27 | (diagnostic, login-node arithmetic; `exploration/p50a_iscoanchor.py`) **B4**: does an ISCO-frequency clock ratio beta_isco = M Omega_isco(chi1) / Omega_isco,NR predict the fitted plunge-clock offsets g (r1 chi1 + r3 chi1^2) on ln beta_r (user, 2026-10-27: theory forms toward <= 12c, direct qnm / remnant calls)? NR side A Kerr ISCO of the remnant (M_f, chi_f from NRSur3dq8Remnant), B Kerr ISCO at X1^2 chi1, C B x (1 + 1.2513 nu) (conservative 1SF shift) | 0 | – | fitted offset (49b lean-12) -0.067 / -0.039 / -0.016 / 0 / +0.010 / +0.015 / +0.013 / +0.005 at chi1 = -0.6 .. +0.8: rises then FALLS (peak near +0.4); every candidate (minus its chi1 = 0 value) is monotone and steep (A at q = 5: -0.033 .. +0.115; B, C: -0.094 .. +0.341) and scales with nu, while the fit is q-flat | | | – | **negative: B4 does not replace r1, r3.** Best-scaled candidate leaves rms 0.025 = the offset's own size at every q. Only A on the retrograde side tracks the fit (q = 3: -0.056 / -0.041 / -0.022 vs -0.067 / -0.039 / -0.016); the prograde turnover is not an ISCO effect. With 42a (not the ringdown frequency) and 45a (not a stretch), r1, r3 stay empirical |
| 50b | 2026-10-27 | pass 43 H2 (1PA theory clock replacing b, b2, P0, P1 by c0, c1; OT onset) with the freeze moved to where pass 43a saw the 1PA clock leave the fitted one: H3 |nu F1/F0| >= 0.10, H4 >= 0.05 (`p43_1paclock.py H3|H4 40`) | 14 / 14 | – | T96 (mean / median / max) H3 7.37 / 4.80 / 55.4 e-4; H4 10.09 / 6.04 / 43.7 e-4 (pass 43 H2 at 0.25: 20.2 / 20.5 / 81; pass 40 lean 2.14 / 1.46 / 10.1) | H270 median H3 4.03e-4, H4 5.13e-4 | q = 2.5 max H3 1.42e-2, H4 2.00e-2 (L FAILS) | H3 c0 +3.38, c1 -6.05 | 3537214, 3537215 | **negative, though the freeze was the main defect: 0.25 -> 0.10 is 4x better (median 2.05e-3 -> 4.8e-4), but still 3.3x worse than the fitted clock and fails L.** Earlier freezing (0.05) is worse again, so there is no threshold that works. Worst rows q = 3 (2.0e-3) and q = 8 (7.6e-4), worst column chi1 = +0.8 (1.4e-3): the Schwarzschild 1PA clock cannot follow the Kerr ppBHPT inspiral, and a flat frozen clock cannot carry the late-inspiral drift that P E(t) supplies. The inspiral coefficients b, b2, P0, P1 stay |
| 50c | 2026-10-27 | H5 = pass 50b H3 (1PA clock frozen at |nu F1/F0| >= 0.10, OT onset) + the energy drift P0 E(t) kept free (c0, c1, P0 replace b, b2, P0, P1); warm start from H3 (`p43_1paclock.py H5 40`) | 15 | – | T96 (mean / median / max) 7.22 / 5.13 / 57.7 e-4 (H3 7.37 / 4.80 / 55.4) | H270 median 4.26e-4 | q = 2.5 max 1.36e-2 (L FAILS) | c0 +3.46, c1 -5.99 | 3537342 | **negative: the free drift buys 2% on the mean and costs 7% on the median.** So the 3.3x gap of the theory clock is not a missing drift: the theory clock's SHAPE through the late inspiral is wrong for the Kerr ppBHPT (worst q = 3 row 1.9e-3, chi1 = +0.8 column 1.3e-3). **The 1PA inspiral-clock route is closed** (passes 43, 50b, 50c): it reproduces the fitted clock only in the adiabatic band (pass 43a) |
| 49c | 2026-10-28 | pass 49b continued 12 -> 9 (user, 2026-10-28: apples-to-apples with flux_anc_spin_with_orbit at 9c), 8 trial drops per step (the Hessian ranking was unreliable by n = 12) (`p49_prune.py 9 8 2.5 10 p40lean`) | 11 / 10 / 9 | – | T96 (median / mean / max) 11c 3.10 / 3.79 / 11.7; 10c 3.31 / 4.53 / 14.7; **9c 3.78 / 5.07 / 15.8 e-4** (flux_orbit 9c 8.86 / 10.92 / 65.3) | H270 median 2.76 / 3.02 / **3.51** e-4 (flux_orbit 7.20) | 9c: G1, G2, V (0.93), L (q = 2.5 max 3.4e-3), P pass; G3 fails (max 1.58e-3); U fails (chi1 = -0.6 column 1.23e-3 = 3.3x median, spread 0.70) | dropped pa0, r3, r1; kept b, b2, P0, P1, s2, s3, l0, pa2, pb0 | 3538867 | **lean-9 beats flux_orbit at the same 9 coefficients by 2.3x (T96 median), 2.2x (R100 median 3.50e-4 vs 7.88e-4), 4.1x (T96 max), 2.2x (L20 median 9.7e-4 vs 2.16e-3).** Degradation 12 -> 9 is smooth (+14 / +7 / +14 % per step on the median), no jump back to the single-regime level: the two-regime phase extends to 9c. The plunge-clock offsets r1, r3 went out before the shape terms pa2, pb0. Sealed test scored once per size (selection on T96 fit-mean only); not on the Pareto front until reported |
| 51 | 2026-10-30 | (scoring only; user) (a) Zhong, Han & Sun 2025 (arXiv:2503.04534, LaTeX source Tables II l = 2, III) extracted (2,2) alpha'(q), beta'(q) = 1 + A/q + ... + D/q^4 (8 coef, spin-independent), their literal phase map h = alpha' A_pp(t) exp(-i phi_pp(t) / beta') on the raw ppBHPT grid, applied to the spinning BHPTNRSur2dq1e3 input; beta' as a time scaling as a diagnostic (`exploration/fdm/zhong_score.py`, gate: published Rink through the harness = rink_scores.json 1.00000). (b) frequency-domain (2,2) mismatch, aLIGO zero-detuned high-power PSD, f_low 20 Hz, M = 100 / 200 Msun, maximised over time and template polarisation / phase (2-D quadrature span, exact), mean over 4 signal polarisations, 126 points q = 3..8 (0.25) x chi1 in {-0.6, -0.3, 0, 0.3, 0.6, 0.8} (`fdm/fd_run.py`, `fd_core.py`) | 8 / – | – | Zhong T96 8.52 / 12.2 / 30.9 e-2 (median / mean / max), chi1 = 0 median 1.08e-2; V300 median 7.90e-2; beta'-as-time 0.50 (wrong convention). FD median / max at M = 100: pass 40 4.7e-4 / 6.1e-3, lean-16 4.8e-4 / 6.0e-3, lean-12 8.4e-4 / 7.8e-3, lean-9 1.3e-3 / 7.1e-3, flux_orbit 3.7e-3 / 2.4e-2, Rink 7.3e-3 / 4.0e-2, Zhong 1.5e-2 / 1.9e-1 (M = 200 in fdm/results/fd_summary.json) | **sealed test, one look (zhong): R100 6.12e-2 / 9.92e-2 / 2.91e-1, L20 median 1.12e-1** | self-tests: match(NR, NR) 1e-16; explicit F+ h+ + Fx hx strain = Re[e^{i delta} h22] form to 7 digits (single-mode polarisation = phase); white-PSD mismatch / E 1.1 for the pass-40 models; f_low never raised; every model reproduces its stored T96 at 1.0000 | 3543820 | **The ordering survives the detector metric; the ratios grow at the spin edges.** FD mismatch is 3-4x E (the PSD weights the merger). lean-9 is 2.9x better than flux_orbit at 9c (M = 100 median), but at chi1 = -0.6 the pruned models rise to ~6e-3 like flux_orbit: the dropped shape terms carried that edge. Zhong's coefficients are a non-spinning, 2000 M-window fit; as a spin model they are spin-blind, so this scores their extracted formula, not their network. Pareto: Zhong added (published) |
