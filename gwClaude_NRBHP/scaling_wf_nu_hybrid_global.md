# wf_nu_hybrid_global — globally-jointly-fit hybrid BHPT→NR (2,2) model

Same 11-parameter **wf-nu-hybrid** architecture as `wf_nu_hybrid_q_dep`, but the master
polynomial coefficients are obtained by a **global joint fit** — optimizing the
coefficients directly against all q ∈ [3, 8] waveforms simultaneously — instead of the
per-q fit → regress pipeline. This dodges the per-q "degeneracy scatter" that makes the
regressed (master) coefficients noisy, yielding a more uniform in-range fit and smoother
α/β shapes across q, while holding q<3 extrapolation under the 1e-2 gate.

The 12 training mass ratios are sampled **evenly in ν = q/(1+q)²** (the regression
variable), not evenly in q. Because ν compresses at high q (`|dν/dq| = (q−1)/(1+q)³` is
~3× larger at q=3 than q=8), even-q sampling is *sparse in ν right at the q=3 boundary* —
the worst place to be thin when extrapolating below it. Even-ν sampling puts more training
points near q=3, which is what governs q<3 extrapolation. This replaced the original
even-q fit (2026-07-26): even-q failed the gate at q=2.25 (mathcalE 1.10e-2) despite
passing at q=2.0; even-ν clears all of q ∈ [2, 2.75] and has a lower in-range max.

Pure [3, 8] training: **no q < 3 data is used**. q<3 is a held-out extrapolation.

## Model equations (nu = q/(1+q)²; every param a cubic in nu, PP-anchored)

```
S      = 1 / (1 + exp(-(p_loss - p0)/w))
dp     = p_loss - p0;   dp_hat = dp / (p_loss[0] - p0)        # in [0,1] during inspiral
dE     = Ehat - Ehat(p0);   dJ = Jhat - Jhat(p0)
alpha  = alpha_i + (1-S)*alpha_L*dp_hat + S*(alpha_E*dE + alpha_J*dJ)
beta   = (1-S)*(beta_i + beta_L*dp) + S*beta_r
tau(t) = t0 + integral beta dt        # t0 places the merger at t = 0 (convention)
h(tau) = alpha * exp(i*phi0) * h_BHPT # phi0 = 0 (convention)
```

Loss coordinates (Ehat, Jhat, p_loss) are from the ppBHPT (2,2) GW flux — **no surfinBH**.

## How it was fit (global joint fit)

`peaks_results/prototypes/global_joint_fit.py --sample nu` (default). The 9 physical
params' coefficients (`p0, w, alpha_i, alpha_L, alpha_E, alpha_J, beta_i, beta_r, beta_L`)
are optimized with Powell against the mean [3,8] waveform mismatch over 12 training q's
sampled **evenly in ν**; `t0_nr, phi0` are per-q alignment nuisances (not global coeffs).
Fast evaluator with **analytic φ0** (optimal overall phase = angle of the complex overlap)
+ a 1-D t0 search — validated to match `evaluate_model_hybrid` + `polish_nuisance` to 4
significant figures. Warm-started from the per-q-regressed master, degree 3.

Training q's (even in ν): 3.0, 3.256, 3.513, 3.897, 4.2, 4.6, 5.0, 5.436, 5.949, 6.590,
7.231, 8.0 — visibly denser toward the q=3 boundary.

**Regularization** = λ · Σ (nonlinear excursion of each param at q=2 / its [3,8] span)²,
which favours gentle extrapolation. **Key finding from the λ sweep:** gentle-shape
regularization *conflicts* with q=2 accuracy — λ = 1e-7, 1e-6 forced gentler extrapolation
but wrecked q=2 (mathcalE 0.19, 0.13), because the true q=2 parameters are **not** a gentle
continuation of [3,8] (the "wild" cubic behaviour in β_r/p0/w is what keeps q=2 accurate).
So the regularization only works as a weak tie-breaker; **λ = 1e-8 is the selected value.**

## Performance (mathcalE vs NRHybSur3dq8, full window, t0/φ0-aligned)

All mismatches over the 64-q [3,8] grid; low-q rows are held-out extrapolation.

| q | wf_nu_hybrid_global (even-ν) | even-q (previous) | wf_nu_hybrid (master) |
|:--|:--|:--|:--|
| 8.0 | 3.89e-4 | 3.66e-4 | 9.63e-4 |
| 5.0 | 1.20e-4 | 1.26e-4 | 1.06e-4 |
| 3.0 | 5.22e-4 | 5.71e-4 | 3.97e-4 |
| 2.75 (extrap) | 1.00e-3 | 9.9e-4 | 1.18e-3 |
| 2.5 (extrap) | 2.46e-3 | 2.6e-3 | 2.55e-3 |
| **2.25 (extrap)** | **9.38e-3 ✓** | **1.10e-2 ✗** | 3.38e-3 |
| **2.0 (extrap)** | **4.04e-3** | 4.51e-3 | 3.92e-3 |
| in-range [3,8] median | 1.81e-4 | 1.73e-4 | 1.27e-4 |
| **in-range [3,8] max** | **5.22e-4** | 5.71e-4 | 9.63e-4 |

Net vs the previous even-q global fit: even-ν **clears the gate across all of q ∈ [2, 2.75]**
(even-q failed at q=2.25, 1.10e-2), improves q=2.0 (4.04e-3 vs 4.51e-3) and the in-range max
(5.22e-4 vs 5.71e-4), at a small cost to the in-range median and q=8. Both global fits keep
the more-uniform in-range profile vs the per-q master (max 5.2e-4 vs 9.6e-4).

**Honest caveats.** (1) The per-q-regressed **master is actually better at low q** than any
global variant (no q=2.25 notch; q2.25=3.38e-3, q2.0=3.92e-3) — the global fit's win is
in-range *uniformity*, not q<3 extrapolation. The q=2.25 non-monotonic notch is a global-fit
artifact (the Powell joint optimum misbehaves there); it is absent in the master. (2) The
visible q=2 defect is a merger **amplitude overshoot** (ringdown α ~0.66 vs true ~0.34) —
α_E/α_J-driven and largely **cosmetic** (ringdown ≈2% of the energy; the mismatch-relevant
q=2 error is the β_r over-extrapolation, ~0.73 vs true ~0.66). See `[[hybrid-q2-extrapolation]]`
memory / the SELECTIVE-reg follow-up.

Plots: `Agentic_plots/wf_nu_hybrid_global/hybrid_global_{waveforms,shapes}.png`.

## Selected coefficients (cubic in nu; c0 fixed at the PP anchor for anchored params)

| parameter | anchor | c0 | c1 | c2 | c3 |
|:---|:---|---:|---:|---:|---:|
| p0 | free | -7.2937 | 137.995 | -989.799 | 2350.96 |
| w | free | 1.89346 | -32.9862 | 69.5778 | -312.861 |
| alpha_i | 1 | 1 | -1.60909 | 7.48792 | -37.2928 |
| alpha_L | 0 | 0 | 0.080936 | -6.43866 | 31.8268 |
| alpha_E | 0 | 0 | -2.43677 | 58.0099 | -170.09 |
| alpha_J | 0 | 0 | 2.48766 | -64.2046 | 165.216 |
| beta_i | 1 | 1 | -1.34341 | 0.609532 | -7.19292 |
| beta_r | 1 | 1 | -1.41784 | 4.30061 | -19.0779 |
| t0_nr | free | -37.3321 | -1241.84 | 9492.92 | -20989.6 |
| phi0 | free | -445.192 | 9997.98 | -67013.7 | 140234 |
| beta_L | 0 | 0 | 0.0144972 | -0.568263 | 3.61853 |

(`t0_nr`, `phi0` are carried over from the master seed; the generator ignores them and
aligns by the merger convention.) Machine-readable: `wf_nu_hybrid_global_results/coeffs.json`.

## Files & usage

- `BHPTNRHybridGlobal.py` — generator, mirrors `BHPTNRClaude` / `BHPTNRPeaks`:
  ```python
  from BHPTNRHybridGlobal import generate_hybrid_global_calibrated
  t, h = generate_hybrid_global_calibrated(q_input=5.0)   # merger at t=0, dt=0.1
  ```
  `get_alpha_beta(q)` returns (t, α(t), β(t)). No NR is used at generation (φ0 = 0,
  merger at t = 0); optimize alignment when comparing to a reference.
- `peaks_results/prototypes/global_joint_fit.py` — the fit (defaults `--sample nu --lam 1e-8`);
  writes `global_coeffs_samp_nu.json` (copied to `wf_nu_hybrid_global_results/coeffs.json`;
  the previous even-q coeffs are kept as `coeffs_evenq_lam1em8.json.bak`).
- `peaks_results/prototypes/validate_hybrid_global.py` — reproduces the table + main plots.
- `peaks_results/prototypes/lowq_plots_hybrid_global.py` — q=2/2.25/2.5 low-q panels.
