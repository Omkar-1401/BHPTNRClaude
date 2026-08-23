# Peaks-based BHPT→NR (2,2) calibration — Stage-2 model

Peak-ratio calibration of the raw ppBHPT surrogate to NR, following Islam & Khanna
(arXiv:2307.03155, Sec. II.A.3). The amplitude/time rescalings α, β are read off
directly as NR/ppBHPT ratios **at the waveform peaks**, then regressed across mass
ratio — no L2 waveform fit. This is the companion to the L2-fit `PN_opt_*` /
`wf_nu_*` models; its purpose is methodological (the paper's physical peak method)
and low-q, not to beat the L2 models in-band.

## Method

At each GW-phase peak k (extrema π apart in unwrapped phase, k=0 at merger):
- `α(k) = |h|_NR / |h|_ppBHPT`  (amplitude ratio)
- `β_abs(k) = t_NR / t_ppBHPT`  (merger-relative time-stretch, paper-literal)
- each peak tagged with `x = (M ω_orb)^(2/3)` from the ppBHPT side.

Stage 1 (`fit_scaling_peaks.py`) measures and caches these point clouds
(`peaks_results/per_q/`). Stage 2 (`fit_scaling_peaks_stage2.py`) builds the master model.

## Architecture (Stage 2)

**Stage A — per q.** Fit `α(x)−1` and `β_abs(x)−1` as **Chebyshev** polynomials in a
centered/scaled coordinate `u = (clip(x,XLO,XHI) − XMID)/XHALF`. Centering is
essential: on the raw range `x∈[0.05,0.22]` a degree-5 *monomial* fit has O(10⁴)
coefficients with catastrophic cancellation; the centered Chebyshev coefficients are
O(0.1) and regress smoothly.
- α x-degree 5, β x-degree 4.

**Stage B — across q.** Regress each Chebyshev coefficient in `ν = q/(1+q)²`,
**PP-anchored** so every correction vanishes in the point-particle limit:
`coeff_m(ν) = Σ_{k≥1} g[m,k] ν^k` ⟹ `coeff_m(0)=0` ⟹ `α,β→1` as `ν→0`.
- α, β both ν-degree 4. (ν beats 1/q and δ=√(1−4ν) decisively; β ν-degree 4 is the
  sweet spot — degree 3 underfits, degree 5 overfits and oscillates below q≈2.5.)

**Reconstruction — TIME MAP (not the per-sample frequency map).** Evaluate the model
α, β *only at the BHPT phase peaks* (where x is clean and monotone), set
`t_NR(k) = β_k · t_BHPT(k)`, build the BHPT→NR time map by interpolating **through the
peaks in time**, and interpolate α in time. This avoids the near-merger x-scrambling
that structurally caps a per-sample x-map at ~2% at q=2 (any β x-degree — see below).

**Training range [2.25, 8].** Pure [3,8]→q=2 extrapolation tops out at ~2–5% and is
*fragile* to the x-clip bound (q=2's long early inspiral falls below the training
x-range, so the frozen boundary β drives the phase). Training on the cached low-q
anchors as well ([2.25, 8]) makes the x-normalisation lower bound *real training data*
(q=2.25), so q=2 is a robust one-step extrapolation.

## Performance (mathcalE vs NRHybSur3dq8, inspiral-through-merger window, 98% coverage)

| q | mathcalE | note |
|:--|:--|:--|
| 3–8 (in-range) | **median 0.26%**, max 0.37% | regression-limited (α is the bottleneck) |
| 2.75 / 2.5 / 2.25 | 0.38% / 0.33% / 0.27% | low-q anchors (in training) |
| **2.0** | **0.55%** | **one-step extrapolation — clears the <1% gate** |
| 1.75 | 4.6% | two steps out — fails |
| 1.5 | 33% | fails |

Validation waveforms: `peaks_results/plots/stage2_validation.png`.
(Recalibrating with q=2 in training pushes q=2 to ~0.34%, but then it is no longer a
held-out prediction.)

## Honest limitations

1. **In-band accuracy is ~0.26%, ≈40× worse than the L2-fit models (~7e-5).**
   Peak-ratio α, β are inherently coarser than L2-optimised scalings. This model is
   **not** an in-band replacement for `PN_opt_creative` / `remnant_partial`; its value
   is the physical peak method and a robust low-q extrapolator.
2. **Ringdown not modelled.** The peaks method is an inspiral method: coverage is
   [~−5000 M, merger] ≈ 98% of the (2,2) energy. The post-merger ringdown (~2% of
   energy) is truncated at merger. Reported mismatches are over the covered window, so
   they are **not** directly comparable to the L2 models' full-window numbers (e.g.
   `remnant_partial` q=2 = 0.37% full-window). A QNM/remnant ringdown branch
   (`r_beta(χ_f)·β_r_physical`, logistic blend near merger) is the planned next step —
   machinery exists in `fit_scaling_PN_opt_remnant_partial.py`. See `peaks_results/RESUME.md`.
3. **q < 2.25 is extrapolation** and degrades fast (q=1.75 already 4.6%). Below the
   training frequency range the early-inspiral α, β are frozen at the clip boundary.

## Key findings that shaped the architecture (full trail in `peaks_results/RESUME.md`)

- The per-sample **x-map** reconstruction is a structural wall at q=2 (~2%, any β
  x-degree 3→10) — a smooth β(x) leaves ~1e-3 RMS that accumulates fatal phase over
  q=2's long inspiral. The **time-map** reconstruction fixes it.
- **β is the sole q=2 bottleneck**; regressed α matches the measured-recon ceiling.
- The measured per-q **time-map ceiling** at q=2 is 0.066% (interp of q=2's own peaks),
  so the method itself is not the limit — the ν-regression of β is.

## Files & usage

- `fit_scaling_peaks_stage2.py` — Stage-2 fit + validation; writes
  `peaks_results/stage2_coeffs.json` (24 α + 20 β coefficients, x-range, degrees).
- `BHPTNRPeaks.py` — generator, mirrors `BHPTNRClaude.py`:
  ```python
  from BHPTNRPeaks import generate_peaks_calibrated
  t, h = generate_peaks_calibrated(q_input=5.0)   # t ends at merger (t=0), dt=0.1
  ```
  Overall phase convention φ0 = 0 (no NR used at generation). `get_alpha_beta(q)`
  returns the per-peak α, β nodes. Output capped at the NR-validated window by default;
  `t_start=None` emits the full (unvalidated) BHPT inspiral.
- `peaks_results/prototypes/` — the diagnostic scripts behind every decision above.
