# beta pinned between two derived anchors (gwr_betaqnm)

**5 coefficients, of which beta gets 2.**  The first model in this workspace whose beta is
built between two *derived* endpoints — the Newtonian chirp anchor at early inspiral and the
remnant QNM frequency ratio at ringdown — rather than fitted as a free nu-polynomial.

It is **NOT competitive on mismatch**: in-range median 2.93e-03 against `fluxanchored`'s
4.75e-04 (6.2x), and q=2 5.56e-02 against 1.31e-03 (43x).  It is recorded because it settles
a physics question that eight months of L2-fitted models could not, and because its failure
mode is specific and measured rather than mysterious.  See "Why it is here" below.

## Model

```python
Ehat(t) = Eoft(t)/E_tot                      in [0,1], gw_remnant, monotone
Ehat_m  = Ehat at the |h| peak               measured: 0.597 (q=2) .. 0.753 (q=8)
Ehat_r  = Ehat at t_merger + 30 M            start of the measured QNM plateau

u = clip(Ehat/Ehat_m, 0, 1)                        0 -> 1 by merger
v = clip((Ehat-Ehat_m)/(Ehat_r-Ehat_m), 0, 1)      0 -> 1 by +30 M, then HELD

beta (t,q) = B0 + (Bm - B0)*u + (B1 - Bm)*v
alpha(t,q) = X1^(6/5) * (1 + c0*nu + c1*nu^2) * (1 + (A/nu)*E(t))

B0 = X1^(6/5) * (1 + b*nu)          early-inspiral anchor        b  FITTED (1)
Bm = B0 * (1 + m*gb)                merger value                 m  FITTED (1)
B1 = W_Schw * Mf(q) / omega_220(chi_f(q))       ringdown         IMPOSED, 0 params
gb = (B1 - B0)/B0                   > 0 at every q in [2,8]
```

`beta(start) = B0`, `beta(merger) = Bm`, `beta(t >= +30 M) = B1` exactly and as a **plateau**
(v saturates), which is what the data requires: in the ringdown both waveforms sit on QNM
frequencies so `beta = omega_pp/omega_NR` is constant.

Fitted coefficients: `c0, c1, A` (alpha) and `m, b` (beta) = **5**.

## The two anchors

**Early inspiral, `X1^(6/5)`** — derived, not fitted.  From `alpha_beta_pn_scaling_note_revised.pdf`
§3.1: matching the Newtonian chirp rate under `tau = beta*t` gives `eta_pp*beta^(5/3) = nu`,
and since `nu/eta_pp = m1^2/M^2 = X1^2`, `beta_LO = (nu/eta_pp)^(3/5) = X1^(6/5)`.  Pure
mass-ratio algebra; the only physical input is that the flux is (mass ratio) x (universal
function of x) at leading order, which is measurable from `Edot`.

**Ringdown, the QNM frequency ratio** — derived from the remnant.  The ppBHPT rings at the
*primary's* Schwarzschild frequency (the background never changes in the test-mass limit);
NR rings at the *remnant's* Kerr frequency.  So

    B1 = omega_220^Schw / omega_220(Mf, chi_f)  =  W_Schw * Mf / omega_fit(chi_f)

with `Mf, chi_f` from **gwModelRemS** (`gwModels.remnants.gwModelRemS(q, 0, 0)`; installed
2026-08-19) and `W_Schw = 0.3737`.

**Both were verified against direct measurement** (`beta_drift_tests/ringdown_beta_vs_qnm.py`):
reading `omega_pp` and `omega_NR` off the waveforms over merger+30..90 M gives a ringdown
`beta` of 0.87923 at q=8 against B1 = 0.87673 (+0.29%); 0.82848 vs 0.82325 at q=5 (+0.68%);
0.76396 vs 0.75818 at q=3 (+0.91%).  Measured `omega_pp` is **0.3740 +- 0.27%, q-independent**,
confirming the Schwarzschild numerator.

## Results

```
python fit_scaling_gwr_betaqnm.py --bfree
```
Coefficients `c0, c1, A, m, b` = **0.2536, -0.3198, -1.1390, 0.6319, -0.0150**
(`gwr_betaqnm_results/coeffs_bfree.json`).

| model | coef | in-range med | max | q=2.75 | q=2.5 | q=2.25 | q=2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **betaqnm (bfree)** | 5 | 2.929e-03 | 6.749e-03 | 1.109e-02 | 1.864e-02 | 3.188e-02 | 5.556e-02 |
| betaqnm (parfree) | 3 | 1.438e-02 | 1.675e-02 | 5.535e-03 | 3.266e-03 | **1.780e-03** | **3.182e-03** |
| fluxanchored | 7 | **4.752e-04** | **7.150e-04** | **6.347e-04** | **5.937e-04** | 6.867e-04 | 1.306e-03 |
| anchored | 6 | 6.255e-04 | 9.788e-04 | 9.477e-04 | 9.279e-04 | 1.016e-03 | 1.529e-03 |

Per-q in range: 6.75e-03 (q=3), 1.87e-03, 1.94e-03, 2.87e-03, 4.00e-03, 2.99e-03 (q=8).
**Non-monotonic in q, worst at q=3** — see the right panel of `..._corners_vs_q.pdf`.

### beta's three corners

| q | B0 (imposed) | Bm (fitted) | B1 (QNM, imposed) | inspiral rise | merger -> ringdown |
|---:|---:|---:|---:|---:|---:|
| 2.00 | 0.61269 | 0.67463 | 0.71071 | +10.11% | +5.35% |
| 2.50 | 0.66576 | 0.70992 | 0.73565 | +6.63% | +3.62% |
| 3.00 | 0.70607 | 0.73900 | 0.75818 | +4.66% | +2.60% |
| 5.00 | 0.80182 | 0.81536 | 0.82325 | +1.69% | +0.97% |
| 8.00 | 0.86691 | 0.87312 | 0.87673 | +0.72% | +0.41% |

beta **rises monotonically at every q** and lands on the QNM plateau.  `m = 0.632 < 1`, so the
fit puts ~2/3 of the climb before merger and finishes over the ringdown ramp.  Note it does
NOT choose the peaky overshoot the peaks method measures (tangent beta 0.938 at q=8 vs QNM
0.877) — with B1 pinned, the fit approaches it from below.

## Why it is here, given it loses on mismatch

**1. It shows the rising-beta requirement is satisfiable.**  Five earlier routes could not
produce a beta that rises at every q: `b_E >= 0` collapsed to flat above q=4 (2x cost); the
four-drive test (E, accumulated phase, log frequency, raw time) gave the same falling sign for
all four; the merger-aligned phase gauge made q=5 worse; and `gwr_anchored2`/`3` cost 4-37x.
Here beta rises at all q by construction, with 2 coefficients.

**2. It prices the QNM anchor.**  Imposing B1 costs 6.2x in range.  That is now measured three
independent ways — `w = +1` in `gwr_anchored2` (10x over the free optimum), the `w` scan in
`gwr_qnmshape` (positive branch rising steeply from a minimum near w = -0.5), and this model —
and the cost is **insensitive to how the path gets there**.

**3. The `parfree` variant inverts the q-trend.**  With beta fully parameter-free (m=1, b=0)
low q becomes 5-8x BETTER than in range (1.78e-03 at q=2.25 against 1.44e-02 in range), the
opposite of every fitted model here.  For beta, q<3 is then **not extrapolation at all**:
`X1^(6/5)` is algebra, gwModelRemS is valid to q ~ 1000, and `Ehat(t)` is measured at whatever
q is asked.  Only alpha's 3 coefficients extrapolate.  That is direct evidence the fitted
models' low-q trouble is coefficient extrapolation rather than physics.

## Caveats

1. **Not PN-free.**  `X1^(6/5)` is a derived PN result.  Keep `gwr_energy_stiff` frozen as the
   PN-free evidence run.
2. **alpha is doing compensation work.**  With beta heavily constrained, alpha's coefficients
   are not descriptions.  In `parfree` (beta fully frozen) `c1` flips sign to +0.85 and alpha
   drops to 0.32 at q=2.  Do not read physics off them.
3. **alpha's coupling is `A/nu` with a single fitted `A`** and it extrapolates hardest at q=2 —
   visibly the largest single error source there (`..._q2_extrap_params.pdf`).  A second
   nu-coefficient on alpha is the obvious next change and is independent of beta.
4. **The E-saturation split was tried and is WORSE.**  Redefining "merger" as the point where
   E stops growing (t_sat = +38.7 .. +40.5 M, q-universal) gave median 3.83e-03 vs 3.78e-03 at
   4 coefficients and 3.34e-03 vs 2.93e-03 at 5.  Do not retry; the comment in
   `fit_scaling_gwr_betaqnm.py:prep` records it.
5. **The QNM fit's chi=0 limit is 1.45% off.**  `pn_anchored`'s three constants give 0.36830 at
   chi=0 against the true Schwarzschild 0.37370, so `B1 -> 1.0146` rather than 1 in the
   PP limit (now testable thanks to gwModelRemS reaching q=10^4).  At the chi_f values actually
   used the fit is good to +0.29% (q=8) .. +0.91% (q=3), so in-range B1 is fine, but the PP
   limit needs a proper Kerr QNM evaluation (`qnm` package or Berti tables).
6. **q=2 has a doubly-extrapolated input** — `BHPTNRSur1dq1e4` sets `X_min = log10(2.5)`, so the
   ppBHPT waveform itself is out of domain there.  Shared by every model in this workspace.

## Files

- fit: `python fit_scaling_gwr_betaqnm.py --bfree` (also `--parfree`, and plain 4-coef)
- coeffs: `gwr_betaqnm_results/coeffs_bfree.json` (also `coeffs_parfree.json`, `coeffs.json`)
- plots: `python NRBHP_gwr_betaqnm_plots.py bfree` -> `Agentic_plots/gwr_betaqnm/`
  (per-q params / waveform / zoomed at q=2,3,5,8; `corners_vs_q`; 14 PDFs)
- overlay: `python NRBHP_betaqnm_overlay.py {bfree|parfree|plain}` -> same folder, house
  format (1x2, tab colours by q, xlim (-1000, 80), default rcParams)
- anchor verification: `beta_drift_tests/ringdown_beta_vs_qnm.py`
- the five superseded routes: `beta_drift_tests/{scan_bE,monotone_beta,gauge_aligned,
  beta_drive_pnfree,inspiral_only_sign,segment_bE_scan,two_ingredient_test}.py`
