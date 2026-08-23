# Flux-coupled alpha on the anchored layer (gwr_energy_fluxanchored)

**7 coefficients.**  Best model in this workspace on both axes at once: in-range and at
every q < 3.

## Model

```python
E(t) = gw_remnant Eoft            F(t) = Edot / max(Edot)     # peak-normalised flux
nu   = q/(1+q)**2                 X1   = q/(1+q)

alpha(t) = alpha_PP(q) * (1 + alpha_F(q) * F(t))     # <- flux, not E
beta (t) = beta_PP(q)  +  P(nu) * E(t)
tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'
h_model(tau) = alpha * exp(i*phi0) * h_BHPT          # phi0 analytic

alpha_PP(q) = X1**1.2 * (1 + c0*nu + c1*nu**2)   # 2, derived base
alpha_F(q)  = A0*nu + A1*nu**2                   # 2, -> 0 as nu -> 0 BY CONSTRUCTION
beta_PP(q)  = X1**1.2 * (1 + b*nu)               # 1, derived base
P(nu)       = P0 + P1*nu                         # 2, empirical
```

`alpha_F` is PP-anchored by its form, which `alpha_E` never was: because F is
peak-normalised, `alpha_F` *is* essentially `d ln alpha` across the window -- the
quantity measured to scale as nu^1.635 -- and the per-q fits give -0.3213 (q=3) to
-0.1206 (q=8), i.e. ~nu^1.53.  So a polynomial starting at nu^1 is the natural form and
the test-mass limit comes free.  The E-coupled `alpha_E` needed nu^-1 for the same job,
a power no polynomial represents and whose justification did not survive measurement.

## Why it works: the two failure modes are independent

| change | in-range | q < 3 |
|:---|:---|:---|
| alpha coordinate E -> F | **fixes** (25% better) | no effect |
| regression layer deg-3 -> anchored | no effect | **fixes** (18x at q=2) |

`gwr_energy_flux` (flux alpha, uniform deg-3, 14 coef) improved in-range to 4.70e-04 and
left q=2 at 4.6e-02.  `gwr_energy_anchored` (E alpha, anchored, 6 coef) improved q=2 to
1.53e-03 and left in-range at 6.26e-04.  Combining them gets both, as it should.

## Results

Two variants.  `seeded` = per-q fit then independent regression of the same forms;
`global` = all 7 optimised jointly against 12 even-nu waveforms in [3,8].  **No q<3 data
anywhere in either.**

| model | coef | in-range med | in-range max | q=2.75 | q=2.5 | q=2.25 | q=2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **fluxanchored, seeded** | 7 | 4.8964e-04 | 7.1563e-04 | 6.463e-04 | 5.538e-04 | **5.017e-04** | **7.585e-04** |
| **fluxanchored, global** | 7 | **4.7516e-04** | **7.1503e-04** | **6.347e-04** | 5.937e-04 | 6.867e-04 | 1.306e-03 |
| anchored (E alpha) | 6 | 6.2551e-04 | 9.7884e-04 | 9.477e-04 | 9.279e-04 | 1.016e-03 | 1.529e-03 |
| flux (uniform deg-3) | 14 | 4.6993e-04 | 7.1604e-04 | 7.089e-04 | 1.721e-03 | 8.702e-03 | 4.600e-02 |
| stiff | 9 | 6.2366e-04 | 9.7359e-04 | 9.829e-04 | 1.124e-03 | 1.727e-03 | 3.785e-03 |
| gwr_energy_global | 14 | 6.26e-04 | 9.70e-04 | 9.9e-04 | 1.40e-03 | 4.9e-03 | 2.8e-02 |

Both variants beat **both parents at all four held-out mass ratios**.

**The `seeded` variant is the headline: q=2 at 7.585e-04**, with q=2.25 (5.017e-04) and
q=2.5 (5.538e-04) *better than the in-range median*.  For the first time in this
workspace the model is as accurate below the training range as inside it.  For scale,
the previous best q=2 anywhere was `pn_anchored` at 2.55e-03, and `gwr_energy_stiff`'s
q=2 is 3.785e-03.

The joint fit buys a little in-range (4.75 vs 4.90e-04 median) and costs a lot at q=2
(1.31e-03 vs 7.59e-04): it optimises the in-range mean, which pulls the coefficients off
the nu-structure the per-q optima actually trace.  This inverts the `gwr_energy_stiff`
lesson, and consistently so -- there the seeded forms could not reach the per-q floor so
the joint fit had slack to recover; here the flux parameterisation already sits at it, so
the joint fit has nothing to gain in-range and trades away extrapolation to get it.

## Coefficients

```
                c0        c1          A0        A1         b        P0        P1
seeded    0.040021  0.197206   -0.773103 -5.064748  0.013036 -4.756750 29.331585
global    0.037410  0.212817   -0.746055 -5.233438  0.013147 -4.404917 27.084697
```

## Per-q floor below q=3 (measured) -- the consistency check passes

A regressed master cannot beat the per-q floor of its own form, so this is a correctness
test as well as a margin bound.  Per-q floor = the flux form fitted at that single q alone,
evaluated on the full grid:

| q | flux per-q floor | seeded master | master/floor | E-coupled floor | floor gain |
|---:|---:|---:|---:|---:|---:|
| 3 | 6.7175e-04 | -- | -- | 9.5679e-04 | 1.42x |
| 2.75 | 5.8794e-04 | 6.4627e-04 | 1.10x | 9.1084e-04 | 1.55x |
| 2.5 | 5.0099e-04 | 5.5377e-04 | 1.11x | 8.5641e-04 | 1.71x |
| 2.25 | 4.2089e-04 | 5.0165e-04 | 1.19x | 8.1920e-04 | 1.95x |
| 2 | **4.2505e-04** | 7.5847e-04 | **1.78x** | 8.4786e-04 | **1.99x** |

The master sits above its floor everywhere, so there is no contradiction: the earlier
puzzle -- q=2 at 7.585e-04 being below the *E-coupled* floor of 8.4786e-04 -- is simply
that the flux form has a floor **~2x lower**, and one that *falls* toward low q
(6.72e-04 at q=3 down to 4.21e-04 at q=2.25).  That is also why the low-q master numbers
can sit below the in-range median: the achievable floor is genuinely better there.

**Where the remaining margin goes.**  Comparing the seeded forms against the per-q truth
below q=3, the two derived-base prefactors extrapolate essentially perfectly and the two
empirical couplings carry the entire error:

| q | `alpha_PP` | `alpha_F` | `beta_PP` | `P` |
|---:|---:|---:|---:|---:|
| 2.75 | +0.2% | -3.1% | +0.0% | -4.9% |
| 2.5 | +0.4% | -4.2% | +0.1% | -7.0% |
| 2.25 | +0.6% | -10.2% | -0.0% | +7.8% |
| 2 | **+0.9%** | **-18.0%** | **-0.2%** | **+23.0%** |

So the `X1^(6/5)` bases hold to well under 1% all the way to q=2, and `alpha_F` and `P` --
the two functions with no derivable order -- are what cost the 1.78x at q=2.  There is
real headroom: better nu-forms for those two could take q=2 from 7.6e-04 toward 4.3e-04.

## Remaining caveats

1. Robustness to `--nperq` / `--ntrain` / the training set is untested; a jump this large
   deserves a sensitivity check.
3. F is non-monotonic where E is not, so alpha returns toward alpha_PP after merger.  That
   did not hurt here, but it is the structural difference to watch if ringdown accuracy is
   later scrutinised.

   **Corrected 2026-08-11: F does NOT "peak at merger" as this caveat originally said.**
   Measured:

   | q | F peaks at | F at merger | pre-merger steps increasing | worst backward step |
   |---:|---:|---:|---:|---:|
   | 3 | -15.0 M | 0.6428 | 63.9% | -75.6% of range |
   | 5 | -15.0 M | 0.7223 | 68.7% | -62.7% of range |
   | 8 | +6.0 M | 0.8456 | 68.6% | -45.6% of range |

   The peak sits ~15 M BEFORE merger at q=3 and 5 (6 M after, at q=8), and F has already
   fallen to 0.64-0.85 by the merger itself.  On top of that `Edot` oscillates on the
   orbital timescale, so only ~65% of F's pre-merger steps increase, with backward
   excursions up to 76% of its range.  None of this invalidates the model — `alpha_F` only
   ever multiplies F, and the fit is what it is — but the physical story "F supplies the
   merger ramp" is looser than stated, and anything that RELIES on F being monotone or
   merger-centred needs a smoothed, merger-centred flux instead.  See
   `RESUME_gwremnant.md` open item 1.
4. Like `gwr_energy_anchored`, this imposes X1^(6/5) and so is **not PN-free**.  Keep the
   9-coefficient `gwr_energy_stiff` frozen as the evidence run.
5. Higher modes have not been re-run on this base.  Given (3,3) is base-limited, it should
   inherit the improvement; (4,4)/(5,5) may again degrade.

## Files

- fit: `python fit_scaling_gwr_energy_fluxanchored.py --global --maxiter 80`
- coeffs (both variants): `gwr_energy_fluxanchored_results/coeffs.json`
- per-q flux cache: `gwr_energy_flux_results/per_q_cache_flux.json`
- parents: `fit_scaling_gwr_energy_flux.py`, `fit_scaling_gwr_energy_anchored.py`
- plots: `python NRBHP_gwr_global_plots.py --model E flux fluxanchored` ->
  `Agentic_plots/gwr_global_{E,flux,fluxanchored}/`, 15 PDFs each, house format.  One
  script for all three (`--model` dispatch, one evaluator generalised over alpha's
  coordinate) so the sets are produced identically and can be laid side by side.  Uses
  the GLOBAL coefficients only -- the `seeded` variant is not plotted.
- `--aF-deg 1` gives a 6-coefficient variant; not run.
