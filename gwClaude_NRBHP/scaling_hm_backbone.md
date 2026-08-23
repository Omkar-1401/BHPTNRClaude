# Higher modes via the (2,2) backbone transfer — Step 1 (zero-parameter diagnostic)

`fit_scaling_hm_backbone.py`.  Extends `gwr_energy_stiff` from (2,2) to the 9 modes
BHPT and NR have in common, **adding no fitted parameters at all**.

## This is not the first higher-mode test in this workspace

The same backbone was applied to the **`pn_anchored`** (2,2) base on 2026-07-28 for
(3,3) and (4,4) only — prototypes `peaks_results/prototypes/hm_backbone_test.py`,
`hm_residual_fit.py`, `hm_localize_q2_44.py`, `hm_q{2,4}_fig8.py`; plots
`Agentic_plots/pn_anchored_hm/`.  Nothing was committed to model files.  What is new
here is (a) the base — `gwr_energy_stiff`, which uses **no PN input and no logistic
gate**, where `pn_anchored` uses both; (b) coverage of all 9 available modes rather
than two, including the off-diagonal set and (2,1); (c) the power weighting.

Apples-to-apples, both **pure** backbones (no residuals fitted):

| | pn_anchored base | gwr_energy_stiff base (here) |
|:---|---:|---:|
| (3,3) in-range median | **1.98e-3** | 2.69e-3 |
| (3,3) q=2 | **6.5e-3** | 1.73e-2 |
| (4,4) in-range median | **9.1e-3** | 1.34e-2 |
| (4,4) q=2 | 7.0e-2 | **4.62e-2** |

`pn_anchored` transfers better at (3,3), which is expected: its (2,2) q=2 is 2.55e-3
against this model's 3.78e-3, and the higher modes inherit and amplify that.  So the
PN-free base costs higher-mode accuracy at low q, and the choice between them is the
same PN-vs-no-PN trade as for (2,2).

**Two findings from that earlier round carry over and should not be re-derived:**
(1) for (4,4) at q=2 the failure is in the **amplitude, not the phase** — the
`(m/2)phi_22` + shared-time-map phase transfer was clean, and an own-residual in-sample
fit reached only 2.06e-2 against 2.72e-2 extrapolated, so it is structural (likely
spherical-spheroidal mode mixing), not a residual-discipline problem.  (2) The
structural 2% is with respect to a 2-coefficient rho; the note's is 3-coefficient and
gated, so it is not proven fundamental.

## Construction (note §6.3, Eq. 43)

```
alpha_lm(t,q) = alpha_22(t,q) * C_lm(nu) * beta(t,q)^(-(l+eps_lm-2)/3) * rho_lm(t,q)

eps_lm = (l+m) mod 2
C_lm   = | X2^(l+eps_lm-1) + (-1)^m X1^(l+eps_lm-1) |        X1 = q/(1+q), X2 = 1/(1+q)
rho_lm -> 1 in the point-particle limit                       <- the only free part
```

`alpha_22(t)` and `beta(t)` are taken **unchanged** from `gwr_energy_stiff` (9
coefficients).  `C_lm` is closed form.  The `beta^(...)` factor is the frequency
rescaling the common time map already implies (`x_NR/x_pp = beta^(-2/3)`, note Eq. 42),
so it is not a parameter either.  Setting `rho_lm = 1` therefore gives a **pure
prediction**: every higher mode below is obtained with zero additional fitting.

Verified: `C_lm` reproduces the note's Eq. (40) closed forms (`C_33 = Delta`,
`C_44 = 1-3nu`, `C_55 = Delta(1-2nu)`, `C_22 = 1`) exactly at every q.  Because
`C_22 = 1` and its exponent is 0, the (2,2) mismatch is reproduced to the digit
(9.7360e-04 / 7.0099e-04 / 3.5419e-04 / 3.7850e-03 at q = 3/5/8/2), which is the
built-in consistency check against `scaling_gwr_energy_stiff.md`.

## What is shared

`beta(t)` and the time map `tau(t) = t0_nr + int beta dt'` are **common to all modes**.
That is the structure of the original BHPTNRSur1dq1e4 calibration (note Eq. 1: one
`beta(q)`, mode-dependent `alpha_l(q)`) and of the reference implementation in
`BHPTutils/bhpt_utils/nrcalib/alpha_beta_calibration.py` (`AlphaBetaOptimizer.
_optimize_func`: a single beta scales the time axis, one alpha per mode scales the
strain).  So this extension **adds no time-map freedom and cannot degrade (2,2)**.
Per mode the only freedom used is the constant phase, solved analytically as for (2,2).

## Results, rho = 1 (no fitting)

`k` = the best single multiplicative constant per mode, also closed form
(`k = |z|/n2`), reported to show how much a 1-parameter-per-mode residual would buy.

| mode | power frac (q=5) | q=3 | q=5 | q=8 | q=2 | k (q=3 / 5 / 8) |
|:---|---:|---:|---:|---:|---:|:---|
| (2,2) | 0.965 | 9.74e-4 | 7.01e-4 | 3.54e-4 | 3.79e-3 | 1.000 / 0.999 / 1.000 |
| **(3,3)** | 2.5e-2 | **4.82e-3** | **2.69e-3** | **2.21e-3** | 1.73e-2 | 0.999 / 0.993 / 0.991 |
| (4,4) | 2.0e-3 | 3.03e-2 | 1.34e-2 | 1.00e-2 | 4.62e-2 | 0.907 / 0.936 / 0.955 |
| (5,5) | 2.2e-4 | 8.10e-2 | 4.25e-2 | 2.97e-2 | 4.12e-2 | 0.794 / 0.838 / 0.876 |
| (2,1) | 7.6e-3 | 2.69e-2 | 1.63e-2 | 1.26e-2 | 3.85e-2 | 0.914 / 0.924 / 0.937 |
| (3,2) | 4.4e-4 | 2.83e-1 | 1.18e-1 | 6.03e-2 | 3.36e-1 | 0.639 / 0.755 / 0.830 |
| (4,3) | 4.0e-5 | 4.52e-1 | 2.78e-1 | 1.35e-1 | 2.33e-1 | 0.525 / 0.611 / 0.722 |
| (3,1) | 2.3e-5 | 7.16e-1 | 3.67e-1 | 2.10e-1 | 1.04e+0 | 0.422 / 0.561 / 0.675 |
| (4,2) | 5.0e-6 | 9.45e-1 | 4.61e-1 | 2.49e-1 | 1.26e+0 | 0.367 / 0.664 / 0.288 |

Errors fall monotonically with q for every mode, as a point-particle-limit
construction should.

## Power-weighted mode-sum error (rho = 1)

`sum(n1_lm * E_lm) / sum(n1_lm)` — a proxy for the full strain (a true multi-mode
mismatch depends on inclination), with the power fraction each set captures.

| mode set | q=3 | q=5 | q=8 | q=2 | power captured (q=5) |
|:---|---:|---:|---:|---:|---:|
| (2,2) only | 9.74e-4 | 7.01e-4 | 3.54e-4 | 3.79e-3 | 0.9646 |
| + (3,3) | 1.03e-3 | 7.51e-4 | 4.18e-4 | 3.87e-3 | 0.9896 |
| + (3,3), (2,1) | 1.14e-3 | 8.70e-4 | 5.47e-4 | 3.94e-3 | 0.9973 |
| diagonal l=m | 1.07e-3 | 7.86e-4 | 4.58e-4 | 3.90e-3 | 0.9919 |
| **all 9** | **1.27e-3** | **9.78e-4** | **6.45e-4** | **4.05e-3** | **1.0000** |

**Going from (2,2) alone to all 9 modes costs 31% / 40% / 82% / 7% at q = 3/5/8/2 and
buys the remaining 2-5% of the power, for zero parameters.**  The badly-predicted
off-diagonal modes barely register because they carry <0.1% of the power.

## What the q=2 column is, and what it decomposes into

**The q=2 numbers above are the EXTRAPOLATED MASTER, not per-q truth.**  The (2,2)
parameters come from the shipped 9 coefficients fitted on [3,8] only and evaluated at
q=2; `rho_lm = 1` everywhere.  The only thing refit at q=2 is `t0_nr`, the 1-D alignment
nuisance, exactly as in the (2,2) table — hence the (2,2) entry is 3.7850e-3, the
`global` value from `scaling_gwr_energy_stiff.md`, not its per-q floor of 8.4786e-4.
Nothing per-mode is fitted anywhere; the `k` column is the sole exception, and it is an
in-sample constant at that q, reported for diagnosis only.

So each q=2 higher-mode number carries three stacked extrapolations: the BHPT surrogate
input (q=2 < X_min = 2.5), the (2,2) coefficient extrapolation, and an unfitted residual.

`--perq` swaps in the per-q-optimal (2,2) parameters (`gwre.optimize_case`) to separate
the second of those.  It is a diagnostic, not a model — it uses NR at q=2.

| q=2, rho=1 | master (2,2) base | per-q (2,2) base | change |
|:---|---:|---:|:---|
| (2,2) | 3.7850e-3 | 8.4786e-4 | 4.5x better (by definition) |
| **(3,3)** | 1.7348e-2 | **6.4815e-3** | **2.7x better** |
| (4,4) | 4.6150e-2 | 5.8224e-2 | 1.3x **worse** |
| (5,5) | 4.1206e-2 | 6.9133e-2 | 1.7x **worse** |
| (2,1) | 3.8491e-2 | 3.5403e-2 | 1.1x better |
| (3,2) | 3.3586e-1 | 3.9634e-1 | worse |

Two clean conclusions:

1. **(3,3) at q=2 is limited by the (2,2) base, not by the transfer.**  Give it a
   per-q-accurate quadrupole and it drops to 6.48e-3, and `k` becomes 0.9999 with
   `E(rho=const)` equal to `E(rho=1)` to five digits — i.e. with a good base, (3,3)
   needs *no residual at all* even at q=2.  Anything that improves (2,2) at low q
   improves (3,3) proportionally and for free.  Consistent with `pn_anchored` reaching
   6.5e-3 on (3,3) from a (2,2) that is 2.55e-3.

2. **(4,4) and (5,5) get WORSE with a better (2,2) base.**  Their q=2 error is not
   inherited at all; the master's extrapolated `alpha_22`/`beta` were *accidentally
   compensating* part of the transfer error, and removing that error exposes it.  This
   independently corroborates the July `pn_anchored` finding that the (4,4) q=2 failure
   is structural (amplitude/mode-mixing) rather than a matter of coefficient accuracy —
   now reached from a different base and without fitting anything.

## Residual structure — what Step 2 should fit

1. **(3,3) needs no residual.**  `k` is within 0.1-0.9% of 1 for q>=3, and the best
   constant removes only 0.02-1.9% of the error.  `rho_33 = 1` is already correct;
   its 2-5e-3 error is inherited from the (2,2) base and the shared time map, not from
   a missing mode factor.

2. **The residual is O(nu) with an l-dependent coefficient — the note's Eq. (52) form,
   confirmed at the waveform level.**  `(1-k)/nu` is nearly q-independent:

   | mode | q=3 | q=5 | q=8 |
   |:---|---:|---:|---:|
   | (3,3) | 0.008 | 0.050 | 0.092 |
   | (4,4) | 0.497 | 0.464 | 0.455 |
   | (5,5) | 1.098 | 1.168 | 1.257 |
   | (2,1) | 0.459 | 0.548 | 0.636 |

   So `rho_lm ~ 1 - nu*c_l` with `c_l ~ {0, 0.46, 1.2}` for `l = 3, 4, 5` and 0.55 for
   (2,1) — i.e. roughly `c_l ~ 0.6 (l-3)` on the diagonal.  This is a 1-parameter-per-mode
   ansatz whose value is already measured.

3. **But a constant rho is not enough for l>=4.**  The best constant removes only
   11-42% of the error for (4,4), (5,5), (3,2) versus ~1% for (3,3):

   | 1 - E(rho=const)/E(rho=1) | q=3 | q=5 | q=8 | q=2 |
   |:---|---:|---:|---:|---:|
   | (3,3) | 0.02% | 0.92% | 1.89% | 0.98% |
   | (4,4) | 16.6% | 17.3% | 10.9% | 12.9% |
   | (5,5) | 37.3% | 41.9% | 32.4% | 30.3% |
   | (2,1) | 15.7% | 20.3% | 17.4% | 10.8% |

   So `rho_lm` must be **time-dependent**.  The note writes it in the PN variable x
   with the logistic switch (Eqs. 52-53), both of which `gwr_energy_stiff` deliberately
   avoids; the translation into this model's own idiom is
   `rho_lm(t) = 1 + rho_E,lm * E(t)`, keeping the extension PN-free and gate-free.

4. **Off-diagonal modes fail, as the note predicted.**  (3,1), (3,2), (4,2), (4,3) run
   6e-2 to 1.26; (3,1) and (4,2) exceed 1 at q=2, i.e. worse than predicting zero.  The
   note explicitly warns that spherical-spheroidal mixing means these "should be tested
   separately rather than assumed to follow the same residual hierarchy as l=m modes".
   Confirmed.  **(2,1) is the exception**: it is off-diagonal but behaves like a diagonal
   mode (`k` ~ 0.92-0.94, smooth in q), and it is the third-strongest mode (0.4-1.1% of
   power), so it belongs with the diagonal set.

## Caveats

* Per-mode mismatch is normalised by that mode's own norm, so weak modes show large
  relative errors that are negligible in the summed strain — read the table above
  together with the power fractions.
* q=2 carries the usual double extrapolation (`BHPTNRSur1dq1e4` X_min = log10(2.5) plus
  coefficient extrapolation); q=2.5 is the honest low-q gate.
* NR higher modes come from NRHybSur3dq8, whose own higher-mode accuracy is worse than
  its (2,2); part of the (5,5) and off-diagonal error may be the reference, not the model.
  Not yet quantified.

## Mode availability

BHPT `BHPTNRSur1dq1e4`: (2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),
(5,5),(6,4)...(10,9).  NR `NRHybSur3dq8`: (2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),
(4,2),(4,3),(4,4),(5,5).  Intersection = the 9 modes above.  NR-only: (2,0),(3,0).
BHPT-only: (5,3),(5,4) and all l>=6 — not calibratable against this NR model.

Command: `python fit_scaling_hm_backbone.py --q 3 5 8 2`
Diagnostic: `python fit_scaling_hm_backbone.py --q 2 --perq` (per-q (2,2) base; uses NR at q=2)
Results: `hm_backbone_results/backbone_rho1.json`, `backbone_rho1_perq.json`; cache `.cache/hm/`.

## Consolidated four-base comparison (2026-08-03)

`python NRBHP_hm_backbone_fig8_plots.py --q 2 4 --model E_deg3 flux_deg3 E_anchored
flux_anchored --outdir consolidated_gwremnant_hm`
-> `Agentic_plots/consolidated_gwremnant_hm/hm_q{2,4}_<base>.png`, 8 figures.

The backbone is unchanged and still fits nothing; only the (2,2) quadrupole it transfers
differs.  Each panel's (2,2) value reproduces that base's own documented mismatch.

| base | q=4 (2,2) | (3,3) | (4,4) | q=2 (2,2) | (3,3) | (4,4) |
|:---|---:|---:|---:|---:|---:|---:|
| `E_deg3` (14c) | 9.336e-04 | 3.823e-03 | 1.977e-02 | 2.762e-02 | 6.322e-02 | 2.481e-01 |
| `flux_deg3` (14c) | **7.160e-04** | **3.482e-03** | **1.738e-02** | 4.600e-02 | 1.102e-01 | 3.418e-01 |
| `E_anchored` (6c) | 9.311e-04 | 3.883e-03 | 2.022e-02 | 1.529e-03 | 9.986e-03 | **4.869e-02** |
| `flux_anchored` (7c) | **7.150e-04** | **3.472e-03** | **1.722e-02** | **1.306e-03** | **9.411e-03** | 6.407e-02 |

**In range (q=4) the flux bases win on every mode** -- (3,3) 3.47e-03 against 3.88e-03,
(4,4) 1.72e-02 against 2.02e-02 -- and the regression layer barely matters, exactly as
for the (2,2) alone.

**At q=2 the anchored layer is what matters**, and the two uniform-deg-3 bases are
hopeless for higher modes (up to 0.34 for (4,4)).  Between the two anchored bases the
flux one is better on (2,2) and (3,3) but **worse on (4,4)** (6.41e-02 vs 4.87e-02).

That (4,4) inversion is the ringdown defect made visible.  Because F is peak-normalised
and non-monotonic, `alpha = alpha_PP (1 + alpha_F F)` returns to `alpha_PP` once the flux
decays, while the empirical alpha keeps falling through ringdown (see the q=2 parameter
panel in `Agentic_plots/gwr_global_fluxanchored/`).  (4,4) carries relatively more
merger-ringdown weight than (2,2) or (3,3), so it is the mode that pays for it -- and
here the penalty exceeds the flux coordinate's in-band gain.  A two-term alpha (E for the
post-merger floor, F for the merger ramp) is the indicated fix; E and F are not degenerate
with each other the way `wf_nu_fluxes`'s two flux terms were.

## Plots

`python NRBHP_hm_backbone_fig8_plots.py --q 2 4` ->
`Agentic_plots/hm_backbone/q{2,4}_waveforms_224.png`.  Fig-8-style stacked
(2,2)/(3,3)/(4,4) Re(h) panels, format identical to the pn_anchored prototypes
`peaks_results/prototypes/hm_q{2,4}_fig8.py` so the two can be laid side by side.  The
mathcalE on each panel comes from this model's own evaluator, so it matches the tables
above (q=2: 3.785e-3 / 1.735e-2 / 4.615e-2; q=4: 9.352e-4 / 3.860e-3 / 2.023e-2 — the
q=4 (2,2) value being the 9.3518e-4 of `scaling_gwr_energy_stiff.md`).

**Ordering trap, worth knowing before writing any other plotting script off this model:**
`rcdefaults()` must be called **after** the model imports, not before.  The pn_anchored
prototypes call it first, which is safe for *their* import chain, but this model's chain
pulls in `gw_remnant`, and `gw_remnant/gw_utils/gw_plotter.py` sets `font.size=18`,
`font.family='STIXGeneral'`, `axes.linewidth=1` and `figure.figsize=(14,10)` at import
time.  Reset-then-import silently yields 18pt STIX figures instead of 10pt DejaVu Sans.
`NRBHP_gw_remnant_energy_plots.py` already uses the correct ordering.

What the figures show, consistent with the numbers: at q=4 the (2,2) and (3,3) panels
track NR through ringdown, while (4,4) overshoots in amplitude at merger with its zero
crossings still aligned — the visual form of the "amplitude, not phase" localisation.
At q=2 the same (4,4) merger overshoot is larger and (3,3) picks up a ringdown
amplitude error, while both inspirals remain clean.
