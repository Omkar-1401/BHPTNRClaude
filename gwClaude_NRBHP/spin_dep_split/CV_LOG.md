# Split variants: selection on T96 cross-validation (2026-10-13)

User (2026-10-13): "Why are you choosing with 270? Those are literally held out points?"  Correct: EXPLORE_LOG rounds 1-4
chose variants on H270 (auto_round2 took the round-1 winner by H270 mean; the window scan and per-spin T_start
optima were read off H270).  From here:
* **Selection and tuning on T96 only**: 4-fold CV, fold of T96 point (iq, ic) = (iq + ic) % 4 (24 per fold, each q twice,
  each chi1 three times); refit from scratch on 72, score the 24 held out; pooled 96 held-out full-window E (full alignment).
  `cv_split.py`.  Caveat: optimisers start from the full-T96 coefficients (as round 5) -- a start, not a fit.
* **H270 is now spent for these variants** (it was looked at ~34 times).  Report it, but not as an untouched test.
* **F100 = new held-out set** (`fresh_set.py`; 100 uniform points in the H270 box, seed 20261013; caches in fresh_cache/).
  No fit or choice uses it.  Scored once, for the final pick(s) only; every look recorded below.
* T_start(chi1) line ("tline", 14c: inspiral b = 0, slim MR, start = min(T0 + T1 chi1, T_peak - 0.55)): (T0, T1) chosen on
  the training points by mean full-window E over a 4 x 4 grid, nested inside each CV fold; `--fold all` gives the candidate.

## Jobs
2026-10-13: cvA0-3 (11 variants each: slim, windows, inspiral trades), cvB0-3 (11 MR forms), cvT0-3 + cvTall (tline),
f100 (development; cache build: done 10:46, 100 + 100 caches, exit 0).  Jobs 3507505-3507518.  Outputs: results/cv/<name>__fold<K>.json, logs/cv.*.log.

## Results (all 13 jobs exit 0, 2026-10-13/14)
Pooled 96 held-out full-window E (median / mean / max), ranked by mean.  All 14c (tline: 13 fitted + T0 chosen on grid).
| variant | held-out median | mean | max |
|:--|--:|--:|--:|
| tline (chose T0 = -3, T1 = 0 on EVERY fold and on full T96) | 6.24e-4 | **8.55e-4** | **5.9e-3** |
| slim__win-3.0 | 5.15e-4 | 8.75e-4 | 6.8e-3 |
| insp8b__win-2.0 | 6.43e-4 | 8.77e-4 | 6.3e-3 |
| insp8__win-2.0 | 6.55e-4 | 8.79e-4 | 6.0e-3 |
| slim__win-2.0 | **4.76e-4** | 8.87e-4 | 7.0e-3 |
| slim__win-4.0 | 5.97e-4 | 8.95e-4 | 6.8e-3 |
| slim__win-1.5 | 4.90e-4 | 9.11e-4 | 7.2e-3 |
| insp8_mr6 | 6.62e-4 | 9.16e-4 | 6.2e-3 |
| insp8b_mr6 | 5.92e-4 | 9.50e-4 | 6.5e-3 |
| slim (T = -1) | 5.44e-4 | 9.53e-4 | 7.3e-3 |
| aF_P_S | 6.84e-4 | 1.01e-3 | 7.2e-3 |
| slim__win-0.5 | 6.18e-4 | 1.02e-3 | 7.5e-3 |
| insp7_mr7 | 5.67e-4 | 1.02e-3 | 7.1e-3 |
| rem_dchi2 | 5.57e-4 | 1.05e-3 | 6.7e-3 |
| rem_dbr_s1, aF_bPP_P, aF_bPP_P0_s1, alpha_spin, bump, bump_s1, rem_dbr2, rem_dchi, rem_dbr | 7.7e-4 - 1.6e-3 | 1.26e-3 - 2.11e-3 | |

Findings: (1) any window start T = -1.5 .. -3 beats T = -1 on held-out median AND mean; the MR form choice (slim, chi1^2
term) is confirmed -- every other MR form is worse.  (2) The T_start slope T1 > 0 is never chosen by training MEAN (it lowers
the training median but raises the mean: the chi1 = +0.8 points want an EARLY start, the slope makes it late).
(3) Per point vs slim: earlier start helps chi1 < 0 (tline median ratio 0.54) and hurts chi1 > 0 (1.19).
(4) Ranking by median -> slim__win-2.0; by mean/max -> tline (= inspiral b = 0, constant T = -3).  Criterion = user's call.

## Looks at F100

* look 2026-10-15 10:04: slim__win-2.0 (14c): F100 4.986e-04 / 7.147e-04 / 2.74e-03
* look 2026-10-15 10:04: tline (14c): F100 4.847e-04 / 6.822e-04 / 1.94e-03; H270 4.083e-04 / 5.360e-04 / 2.18e-03

## Final (2026-10-15): A = slim__win-2.0, B = tline (b = 0, T = -3), both chosen by the user from the CV ranking
| | T96 (training) | H270 (validation, spent) | F100 (test, one look) |
|:--|:--|:--|:--|
| A | 4.64e-4 / 8.54e-4 / 6.5e-3 | 3.98e-4 / 5.41e-4 / 2.7e-3 | 4.99e-4 / 7.15e-4 / 2.74e-3 |
| B | 6.21e-4 / 8.49e-4 / 5.6e-3 | 4.08e-4 / 5.36e-4 / 2.2e-3 | **4.85e-4 / 6.82e-4 / 1.94e-3** |
Figures (plot_split_AB.py, layout of exploration/plot_p36.py): Agentic_spin_split_plots/split_A_slim_T-2/, split_B_b0_T-3/.
Open: narrow alpha dips just after the peak (s ~ +10..+20 m1) for chi1 < 0 in alpha_beta plots -- not yet traced.
Model docs: scaling_split_A_slim_T-2.md, scaling_split_B_b0_T-3.md (2026-10-15).  B at q = 2.5 / 2: results/diag_AB/lowq_B.json.
2026-10-16: Fig. 10 t_end: B chi1 = -0.5, -0.4 at t_end = -100 re-fitted from the t_end = -50 solution (1.07e-4 -> 2.31e-5, 1.05e-4 -> 2.12e-5; fix_tend_B.py); all figures redrawn.
