# Training-size study (started 2026-10-18)

**Question (user):** published models are trained on the 12 x 8 grid T96. How do the mismatches change with 24, 48 and 192
similarly uniform training points?  Models: pn_anchored_spin, flux_anc_spin_with_orbit, Rink et al. 2024 (form refit),
split A, split B, exploration pass 39 L4 (pass 40 = L4 with a closed-form R2 table, frozen coefficients; add if adopted).

## Decisions (user, 2026-10-18)
* Grid aspect kept ~3 q : 2 chi1: G24 = 6 x 4, G48 = 8 x 6, G96 = T96 (12 x 8), G192 = 16 x 12.  q from 3 to 8 even in nu,
  chi1 from -0.6 to 0.8 evenly spaced.  (T96's q are the historical list, even in nu only to ~1e-3.)
* Scored on a NEW set V300 (300 uniform points, drawn in `grids.py`), not H270 (used for selection) and not the sealed
  R100 / F100.  **V300 is score-only: no fit, start or form choice may use it.**  Closest V300 point to a training point:
  0.0003 in the unit box (q/5, chi1/1.4), to a G192 point.

## Protocol
* Each model refit with its own objective and stages; the shipped T96 coefficients must first reproduce their stored
  per-case T96 mismatches (gate, every run).  Score = full-window E, full alignment (each model's stored scorer).
* Starts: `t96` (shipped coefficients, as every past fit) and `pertN` (perturbed, seed N).  Which run to report per grid
  is chosen on the TRAINING objective only.  Comparing them shows whether small grids just stay in the T96 basin.
* Per model:
  * `pn13`: b1, b2, a2 stay at the nonspinning pn_anchored values (inherited, not from T96); 10 refit.  NM restarts + Powell.
  * `flux_orbit`: all 9 refit.  NM restarts + Powell.
  * `splitA` / `splitB`: stage 1 (inspiral, mean E_insp) ALWAYS refit on the grid, then stage 2 (slim MR, mean E_MR).
    B's window line held at the chosen T0 = -3, T1 = 0 (not re-tuned per grid).
  * `rink`: their Sec. IV procedure on our grid: per-point constant alpha, beta, then least squares of the 12-coefficient
    1/q x chi1 form.  Our scorer and window ([-5000.1, 100] M), not theirs ([-10000, 100] M, q = 3..10).
  * `p39L4`: 18 coefficients refit with A.refit (Powell, budget).  **Caveat: its basis shapes and term choices were derived
    on T96**, so this measures coefficient sensitivity only.

## Fresh-G96 reproduction (user, 2026-10-18) -- must pass before any 24 / 48 / 192 number is trusted
1. **Fresh waveforms:** T96 regenerated from scratch (`TS_FRESH=1`, `cache_fresh/`) exactly as the new grid points were;
   every array compared with spin_dep's old cache (`check_fresh96.py`), and every harness gate rerun on the fresh build.
2. **Fresh refit:** each model refit on G96 by its harness (t96 and perturbed starts) must return the shipped T96
   median / mean / max (training mean no worse than shipped; per-case agreement reported).  Exception: Rink, whose
   original is the published fit (other points, other window); its G96 refit vs published is reported, not gated.

## Files
`grids.py` (grids, V300, cache routing: new points in `cache/`, nothing written to spin_dep/), `common.py`,
`fit_simple.py` (pn13, flux_orbit), `fit_split.py`, `fit_rink.py`, `fit_p39.py`; results `results/<model>__G<N>__<start>.json`;
jobs `../ls6/training_size/`; logs `logs/`.

## Log
* 2026-10-18 job 3518574: caches built (544 new pairs, 37 s); guard clear.
* 2026-10-18 job 3518582: smoke (gates + one short G24 fit per model; results -> results/smoke/).
* 2026-10-18 smoke 3518582: gates pass for pn13, flux_orbit, p39L4, splitA, splitB (shipped / stored T96 within 1e-4);
  Rink crashed (coefficients set after the fork; fixed: they travel with the task).  G24 fit times: pn13 1.3 min,
  flux_orbit 0.7 min, split ~6.5 min, p39 not converged in 8 min (budget raised to 75).  Smoke results in results/smoke/.
* 2026-10-18 main campaign (normal queue), one job per model, G96 first: rink 3518662, pn13 3518663, flux_orbit 3518664,
  splitA 3518665, splitB 3518666, p39 3518667.  Starts t96, pert1, pert2 (rink: t96, pert1).
* 2026-10-18 fresh96 3518596: fresh T96 vs old spin_dep cache differ by <= 2.3e-5 (waveforms) / 8e-6 (drives) relative, all
  96 cases.  On the FRESH build every gate (incl. Rink) reproduces the stored T96 per-case numbers to 1e-6; on the old cache
  only to 1e-4 -- so the stored numbers were made from a build matching today's, not from the 2026-08-30 cache files.
  Decision: the campaign uses TS_FRESH=1 (all grids built by today's code).  Jobs 3518662-67 cancelled before start;
  resubmitted as rink 3518686, pn13 3518687, flux_orbit 3518688, splitA 3518689, splitB 3518690, p39 3518691.
* 2026-10-18 rink 3518686 done (12 min).  Gate 1e-6.  pert1 = t96 at every grid (linear-ish LSQ; start-independent).
  V300 median / mean / max: G24 1.81e-3 / 2.34e-3 / 1.0e-2; G48 1.64e-3 / 2.30e-3 / 1.1e-2; G96 1.69e-3 / 2.30e-3 / 1.1e-2;
  G192 1.65e-3 / 2.29e-3 / 1.1e-2; published (q 3..10, their window) 1.95e-3 / 2.75e-3 / 1.1e-2.  Per-point constant
  (alpha, beta) floor on the training points: median ~1.5e-3 at every grid -- the form, not the data, limits Rink beyond 48.
* 2026-10-18 p39 3518691 (12 runs in series, ~16 h) cancelled before start and split into 12 one-run jobs
  3519392-3519403 (ls6/training_size/run_p39_G<N>_<start>.slurm; logs/p39_G<N>_<start>.<job>.log).
* 2026-10-18 (user: fix it) spin_dep/fit_pn_anchored_spin.py:471: `r = minimize(...)` (was unassigned -> NameError at :475
  on a converged --refit, before writing coeffs_spin.json).  Backup fit_pn_anchored_spin.py.bak_20261018.  pyflakes over
  spin_dep, spin_dep_split, exploration, training_size, root: no other undefined name (one false positive:
  spin_dep_split/fit_pn_anchored_spin_split.py:220 `prev`, only read when rep >= 1, after it is set).
* 2026-10-19 campaign done (pn13, flux_orbit 38 min; split 2.2 h; p39 12 jobs 2-58 min, none budget-limited; 68 results).
  All gates pass (<= 4e-6 on the fresh build).  `summarize.py` -> SUMMARY.md.  Fresh-G96 reproduction: pn13, flux_orbit,
  p39L4 reproduce shipped T96 at ratio 1.000 from all starts; Rink G96 refit 1.99e-3 vs published 2.35e-3 median (not gated).
  **Split A / B did NOT reproduce (2-4% worse; all starts agree).**  Diagnosis (check_split_stage.py, job 3521130):
  not the cache (old and fresh identical); the stage-1 objective depends on the WINDOW (the MR copy carries no SO term,
  so the blend matters even with theta_m = theta_i).  Shipped B's stage 1 ran at the default T = -1 (cv_split, before
  set_tline); the harness ran it at T = -3.  Shipped A's theta_i came from fit_flux_spin_split's iterated insp/mr pipeline
  at T = -1 (with the 9-coefficient MR copy), the harness used explore_split.stage1 at T = -2.  At T = -1 the shipped
  stage-1 values beat the refits (B 3.998e-4 vs 4.044e-4; A origin objective 2.391e-4 vs 2.464e-4).
  Fix: fit_split.py now replays each shipped pipeline (stage 1 at T = -1; A = the origin insp/mr sequence).  Old split
  results -> results/split_v1_stage1window/.  Rerun: splitA G96 3521154 (dev), G24 3521155, G48 3521156, G192 3521163;
  splitB G96 3521165 (dev), G24 3521166, G48 3521167, G192 3521168.
* p39L4 G48 (chosen by training mean, t96 start): V300 median 1.95e-4 vs 1.56e-4 (G24), 1.42e-4 (G96).  Loss at
  chi1 in [-0.25, 0.1) (1.57x G96) and q 5..8; five spin-group coefficients moved 20-30%.  G48's chi1 nodes leave
  (-0.32, -0.04) empty.  pert1 (higher training mean) generalises better (1.75e-4): an overfit of the 18c form at 48 points.
* 2026-10-20 split rerun done (8 jobs, 0 errors).  **Fresh-G96 reproduction now passes for all five gated models**
  (T96 median / mean / max ratio refit / shipped): pn13 1.005 / 1.000 / 0.998; flux_orbit 1.000 / 1.000 / 1.000;
  split A 0.997 / 1.005 / 1.004; split B 1.000 / 1.000 / 1.000; p39L4 1.000 / 1.000 / 1.000.  Rink (not gated): G96 refit
  0.85x the published median.  SUMMARY.md regenerated; figures plots/learning_curve, plots/learning_ratio (plot_curves.py).
  Findings: G24 costs 5-11% on the V300 median (Rink 7%), G48 within 2-4% of G96 (p39 +37%: the overfit above),
  G192 gains <= 3% over G96 for every model.  V300 MAX rises with N for every gradient-fitted model (pn13 0.76 -> 1.14 of
  G96 from G24 to G192): the objective is the training MEAN and the boundary share falls from 67% (6x4) to 27% (16x12),
  so small grids weight the edge corners (where the max lives) more -- a hypothesis, not tested.

**2026-10-22.** User: add the two current exploration candidates to the campaign.  `fit_p40.py p40|lean` (as fit_p39.py):
pass 40 = L4 coefficients + closed-form R2 O2n (gate: exploration/p40/p40.json O2n T96_cases); lean = pass 42 K1, 16c, OT
onset at T_start = -1 (gate: exploration/p42/p42_K1.json).  p42_otonset is imported only for 'lean' (its import widens
the onset guard).  24 jobs 3528234-3528257 (G24/48/96/192 x t96/pert1/pert2, budget 75 min, TS_FRESH=1).
summarize.py: models p40, lean added (skipped until results exist).  Side study the same week: Rink form with chi^3 /
chi^4 in the spin factor, G96 only (fit_rink_chi.py, results_rink_chi/, job 3527185): V300 median -1.4% / -1.2%, mean
-0.5%, max +-1% -- no gain; the fitted spin factor is already ~flat (B ~ 1e-2), the residual is a q x chi1 interaction.
  2026-10-22 results (all 24 done 19:40; every gate <= 5e-6).  G96 refit / shipped T96 median/mean/max: p40 0.999 / 1.000 /
  1.003, lean 1.000 / 1.000 / 1.000.  V300 median G24 / G48 / G96 / G192: p40 1.59 / 1.96 / 1.44 / 1.42 e-4 (p39 1.56 /
  1.95 / 1.42 / 1.39); lean 1.60 / 2.11 / 1.47 / 1.45 e-4.  The G48 bump is in all three (+36-44%) and for lean every start
  reaches the same training optimum (train mean equal to 4 digits) -- so it is the grid (no chi1 node in (-0.32, -0.04)),
  not the optimiser.  Plots not yet updated for p40 / lean.

### 2026-10-28 -- lean12 / lean9 (pass 49b / 49c) on the four grids (user)
- `fit_p40.py` extended (backup `fit_p40.py.bak_20261028`): models `lean12`, `lean9` = pass 40 lean chain, coefficients and
  free set from `exploration/p49b/p49.json` steps n = 12 / 9; dropped coefficients held at 0; gate on that step's T96_cases.
- **Caveat (stronger than for the other models):** which coefficients survive was itself chosen on T96, not only the basis.
  User: if the reruns are bad, settle it later with a pruning run whose drops are chosen on G24 only.
- Jobs: gate 3539001 (development, `ls6/training_size/run_lean129_gate.slurm`); 24 fits 3539002-3539025 (afterok on the gate).
- **Results (2026-10-28).** Gate 3539001: stored lean12 / lean9 reproduce their T96 cases to [0.999998, 1.000002].  All 24 fits
  exit 0; every start reaches the same optimum.  Fresh G96 / shipped (median/mean/max): lean12 0.999/1.000/0.997, lean9 1.001/1.000/1.000.
  V300 median vs own G96 refit (G24 / G48 / G192): lean12 1.12 / 1.24 / 0.98; lean9 1.07 / 1.06 / 0.97
  (lean16 1.09 / 1.44 / 0.98; p40 1.10 / 1.36 / 0.99).  The G48 bump shrinks with the coefficient count (44 -> 25 -> 7 %).
  V300 max rises 11-14 % from G96 to G192 for all four.
- summarize.py / plot_curves.py: lean12, lean9 added (backups *.bak_20261028; old plots in plots/v2_20261022/; the 7-series
  figures are byte-identical).  New figures plots/learning_{curve,ratio}_lean.{pdf,png}: p40, lean, lean12, lean9, slots 1-4.
