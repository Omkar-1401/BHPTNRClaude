# ml_bench — where do generic ML paradigms get on the spin alpha/beta calibration?

Started 2026-10-31 (user).  A **benchmark**, not a race: give each ML paradigm a fair, comparable compute budget,
starting from the same day-1 position as the hand-built models, and report where it lands next to pass 40/48,
the lean models and flux_anc_spin_with_orbit.  **Do not spend extra compute pushing an ML method toward our
baseline** (user, 2026-10-31).  Nothing here changes an installed model or `exploration/`.

## Day-1 starting point (what the ML methods get)

- ppBHPT (BHPTNRSur2dq1e3, uncalibrated) / NR (NRHybSur3dq8, chi2 = 0) (2,2) waveform pairs; training set size free
  (user: not restricted to 96 points; cf. Zhong+ 2503.04534 who used a larger set).
- The map h_NR(tau) ~ alpha e^{i phi0} h_pp(t), tau = t0 + int beta dt; phi0 analytic, t0 per case.
- The repo's error E (`spin_dep/fit_pn_anchored_spin._err_at_t0`, imported, never re-implemented).
- Generic choices only: time aligned at the ppBHPT peak (s = t - t_peak,pp); feature scaling; nu -> 0 limit as a
  variant.  NOT given: X1^(6/5), beta_r, R2, OT/ISCO onsets, flux/energy drives, any fitted coefficient.

## Budget (grounded in the hand-built fits)

Production hand fits on one LS6 node: pass 40 T96 5.7-28 min, flux_orbit 38 min (sacct).  So per paradigm:
**<= ~1 node-hour of training** (+ waveform generation for a larger training set at similar scale), a small fixed
hyperparameter search, no marathons.  Whole line ~30 SU.  Report training and evaluation cost with every result.

## Held-out discipline (as in the rest of the repo)

Selection / hyperparameters on training-set CV only.  **V300 (`training_size/V300.txt`) is score-only.**  R100 / L20
sealed (`exploration/test_guard.py` refuses them; never train on them); one look at the end.  H270 may be used as
training data here.

## Paradigms (order may change with results)

1. per-case free alpha(s), beta(s) splines (from alpha = beta = 1) -> regress knot values / PCA coefficients over
   (q, chi1) with KNN, RF, ExtraTrees, XGBoost, GPR, RBF, polynomial, PySR; feature sets (q|nu|1/q|log q, chi1).
2. end-to-end neural (MLP / RNN in the spirit of Zhong+) trained directly on the waveform loss, if a differentiable
   map is cheap enough within budget.
3. PySR closed forms for whatever representation works best (interpretability comparison with the hand forms).

## Environment

`ut_claude` + pysr 2.7.2, juliacall, xgboost-cpu 3.2.0 (added 2026-10-31; freeze diff: new packages only, mpmath
conda build -> pip 1.3.0).  Julia depot `$WORK/../stampede3/julia_depot`, project `.../julia_env_pysr` (keep off
$HOME).  Job scripts in `ls6/ml_bench/`.

## Log

### Reference: hand-built models on V300 (training_size/SUMMARY.md, fit on T96; median / mean / max)

| model | coef | V300 | fit cost (1 LS6 node) |
|:--|--:|:--|:--|
| pass 40 | 18 | 1.44e-4 / 1.89e-4 / 9.4e-4 | 6-28 min |
| lean-16 (pass 42 K1) | 16 | 1.47e-4 / 1.96e-4 / 8.9e-4 | similar |
| lean-12 / lean-9 | 12 / 9 | 2.29e-4 / 2.96e-4 / 8.9e-4; 3.59e-4 / 4.27e-4 / 1.1e-3 | similar |
| flux_anc_spin_with_orbit | 9 | 7.79e-4 / 8.75e-4 / 4.7e-3 | 38 min |
| pn_anchored_spin | 13 | 7.99e-4 / 9.17e-4 / 4.5e-3 | |
| Rink et al. form (refit G96) / published | 12 | 1.69e-3 / 2.30e-3 / 1.1e-2; 1.95e-3 / 2.75e-3 / 1.1e-2 | |

(The hand forms were found over ~51 passes / ~300 SU of exploration; the fit cost above is for refitting a known form.)

### 2026-10-31  setup, targets, scorer

- **Scorer** `common.py`: the chain's protocol (p6_m2_global.err) around S._err_at_t0.  `gate.py`: pass 40 through
  score_ab reproduces exploration/p40/p40.json T96_cases at 0.99997-1.00001 (4 cases; the stand-alone generator equals
  the chain to 2e-9 in alpha, beta, the rest is t0-search detail); score_wave (model rendered as a waveform) within 2e-5.
- **Training pool** 714 cached in-range pairs (all caches) minus V300 minus sealed.  V300 = score set (300).
- **Paradigm A targets** (`direct_targets.py`, Zhong-style, no map, no fitting): dlnA, dphi on a peak-aligned grid in M.
  Grid must be 0.2 M from -200 M: retrograde ringdown pp |h| -> 5e-6 (own-target E at q = 3, chi1 = -0.6: 1 M 5.8e-5,
  0.25 M 3.7e-7); with it own-target E <= 1.2e-7.  Extracted 714 + 300 in ~1 min (job 3545184).
- **Paradigm B targets** (`ab_targets.py`): equal-phase alpha(t), beta(t) on the hand models' clock s (m1) -- exact through
  the repo's map, one gauge for all cases, no fitting.  Own-target E 7e-9..3e-8, except the retrograde ringdown
  (q = 3, chi1 = -0.6: 4.2e-5; q = 8: 8.5e-7; non-monotone phase handled by a running max) -- a known floor of this target.
- **Paradigm C** `nn_direct.py`: pointwise MLP (277k params) and Zhong-style CfC (ncps AutoNCP 64, 16.1k params = Zhong's
  spin model size) on the paradigm-A targets, 25 min each per loss variant.  ncps 1.0.1 batched-timespans bug: step size fed
  as an input channel instead.  numpy must be imported before torch (system libstdc++).
- **Paradigm D** `pysr_bench.py`: PySR on the PCA coefficients over (nu, chi1).
- Packages added: torch 2.15.0+cpu, ncps 1.0.1 (+ Jinja2, MarkupSafe, networkx); new packages only.
- Smoke (150 training cases, K = 6, nu features, 3 test cases): poly 1.6-3.2e-4, rbf 2.6-3.9e-4, gpr 3.0-4.6e-4,
  xgb 4.5-7.2e-4, knn 0.6-3.5e-3.

### 2026-10-31/11-01  paradigm A (direct dlnA, dphi; PCA K + regressor), job 3545186

CV (5-fold on the 714 pool) chose K = 20 for every method (largest tried; K = 6 / 12 / 20 monotone).  Configs: poly q, knn logq, rf q, et invq, xgb q, gpr nu, rbf q, mlp invq.

V300 learning curve (random nested subsets of the pool, seed 20261031; median / mean / max).  pass 40 column: its
refit on the training_size grid of the same size (G48 / T96 / G192), from training_size/SUMMARY.md.

| method | N = 48 | N = 96 | N = 192 | N = 384 | N = 714 | train s (714) |
|:--|--:|--:|--:|--:|--:|--:|
| poly | 2.33e-04 / 3.03e-04 / 4.0e-03 | 4.73e-05 / 9.42e-05 / 1.5e-03 | 3.34e-05 / 8.70e-05 / 1.9e-03 | 2.38e-05 / 7.11e-05 / 1.6e-03 | 2.44e-05 / 6.83e-05 / 1.0e-03 | 2 |
| knn | 4.49e-04 / 2.41e-03 / 5.7e-02 | 1.79e-04 / 4.97e-04 / 9.5e-03 | 9.18e-05 / 2.22e-04 / 7.6e-03 | 6.16e-05 / 1.56e-04 / 5.3e-03 | 3.14e-05 / 7.67e-05 / 1.1e-03 | 2 |
| rf | 1.45e-03 / 3.86e-03 / 8.0e-02 | 7.14e-04 / 2.98e-03 / 5.6e-02 | 1.10e-04 / 3.77e-04 / 8.2e-03 | 5.71e-05 / 1.86e-04 / 7.7e-03 | 3.68e-05 / 8.67e-05 / 3.7e-03 | 19 |
| et | 5.98e-04 / 2.53e-03 / 4.4e-02 | 3.30e-04 / 1.25e-03 / 1.5e-02 | 3.92e-05 / 1.48e-04 / 4.1e-03 | 1.60e-05 / 7.02e-05 / 5.1e-03 | 1.17e-05 / 2.74e-05 / 1.0e-03 | 18 |
| xgb | 2.78e-04 / 8.60e-04 / 1.1e-02 | 2.34e-04 / 4.44e-04 / 4.3e-03 | 1.94e-04 / 4.23e-04 / 5.2e-03 | 1.58e-04 / 3.18e-04 / 4.3e-03 | 1.26e-04 / 2.60e-04 / 2.5e-03 | 21 |
| gpr | 1.03e-04 / 2.79e-04 / 2.7e-03 | 2.49e-05 / 1.69e-04 / 1.9e-03 | 1.01e-05 / 5.01e-05 / 1.0e-03 | 1.13e-05 / 4.64e-05 / 1.6e-03 | 1.16e-05 / 3.86e-05 / 1.6e-03 | 391 |
| rbf | 4.32e-05 / 1.00e-04 / 8.3e-04 | 2.98e-05 / 7.40e-05 / 9.1e-04 | 1.44e-05 / 5.62e-05 / 1.0e-03 | 9.24e-06 / 3.55e-05 / 1.1e-03 | 1.03e-05 / 2.24e-05 / 7.7e-04 | 2 |
| mlp | 2.81e-04 / 8.00e-04 / 1.2e-02 | 1.33e-04 / 6.72e-04 / 1.9e-02 | 4.91e-05 / 2.38e-04 / 4.8e-03 | 5.38e-05 / 2.15e-04 / 9.6e-03 | 4.76e-05 / 1.59e-04 / 4.5e-03 | 193 |
| *pass 40 (18c)* | 1.96e-4 / 2.20e-4 / 7.5e-4 | 1.44e-4 / 1.89e-4 / 9.4e-4 | 1.42e-4 / 1.86e-4 / 1.1e-3 |  |  | 340-1700 |

Reading: (1) with the dense 714 pool every method beats pass 40 on V300 median and mean; RBF also on max (7.7e-4).
V300 points sit a median 0.019 (unit box) from the pool vs 0.049 from T96 -- dense-data interpolation of a smooth
surrogate (NRHybSur3dq8), ~2.5e5 stored numbers vs 18.  (2) Data efficiency splits the methods: RBF beats pass 40 on
median / mean already at N = 48 (max within 11%); GPR and poly at 96 on median (max 2x worse); trees, KNN, MLP, XGBoost
need 192-384+ and keep 4-20x worse maxima.  (3) CHECK PENDING (job 3545216): same exact grids as the hand refits
(G24 / G48 / T96 / G192) + 5 random draws per size, and extrapolation (q < 3, q = 9, 10) on the same points as pass 40.

### 2026-11-01  robustness: SAME grids as the hand refits (job 3545216)

V300 median / mean / max.  'grid' = trained on exactly G24 / G48 / T96 / G192 (the training_size grids pass 40 was
refit on); 'random' = median over 5 random pool draws of the same size (worst max in brackets).  Inner hyperparameters
re-chosen on each training set; features / K from the 714-pool CV.

| N | pass 40 (refit, same grid) | RBF grid | RBF random | GPR grid | GPR random | poly grid | poly random |
|--:|:--|:--|:--|:--|:--|:--|:--|
| 24 | 1.6e-04 / 2.0e-04 / 1.2e-03 | 1.2e-04 / 1.8e-04 / 1.2e-03 | 6.8e-05 / 1.8e-04 (8.9e-03) | 3.8e-04 / 4.1e-04 / 2.1e-03 | 3.9e-04 / 7.2e-04 (1.9e-01) | 4.6e-04 / 5.6e-04 / 2.5e-03 | 2.2e-04 / 3.2e-04 (7.7e-03) |
| 48 | 2.0e-04 / 2.2e-04 / 7.5e-04 | 7.4e-05 / 1.7e-04 / 1.4e-03 | 3.1e-05 / 1.3e-04 (4.9e-03) | 4.0e-04 / 6.3e-04 / 3.4e-03 | 7.1e-05 / 3.0e-04 (3.1e-02) | 3.1e-04 / 4.0e-04 / 2.5e-03 | 1.9e-04 / 3.0e-04 (5.4e-03) |
| 96 | 1.4e-04 / 1.9e-04 / 9.4e-04 | 2.5e-05 / 8.0e-05 / 1.0e-03 | 2.4e-05 / 7.6e-05 (5.0e-03) | 7.2e-05 / 1.9e-04 / 3.7e-03 | 2.3e-05 / 1.2e-04 (6.0e-03) | 2.7e-04 / 3.4e-04 / 3.3e-03 | 4.3e-05 / 1.2e-04 (4.5e-03) |
| 192 | 1.4e-04 / 1.9e-04 / 1.1e-03 | 1.8e-05 / 6.9e-05 / 9.5e-04 | 1.2e-05 / 5.6e-05 (1.6e-03) | 6.6e-05 / 2.0e-04 / 5.8e-03 | 1.1e-05 / 6.0e-05 (2.0e-03) | 4.4e-05 / 9.1e-05 / 1.4e-03 | 3.2e-05 / 8.6e-05 (2.2e-03) |

Reading: on the identical training grids, RBF beats pass 40 on median at every N (1.3x at 24 -> 8x at 192) and on mean
(1.1-2.7x), but its max is level or worse up to T96 (1.9x worse at G48) and only better at G192.  GPR and poly do WORSE
on the regular grids than on random draws (inner CV picks degree 2 for poly on G24-T96; GPR length scales poorly
constrained on a grid) and lose to pass 40 on the grids up to T96 on mean and max (poly on all three statistics).  The
earlier 'poly / GPR beat pass 40 at 96' came from random point placement.  Random draws have much worse tails
(maxima up to 1e-2-1e-1 at N <= 48).

### 2026-11-01  extrapolation on the same points as pass 40 (job 3545228)

Sets: lowq 2.5 <= q < 3 (48), vlowq q = 2.0, 2.32 (46; BHPT input itself extrapolated), highq q = 9, 10 (16, the pass-44
points; NRHybSur3dq8 extrapolated).  ML trained on the [3, 8] pool (714) or 96 random pool points; median / mean / max.

| model | N | lowq | vlowq | highq | V300 |
|:--|--:|:--|:--|:--|:--|
| **pass 40** (18c) | 96 (T96) | 8.9e-04 / 1.0e-03 / 4.6e-03 | 8.4e-03 / 8.9e-03 / 2.3e-02 | 3.2e-04 / 1.3e-03 / 1.2e-02 | 1.4e-04 / 1.9e-04 / 9.4e-04 |
| gpr | 714 | 1.7e-04 / 5.0e-04 / 7.6e-03 | 3.2e-03 / 3.9e-03 / 2.8e-02 | 9.5e-05 / 2.4e-04 / 1.5e-03 | (final table) |
| gpr | 96 | 8.5e-04 / 2.0e-03 / 1.6e-02 | 1.1e-02 / 1.5e-02 / 9.0e-02 | 1.9e-04 / 4.1e-04 / 2.8e-03 | 2.5e-05 / 1.7e-04 / 1.9e-03 |
| rbf | 714 | 2.4e-04 / 6.3e-04 / 1.1e-02 | 5.8e-03 / 7.4e-03 / 5.3e-02 | 1.3e-03 / 8.7e-03 / 1.2e-01 | (final table) |
| rbf | 96 | 2.2e-04 / 7.4e-03 / 2.5e-01 | 2.3e-03 / 3.4e-02 / 5.7e-01 | 5.6e-03 / 7.9e-03 / 5.6e-02 | 3.0e-05 / 7.4e-05 / 9.1e-04 |
| poly | 714 | 1.6e-04 / 2.2e-03 / 6.7e-02 | 1.7e-03 / 2.0e-02 / 6.5e-01 | 2.7e-01 / 3.0e-01 / 7.0e-01 | (final table) |
| poly | 96 | 9.1e-04 / 1.2e-03 / 9.2e-03 | 9.3e-03 / 9.5e-03 / 2.3e-02 | 6.7e-03 / 1.2e-02 / 4.5e-02 | 4.7e-05 / 9.4e-05 / 1.5e-03 |
| et | 714 | 1.1e-02 / 1.1e-02 / 3.4e-02 | 1.2e-01 / 1.0e-01 / 2.3e-01 | 7.1e-03 / 7.8e-03 / 3.5e-02 | (final table) |
| et | 96 | 3.0e-02 / 3.9e-02 / 1.0e-01 | 1.6e-01 / 1.7e-01 / 3.0e-01 | 1.5e-02 / 2.2e-02 / 9.1e-02 | 3.3e-04 / 1.2e-03 / 1.5e-02 |
| knn | 714 | 1.0e-02 / 9.3e-03 / 3.0e-02 | 1.2e-01 / 9.7e-02 / 1.7e-01 | 7.2e-03 / 8.3e-03 / 3.9e-02 | (final table) |
| knn | 96 | 2.0e-02 / 2.6e-02 / 7.5e-02 | 1.3e-01 / 1.4e-01 / 2.7e-01 | 8.5e-03 / 1.3e-02 / 7.0e-02 | 1.8e-04 / 5.0e-04 / 9.5e-03 |
| rf | 714 | 1.3e-02 / 1.3e-02 / 4.0e-02 | 1.2e-01 / 1.1e-01 / 2.2e-01 | 7.9e-03 / 9.3e-03 / 4.1e-02 | (final table) |
| rf | 96 | 4.7e-02 / 5.2e-02 / 1.5e-01 | 1.8e-01 / 1.9e-01 / 3.1e-01 | 1.4e-02 / 2.3e-02 / 8.6e-02 | 7.1e-04 / 3.0e-03 / 5.6e-02 |
| xgb | 714 | 1.0e-02 / 9.5e-03 / 3.1e-02 | 1.1e-01 / 9.9e-02 / 1.8e-01 | 7.4e-03 / 8.0e-03 / 3.6e-02 | (final table) |
| xgb | 96 | 1.2e-02 / 1.6e-02 / 5.0e-02 | 1.2e-01 / 1.2e-01 / 2.1e-01 | 7.2e-03 / 1.1e-02 / 5.6e-02 | 2.3e-04 / 4.4e-04 / 4.3e-03 |
| mlp | 714 | 9.2e-04 / 3.1e-03 / 2.1e-02 | 3.9e-02 / 4.8e-02 / 1.2e-01 | 1.6e-03 / 3.3e-03 / 1.9e-02 | (final table) |
| mlp | 96 | 8.3e-03 / 2.0e-02 / 1.9e-01 | 8.0e-02 / 1.1e-01 / 2.9e-01 | 3.5e-03 / 9.1e-03 / 6.3e-02 | 1.3e-04 / 6.7e-04 / 1.9e-02 |

Reading: trees, KNN, XGBoost hold the edge value -> 1e-2 at q < 3 and 7e-3 at q = 9, 10 regardless of N (10-25x pass 40).
poly (714) and RBF lose control of the tails (poly highq median 0.27; RBF-96 lowq max 0.25, vlowq max 0.57).  MLP 3-20x
worse.  **GPR is the exception**: with nu features the extrapolation distance is short in nu, and at N = 714 it beats pass 40
at high q on all three statistics and at low q on median / mean (max 1.6x worse); at N = 96 it still beats pass 40 at high q
(1.9e-4 / 4.1e-4 / 2.8e-3) but loses at low q on mean / max (2x / 3.4x).

### 2026-11-01  paradigm B: ML on alpha, beta through the repo's map (jobs 3545225, 3545235)

Targets: equal-phase alpha(t), beta(t) on s (m1).  'ln beta' = regress ln alpha, ln beta (PCA K); 'clock' = regress ln alpha
and tau(s) - s, beta = dtau/ds (floor 1e-3).  714 pool, CV-best config per method, V300 median / mean / max.
Own-PCA-projection floor (no regression), 5 cases: ln beta K = 20 3e-4..6e-2, K = 60 1e-5..1.7e-4 -- the K <= 20
ln beta runs were REPRESENTATION-limited (beta errors integrate into the clock; the pass-37 lesson).

| method | ln beta (K <= 60, CV-best) | clock tau (K <= 20) | direct dlnA, dphi (paradigm A) |
|:--|:--|:--|:--|
| rbf | 1.1e-04 / 2.1e-04 / 2.5e-03 (rbf|invq|60) | 3.5e-05 / 5.1e-05 / 9.0e-04 (rbf|invq|20) | 1.0e-05 / 2.1e-05 / 6.3e-04 |
| poly | 3.5e-04 / 4.9e-04 / 7.6e-03 (poly|q|60) | 1.1e-04 / 2.4e-04 / 5.2e-03 (poly|q|20) | 2.4e-05 / 6.8e-05 / 1.0e-03 |
| et | 2.3e-04 / 1.5e-03 / 9.2e-02 (et|invq|60) | 2.9e-04 / 5.3e-03 / 2.9e-01 (et|nu|12) | 1.2e-05 / 2.7e-05 / 1.0e-03 |
| knn | 7.7e-04 / 2.3e-03 / 4.9e-02 (knn|invq|40) | 9.1e-04 / 2.7e-03 / 3.5e-02 (knn|invq|20) | 3.1e-05 / 7.7e-05 / 1.1e-03 |
| rf | 1.4e-03 / 5.4e-03 / 2.4e-01 (rf|nu|40) | 1.2e-03 / 4.5e-03 / 1.2e-01 (rf|q|12) | 3.6e-05 / 8.6e-05 / 3.4e-03 |
| gpr | 2.4e-02 / 4.8e-02 / 3.3e-01 (gpr|invq|60) | 5.0e-03 / 1.2e-02 / 1.2e-01 (gpr|logq|20) | 1.2e-05 / 3.9e-05 / 1.6e-03 |
| xgb | 3.2e-03 / 1.1e-02 / 1.7e-01 (xgb|nu|60) | 5.6e-03 / 1.4e-02 / 1.6e-01 (xgb|q|20) | 1.2e-04 / 2.4e-04 / 2.1e-03 |
| mlp | 3.5e-03 / 7.1e-03 / 1.0e-01 (mlp|nu|60) | 1.6e-03 / 3.1e-03 / 3.3e-02 (mlp|q|6) | 5.5e-05 / 1.1e-04 / 3.5e-03 |
| *pass 40 (T96, 18c)* | 1.4e-04 / 1.9e-04 / 9.4e-04 | | |

Reading: the target matters more than the regressor.  Regressing the clock instead of ln beta gains 3-70x; with it RBF
beats pass 40 on all three V300 statistics (max marginally) at N = 714.  Through the map, every method is still 2-10x
worse than the same method on the direct amplitude/phase targets: the equal-phase gauge puts the retrograde-ringdown
phase plateaus and window-edge holds into the curves, which PCA spends components on.

### 2026-11-01  paradigm D: PySR on the paradigm-A PCA coefficients (K = 12, job 3545274)

24 searches (one per coefficient), inputs (nu, chi1), operators + - * / square cube sqrt exp, maxsize 30, 150 s each
(70 min total); expression per coefficient chosen on a 20 % validation split of the pool.

**V300 1.04e-04 / 8.08e-03 / 6.39e-01**, total complexity 619 (poly at the same K = 12, CV: 5.1e-5 / 8.8e-5 / 2.3e-3; pass 40: 1.44e-4 / 1.89e-4 / 9.4e-4).

| coef | complexity | val MSE / var |
|:--|--:|--:|
| A0 | 29 | 4.83e-03 |
| A1 | 30 | 4.64e-03 |
| A2 | 27 | 1.80e-01 |
| A3 | 30 | 3.54e-01 |
| A4 | 29 | 1.61e-01 |
| A5 | 26 | 4.93e-01 |
| A6 | 30 | 4.98e-01 |
| A7 | 23 | 3.29e-01 |
| A8 | 6 | 9.51e-01 |
| A9 | 24 | 5.52e-01 |
| A10 | 30 | 4.73e-01 |
| A11 | 1 | 1.01e+00 |
| P0 | 27 | 3.84e-03 |
| P1 | 29 | 4.42e-03 |
| P2 | 27 | 1.57e-02 |
| P3 | 30 | 1.07e-01 |
| P4 | 30 | 1.74e-01 |
| P5 | 22 | 1.10e-01 |
| P6 | 27 | 2.37e-01 |
| P7 | 29 | 4.16e-01 |
| P8 | 30 | 3.76e-01 |
| P9 | 28 | 8.38e-01 |
| P10 | 30 | 1.05e-01 |
| P11 | 25 | 5.21e-01 |

Reading: the leading coefficients are captured (A0, P0: 99.5 % of variance) but from the third coefficient on the searches
leave 16-50 % unexplained and reach for numerically pathological forms (exp(-3e22 nu^27), 18th powers) that act as hidden
steps -> the mean is 43x pass 40's and the max 0.64.  619 nodes over 24 abstract PCA coefficients are not interpretable;
pass 40's 18 coefficients each sit on a named physical term.  Within the budget, symbolic regression does not compete on
either axis.  Expressions: results/pysr_direct_K12.json.

### 2026-11-01  paradigm C: neural networks on the direct targets (jobs 3545275, 3545276, 3545368)

Split 572 train / 142 validation of the pool; 25 min CPU training per loss variant (64 threads); variant chosen on
validation E.  'w' = loss weighted by the mean NR power density |h_NR(s)|^2 (what E weights); 'mse' = plain (Zhong+).

| model | params | steps | val E (mse) | val E (w) | V300 (chosen) |
|:--|--:|--:|:--|:--|:--|
| pointwise MLP | 276994 | 54518 | 2.1e-05 / 2.8e-05 / 1.6e-04 | 2.8e-06 / 7.2e-06 / 7.8e-05 | 3.1e-06 / 2.1e-05 / 5.7e-04 (w) |
| CfC (Zhong-style) | 16126 | 366 | 1.1e-02 / 1.9e-02 / 1.0e-01 | 6.9e-03 / 1.5e-02 / 9.7e-02 | 6.1e-03 / 1.1e-02 / 1.0e-01 (w) |

Weighted MLP, extrapolation and pass 40's own training set (same points as the pass-40 rows):

| model | training | lowq | vlowq | highq | V300 |
|:--|:--|:--|:--|:--|:--|
| pass 40 (18c) | T96 | 8.9e-04 / 1.0e-03 / 4.6e-03 | 8.4e-03 / 8.9e-03 / 2.3e-02 | 3.2e-04 / 1.3e-03 / 1.2e-02 | 1.4e-04 / 1.9e-04 / 9.4e-04 |
| MLP-w | 572 pool | 9.8e-05 / 2.3e-04 / 3.3e-03 | 4.1e-03 / 7.7e-03 / 9.8e-02 | 6.2e-05 / 1.4e-04 / 1.3e-03 | 3.1e-06 / 2.1e-05 / 5.7e-04 |
| MLP-w | T96 (same as pass 40) | 2.8e-04 / 7.8e-04 / 9.2e-03 | 1.9e-02 / 2.0e-02 / 1.2e-01 | 3.3e-05 / 1.2e-04 / 1.1e-03 | 2.8e-05 / 2.1e-04 / 4.5e-03 |

Reading: the CfC is badly under-trained within the budget (~360 steps: the recurrence over ~1500 steps costs ~4 s/step on
CPU) and lands at Zhong+'s quoted level (~5e-3).  The pointwise MLP with the E-weighted loss is the strongest ML model
found: at N = 572 it beats pass 40 on every statistic in interpolation and at low and high q (only the vlowq max is worse,
where the BHPT input itself is extrapolated).  **Trained on exactly T96** it beats pass 40 on the V300 median (5x) and at
high q on all three statistics, ties the V300 mean, and loses the V300 max (4.8x), the lowq max (2x) and all of vlowq.
The loss weighting, not the architecture, made the difference (val median 2.1e-5 -> 2.8e-6).

## SUMMARY (2026-11-01; 5.2 node-hours ~ 5 SU of the ~30 SU budget)

Day-1 start for every method (raw ppBHPT / NR pairs, the map ansatz or none, no hand-built physics); E identical to the
exploration's (gate.py); V300 score-only; R100 / L20 still sealed (not opened by this line).

1. **Dense data, interpolation (V300, N = 572-714):** nearly every generic method beats pass 40 on median and mean; RBF,
   GPR, ExtraTrees reach ~1e-5, the E-weighted MLP 3.1e-6 / 2.1e-5 / 5.7e-4 (all three statistics better).  The truth
   is itself a smooth surrogate (NRHybSur3dq8) sampled densely (V300 sits 0.019 from the pool vs 0.049 from T96).
2. **Same training set as pass 40 (T96, and G24 / G48 / G192):** only RBF (PCA K = 20) and the E-weighted MLP beat pass 40
   on the median (5-6x); neither beats its worst case (V300 max 1.0e-3 and 4.5e-3 vs 9.4e-4).  GPR and poly lose on the
   grids on mean and max; their small-N 'wins' came from random point placement.
3. **Extrapolation (q = 2.5-3, 2-2.3, 9-10):** trees / KNN / XGBoost hold the edge value (10-25x worse); poly and RBF lose
   the tails (max up to 0.7).  GPR (nu features) and the E-weighted MLP extrapolate well with dense data; on T96 the MLP
   still wins at high q but loses at low q on the max and at q < 2.5 everywhere.
4. **Through the alpha-beta map:** the target choice dominates -- regressing the clock tau instead of ln beta gains 3-70x
   (beta errors integrate, as pass 37 found); best (RBF, 714) 3.5e-5 / 5.1e-5 / 9.0e-4.
5. **Interpretability:** PySR on the PCA coefficients gives 24 expressions, complexity 619, V300 1.0e-4 / 8.1e-3 / 0.64
   (pathological forms); the Zhong-style CfC is under-trained at this budget (6.1e-3).
6. **Size:** generic models store ~1e5 numbers (PCA bases + regressor; the MLP 2.8e5 weights) vs pass 40's 18 coefficients
   with physical meaning, the nu -> 0 limit built in, and the best worst-case on its own training set.

Bottom line for the paper: with many cheap training waveforms, generic interpolation (especially an E-weighted MLP)
surpasses the 18-coefficient forms on typical accuracy, because the hand forms sit at a form-limited floor (~1.4e-4, flat
from 96 to 192 points).  On the hand models' own training set the generic methods win the median but not the worst case,
lose at q < 2.5, and offer no physical content.

## Paradigm E: ML alpha, beta fitted THROUGH THE MAP (2026-11-01; the intended benchmark)

User clarification: the benchmark must replace only the hand-written forms of the exploration.  Same map
(tau = t0 + int beta, h_NR ~ alpha e^{i phi0} h_pp), same objective (mean E over T96 at fit stride, phi0 analytic, t0 per
case by the scorer's grid search), global alpha(t; nu, chi1), beta(t; nu, chi1), V300 scored by common.score_ab.  The
earlier paradigms A-D (direct corrections; regression on model-free equal-phase alpha, beta) are NOT this; they stay as
diagnostics / initialisation.  Gauge note (ab_compare.py): alpha, beta are defined only up to the phi0 / t0 freedom;
pass 40's fitted pair sits ~2.5-3 rad from the peak-zeroed equal-phase gauge, so curves are compared in a matched gauge.

* efit.py: torch E == fit_pn_anchored_spin._err_at_t0 (gate: 4 T96 cases, pass 40 alpha/beta, ratio 1 +- 6e-11).
  Time grid S_C: 1500 pts uniform in warp(s) on [-8000, 0) m1 + 400 on [0, 130]; representation floor (model-free
  alpha, beta interpolated from S_C) T96 2.8e-7 / 2.7e-6 / 3.9e-5.  (600 pts: 1.2e-4 median -- a first MLP/GRU job on that
  grid was cancelled; a 300-function B-spline basis: ~2e-4 by projection, so no basis is used.)
  mlp (pointwise, 54k params) and gru (bidirectional, sequence) trained end-to-end on T96, start = identity map
  (alpha = 1, beta = X1), t0 learnable + re-seeded by grid search every 150 steps.
* efit_sr.py: PySR distils ln alpha, ln(beta/X1) in (u, nu, chi1) from the trained mlp; <= 4 candidate pairs (complexity
  <= 12/20/28/35) refitted through E on T96; chosen on T96 mean E; V300 reported for every candidate (trade-off only).
* efit_lin.py: KNN / RBF / GPR / RF / ET / XGB / sparse library -- each linear in a value matrix W once its structure is
  fixed (neighbours, kernel, splits, monomials); structure + start from a fit to the model-free T96 alpha, beta; W refined
  through E on T96 (8 min each); sparse: L1 then prune < 1 % of max, refit.
Jobs 3545718 (mlp, gru, sr), 3545720 (lin).

### Paradigm E results log (V300 median / mean / max; pass 40 1.44e-4 / 1.89e-4 / 9.41e-4)
* mlp, identity start, 30 min (3200 steps): T96 7.3e-4 / 3.2e-3 / 3.0e-2; V300 6.1e-4 / 2.1e-3 / 3.2e-2.  Settles in a
  different solution from pass 40 (beta spikes to ~2.4 at merger, alpha ~10 in the ringdown), rough curves.
* gru, identity start, 30 min, batch 32 (~900 steps): V300 1.6e-2 / 1.8e-2 / 5.3e-2 (under-trained at this budget).
* PySR distilled from that mlp (rel. MSE of the distillation 1.4e-2 / 3.2e-2), refit through E: complexity 24 / 40 / 56 / 70
  -> V300 median 1.1e-1 / 6.3e-2 / 5.4e-2 / 4.5e-2 (chosen on T96: 70, 13 constants).
* knn 2.5e-4 / 3.9e-4 / 5.3e-3 (init 2.8e-4); rbf 7.9e-5 / 1.8e-4 / 1.6e-3 (init 9.4e-5); gpr 5.4e-5 / 1.4e-4 / 1.7e-3
  (init 6.3e-4); rf 4.5e-4 / 9.0e-4 / 8.8e-3 (init 1.7e-3); et 1.6e-4 / 4.3e-4 / 3.7e-3 (init 1.7e-3); xgb 9.8e-4 / 1.5e-3 /
  9.9e-3 (init 1.1e-3); sparse Legendre (15 terms, group pruning kept 15 + 14) dense 2.1e-4 / 2.8e-4 / 1.6e-3 -> pruned +
  refit 1.9e-4 / 2.5e-4 / 1.2e-3 (init 7.4e-4).  T96 medians: knn / rbf / xgb / gpr ~1e-6, et / rf ~1e-5, sparse 1.6e-4.
* Bugs found and fixed on the way (each re-run): rbf and xgb refined in their raw weights diverged (E -> 0.1-0.4) ->
  cardinal / dual form (values at the T96 points); sparse with elementwise L1 + pruning broke beta (E 0.22) -> group
  pruning; efit.py json key clash (models saved; re-scored by efit_rescore.py); the model-free-start mlp diverged at
  lr 2e-3 (fine-tuning needs 2e-4; 8-case check: 2e-3 -> 0.48, 2e-4 -> 7e-4 in 30 s).
* mlp, model-free start (pre-fit 8 min to the model-free alpha, beta, then E at lr 2e-4, 25 min): T96 4.1e-4 / 5.4e-4 /
  2.6e-3; V300 3.9e-4 / 5.2e-4 / 2.8e-3 (better than pass 40 in 4 % of V300).
* PySR from it (job 3546008, with step rejection): complexity 24 / 40 / 56 / 70 -> V300 median 2.8e-1 / 1.0e-1 / 3.2e-2 /
  3.6e-2 (70: one V300 case E = 1); chosen on T96: 56 (12 constants), V300 3.2e-2 / 3.7e-2 / 1.0e-1.
* efit.train now has step rejection (restore last good snapshot, halve lr) after the no-gradient crash: a divergent step
  can push every tau out of the NR window.

## SUMMARY, paradigm E (2026-11-01; ~5 node-hours; figures plot_efit.py -> plots/figE*.png; table results/efit_summary.json)

Same map, objective (mean E on T96), phi0 / t0 handling and scorer as pass 40; only the function class changes.
V300 median / mean / max (pass 40 1.44e-4 / 1.89e-4 / 9.41e-4, 18 coefficients):
    gpr 5.4e-5 / 1.4e-4 / 1.7e-3   rbf 7.9e-5 / 1.8e-4 / 1.6e-3   et 1.6e-4 / 4.3e-4 / 3.7e-3   sparse 1.9e-4 / 2.5e-4 / 1.2e-3
    knn 2.5e-4 / 3.9e-4 / 5.3e-3   mlp-mf 3.9e-4 / 5.2e-4 / 2.8e-3   rf 4.5e-4 / 9.0e-4 / 8.8e-3   mlp-id 6.1e-4 / 2.1e-3 / 3.2e-2
    xgb 9.8e-4 / 1.5e-3 / 9.9e-3   gru 1.6e-2   PySR 3.2e-2 (mf) / 4.5e-2 (id)
1. No ML class beats pass 40's worst case or its smoothness (rms 2nd difference of ln beta pre-merger 25-5000x larger).
2. GPR / RBF (cardinal form, 364,800 values) win the typical case (2.7x / 1.8x on the median; better in 79 % / 73 % of
   V300) and lose at the edges of the range (chi1 ~ -0.6, highest spins).
3. The sparse Legendre library (15 terms in (nu, chi1), each a time curve) is closest overall: within ~1.3x of pass 40 on
   all three statistics, no train / test gap.  Group pruning at 5 % removes almost nothing (15 + 14 terms kept).
4. E refinement matters for the smooth fits (gpr 11.6x, et 10x, sparse 4x, rf 3.7x better on V300 than the same
   regressor fitted to model-free alpha, beta); not for the exact interpolators (knn, rbf, xgb).
5. Networks: start matters -- identity start lands in a spiky gauge (beta ~ 2.4 at merger), the model-free start reaches
   3.9e-4; budget-limited, not overfitting.  PySR stays >= 200x above pass 40: beta errors integrate.
