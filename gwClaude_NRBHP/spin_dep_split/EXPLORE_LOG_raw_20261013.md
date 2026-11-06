# Split exploration — 2026-10-12 (autonomous; user detached)

## SUMMARY (read this first)
_In progress._  Updated as rounds finish.

## Brief (user, 2026-10-12)
"Use these models as your base and try to explore a bit with the split, making sure the number of coeffs is less than
15 and validating on the 270."  Base = the flux split line (18c and slim 14c, `slim_split.py`).

## Rules
* Split protocol unchanged (README "Decisions"): Kerr-ISCO cut, E_insp / E_MR with their own NR norms, one shared t0/phi0,
  Ori–Thorne window T = -1 to the ppBHPT peak (eta = 1/q), two-stage fit on T96.
* <= 14 coefficients (asserted in the driver).  Derived terms (kappa_SO) are not coefficients.
* Validation = H270 full-window E (full alignment, the published comparison) median / mean / max; also T96, and q = 2, 2.5
  (8 spins; ppBHPT extrapolated -- ranking only).  Caveat: H270 is used to CHOOSE between variants, so the winner's H270
  number is mildly optimistic; q = 2, 2.5 are never used for choosing.
* Variants only: nothing installed, `spin_dep/` not edited.  Compute cap 30 SU (normal queue, 1 node per variant).
* Gate: variant `slim` must reproduce `results/slim_confirm_aF_bPP_S.json` (H270 4.88e-4 / 5.86e-4 / 2.8e-3).
  Smoke test on (5, +0.6): the slim form reproduces the stored case to 1e-5 relative.

## Driver
`explore_split.py --variant NAME` (slurm `ls6/spin_dep_split/run_explore.slurm NAME`) -> `results/explore/NAME.json`,
log `logs/explore.<job>.log` (one `RESULT` line per variant).

## Round 1 (submitted 2026-10-12, jobs 3506164-73)
2026-10-12 22:50: normal queue jammed (500/500 nodes busy, 860 pending; nothing started in 8 h).  The 7 fixed-inspiral
variants were cancelled there and resubmitted as ONE development job 3506710 (sequential, slim first); the 3 insp*
variants (3506171-73) stay on normal.
Inspiral copy fixed at the T96 stage-1 fit (9c) unless stated.  MR branch, 5 coefficients unless stated:
| variant | MR branch | total |
|:--|:--|--:|
| slim | copy, free A0 A1 b s1 s2 (gate) | 14 |
| aF_P_S | copy, free A0 A1 P0 s1 s2 | 14 |
| aF_bPP_P | copy, free A0 A1 b P0 P1 | 14 |
| aF_bPP_P0_s1 | copy, free A0 A1 b P0 s1 | 14 |
| bump | copy A0 A1 b + flux-driven merger bump beta_m *= 1 + nu (k0 + k1 chi1) F | 14 |
| bump_s1 | copy A0 A1 b s1 + bump nu k0 F | 14 |
| alpha_spin | copy A0 A1 b s1 + spin in the MR amplitude coupling a_F + nu a_s chi1 | 14 |
| insp7_mr7 | inspiral c0 = c1 = 0 (7c, refit) + copy A0 A1 b s1 s2 P0 P1 | 14 |
| insp8_mr6 | inspiral s2 = 0 (8c, refit) + copy A0 A1 b s1 s2 P0 | 14 |
| insp8b_mr6 | inspiral b = 0 (8c, refit) + copy A0 A1 b s1 s2 P0 | 14 |

## Results
### Round 1 (job 3506710, development, 2026-10-12/13; all 14c, inspiral = stored T96 stage-1 fit)
Full-window E, full alignment, median / mean / max; q = 2, 2.5: medians over 8 spins.  Gate PASSED: `slim` reproduces
H270 median 4.882e-4 (stored 4.882e-4) and every other stored statistic.
| variant | MR free | T96 | H270 | q2.5 | q2 | H270 E_MR (insp-aligned) |
|:--|:--|:--|:--|--:|--:|--:|
| **slim** | A0 A1 b s1 s2 | 5.43e-4 / 9.44e-4 / 6.9e-3 | **4.88e-4 / 5.86e-4 / 2.8e-3** | 4.96e-3 | 2.13e-2 | 7.4e-3 |
| aF_P_S | A0 A1 P0 s1 s2 | 7.18e-4 / 1.06e-3 / 6.7e-3 | 6.30e-4 / 6.86e-4 / 2.5e-3 | 6.03e-3 | 2.30e-2 | 1.4e-2 |
| aF_bPP_P | A0 A1 b P0 P1 | 9.64e-4 / 1.43e-3 / 7.4e-3 | 7.63e-4 / 8.65e-4 / 3.3e-3 | 3.86e-3 | 1.66e-2 | 1.8e-2 |
| aF_bPP_P0_s1 | A0 A1 b P0 s1 | 9.96e-4 / 1.43e-3 / 7.1e-3 | 7.68e-4 / 8.79e-4 / 3.1e-3 | 4.14e-3 | 1.75e-2 | 1.9e-2 |
| alpha_spin | A0 A1 b s1 + a_s | 1.02e-3 / 1.51e-3 / 7.6e-3 | 8.08e-4 / 9.22e-4 / 3.8e-3 | 4.35e-3 | 1.83e-2 | 1.8e-2 |
| bump | A0 A1 b + k0 k1 | 1.10e-3 / 1.53e-3 / 7.5e-3 | 8.83e-4 / 9.34e-4 / 3.7e-3 | 4.77e-3 | 1.97e-2 | 1.9e-2 |
| bump_s1 | A0 A1 b s1 + k0 | 1.12e-3 / 1.55e-3 / 8.0e-3 | 9.04e-4 / 9.46e-4 / 3.9e-3 | 4.63e-3 | 2.00e-2 | 2.0e-2 |

Reading: the slim split stays best by a wide margin.  What it has and the others lack is a CURVED spin dependence of the
MR energy coupling (P_m spin part nu (s1 chi1 + s2 chi1^2), fitted s1 = +50.9, s2 = -70.9; aF_P_S, the only other variant
with both, is second).  A flux bump in beta or a spin term in alpha_F does not replace it (E_MR 2-2.5x worse).  Trade-off:
the variants WITHOUT s2 extrapolate better to q = 2, 2.5 (aF_bPP_P 3.9e-3 vs slim 5.0e-3) -- the chi1^2 MR term is
what costs low q.  Hence round 3: give the MR branch its spin dependence through DERIVED remnant functions
(D_chi = chi_f(q, chi1) - chi_f(q, 0); D_br = beta_r(q, chi1)/beta_r(q, 0) - 1), which are evaluated, not extrapolated, at low q.

### Round 3 (job 3506881, development, after 3506717; submitted 2026-10-13 00:15)
MR copy with s1 = s2 = 0 and P_m += nu (r1 D + r2 D^2), starts = least-squares projection of slim's spin term onto D, D^2.
rem_dchi (13c), rem_dchi2 (14c), rem_dbr (13c), rem_dbr2 (14c), rem_dbr_s1 (A0 A1 b s1 + r1 D_br, 14c); plus the three
trimmed-inspiral variants moved off the jammed normal queue (stage 1 on T96 takes ~70 s): insp8_mr6, insp7_mr7, insp8b_mr6.

## Round 2 (automatic, chained in SLURM; submitted 2026-10-12 ~23:10)
* 3506716 `run_explore_auto.slurm windows` (after 3506710): picks the round-1 winner by H270 mean (<= 14c, gate P clean;
  ABORTS if `slim` does not reproduce H270 median 4.882e-4 within 2%) and refits its MR branch with the Ori–Thorne window
  starting at T = -2 and at T = -0.5 (end still the ppBHPT peak).  Window start is a choice, not a coefficient.
* 3506717 `run_explore_auto.slurm joint` (after 3506716): the same winner with ALL its coefficients polished together on
  mean full-window E (full alignment), 85 min budget.  A trade-off measurement on the split form, not the split protocol.
* Both append their winner and RESULT lines below this section automatically.  Insp* variants (normal queue) are
  included in the winner choice only if they have finished by then.

AUTO round 2 (windows, 2026-10-13 00:07): winner of 7 round-1 variants by H270 mean = **slim** (H270 4.882e-04 / 5.860e-04 / 2.78e-03); running ['slim__win-2.0', 'slim__win-0.5']
* slim__win-2.0 (14c): T96 4.643e-04 / 8.539e-04 / 6.51e-03;  H270 3.983e-04 / 5.413e-04 / 2.69e-03;  q2.5 4.80e-03, q2 2.14e-02
* slim__win-0.5 (14c): T96 6.280e-04 / 1.024e-03 / 7.14e-03;  H270 5.814e-04 / 6.407e-04 / 2.82e-03;  q2.5 5.18e-03, q2 2.13e-02

AUTO round 2 (joint, 2026-10-13 00:25): winner of 7 round-1 variants by H270 mean = **slim** (H270 4.882e-04 / 5.860e-04 / 2.78e-03); running ['slim__joint']
* slim__joint (14c): T96 5.351e-04 / 7.866e-04 / 5.22e-03;  H270 4.513e-04 / 5.230e-04 / 1.87e-03;  q2.5 5.34e-03, q2 2.45e-02

### Round 2 results (jobs 3506716 windows, 3506717 joint; automatic winner = slim)
| variant | T96 | H270 | q2.5 | q2 | H270 E_insp / E_MR (insp-aligned) |
|:--|:--|:--|--:|--:|:--|
| slim (T = -1, reference) | 5.43e-4 / 9.44e-4 / 6.9e-3 | 4.88e-4 / 5.86e-4 / 2.8e-3 | 4.96e-3 | 2.13e-2 | 1.49e-4 / 7.4e-3 |
| slim__win-2.0 | 4.64e-4 / 8.54e-4 / 6.5e-3 | **3.98e-4** / 5.41e-4 / 2.7e-3 | 4.80e-3 | 2.14e-2 | 1.52e-4 / 7.1e-3 |
| slim__win-0.5 | 6.28e-4 / 1.02e-3 / 7.1e-3 | 5.81e-4 / 6.41e-4 / 2.8e-3 | 5.18e-3 | 2.13e-2 | 1.57e-4 / 8.9e-3 |
| slim__joint (not two-stage) | 5.35e-4 / 7.87e-4 / 5.2e-3 | 4.51e-4 / **5.23e-4 / 1.9e-3** | 5.34e-3 | 2.45e-2 | 1.38e-4 / 8.9e-3 |
An earlier window start helps monotonically over T = -0.5, -1, -2 (H270 median 5.81 -> 4.88 -> 3.98e-4); the window START
is the user's choice (T = -1, OT's departure from adiabatic inspiral), so T = -2 is reported as a variant.  The joint polish
converged in 3 min (mean full E 9.42e-4 -> 7.86e-4 on the fit grid): it buys mean and max (-11%, -33%) at a small cost in
E_MR and low q.

### Round 3 results (job 3506881)
| variant | coef | T96 | H270 | q2.5 | q2 | H270 E_insp / E_MR |
|:--|--:|:--|:--|--:|--:|:--|
| rem_dchi | 13 | 1.22e-3 / 1.81e-3 / 1.1e-2 | 8.85e-4 / 1.05e-3 / 5.1e-3 | 4.99e-3 | 1.94e-2 | 1.50e-4 / 2.2e-2 |
| rem_dbr | 13 | 1.61e-3 / 2.12e-3 / 1.4e-2 | 1.02e-3 / 1.26e-3 / 6.9e-3 | 5.96e-3 | 2.09e-2 | 1.49e-4 / 2.9e-2 |
| rem_dchi2 | 14 | 5.54e-4 / 1.05e-3 / 6.4e-3 | 4.80e-4 / 6.42e-4 / 3.0e-3 | 5.66e-3 | 2.23e-2 | 1.47e-4 / 9.2e-3 |
| rem_dbr2 | 14 | 9.66e-4 / 1.90e-3 / 1.8e-2 | 7.80e-4 / 8.85e-4 / 8.3e-3 | 5.50e-3 | 2.20e-2 | 1.52e-4 / 1.9e-2 |
| rem_dbr_s1 | 14 | 8.23e-4 / 1.29e-3 / 8.8e-3 | 6.89e-4 / 7.69e-4 / 3.3e-3 | **3.49e-3** | 1.78e-2 | 1.53e-4 / 1.9e-2 |
| insp8_mr6 (inspiral s2 = 0) | 14 | 6.64e-4 / 9.10e-4 / 6.0e-3 | 5.39e-4 / 6.68e-4 / 2.5e-3 | 3.58e-3 | 1.86e-2 | 1.94e-4 / 9.3e-3 |
| insp7_mr7 (inspiral c0 = c1 = 0) | 14 | 5.61e-4 / 1.01e-3 / 6.8e-3 | 5.11e-4 / 6.51e-4 / 2.9e-3 | 5.34e-3 | 2.26e-2 | 1.92e-4 / 7.7e-3 |
| **insp8b_mr6 (inspiral b = 0)** | 14 | 5.88e-4 / 9.43e-4 / 6.2e-3 | **4.35e-4 / 5.66e-4 / 2.3e-3** | 6.72e-3 | 2.39e-2 | 1.24e-4 / 6.9e-3 |
Reading: the remnant functions do NOT replace the chi1 polynomial (the 1-term forms lose 2-3x in E_MR; the 2-term D_chi form
only matches slim, D_br is worse) -- remnant-spin shift and QNM ratio are nearly linear in chi1 where the fit wants the
curvature of s1 chi1 + s2 chi1^2 (fitted chi1-peak near +0.36).  No low-q payoff except rem_dbr_s1 (3.5e-3 at q2.5, but
H270 +41%).  Trading the inspiral's b (a 0.8% nu correction to b_pp) for the MR's P0 IS a gain: insp8b_mr6 beats slim on
H270 median (-11%), mean (-3%) and max (-17%), and has the lowest E_insp and E_MR of any variant, at a cost below q = 3
(q2.5 6.7e-3 vs 5.0e-3).  Dropping c0, c1 or s2 from the inspiral costs E_insp (+30%).

### Round 4 (job 3507006, development, submitted 2026-10-13 02:00): combine the gains
slim__win-3.0, slim__win-1.5 (window scan), insp8b__win-2.0 (both gains), and joint polishes of slim__win-2.0, insp8b_mr6,
insp8b__win-2.0 (40 min budget each).

### Rounds 2-3 results (jobs 3506716, 3506717, 3506881; all 14c unless stated; H270 / T96 full-window, full alignment)
| variant | what changed vs slim | T96 | H270 | q2.5 | q2 |
|:--|:--|:--|:--|--:|--:|
| slim (reference) | -- | 5.43e-4 / 9.44e-4 / 6.9e-3 | 4.88e-4 / 5.86e-4 / 2.78e-3 | 4.96e-3 | 2.13e-2 |
| **slim__win-2.0** | window starts at T = -2 (0 coef) | 4.64e-4 / 8.54e-4 / 6.5e-3 | **3.98e-4 / 5.41e-4 / 2.69e-3** | 4.80e-3 | 2.14e-2 |
| slim__win-0.5 | window starts at T = -0.5 | 6.28e-4 / 1.02e-3 / 7.1e-3 | 5.81e-4 / 6.41e-4 / 2.82e-3 | 5.18e-3 | 2.13e-2 |
| slim__joint | all 14 polished on full-window E (not the split protocol) | 5.35e-4 / 7.87e-4 / 5.2e-3 | 4.51e-4 / 5.23e-4 / **1.87e-3** | 5.34e-3 | 2.45e-2 |
| **insp8b_mr6** | inspiral b = 0 (refit), MR frees P0 too | 5.88e-4 / 9.43e-4 / 6.2e-3 | 4.35e-4 / 5.66e-4 / 2.30e-3 | 6.72e-3 | 2.39e-2 |
| insp8_mr6 | inspiral s2 = 0, MR frees P0 | 6.64e-4 / 9.10e-4 / 6.0e-3 | 5.39e-4 / 6.68e-4 / 2.54e-3 | **3.58e-3** | 1.86e-2 |
| insp7_mr7 | inspiral c0 = c1 = 0, MR frees P0 P1 | 5.61e-4 / 1.01e-3 / 6.8e-3 | 5.11e-4 / 6.51e-4 / 2.93e-3 | 5.34e-3 | 2.26e-2 |
| rem_dchi2 | MR spin via D_chi, D_chi^2 (derived) | 5.54e-4 / 1.05e-3 / 6.4e-3 | 4.80e-4 / 6.42e-4 / 2.97e-3 | 5.66e-3 | 2.23e-2 |
| rem_dbr_s1 | MR spin via s1 chi + r1 D_br | 8.23e-4 / 1.29e-3 / 8.8e-3 | 6.89e-4 / 7.69e-4 / 3.30e-3 | **3.49e-3** | 1.78e-2 |
| rem_dchi (13c) / rem_dbr (13c) / rem_dbr2 | one D term, or D_br^2 | -- | 8.9e-4 .. 1.0e-3 medians | 5.0-6.0e-3 | ~2e-2 |

Reading:
* The Ori–Thorne window START matters at zero coefficients: earlier is better (T = -2: H270 median -18%, mean -8%, and q2.5
  slightly better); later is worse.  The user's choice was T = -1 -- reported as a variant, not adopted.  Round 4 scans -1.5, -3, -4.
* Derived remnant functions do NOT replace the chi1^2 MR spin term in range (rem_dchi2 ties the median but loses on the mean and
  max); the D_br forms are poor in range.  rem_dbr_s1 and insp8_mr6 extrapolate best to q = 2.5 (3.5-3.6e-3 vs slim 5.0e-3)
  but cost 10-40% in range: the in-range / low-q trade-off is the same one seen in round 1.
* Moving the inspiral b (a 0.8%-level nu correction to b_pp) into the MR branch's P0 (insp8b_mr6) is the best two-stage
  variant on H270 mean and max, at a low-q cost.
* The joint polish buys the max (2.78e-3 -> 1.87e-3) and the mean but not low q: a split-form model fitted on the
  full-window objective, not the split protocol.

### Round 4 (submitted 2026-10-13 ~00:50): jobs 3507007 (windows + insp8b at T = -2), 3507008 (joint polishes, after 3507007)
slim__win-1.5, slim__win-3.0, slim__win-4.0, insp8b__win-2.0, insp8__win-2.0; then slim_win-2.0__joint, insp8b__joint,
insp8b_win-2.0__joint (30 min budget each).

### Round 4 results (job 3507006, 32 min)
| variant | T96 | H270 | q2.5 | q2 | H270 E_insp / E_MR |
|:--|:--|:--|--:|--:|:--|
| slim__win-1.5 | 4.89e-4 / 8.90e-4 / 6.7e-3 | 4.31e-4 / 5.56e-4 / 2.7e-3 | 4.85e-3 | 2.13e-2 | 1.42e-4 / 7.2e-3 |
| slim__win-3.0 | 4.55e-4 / 8.21e-4 / 6.2e-3 | 3.92e-4 / 5.42e-4 / 2.6e-3 | 4.71e-3 | 2.16e-2 | 1.61e-4 / 7.5e-3 |
| insp8b__win-2.0 (two-stage) | 6.40e-4 / 8.70e-4 / 6.0e-3 | **3.62e-4** / 5.32e-4 / 2.3e-3 | 6.12e-3 | 2.48e-2 | 1.42e-4 / 6.8e-3 |
| slim_win-2.0__joint | 4.59e-4 / 7.08e-4 / 4.9e-3 | 3.85e-4 / **4.64e-4 / 1.8e-3** | 5.35e-3 | 2.42e-2 | 1.33e-4 / 7.3e-3 |
| insp8b__joint | 5.77e-4 / 8.40e-4 / 4.4e-3 | 4.43e-4 / 5.37e-4 / 1.9e-3 | 5.57e-3 | 2.45e-2 | 1.50e-4 / 6.9e-3 |
| insp8b_win-2.0__joint | 5.22e-4 / 7.56e-4 / 4.3e-3 | **3.47e-4** / 4.74e-4 / 1.9e-3 | 5.58e-3 | 2.44e-2 | 1.25e-4 / 6.9e-3 |
Window-start gain saturates by T = -2 (H270 median: -0.5 5.81, -1 4.88, -1.5 4.31, -2 3.98, -3 3.92 e-4).  The b-trade and
the earlier window combine (insp8b__win-2.0: -26% median vs slim), but the b-trade RAISES the T96 median (6.4e-4 vs 5.4e-4)
while lowering H270 -- a possible sign of selection on H270, hence round 5.  No variant improves q < 3.

### Round 5 (job 3507041, development, submitted 2026-10-13 02:40): half-split cross-validation
Each of slim, slim__win-2.0, insp8b_mr6, insp8b__win-2.0 refit (stage 1 AND stage 2) on one 48-point half of T96 (seed-1
split of spin_dep/overfit_checks_flux_orbit.py), scored on the other half ("T96_other_half") and on H270; both halves.
Caveat: the starts are the full-T96 coefficients (refit, not reused).
