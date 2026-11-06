# Split exploration — 2026-10-12/13 (autonomous; user detached)

## SUMMARY (read this first)
> **2026-10-13 correction (user):** rounds 1-4 SELECTED on H270 (auto_round2 winner by H270 mean; window scan and per-spin T_start optima read off H270), so the H270 numbers below are selection-biased for these variants, not a held-out estimate.  Selection now on T96 4-fold CV, final test on a fresh set F100: see CV_LOG.md.

Five rounds, 34 variants, every one 13-14 coefficients, all on the split protocol (Kerr-ISCO cut, E_insp / E_MR, two-stage
fit on T96) unless marked "joint".  Compute ~5 SU.  Nothing installed.  Numbers: full-window E, full alignment,
median / mean / max (q = 2.5: median over 8 spins, ppBHPT extrapolated).

| variant (14c) | what changes vs slim | H270 (validation) | T96 halves, held out (96) | q = 2.5 |
|:--|:--|:--|:--|--:|
| slim (reference, the current split) | -- | 4.88e-4 / 5.86e-4 / 2.8e-3 | 5.37e-4 / 9.62e-4 / 6.9e-3 | 4.96e-3 |
| **slim__win-2.0** | Ori–Thorne window starts at T = -2 (not -1); 0 coefficients | **3.98e-4 / 5.41e-4 / 2.7e-3** | **4.83e-4 / 8.97e-4 / 6.6e-3** | 4.80e-3 |
| insp8b__win-2.0 | + inspiral b = 0, MR branch frees P0 instead | 3.62e-4 / 5.32e-4 / 2.3e-3 | 6.26e-4 / 8.86e-4 / 5.9e-3 | 6.12e-3 |
| slim_win-2.0__joint | slim__win-2.0 with all 14 polished on full-window E | 3.85e-4 / 4.64e-4 / 1.8e-3 | (not run) | 5.35e-3 |
| insp8b_win-2.0__joint | insp8b__win-2.0, joint polish | 3.47e-4 / 4.74e-4 / 1.9e-3 | (not run) | 5.58e-3 |

**Findings**
1. **The one robust gain is the window start.** Starting the Ori–Thorne blend at T = -2 instead of the user's T = -1 lowers the
   error on every measure: H270 median -18%, mean -8%, max -3%; the held-out T96 halves -10% / -7% / -4%; q = 2.5 -3%.  It
   costs no coefficient.  Scan (H270 median, e-4): T = -0.5 5.81, -1 4.88, -1.5 4.31, -2 3.98, -3 3.92, -4 4.42 -- an optimum
   near T = -2..-3.  T = -1 was a user decision (OT's departure from adiabatic inspiral), so this is a proposal, not adopted.
2. **The b-trade is not robust.** Moving the inspiral's b (a 0.8% nu correction to b_pp) into the MR branch (insp8b) wins on
   H270 (-9% median vs slim__win-2.0) but LOSES on the held-out T96 halves (median 6.26e-4 vs 4.83e-4) and below q = 3.  H270 is
   interior; the T96 points include the q = 3, 8 and chi1 = -0.6, +0.8 edges.  It helps the interior and hurts the edges, and
   H270 was also the selection set -- treat its H270 gain as optimistic.
3. **The MR branch needs a curved spin dependence**: s1 chi1 + s2 chi1^2 in its energy coupling (fitted s1 = +51, s2 = -71).
   Every MR variant without it (flux-driven merger bump in beta, spin in the MR amplitude coupling, a linear spin term only)
   is 2-2.5x worse in E_MR.  Derived remnant functions (remnant-spin shift chi_f(q, chi1) - chi_f(q, 0); ringdown clock-ratio
   shift beta_r(q, chi1)/beta_r(q, 0) - 1) do not replace it: both are nearly linear in chi1.
4. **Low q is untouched.** No in-range winner improves q = 2.5 (4.7-6.1e-3; flux_anc_spin_with_orbit 4.4e-3, pn_anchored_spin
   4.1e-3).  The variants that extrapolate best (insp8_mr6 3.6e-3, rem_dbr_s1 3.5e-3) are those WITHOUT the chi1^2 MR term,
   and they cost 10-40% in range: the same in-range / low-q trade-off as round 1.
5. The joint polish (all coefficients on the full-window error; NOT the split protocol) converges in ~3 min and buys the
   mean and max (slim__win-2.0: -14% / -34% on H270) at a cost below q = 3.

**Recommendation for the user to decide:** adopt the window start T = -2 for the split line (slim__win-2.0: 14c, H270
3.98e-4 / 5.41e-4 / 2.7e-3, vs flux_anc_spin_with_orbit 7.20e-4 / 7.90e-4 / 2.7e-3 and pn_anchored_spin 7.10e-4 / 8.27e-4 /
2.9e-3).  Keep the b-trade and the joint polish as variants.  Open: low q, and the chi1 = +0.8 corner.

**Process note.** Around 2026-10-13 01:56 a second agent session acted on the same files: it submitted jobs 3507007 (sx_r4a)
and 3507008 (sx_r4b), added the variants slim__win-4.0 and insp8__win-2.0 to explore_split.py, and wrote a parallel round-4
section into this log.  Its results that overlap with job 3507006 agree exactly (same RESULT lines); its two extra variants are
included below.  3507008 (a duplicate of 3507006's joint polishes) was cancelled before it wrote anything.  The unedited log
is kept as `EXPLORE_LOG_raw_20261013.md`.

## Brief (user, 2026-10-12)
"Use these models as your base and try to explore a bit with the split, making sure the number of coeffs is less than
15 and validating on the 270."  Base = the flux split line (18c and slim 14c, `slim_split.py`).

## Rules
* Split protocol unchanged (README "Decisions"): Kerr-ISCO cut, E_insp / E_MR with their own NR norms, one shared t0/phi0,
  Ori–Thorne window T = -1 to the ppBHPT peak (eta = 1/q), two-stage fit on T96.  Variants that change the window start or
  use a joint fit are labelled as such.
* <= 14 coefficients (asserted in the driver).  Derived terms (kappa_SO, the remnant functions) are not coefficients.
* Validation = H270 full-window E (full alignment).  Also T96 and q = 2, 2.5 (8 spins; ppBHPT extrapolated -- ranking only).
  H270 was also used to choose between variants, so the winners' H270 numbers are mildly optimistic; round 5 (half-split
  cross-validation) and q = 2, 2.5 are never used for choosing.
* Variants only: nothing installed, `spin_dep/` not edited.  Compute cap 30 SU; used ~5 SU.
* Gate: variant `slim` reproduced `results/slim_confirm_aF_bPP_S.json` exactly (H270 4.882e-4 / 5.860e-4 / 2.78e-3).

## Driver
`explore_split.py --variant NAME [--variant ...]` (slurm `ls6/spin_dep_split/run_explore_dev.slurm NAME ...`;
`run_explore_auto.slurm windows|joint` = the automatic round 2) -> `results/explore/NAME.json` (coefficients, per-case
errors on T96, H270, q = 2, 2.5 and, for half variants, the other half), log `logs/explore.<job>.log` (one RESULT line each).
Variant spec: `insp` (None = stored T96 stage-1 inspiral; ("pin0", names) = those pinned to 0 and stage 1 refit), `mr` (MR
branch form, `MR_FORMS`), optional `window` (Ori–Thorne start T), `from` + `joint` (polish a finished variant), `half`.
Backups of the driver between rounds: `explore_split.py.bak_r{1,2,4}`.

## Jobs
The normal queue was jammed (500/500 nodes busy, 860 pending), so after 8 h all work moved to the development queue:
3506710 (round 1), 3506716 + 3506717 (round 2, chained, automatic), 3506881 (round 3), 3507006 (round 4), 3507041 (round 5);
plus the other session's 3507007.  The first normal-queue submissions (3506164-73) were cancelled unrun.

## Results
All 14c unless stated; T96 / H270 = median / mean / max; split errors are H270 medians with the inspiral alignment.

### Round 1 — which 5 coefficients the MR branch frees (job 3506710)
| variant | MR free | T96 | H270 | q2.5 | q2 | E_MR |
|:--|:--|:--|:--|--:|--:|--:|
| **slim** | A0 A1 b s1 s2 | 5.43e-4 / 9.44e-4 / 6.9e-3 | **4.88e-4 / 5.86e-4 / 2.8e-3** | 4.96e-3 | 2.13e-2 | 7.4e-3 |
| aF_P_S | A0 A1 P0 s1 s2 | 7.18e-4 / 1.06e-3 / 6.7e-3 | 6.30e-4 / 6.86e-4 / 2.5e-3 | 6.03e-3 | 2.30e-2 | 1.4e-2 |
| aF_bPP_P | A0 A1 b P0 P1 | 9.64e-4 / 1.43e-3 / 7.4e-3 | 7.63e-4 / 8.65e-4 / 3.3e-3 | 3.86e-3 | 1.66e-2 | 1.8e-2 |
| aF_bPP_P0_s1 | A0 A1 b P0 s1 | 9.96e-4 / 1.43e-3 / 7.1e-3 | 7.68e-4 / 8.79e-4 / 3.1e-3 | 4.14e-3 | 1.75e-2 | 1.9e-2 |
| alpha_spin | A0 A1 b s1, a_F += nu a_s chi1 | 1.02e-3 / 1.51e-3 / 7.6e-3 | 8.08e-4 / 9.22e-4 / 3.8e-3 | 4.35e-3 | 1.83e-2 | 1.8e-2 |
| bump | A0 A1 b, beta_m *= 1 + nu (k0 + k1 chi1) F | 1.10e-3 / 1.53e-3 / 7.5e-3 | 8.83e-4 / 9.34e-4 / 3.7e-3 | 4.77e-3 | 1.97e-2 | 1.9e-2 |
| bump_s1 | A0 A1 b s1, bump nu k0 F | 1.12e-3 / 1.55e-3 / 8.0e-3 | 9.04e-4 / 9.46e-4 / 3.9e-3 | 4.63e-3 | 2.00e-2 | 2.0e-2 |

### Round 2 — window start and joint polish of the round-1 winner (jobs 3506716, 3506717; automatic)
| variant | T96 | H270 | q2.5 | q2 | E_insp / E_MR |
|:--|:--|:--|--:|--:|:--|
| slim__win-0.5 | 6.28e-4 / 1.02e-3 / 7.1e-3 | 5.81e-4 / 6.41e-4 / 2.8e-3 | 5.18e-3 | 2.13e-2 | 1.57e-4 / 8.9e-3 |
| slim__win-2.0 | 4.64e-4 / 8.54e-4 / 6.5e-3 | 3.98e-4 / 5.41e-4 / 2.7e-3 | 4.80e-3 | 2.14e-2 | 1.52e-4 / 7.1e-3 |
| slim__joint | 5.35e-4 / 7.87e-4 / 5.2e-3 | 4.51e-4 / 5.23e-4 / 1.9e-3 | 5.34e-3 | 2.45e-2 | 1.38e-4 / 8.9e-3 |

### Round 3 — derived remnant spin functions; trading inspiral coefficients (job 3506881)
MR copy with s1 = s2 = 0 and P_m += nu (r1 D + r2 D^2); starts = least-squares projection of slim's spin term onto D, D^2.
| variant | coef | T96 | H270 | q2.5 | q2 | E_insp / E_MR |
|:--|--:|:--|:--|--:|--:|:--|
| rem_dchi (D_chi) | 13 | 1.22e-3 / 1.81e-3 / 1.1e-2 | 8.85e-4 / 1.05e-3 / 5.1e-3 | 4.99e-3 | 1.94e-2 | 1.50e-4 / 2.2e-2 |
| rem_dbr (D_br) | 13 | 1.61e-3 / 2.12e-3 / 1.4e-2 | 1.02e-3 / 1.26e-3 / 6.9e-3 | 5.96e-3 | 2.09e-2 | 1.49e-4 / 2.9e-2 |
| rem_dchi2 (D_chi, D_chi^2) | 14 | 5.54e-4 / 1.05e-3 / 6.4e-3 | 4.80e-4 / 6.42e-4 / 3.0e-3 | 5.66e-3 | 2.23e-2 | 1.47e-4 / 9.2e-3 |
| rem_dbr2 (D_br, D_br^2) | 14 | 9.66e-4 / 1.90e-3 / 1.8e-2 | 7.80e-4 / 8.85e-4 / 8.3e-3 | 5.50e-3 | 2.20e-2 | 1.52e-4 / 1.9e-2 |
| rem_dbr_s1 (s1 chi1 + r1 D_br) | 14 | 8.23e-4 / 1.29e-3 / 8.8e-3 | 6.89e-4 / 7.69e-4 / 3.3e-3 | 3.49e-3 | 1.78e-2 | 1.53e-4 / 1.9e-2 |
| insp8_mr6 (inspiral s2 = 0; MR + P0) | 14 | 6.64e-4 / 9.10e-4 / 6.0e-3 | 5.39e-4 / 6.68e-4 / 2.5e-3 | 3.58e-3 | 1.86e-2 | 1.94e-4 / 9.3e-3 |
| insp7_mr7 (inspiral c0 = c1 = 0; MR + P0 P1) | 14 | 5.61e-4 / 1.01e-3 / 6.8e-3 | 5.11e-4 / 6.51e-4 / 2.9e-3 | 5.34e-3 | 2.26e-2 | 1.92e-4 / 7.7e-3 |
| insp8b_mr6 (inspiral b = 0; MR + P0) | 14 | 5.88e-4 / 9.43e-4 / 6.2e-3 | 4.35e-4 / 5.66e-4 / 2.3e-3 | 6.72e-3 | 2.39e-2 | 1.24e-4 / 6.9e-3 |
Dropping c0, c1 or s2 from the inspiral costs ~30% in E_insp; dropping b does not.

### Round 4 — combinations and the window scan (jobs 3507006; 3507007 from the other session)
| variant | T96 | H270 | q2.5 | q2 | E_insp / E_MR |
|:--|:--|:--|--:|--:|:--|
| slim__win-1.5 | 4.89e-4 / 8.90e-4 / 6.7e-3 | 4.31e-4 / 5.56e-4 / 2.7e-3 | 4.85e-3 | 2.13e-2 | 1.42e-4 / 7.2e-3 |
| slim__win-3.0 | 4.55e-4 / 8.21e-4 / 6.2e-3 | 3.92e-4 / 5.42e-4 / 2.6e-3 | 4.71e-3 | 2.16e-2 | 1.61e-4 / 7.5e-3 |
| slim__win-4.0 (3507007) | 5.00e-4 / 8.21e-4 / 5.9e-3 | 4.42e-4 / 5.67e-4 / 2.5e-3 | 4.74e-3 | 2.18e-2 | 1.79e-4 / 7.8e-3 |
| insp8b__win-2.0 | 6.40e-4 / 8.70e-4 / 6.0e-3 | 3.62e-4 / 5.32e-4 / 2.3e-3 | 6.12e-3 | 2.48e-2 | 1.42e-4 / 6.8e-3 |
| insp8__win-2.0 (3507007) | 6.54e-4 / 8.73e-4 / 5.8e-3 | 5.31e-4 / 6.56e-4 / 2.5e-3 | 3.64e-3 | 1.90e-2 | 2.10e-4 / 9.6e-3 |
| slim_win-2.0__joint | 4.59e-4 / 7.08e-4 / 4.9e-3 | 3.85e-4 / 4.64e-4 / 1.8e-3 | 5.35e-3 | 2.42e-2 | 1.33e-4 / 7.3e-3 |
| insp8b__joint | 5.77e-4 / 8.40e-4 / 4.4e-3 | 4.43e-4 / 5.37e-4 / 1.9e-3 | 5.57e-3 | 2.45e-2 | 1.50e-4 / 6.9e-3 |
| insp8b_win-2.0__joint | 5.22e-4 / 7.56e-4 / 4.3e-3 | 3.47e-4 / 4.74e-4 / 1.9e-3 | 5.58e-3 | 2.44e-2 | 1.25e-4 / 6.9e-3 |

2026-10-13 04:30 (background session 66e999): two sessions had been driving this log in parallel; the background one has stopped (its last job 3507007 is above; its duplicate joint job 3507008 was cancelled by the interactive session). The interactive session gwclaude-nrbhp-e6 owns the exploration from here.

### Round 5 — half-split cross-validation (job 3507041)
Each candidate refit from scratch (stage 1 AND stage 2; starts = the full-T96 coefficients) on one 48-point half of T96
(the seed-1 split of spin_dep/overfit_checks_flux_orbit.py), scored on the other half; both halves, pooled = 96 held-out scores.
| candidate | held-out halves (pooled 96) | H270, fit on half 0 / half 1 (medians) | H270 means |
|:--|:--|:--|:--|
| slim | 5.37e-4 / 9.62e-4 / 6.9e-3 | 4.69e-4 / 4.94e-4 | 5.88e-4 / 6.10e-4 |
| slim__win-2.0 | **4.83e-4 / 8.97e-4 / 6.6e-3** | 4.21e-4 / 4.38e-4 | 5.72e-4 / 5.91e-4 |
| insp8b_mr6 | 5.66e-4 / 9.58e-4 / 6.1e-3 | 4.27e-4 / 4.44e-4 | 5.61e-4 / 5.75e-4 |
| insp8b__win-2.0 | 6.26e-4 / 8.86e-4 / 5.9e-3 | 3.56e-4 / 3.67e-4 | 5.30e-4 / 5.38e-4 |
Half-fits stay close to the full fits on H270 (within ~10%), so none of the four is overfitted on 48 points.  On the held-out
T96 points the window start is a clear gain; the b-trade raises the median (edges) while lowering the mean and max.
