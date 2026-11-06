# spin_dep_split — spin models with an ISCO-split error and an Ori–Thorne join (started 2026-10-03)

A separate model line.  Nothing in `spin_dep/` is edited; it is only imported (model code, caches).

## Decisions (user, 2026-10-03)
* **Cut**: the primary's Kerr ISCO, x_pp = x_isco(chi1) = Omega_isco(chi1)^(2/3) (Bardeen–Press–Teukolsky,
  m1 units, chi1 signed), found on the ppBHPT waveform and carried to NR time by
  t_NR - t_peak,NR = X1^(6/5) (t_pp - t_peak,pp).  Same cut for every model; not x = 1/6 (that is the
  ISCO only at chi1 = 0).
* **Error**: E_insp on NR [-5000.1, t_isco), E_MR on [t_isco, +100], each normalised by its own NR norm;
  **one shared t0, phi0** for the whole waveform.  The ordinary full-window mathcalE is always reported too.
* **Join** (2026-10-03, user): blend S(t) from 0 to 1 over the **Ori–Thorne time window**, from T = -1
  (departure from adiabatic inspiral) to the **ppBHPT |h22| peak** (user chose the peak over OT's
  T_plunge = 3.412, which falls 32-81 m1 after the peak at our mass ratios -- OT's small-eta divergence).
    T(t) = T* + (dtau/dt)_isco (t - t_isco,pp) / (eta^(-1/5) tau_o),   eta = 1/q,  T* = 0.7154
  t_isco,pp = ppBHPT time where x_pp reaches x_isco(chi1); T* = the T at which OT's universal solution of
  X'' = -X^2 - T (matched to X = sqrt(-T)) crosses X = 0 (computed; T_plunge check 3.412 reproduced);
  tau_o = (alpha beta kappa)^(-1/5) from OT's closed forms, kappa with Finn's Edot_isco interpolated in
  a over OT Table I.  Time unit on our grid: 15-69 m1.  eta = 1/q because in ppBHPT the particle m2
  orbits a Kerr hole m1 and GOT generated the data with it.  TODO (user): compare eta = nu later.
  The blend itself (a partition of unity mixing the inspiral and MR forms of alpha, beta) is OUR
  modelling device, not part of Ori–Thorne; OT supplies its location and duration only.
  beta, alpha = (1 - S) insp + S MR; S smooth (C2), so alpha, beta smooth and tau = int beta carries
  the phase through.  Superseded: tanh in OT X with width w (switches too slowly after the ISCO for
  prograde spins: 23% inspiral weight left at the peak for chi1 = +0.6, w = 1).

* **MR branch** (user, 2026-10-03): beta_MR = r0 * beta_r (exact Kerr QNM ratio), r0 FITTED as a test --
  if it comes out ~1, fix r0 = 1 (zero-parameter MR beta).  Installed fits had r0 = 0.80 (pn_anchored_spin),
  0.56 (pn_anc_spin_physical), read as compensation for their late-engaging energy-loss switch.
  alpha_MR keeps the fitted merger amplitude (m0 + m1 nu).
* **Fit** (user, 2026-10-03): TWO-STAGE.  Stage 1: inspiral coefficients on E_insp only.  Stage 2: freeze
  them, MR coefficients (m0, m1, r0) on E_MR.  One repeat pass if stage 2 moves E_insp through the blend
  overlap (the blend starts 50-220 m1 before the ISCO).  Why not joint E_insp + E_MR: separate
  normalisation gives the MR segment equal weight, ~13x its 7% signal share -- the MR would drive the fit
  even harder than in the published full-window fits.  A joint fit is run ONCE from the two-stage result
  only to measure the trade-off; it is not a model.
* **Finding (2026-10-03): the installed pn_anchored_spin switch is never off or fully on.** Its MR weight
  S = sigma((p_loss - p00 - p01 nu)/w0) is 0.04-0.06 at -5000 m1, 0.08-0.09 at -500, only 0.15-0.19 at the
  peak and 0.19-0.25 at +50 (6 S18 cases).  So the installed model is one ~90/10 mixture of the inspiral
  and MR forms throughout; its r0, m0, m1 act as inspiral-wide corrections, not merger physics (as CLAUDE.md
  already noted for nonspinning pn_anchored, S <= 0.18).  Consequence: its inspiral coefficients alone
  (S = 0) put beta ~1% high and are useless as a start.  The split fit therefore STARTS from the installed
  model projected onto the split form by linear least squares (inspiral coefficients to its alpha, beta on
  ppBHPT t in [-6500, window start]; r0, m0, m1 to its alpha, beta on [peak, peak + 60]).  The projection
  gives r0 = 1.02.
* Watch: the true beta has a ~10% merger bump (-50..+10 M; EXPLORATION.md "already known"); a monotone
  blend from the inspiral polynomial to the constant r0 beta_r cannot produce it.  At the projected start
  E_MR is 0.02-0.9 vs the installed 0.01-0.15 (9 cases).  The S18 screen measures whether the fit closes this.
* Order: pn_anchored_spin first (its parameters already split into inspiral / merger groups); S18 screen,
  then T96, then H270.

## Files
* `isco_split_baseline.py` (+ `run_isco_split.slurm`): the baseline — the four installed spin models scored
  on the split.  Gate: full-window E reproduces stored values (ratio 1.00000, all 384).
  -> `results/isco_split_baseline.json`, `Agentic_spin_split_plots/baseline/isco_split_<model>.png/pdf`.
* `refs/ori_thorne/OriThorne.tex` (gr-qc/0003032), `refs/islam2407/main.tex` (2407.18319v2).
  Ori–Thorne Table I reproduced from its formulas to <= 2.4e-4 (Table II bandwidths to their 2-digit rounding).

## Baseline (installed coefficients, 96 training points, medians; max in brackets)
| model | E_full | E_insp | E_MR | MR share of signal / of error |
|:--|--:|--:|--:|:--|
| pn_anchored_spin | 8.65e-4 | 3.8e-4 (5.2e-3) | 7.4e-3 (5.0e-2) | 7% / 55% |
| pn_anc_spin_physical | 9.64e-4 | 4.2e-4 (4.6e-3) | 7.6e-3 (5.2e-2) | 7% / 54% |
| fluxanchored_spin | 1.30e-3 | 8.7e-4 (1.1e-2) | 8.1e-3 (1.3e-1) | 7% / 38% |
| energy_anchored_spin | 1.57e-3 | 9.5e-4 (1.1e-2) | 9.6e-3 (1.1e-1) | 7% / 42% |
t_isco on the NR clock: -91 .. -6 M.  ISCO crossing inside the 2d join window [-9, -5] m1 at chi1 = +0.8,
q <= 3.9 (4 points) — watch these.

## Overnight 2026-10-03/04 (autonomous; see OVERNIGHT_LOG.md, SUMMARY at its top)
* `fit_flux_spin_split.py` — **fluxanchored_spin_split** VARIANT (not adopted/installed): flux inspiral + derived PN SO term,
  flux-form MR copy, OT window; two-stage.  Results `results/flux_spin_split_{S18,T96}_T-1*.json`, `results/h270_scores.json`,
  figure `Agentic_spin_split_plots/fluxanchored_spin_split/T96_heatmaps.png`.  Beats the installed leader on T96 median/mean
  and on H270 (median -35%); not yet uniform (chi1 = +0.8, low-q retrograde).
* Diagnostics: `diag_inspiral_form.py`, `diag_inspiral_forms_all.py`, `diag_admixture.py`, `diag_so_term.py [--corrected]`,
  `diag_percase.py`, `score_h270.py`.  `ot_join.ETA` ("1/q" default, "nu" comparison only).
* Derived term: delta ln beta_SO = (3/2) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1] (refs/blanchet_lrr; for an x^p term
  in ln beta the chirp matching gives 3 (A_NR - A_pp)/(5 - 2p), not the note's constant-beta 3/5).

## 2026-10-04 — parsimony (user: keep the split, cut coefficients)
Staged in $SCRATCH/spin_dep_split_tmp while /work MDT0 was out of inodes; copied back here (the copied run_*.slurm files
still point at the scratch paths; the .py files write next to themselves).
* (moved to spin_dep/ and promoted 2026-10-05 as **flux_anc_spin_with_orbit**: `fit_flux_anc_spin_with_orbit.py`, md `scaling_flux_anc_spin_with_orbit.md`) — BASELINE, no split: original fluxanchored_spin form + derived SO term, 9c, full-window fit.
  T96 8.86e-4/1.09e-3/6.5e-3, H270 7.20e-4/7.90e-4/2.7e-3 (median/mean/max, full-aligned): level with the 13c leader.
* `slim_split.py --select` — greedy forward selection of the MR branch on S18 (rule K: >= 5% per added coefficient on mean
  full-window E).  Kept aF (A0, A1), bPP (b), S (s1, s2): -19%, -20%, -24%; rejected aPP (c0, c1: -1%) and P (P0, P1);
  beta_r-anchored branch (m, r0) poor (2.77e-3; r0 = 0.993).  `results/slim_select_S18.json`.
* **`slim_split.py --confirm "aF,bPP,S"` — fluxanchored_spin_split, 14 coefficients (9 inspiral + derived SO, 5 MR):**
  T96 5.42e-4 / 9.44e-4 / 6.9e-3;  H270 4.88e-4 / 5.86e-4 / 2.8e-3  (full-aligned E_full, median/mean/max).
  vs 18c split (T96 5.17e-4/9.69e-4/6.7e-3, H270 4.59e-4/5.94e-4/2.8e-3): same accuracy, 4 fewer coefficients.
  vs installed pn_anchored_spin (13c; T96 8.65e-4/1.17e-3/6.6e-3, H270 7.10e-4/8.27e-4/2.9e-3): H270 median -31%, mean -29%;
  T96 max +5% (chi1 = +0.8 corner).  H270 insp-aligned: E_insp 1.5e-4, E_MR 7.4e-3 (leader 1.7e-4, 2.2e-2).
  `results/slim_confirm_aF_bPP_S.json`.  Still a VARIANT: not installed.
* Model md (equations, results incl. Rink et al. 2024 comparison, Anchors with the SO derivation):
  `scaling_fluxanchored_spin_split.md`.  Published-model scores: `score_rink.py` -> `results/rink_scores.json`;
  low-q extrapolation: `extrap_lowq.py` -> `results/extrap_lowq.json`.

## Pipeline (2026-10-04; user: keep, to be discussed before starting)
1. Refit pn_anchored_spin with all 13 coefficients free on T96, score H270 (fair 13 vs 14 comparison; ~1-2 SU).
2. Slim split extrapolation: tie the 5 MR coefficients' q-dependence to something physical (not free nu, nu^2).
3. Derive alpha's spin-orbit term by amplitude matching (alpha gets -(2/3) delta_SO + the amplitude's own SO terms).
4. chi1 = +0.8 corner (worst cell for every model).
5. Overfitting checks: MR-coefficient stability on random halves of T96; learning curve (18 / 48 / 96 -> H270).
State: flux_anc_spin_with_orbit (spin_dep/, 9c) promoted to a full model 2026-10-05; the slim split (here, 14c) is a variant.
Git: spin_dep/ essentials + figures committed on branch spin_dep_calib (e45e929, e20e5ca, 2e40745), pushed by the
user from the laptop (no GitHub SSH access from LS6).  spin_dep_split/ is not committed yet.
2026-10-05 results: item 3 (alpha SO term) -- no effect, not adopted (spin_dep/scaling_flux_anc_spin_with_orbit.md).
Item 2 (anchoring the slim MR coefficients, `slim_mr_anchor.py`, job 3489633) -- all three variants WORSE in range and at
low q: (a) A1' shared 13c: H270 8.5e-4, q2.5 7.8e-3; (b) MR beta prefactor -> beta_r(q, chi1) 14c: 6.0e-4, 7.4e-3;
(c) both 13c: 9.0e-4, 1.0e-2; vs slim14 4.9e-4, 5.0e-3.  The MR branch's own nu^2 term matters, and beta_r is the wrong anchor
for a branch that is active from T = -1.  A real low-q fix needs a different MR form.  Items 1, 4, 5 remain open.
Item 1 RUNNING 2026-10-05 (job 3490864): spin_dep/refit_pn_anchored_spin_13free.py -> spin_dep/test_joint2d_results/coeffs_13free_20261005.json.
Item 5 PREPARED, not submitted: spin_dep/overfit_checks_flux_orbit.py (9c) and spin_dep_split/overfit_checks_slim_split.py
(14c); same 4 random 48/48 halves (seed 1) + learning curve 18/48/96 -> H270.  Submit: sbatch ls6/spin_dep/run_overfit_{flux_orbit,slim_split}.slurm
Item 1 DONE 2026-10-05 (job 3490864, 5 min, converged): pn_anchored_spin with all 13 coefficients fitted on T96
(spin_dep/test_joint2d_results/coeffs_13free_20261005.json; TEST, not installed).  Full-window E median/mean/max:
T96 8.50e-4/1.08e-3/6.01e-3 (installed 8.65e-4/1.17e-3/6.58e-3); H270 6.98e-4/7.88e-4/2.53e-3 (installed 7.10e-4/8.27e-4/2.91e-3);
q2.5 4.08e-3, q2 1.65e-2.  Freeing the 3 inherited coefficients gains only 2-13%; flux_anc_spin_with_orbit (9c) is level with it
on H270 (7.20e-4/7.90e-4/2.66e-3); the slim split (14c) stays ~30% better in range.  Replacing the installed pn_anchored_spin:
user's call.
Selftest 2026-10-05: added S + B checks for all spin models (audit_selftest.py); 41/42 pass; B.junction_spin FAILED (its 09-24
coefficients predate the audit fixes, file unstamped) -> refit job 3490890 (refit + replot + selftest), per the user's rule.
2026-10-12/13 autonomous split exploration (user: < 15 coefficients, validate on H270): `explore_split.py`, results
`results/explore/`, plan + results + SUMMARY in `EXPLORE_LOG.md`.  Robust gain: Ori–Thorne window start T = -2 instead of -1
(slim__win-2.0, 14c: H270 3.98e-4 / 5.41e-4 / 2.7e-3, held-out T96 halves -10% median) -- user decision pending.
