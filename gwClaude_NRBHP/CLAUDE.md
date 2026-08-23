# gwClaude_NRBHP

**START HERE for the gw_remnant energy/flux model line: `RESUME_gwremnant.md`** — current
standings, exact reproduce commands, open items in priority order, the settled negative results,
and the gotchas (rcdefaults ordering, PN-free caveat, q<2.5 double extrapolation).

Workspace for BHPT-to-NR waveform calibration. Mirror of `../gwAgentic_NRBHP/` — **do not write into that directory**, read-only reference only.

## Environment

Use conda env `ut_claude` (python3). Imports numpy/scipy/gwsurrogate/gwtools work directly — no LD_LIBRARY_PATH hack needed (that was only for the old `.venv`/`ut_codex` setup).

Keep BLAS/OMP threads <= 4.

## Running on TACC (Stampede3)

Allocation `-A PHY26026`. The Claude Code CLI itself runs fine on the login node (lightweight: API calls + file edits), but **never run a fit/scaling script directly on the login node** — TACC kills/throttles compute-heavy login-node processes. Submit interactive jobs instead:

- Quick test/debug (<=2h): `idev -p skx-dev -N 1 -n 1 -A PHY26026 -m 60`
- Full production run: `idev -p skx -N 1 -n 1 -A PHY26026 -m <minutes>` (or `sbatch` with the same `-p`/`-A` for a non-interactive batch job if the run is long enough to outlast a terminal session)

`skx-dev` caps at 2:00:00 wall time; use `skx`, `icx`, or `spr` (uncapped queues) for anything longer.

**Env**: build from `environment.yml` (minimal core stack mirroring the local `ut_claude` env — not the full local freeze, which carries unrelated packages) under `$WORK`, not `$HOME` (home has a file-count quota that a conda env, especially with lalsuite, will blow through):
```
module load conda   # check `module spider conda` for the exact module name on Stampede3
conda env create -f environment.yml -p $WORK/envs/ut_claude
conda activate $WORK/envs/ut_claude
```

## Dependencies (at UT_Austin level)

- `BHPTNRSurrogate/surrogates` — add to sys.path, import `BHPTNRSur1dq1e4`
- `BHPTutils` — `bhpt_utils.all_alpha_beta_1D_scaling_params` for plot comparisons

## Current accepted model (q=5, (2,2) mode)

10 parameters, full-window mismatch E = 6.0368e-5 (calibrated=False vs NRHybSur3dq8).
Key files: `fit_scaling_PN_opt_creative.py`, `scaling_PN_opt_creative.md`, `NRBHP_time_dep_PN_opt_creative.py`, `Agentic_plots/`.

Waveform loading is slow (~1 min); both fit and plot scripts cache to `.cache/waveforms_q5_22.npz`.

## Key physics insight

Inspiral beta (time-stretch) drifts linearly in the PN-loss coordinate (beta_L ~ 2.4e-3). Releasing that single degree of freedom halved the error floor vs constant-inspiral models. The ~1.2e-4 floor of constant-beta models is not irreducible.

## Models available

| identifier | files | notes |
|:---|:---|:---|
| PN_opt_creative | `fit_scaling_PN_opt_creative.py` | base model, q=5 only |
| PN_opt_creative_q_dep | `fit_scaling_PN_opt_creative_q_dep.py` | 40-q grid, 1/q polynomial; median 7.66e-5, max 2.13e-4 |
| PN_opt_remnant_partial | `fit_scaling_PN_opt_remnant_partial.py` | 64-q grid [3,8]; beta_r factorised (physical×correction), other 9 params chi_f poly; median 7.48e-5, max 5.98e-4; q=2 extrap: mathcalE~0.37% |
| PN_opt_nu_Pade | `fit_scaling_PN_opt_nu_Pade.py` | 40-q grid; PP-anchored (alpha_i,beta_i,beta_r→1, corrections→0 at 1/q→0); bake-off selected deg-4 polynomial in nu=q/(1+q)²; median 7.05e-5, max 1.68e-4; q=2 extrap mathcalE~6.5% |
| q_dep_classic | `fit_scaling_PN_opt_q_dep_classic.py` | 40-q grid; constant alpha, beta (Islam+22 model class); PP-anchored quartic in 1/q; calibrated to NRHybSur3dq8; target ~1e-3 in-range, ~1e-2 at q=2 |
| PN_opt_full_PN_q_dep | `fit_scaling_PN_opt_full_PN_q_dep.py` | 64-q grid [3,8]; NO logistic switch; direct PN coupling: alpha=alpha_i+alpha_E*Ehat+alpha_J*Jhat, beta=beta_i+beta_E*Ehat+beta_J*Jhat+beta_L*p_loss; 9 params; chi_f regression; target median ~1e-4, q=2 extrap <1e-2 |
| wf_nu_q_dep | `fit_scaling_wf_nu_q_dep.py` | 64-q grid [3,8]; same 10-param logistic-switch architecture; loss coords (Ehat,Jhat,p_loss) from waveform GW flux (not PN); PP-anchored poly in nu=q/(1+q)²; no surfinBH; target median ~1e-4, q=2 extrap <1e-2 |
| wf_nu_hybrid_q_dep | `fit_scaling_wf_nu_hybrid_q_dep.py` | 64-q grid [3,8]; 11-param hybrid: dp_hat=(p_loss-p0)/(p_loss[0]-p0)∈[0,1], alpha=alpha_i+(1-S)*alpha_L*dp_hat+S*(alpha_E*dE+alpha_J*dJ); wf-flux coords; PP-anchored nu poly (alpha_L→0); warm-started from wf_nu_q_dep cache; per-q median 8.27e-5, cubic master median 1.27e-4, max 9.63e-4; q=2 extrap 3.92e-3 |
| wf_nu_fluxes | `fit_scaling_wf_nu_fluxes.py` | 64-q grid [3,8]; 12-param = gated wf_nu + two INSTANTANEOUS-flux terms on alpha: alpha=alpha_i+S*(alpha_E*dE+alpha_J*dJ)+alpha_F*F_e+alpha_G*F_j (F_e=flux_e/max, F_j=flux_j/max, peak-normalized). Instantaneous fluxes are non-monotonic (peak at merger) and F_e≈ω*F_j so distinct time profiles. Everything gated stays (switch anchored, beta unchanged). Warm from wf_nu cache. **Best-ever per-q (median 6.86e-5, beats gated 8.76e-5) but FAILS as master**: cubic master median 6.31e-4 / max 6.38e-3 / q=2=0.11. alpha_F,alpha_G are degenerate (F_e≈ω*F_j, both peak at merger) so the per-q split scatters — kink/basin-jump at q≈3.5-4 (alpha_G flips −0.067→+0.068, alpha_F jumps −0.086→−0.220) is unregressable. Instantaneous fluxes carry real per-q info but the 2-coeff parameterization won't regress. Next: try SINGLE flux term (alpha_F only) to kill the degeneracy. |
| wf_nu_switchless_alpha | `fit_scaling_wf_nu_switchless_alpha.py` | 64-q grid [3,8]; 10-param = wf_nu with S removed from alpha ONLY (beta stays gated). Per-q median 2.3e-4; master median 1.7e-3 / max 1.8e-2; q=2 extrap 8.5e-2, q=3 1.8e-2. Less bad than full switchless (beta gated saves the time-map) but still fails <1e-2 target in-range at q=3 and at q=2. Confirms: alpha also needs the gate for clean regression. Still no peak. |
| **wf_nu_hybrid_global** | `BHPTNRHybridGlobal.py`, `scaling_wf_nu_hybrid_global.md`, `peaks_results/prototypes/global_joint_fit.py` | Same 11-param wf-nu-hybrid form, coeffs from a **GLOBAL JOINT FIT** (optimize master poly coeffs directly vs all [3,8] waveforms at once, Powell + analytic-φ0 fast evaluator; λ=1e-8) instead of per-q→regress. **Training points sampled even-in-ν** (`--sample nu`, dense near the q=3 boundary that governs q<3 extrapolation; replaced even-q on 2026-07-26). More uniform in-range (median 1.81e-4, **max 5.2e-4 vs master 9.6e-4**), q8=3.89e-4/q5=1.20e-4/q3=5.22e-4; **clears the 1e-2 gate across q∈[2,2.75]** (q2=4.04e-3, q2.25=9.38e-3 — even-q had FAILED q2.25 at 1.10e-2). Pure [3,8]. NOTE: the per-q master still beats it at low q (no q2.25 notch; q2.25=3.38e-3) — the global fit's win is in-range uniformity, not q<3. Coeffs `wf_nu_hybrid_global_results/coeffs.json` (prev even-q kept as `coeffs_evenq_lam1em8.json.bak`). Residual q2 defect = merger α overshoot (cosmetic, ~2% energy; the mismatch lever is β_r). |
| **pn_anchored** | `BHPTNRPNAnchored.py`, `fit_scaling_pn_anchored.py`, `scaling_pn_anchored.md` | **PN-anchored (2,2)** per `alpha_beta_pn_scaling_note_revised.pdf`: q-dependence IMPOSED analytically (inspiral α,β = `X1^(6/5)·[1+ν·residual]`, α's leading slope FIXED at 1PN `55/42·νx`; MR via switch to QNM `β_r` + `α_mr`). Only **9 ν-flat residual constants** fit — GLOBAL joint (Powell) on even-ν [3,8]. **q<3 is evaluated not extrapolated → monotonic, NO q2.25 notch**: q2.75/2.5/2.25/2.0 = 6.8e-4/9.0e-4/1.4e-3/**2.55e-3** (beats even-ν global 4.0e-3 AND remnant_partial 3.7e-3 at q2). In-range median 6.1e-4 (3× worse than even-ν — price of ν-flat residuals). Inspiral-only q2=0.21% (87% of power); MR-only 0.57%. CAVEAT: switch stays weak (S≤0.18) so the MR/QNM branch is UNDER-ENGAGED — merger shaped by the inspiral poly + x=0.26 clip, `α_mr(q2)≈1.0` unphysical; inspiral genuinely PN-anchored, MR not yet. Coeffs `pn_anchored_results/coeffs.json`; details+caveats in `scaling_pn_anchored.md`. |
| **peaks (Stage-2)** | `fit_scaling_peaks_stage2.py`, `BHPTNRPeaks.py`, `scaling_peaks.md` | **Peak-ratio method (Islam+Khanna 2307.03155): α,β read off waveform peaks, NOT L2-fit.** Trained [2.25,8]; centered-Chebyshev in x (α deg5, β deg4) → ν-regression PP-anchored (deg4); **TIME-MAP reconstruction** (evaluate model α,β only at peaks, interp map through peaks in time — the per-sample x-map is a structural ~2% wall at q=2). In-range median **0.26%**, q=2 extrap **0.55%** (clears <1% gate). Coverage 98% (inspiral-through-merger; ringdown ~2% energy NOT modelled). NOTE: in-band ~40× worse than L2 models (7e-5) — this is a physical/low-q method, not an in-band replacement. Coeffs in `peaks_results/stage2_coeffs.json`. |
| **gw_remnant_energy** | `BHPTNRGwRemnantEnergy.py`, `fit_scaling_gw_remnant_energy.py`, `scaling_gw_remnant_energy_mult.md`, `NRBHP_gw_remnant_energy_plots.py` | **Switchless, energy-driven, NO PN expressions, no gate, no remnant fits.** `alpha = alpha_PP(nu)*(1+alpha_E(nu)*E(t))`, same for beta, where E(t) is the RAW `gw_remnant` `Eoft` (cumulative radiated energy, units of M, zero at window start; modes (2,2)+(2,-2) of the ppBHPT). phi0 solved analytically (`arg(sum(h_NR*conj(h_model)))`) so **5 free params/q, not 6**. Prefactors PP-anchored to 1 at nu=0; couplings unanchored — E_rad→0 on its own (E_tot ~ nu^2.31 over [3,10]). Per-q then nu-regress, trained q∈[3,8] only, **degree 3** (deg-4 overfits and wrecks low q: q2.5 1.3e-3→6.7e-2). In-range median **6.16e-4** / max **9.87e-4** (max/median 1.6) — matches `pn_anchored` in-range with zero PN input. q2.75/2.5/2.25/2.0 = 1.0e-3/**1.3e-3**/4.1e-3/2.4e-2. E(t) supplies the merger ramp the logistic gate used to fake. **q=2 is limited by degeneracy, not by the form** (per-q at q=2 reaches 8.5e-4): beta_PP and P=beta_PP*beta_E trade off, beta_PP needs ~±0.5% at q=2, and P crosses zero at q≈4.1 leaving its q=3-boundary slope unconstrained. Global joint fit is the indicated fix (next). Coeffs `gw_remnant_energy_results/coeffs_mult.json`; plots `Agentic_plots/gw_remnant_energy/`. Other forms in the same script for comparison: `linear`, `linear_nu`, `mass_power`. |
| **gwr_energy_stiff** | `fit_scaling_gwr_energy_stiff.py`, `scaling_gwr_energy_stiff.md`, `NRBHP_gwr_energy_stiff_plots.py`, `Agentic_plots/gwr_energy_stiff/` | **Best q<3 of this family, and no PN anywhere.** Per-q model identical to `mult` (reuses `per_q_cache_mult.json`); only the regression layer changes — **9 coefficients** with the couplings' nu-structure IMPOSED from measurement instead of fitted as free cubics: `alpha_PP(nu)` deg-3 anchored (3), `alpha_E = (A0+A1 nu)/nu` (2, because alpha_E*nu measures flat to 14%), `beta_PP(X2)` deg-2 anchored in **X2=1/(1+q)** (2, because beta_PP is near-linear there — deg-1 resid 0.196% vs 3.37% in nu), `P = P0+P1 nu` (2, the anchor-independent dbeta/dE). Dropped vs 14-coef: nu^2 and nu^3 of both couplings + nu^3 of beta_PP. With `--global` joint refit: in-range median **6.24e-4** / max **9.74e-4** (both at or better than the 14-coef version), q2.75 9.83e-4, **q2.5 1.12e-3**, q2.25 1.73e-3, **q2 3.79e-3** — 6.4× better at q=2 than 14-coef, level with `remnant_partial` (0.37%), vs `pn_anchored` 2.55e-3 which needs PN. beta_PP now predicts held-out low q to −0.03…−0.46%, inside the ±0.5% tolerance the sensitivity scan demanded. **Key lesson: the global fit helps ONLY once the parameterisation is stiff enough that per-q→regress cannot reach the per-q floor** (14-coef was already at it → wash; 9-coef starts 1.1e-3 vs floor 9.6e-4 → joint fit recovers it and improves every low-q point). |
| **gwr_energy_fluxanchored** | `fit_scaling_gwr_energy_fluxanchored.py`, `scaling_gwr_energy_fluxanchored.md`, `gwr_energy_fluxanchored_results/` | **BEST MODEL ON BOTH AXES. 7 coefficients.** `alpha = alpha_PP(q)*(1+alpha_F(q)*F(t))` with **F = Edot/max(Edot)** (peak-normalised INSTANTANEOUS FLUX, not cumulative E); beta unchanged. `alpha_PP=X1^(6/5)(1+c0 nu+c1 nu^2)` [2], `alpha_F=A0 nu+A1 nu^2` [2], `beta_PP=X1^(6/5)(1+b nu)` [1], `P=P0+P1 nu` [2]. **`alpha_F` is PP-anchored BY CONSTRUCTION** (F peak-normalised → alpha_F ≈ d ln alpha ~ nu^1.53), unlike alpha_E which needed nu^-1. **seeded**: in-range med 4.90e-4 / max 7.16e-4, q2.75/2.5/2.25/2 = 6.46e-4 / 5.54e-4 / **5.02e-4** / **7.59e-4** — q<3 as accurate as in-range, first time in this workspace. **global**: in-range med 4.75e-4, q2 1.31e-3 (joint fit buys in-range, costs q=2 — INVERTS the stiff lesson because the flux forms already sit at the per-q floor). Beats both parents 4/4 at q<3. NOT PN-free. **PER-Q FLOOR NOW MEASURED (`gwr_energy_flux_results/per_q_floor_lowq.json`), consistency check PASSES**: flux floor 6.72e-4 / 5.88e-4 / 5.01e-4 / 4.21e-4 / **4.25e-4** at q=3/2.75/2.5/2.25/2 — i.e. **~2x below the E-coupled floor** (1.42x at q=3 rising to 1.99x at q=2) and *falling* toward low q, which is why low-q master numbers can sit below the in-range median. Master/floor = 1.10/1.11/1.19/**1.78**x. **Where the margin goes:** the X1^(6/5) prefactors extrapolate to <1% at q=2 (a_PP +0.9%, b_PP −0.2%) while the two empirical couplings carry all of it (a_F −18.0%, P +23.0%) — headroom to take q=2 from 7.6e-4 toward 4.3e-4 with better nu-forms for a_F and P. **Remaining caveats: --nperq robustness untested, higher modes not re-run** — see md. |
| **gwr_energy_flux** | `fit_scaling_gwr_energy_flux.py`, `gwr_energy_flux_results/` | Controlled test: flux alpha on the SAME uniform deg-3 14-coef layer as `gwr_energy_global`. **Isolates the coordinate: in-range med 4.70e-4 / max 7.16e-4 vs 6.26e-4 / 9.70e-4 — 25% better with identical parameter count**, but low-q unchanged/worse (q2 4.6e-2). **Conclusion: alpha's COORDINATE controls in-band accuracy; the REGRESSION LAYER controls extrapolation — independent failure modes.** Also settles the open `wf_nu_fluxes` question: a SINGLE flux term regresses cleanly (a_F −0.321→−0.121, smooth monotonic, no basin jump), unlike the two degenerate terms that killed it. |
| **gwr_energy_anchored** | `fit_scaling_gwr_energy_anchored.py`, `scaling_gwr_energy_anchored.md`, `gwr_energy_anchored_results/`, plots `NRBHP_gwr_energy_anchored_plots.py` -> `Agentic_plots/gwr_energy_anchored/` | **6 coefficients (from stiff's 9), BETTER at all four held-out q<3.** Puts the derived `X1^(6/5)` chirp factor into both prefactors: `alpha_PP=X1^(6/5)(1+c0 nu+c1 nu^2)` [2], `alpha_E=A0/nu` [1], `beta_PP=X1^(6/5)(1+b nu)` [1], `P=P0+P1 nu` [2]. Same per-q model (`mult`), same global machinery. In-range median 6.2551e-4 / max 9.7884e-4 (within 0.5% of stiff); q2.75/2.5/2.25/2 = 9.48e-4 / 9.28e-4 / 1.02e-3 / **1.53e-3** vs stiff's 9.83e-4 / 1.12e-3 / 1.73e-3 / 3.79e-3 — **2.47x better at q=2**, beating `pn_anchored` (2.55e-3) and within 1.8x of the per-q floor (8.48e-4). **CAVEAT: this is NO LONGER PN-free** — keep the 9-coef `gwr_energy_stiff` frozen as the evidence run (its value is that it DISCOVERED X1^(6/5) rather than importing it); report both. Note the `seeded` (per-q→regress) variant beats the joint fit at q<=2.25 (1.02e-3 at q=2) but fails q=2.75 and is worse in-range. |
| **gwr_betaqnm** | `fit_scaling_gwr_betaqnm.py`, `scaling_gwr_betaqnm.md`, `gwr_betaqnm_results/`, plots `NRBHP_gwr_betaqnm_plots.py` + `NRBHP_betaqnm_overlay.py` -> `Agentic_plots/gwr_betaqnm/` | **First model with beta pinned between TWO DERIVED anchors** — `X1^(6/5)` at early inspiral and the remnant QNM ratio `W_Schw*Mf/omega_220(chi_f)` at ringdown (Mf, chi_f from **gwModelRemS**, installed 2026-08-19; both anchors verified against a direct `omega_pp/omega_NR` measurement to 0.3-0.9%). `beta = B0 + (Bm-B0)*u + (B1-Bm)*v`, u/v ramps in `Ehat=Eoft/E_tot` split at the |h| peak, B1 reached as a **plateau**. 5 coef, beta gets 2 (`m`=0.632 merger value, `b`=-0.015 anchor tilt). **beta RISES at every q** (+0.72% at q=8 to +10.1% at q=2) — which five earlier routes could not achieve (b_E>=0 collapsed to flat; the 4-drive test gave one sign for all four; the merger-aligned gauge made q=5 worse; gwr_anchored2/3 cost 4-37x). **But NOT competitive**: in-range median 2.93e-3 / max 6.75e-3, q2 5.56e-2 (vs fluxanchored 4.75e-4 / 1.31e-3), non-monotonic in q, worst at q=3. Value is diagnostic: (a) it prices the QNM anchor at 6.2x, now measured 3 independent ways and insensitive to the path shape; (b) the `--parfree` variant (beta FULLY parameter-free, m=1 b=0, 3 coef all in alpha) **INVERTS the q-trend** — low q 5-8x BETTER than in range (1.78e-3 at q2.25 vs 1.44e-2 in range) because for beta q<3 is EVALUATED not extrapolated (X1^(6/5) is algebra, RemS valid to q~1000, Ehat measured per q), which is direct evidence the fitted models' low-q trouble is coefficient extrapolation rather than physics. Settled negative: redefining merger as E-saturation is WORSE (3.34e-3 vs 2.93e-3) — do not retry. alpha's coefficients here are compensators, not descriptions. |
| gwr_energy_stiff — OPEN degree issues | `scaling_gwr_energy_stiff.md` §"Two degree choices the measurements do NOT support" | **Regression-level, needs a `--global` refit before adopting.** (1) `alpha_E` degree 1 is dominated by BOTH neighbours: `A=alpha_E*nu` is non-monotonic in [3,8] (min near q=4), so a line tilts the wrong way — deg 2 beats deg 1 in-range (2.45% vs 5.07% max) AND at q=2 (+7.9% vs +19.4%); even deg 0 extrapolates better (+12.6%). Options: deg 0 (9→8) or deg 2 (9→10). (2) `beta_PP`: the X2 COORDINATE is strongly justified (0.196% vs 3.375% for nu at deg 1), but the 2 coefficients buy in-range accuracy, NOT q=2 — imposing `X1^(6/5)` with ZERO coefficients ties them exactly at q=2 (−0.46% both) and only costs in-range (0.213% vs 0.079%, ≈60 M of accumulated drift). 9→7 is a real trade, not a free win. |
| gwr_energy_global | `fit_scaling_gwr_energy_global.py`, `scaling_gwr_energy_global.md` | **GLOBAL joint fit of `gw_remnant_energy` — tried, and it is a WASH. Do not expect it to fix q<3.** Same `mult` model; the 14 master coefficients (deg-3 in nu, alpha_PP/beta_PP anchored) optimised directly against 12 even-nu waveforms in [3,8] with Powell, analytic phi0, t0_nr a per-q 1-D nuisance. Result vs per-q→regress: in-range median 6.26e-4 vs 6.22e-4, max **9.70e-4 vs 9.82e-4** (1% better); q2.75 9.9e-4 vs 1.0e-3; but q2.5 1.40e-3 vs **1.32e-3**, q2.25 4.9e-3 vs **4.1e-3**, q2 2.8e-2 vs **2.4e-2** — 6–17% WORSE below q=2.5. **Why it doesn't help:** the global fit only has slack to recover when the master polynomial can't reach the per-q optima (wf_nu_hybrid had a ~50% gap: per-q 8.27e-5 → master 1.27e-4). Here per-q→regress was already AT the per-q floor in-range, so there was nothing to gain; and below q=3 both routes extrapolate the same coefficients constrained only by [3,8], so being "on the valley" in-range says nothing about where it is at nu=0.222. **q<3 here is information-limited, not method-limited.** Also: the nonlinear-excursion regulariser (the wf_nu_hybrid_global trick) HURTS monotonically — lam 0 / 1e-8 / 1e-6 gives q2.5 1.40e-3 / 2.39e-3 / 7.53e-3 — because the true low-q behaviour genuinely is nonlinear (beta_E turns sharply up below q=3). Shipped with `--lam 0`. |
| wf_nu_switchless | `fit_scaling_wf_nu_switchless.py` | 64-q grid [3,8]; 8-param, NO switch: alpha=alpha_i+alpha_E*dE+alpha_J*dJ, beta=beta_i+beta_E*dE+beta_J*dJ (dE,dJ ref'd to T_ANCHOR); wf-flux coords; PP-anchored nu poly. **FAILED as master**: per-q converges (median 5.1e-4) but master median 8.8e-3 / max 1.9e-2 (~17× worse than per-q, ~65× worse than gated) — beta flux couplings (beta_E,beta_J ~±0.01) scatter non-monotonically in nu, don't regress, and beta feeds the integrated time-map so residuals blow up (worst at q=8). Diagnostic lessons: (1) monotonic flux coords can't make the non-monotonic alpha peak; (2) the switch structure is what makes beta regress smoothly across q |

## Higher modes

`fit_scaling_hm_backbone.py`, `scaling_hm_backbone.md`, `hm_backbone_results/`, cache `.cache/hm/`,
plots `NRBHP_hm_backbone_fig8_plots.py` -> `Agentic_plots/hm_backbone/q{2,4}_waveforms_224.png`
(Fig-8 format matched to the pn_anchored prototypes) (2026-08-03, base = `gwr_energy_stiff`).
**PLOTTING TRAP for anything built on this model: call `rcdefaults()` AFTER the model imports.**
`gw_remnant/gw_utils/gw_plotter.py` sets font.size=18, font.family='STIXGeneral', axes.linewidth=1,
figure.figsize=(14,10) at import time, so reset-then-import silently gives 18pt STIX figures instead
of 10pt DejaVu Sans. The pn_anchored prototypes reset first — safe for their chain, wrong for this one. **Prior round (2026-07-28, base = `pn_anchored`, (3,3)+(4,4)
only, never committed): prototypes `peaks_results/prototypes/hm_*.py`, plots
`Agentic_plots/pn_anchored_hm/`.** Pure-backbone comparison: pn_anchored gives (3,3) in-range 1.98e-3 /
q2 6.5e-3 and (4,4) 9.1e-3 / q2 7.0e-2, vs stiff's 2.69e-3 / 1.73e-2 and 1.34e-2 / 4.62e-2 — i.e. the
PN-anchored base transfers better at low q (it starts from q2=2.55e-3 vs 3.78e-3), the PN-free one is
better on (4,4) at q2. Carry over from that round, do NOT re-derive: (4,4) at q=2 fails in the
**amplitude, not the phase** (the (m/2)phi22 + shared-map phase transfer is clean), and it is
structural — in-sample own-residual ceiling 2.06e-2 vs 2.72e-2 extrapolated, likely spherical-spheroidal
mixing — though only w.r.t. a 2-coeff rho, so not proven fundamental.

Architecture is settled and **not** per-mode-independent: **beta(t) and the time map are
COMMON to all modes; only alpha is mode-dependent.** That is how BHPTNRSur1dq1e4 itself is
built (note Eq. 1: one `beta(q)`, `alpha_l(q)` per ell) and how the reference
`BHPTutils/bhpt_utils/nrcalib/alpha_beta_calibration.py` `AlphaBetaOptimizer` works (param
vector = `[beta, alpha_mode1, alpha_mode2, ...]`). So higher modes add NO time-map freedom
and cannot degrade (2,2). Per-mode constant phase is analytic, as for (2,2).

Note §6.3 Eq. (43) gives a ZERO-PARAMETER transfer of the (2,2) calibration onto every
higher mode: `alpha_lm = alpha_22 * C_lm(nu) * beta^(-(l+eps_lm-2)/3) * rho_lm`, with
`eps_lm=(l+m)mod2`, `C_lm=|X2^(l+eps-1)+(-1)^m X1^(l+eps-1)|` (verified == note Eq. 40:
C_33=Delta, C_44=1-3nu, C_55=Delta(1-2nu), C_22=1), and `rho_lm->1` the only free part.
`beta^(...)` is not a parameter — it is `x_NR/x_pp = beta^(-2/3)`, implied by the shared map.

**Step-1 result (rho=1, nothing fitted):** power-weighted mode-sum error over all 9 modes is
1.27e-3 / 9.78e-4 / 6.45e-4 / 4.05e-3 at q=3/5/8/2 vs 9.74e-4 / 7.01e-4 / 3.54e-4 / 3.79e-3
for (2,2) alone — i.e. 100% of the power for a 7-82% cost and zero parameters. **(3,3) needs
no residual at all** (2.2-4.8e-3, best-constant rho within 0.1-0.9% of 1). (4,4) 1.0-3.0e-2,
(2,1) 1.3-2.7e-2, (5,5) 3.0-8.1e-2. Off-diagonal (3,1),(3,2),(4,2),(4,3) FAIL (6e-2..1.26,
>1 at q=2) exactly as the note warns for spherical-spheroidal mixing — but they carry <0.1%
of power. (2,1) is off-diagonal yet behaves diagonally; group it with l=m.

**All quoted q=2 numbers are EXTRAPOLATED MASTER, not per-q truth** (only `t0_nr` is refit at q=2,
as in the (2,2) table; nothing per-mode is fitted). `--perq` swaps in the per-q-optimal (2,2) params
as a diagnostic: (3,3) then drops 1.73e-2 -> **6.48e-3** with k=0.9999 (so (3,3) at q=2 is limited by
the (2,2) BASE, not the transfer — improving (2,2) at low q improves (3,3) for free), while (4,4) and
(5,5) get WORSE (4.6e-2->5.8e-2, 4.1e-2->6.9e-2) — the master's extrapolated alpha_22/beta were
accidentally compensating their transfer error. Independent corroboration, from a different base and
with nothing fitted, of the July finding that (4,4) at q=2 is structural.

**Step 2 (next):** residual is measured to be O(nu) with an ell-dependent coefficient —
`(1-k)/nu` is q-flat at ~{0, 0.46, 1.2} for ell=3,4,5 and 0.55 for (2,1), confirming the
note's Eq. (52) nu-suppressed form at waveform level. But a CONSTANT rho removes only 11-42%
of the error for ell>=4, so `rho_lm` must be time-dependent; in this model's idiom that is
`rho_lm(t) = 1 + rho_E,lm * E(t)` (keeps it PN-free and gate-free, unlike the note's x/switch
version). Modes usable = BHPT∩NR = (2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,5);
BHPT-only (5,3),(5,4),l>=6 have no NR counterpart in NRHybSur3dq8.

## Consolidated higher-mode comparison (2026-08-03)

`python NRBHP_hm_backbone_fig8_plots.py --q 2 4 --model E_deg3 flux_deg3 E_anchored flux_anchored
--outdir consolidated_gwremnant_hm` -> `Agentic_plots/consolidated_gwremnant_hm/hm_q{2,4}_<base>.png`
(8 figures). `fit_scaling_hm_backbone.py` now dispatches 4 (2,2) bases via `read_stiff_coeffs(model)`
returning `{params_at, coord}`, with a generalised `_evaluate` over alpha's coordinate — the backbone
still fits NOTHING, only the transferred quadrupole changes. Each panel's (2,2) reproduces that base's
documented mismatch.

| base | q=4 (2,2)/(3,3)/(4,4) | q=2 (2,2)/(3,3)/(4,4) |
|:---|:---|:---|
| E_deg3 (14c) | 9.34e-4 / 3.82e-3 / 1.98e-2 | 2.76e-2 / 6.32e-2 / 2.48e-1 |
| flux_deg3 (14c) | **7.16e-4 / 3.48e-3 / 1.74e-2** | 4.60e-2 / 1.10e-1 / 3.42e-1 |
| E_anchored (6c) | 9.31e-4 / 3.88e-3 / 2.02e-2 | 1.53e-3 / 9.99e-3 / **4.87e-2** |
| flux_anchored (7c) | **7.15e-4 / 3.47e-3 / 1.72e-2** | **1.31e-3 / 9.41e-3** / 6.41e-2 |

In-range the flux bases win on every mode and the layer barely matters; at q=2 the anchored layer is
what matters and the uniform-deg-3 bases are hopeless (up to 0.34). **Between the anchored bases, flux
wins (2,2)+(3,3) but LOSES (4,4)** (6.41e-2 vs 4.87e-2) — that is the **ringdown defect**: F is
peak-normalised and non-monotonic, so `alpha=alpha_PP(1+alpha_F F)` snaps back to alpha_PP after
merger while the empirical alpha keeps falling (visible in `Agentic_plots/gwr_global_fluxanchored/`
q=2 params panel). (4,4) carries more merger-ringdown weight, so it pays. **Indicated fix: two-term
alpha (E for the post-merger floor + F for the merger ramp)** — E and F are NOT degenerate the way
`wf_nu_fluxes`'s two flux terms were. Not yet tried.

## Plots for the three GLOBAL (2,2) models

`NRBHP_gwr_global_plots.py [--model E flux fluxanchored]` — ONE script, `--model` dispatch, one
evaluator generalised over alpha's coordinate (`e_oft` vs `flux_hat`), so the three sets are
produced identically and are directly comparable. Uses the GLOBAL (jointly fitted) coefficients
only (fluxanchored's `seeded` variant is ignored there). 15 PDFs each ->
`Agentic_plots/gwr_global_{E,flux,fluxanchored}/`, prefixes `gwr_global{E,F,FA}_`.
Reproduces the documented mismatches exactly (flux q2 4.6004e-2, fluxanchored q2 1.3056e-3,
E-global q3 9.6973e-4 = its quoted in-range max). The 3rd stacked panel shows the alpha DRIVE
coordinate — E(t) for `E`, F(t)=Edot/max(Edot) for the two flux models — and the `_drive_vs_q`
figure replaces `_Eoft_vs_q` accordingly.

**Plotting-script duplication (known, not yet cleaned up):** `NRBHP_gw_remnant_energy_plots.py`
(pre-existing, the format reference), `NRBHP_gwr_energy_stiff_plots.py` and
`NRBHP_gwr_energy_anchored_plots.py` are near-copies of one another; `NRBHP_gwr_global_plots.py`
is the generalised version and could absorb the middle two. Before deleting them, verify the
generalised evaluator reproduces their numbers — it optimises t0 directly on the full grid,
whereas the copies call `gwre.evaluate_model` with a fit-then-full two-step.

`PN_opt_remnant_partial` and `PN_opt_nu_Pade` both reuse the q_dep per-q cache; only the regression layer changes (chi_f(q) / PP-anchored nu-poly respectively). Waveform cache shared: `.cache/q_dep/`. `PN_opt_nu_Pade` writes machine-readable coeffs to `PN_opt_nu_Pade_results/selected_fit.json` (the plotting script reads that, not the markdown). Plots in `Agentic_plots/Pade_nu/`.

## BHPT surrogate domain limit

`BHPTNRSur1dq1e4.py:63` sets `X_min = [np.log10(2.5)]`, and `common_utils/check_inputs.py:36`
only *prints* a warning when you go outside it. So every q=2 and q=2.25 number in this
workspace — `remnant_partial` 0.37%, `peaks` 0.55%, `pn_anchored` 0.255%, and
`gw_remnant_energy` — has an extrapolated BHPT **input** underneath the extrapolated
coefficients. Shared by all models, so cross-model comparison is unaffected, but it
belongs in the paper's caveats. q=2.5 is the lowest mass ratio where the input is in domain.

Also: `Moore_PN_exprs.dEdt` has prefactor 32 where Moore Eq. (6.3) (arXiv:1605.00304) has
32/5. Harmless in the models built on it (they only ever use peak-normalised flux
coordinates, where the constant cancels) but a factor 5 off as an absolute luminosity —
see `plot_gwremnant_vs_Moore_Eoft.py`, which writes its own 3PN flux and energy.

`gw_remnant` is a pip dependency of the `ut_claude` env (added 2026-08-01).

## q<3 extrapolation key finding

The 1/q polynomial (q_dep) diverges catastrophically at q=2 (phi0 extrapolates to ~369 rad). Two fixes tame q=2: the chi_f polynomial (remnant_partial, 0.37% at q=2) and PP-anchored deg-4 poly in nu (nu_Pade, ~6.5% at q=2). The remnant_partial improvement from ~5.8% to 0.37% came from using a denser 64-q training grid (both the original 40-pt linspace and the supplementary 24 intermediate-q points added during extended q_dep runs). Root cause of q_dep failure is the regression coordinate, not the degree: bounded, physically-monotonic coordinates (chi_f, nu) + endpoint anchoring tame extrapolation. Verified: collapsing the amplitude to one term (alpha_E=alpha_J) is NOT viable — it costs 2-4× accuracy, so both alpha_E and alpha_J carry real information despite their -0.998 cross-q correlation. For q<3 accuracy comparable to training range, explicit calibration at those q values would still be needed.
