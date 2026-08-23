# Global joint fit vs. 2-stage (per-q → regress)

The key thing: **both methods are trying to find the same final object** — the "master"
coefficients `C` that make each parameter (α_i, β_i, p0, …) a smooth function of q
(e.g. a cubic in ν). They differ in *how* they search for `C`.

Notation. The model builds a waveform `h_model(t; q, θ)` from a per-q parameter set `θ`
(the ~11 numbers: p0, w, α_i, …). Those params are supposed to come from the master
coefficients via `θ = params(q; C)` (evaluate the polynomials at q). We have NR waveforms
at training mass ratios `{q1,…,qN}`. Goal: find `C`.

## 2-stage (per-q → regress)

**Stage 1 — N independent waveform fits, one per q, in isolation:**
```
for q in training_qs:
    θ[q] = argmin_θ  mismatch( h_model(t; q, θ),  NR[q] )   # q fitted ALONE
```
This produces a *table* of best-fit params: θ(q1), θ(q2), …. Crucially, q3's fit knows
nothing about q5's fit.

**Stage 2 — P independent 1-D curve fits, one per parameter:**
```
for each param p (alpha_i, beta_i, …):
    C[p] = leastsq_polyfit( nu(training_qs),  [θ[q][p] for q in training_qs] )
```
i.e. fit a smooth cubic *through the scattered per-q values* of each parameter. Those
cubics are `C`.

So the q-dependence is **discovered after the fact** by regressing whatever Stage 1
produced. Two consequences: (a) Stage 1 has no smoothness constraint, so degenerate
params jump between equivalent basins → the points you regress through are jagged;
(b) Stage 2 minimizes error in *parameter space*, not waveform space — a smooth
parameter fit isn't guaranteed to be a good *waveform* fit, and the scatter extrapolates
badly (this is the q<3 wall, the q2.25 notch).

## Global (simultaneous joint fit)

There is **no per-q stage and no regression**. You parameterize `C` directly (the ~30
polynomial coefficients) and minimize **one** objective that aggregates the waveform
mismatch across *all* training q's at once:

```
def objective(C):                       # C = the whole coefficient vector
    errs = []
    for q in training_qs:
        θ_q = params(q; C)               # evaluate the polynomials AT this q
        h   = h_model(t; q, θ_q)         # build the ppBHPT->NR waveform for this q
        errs.append( mismatch(h, NR[q]) )
    return mean(errs)                    # ONE scalar

C_best = minimize(objective, C0, method="Powell")   # ONE optimization, over C
```

**That's the "simultaneous" part.** A single optimizer proposes a value of `C`; to score
it, `objective(C)` sweeps *every* training q, builds each q's waveform (by evaluating the
same shared polynomials at that q), compares each to NR, and folds all those mismatches
into **one number**. The optimizer then nudges `C` to lower that one number, and repeats.

It doesn't solve all q's in closed form at once — it's still an iterative search — but
*every iteration's cost depends on every q together*. So a coefficient tweak that helps
q=5 but hurts q=3 raises the aggregate and gets rejected. The optimizer is therefore
forced to find coefficients that make a good waveform at *all* q simultaneously, and
because each q's params come from the *same low-order polynomial*, smoothness in q is
built in by construction — there's no scatter to regress through, and the objective is
the real waveform mismatch, not a parameter-space proxy.

## Why it matters (the trade)

| | 2-stage | global |
|:--|:--|:--|
| what's optimized | params per q, then curves through them | waveform mismatch over all q jointly |
| q-coupling | none until the regression | every step couples all q |
| smoothness | hoped-for (regress scattered points) | built-in (shared polynomial) |
| failure mode | degeneracy scatter → bad extrapolation | bigger nonlinear opt; needs warm start; can still sit in a degenerate basin for genuinely-degenerate sectors (the MR/switch) |
| cost | cheap per fit | each objective call builds N waveforms |

In practice the global fit is **warm-started from the 2-stage master** — you use the cheap
per-q→regress result as `C0`, then let the joint optimizer polish it against the real
waveforms. And for `pn_anchored` the same global machinery runs but with only **9**
coefficients instead of ~30, because the PN anchors already supply the q-dependence
analytically — so "simultaneous" there is a tiny, fast search.

## The degeneracy behind the 2-stage failure

"Degeneracy" here means: **different parameter combinations produce nearly the same
waveform**, so the mismatch has a flat valley rather than a sharp minimum — and the
2-stage method reacts badly to that.

### What the degeneracy is

The model's parameters don't act independently on the waveform; several combinations do
almost the same thing. In the hybrid model:

- **p0 ↔ w (switch location vs width).** A *later, sharper* switch and an *earlier,
  broader* switch produce almost the same α/β transition. They co-vary directly:
  q=3 → (p0=−0.85, w=0.13); q=8 → (p0=−1.02, w≈0). Traded off, not independently pinned.
- **α_E ↔ α_J (the two flux couplings).** They multiply `dE` and `dJ`, which are both
  smooth monotonic loss coordinates and hence nearly parallel, so `α_E·dE + α_J·dJ` can
  be reproduced by many `(α_E, α_J)` pairs (often opposite signs) — a collinear-regressor
  degeneracy.
- **β_L ↔ α_L (time drift vs amplitude drift).** A time-stretch drift changes the
  frequency sampled at a given mapped time, which mimics an amplitude drift; the two
  drifts partly substitute for each other.

Because of these, the mismatch as a function of θ (at fixed q) is not a bowl with one
bottom — it is a long flat **canyon floor**: sliding along the degenerate direction barely
changes the waveform (hence the mismatch).

### Why this specifically breaks the 2-stage method

**Stage 1 fits each q in isolation**, and the minimum is a flat canyon, so the optimizer
settles at an essentially **arbitrary point along the floor** — depending on the starting
guess, the optimizer's path, and tiny numerical noise. So:

- q=3 might land at (p0=−0.85, w=0.13); the next grid point q=3.13 lands in a *different
  but equally-good* corner of the same canyon. The **waveforms are nearly identical, but
  the parameter values jump** — the "degeneracy scatter". (The per-q p0 trend even has a
  −1.98 outlier where one q jumped basin.)
- Every per-q fit is *good* (low mismatch), yet the good solutions live in **inconsistent
  basins that don't form a smooth sheet** in q.

**Then Stage 2 regresses a smooth cubic through those scattered points.** The scatter is
noise, not physical q-dependence, so the cubic doesn't reproduce any single q's waveform
well, and its higher-order coefficients inflate chasing the jitter → **wild extrapolation
below q=3** (the p0 rail, the q2.25 notch).

Killer subtlety: *good per-q fits do not imply a good master.* The `proto_hybrid_pin_betar`
test showed a genuinely good q=2 solution exists at an on-trend β_r, but it sits in a basin
the [3,8] per-q fits don't smoothly connect to; pinning β_r helped in-range yet q=2 still
failed because the *other* params were still scattering across basins.

### Why global (and anchors) sidestep it

- **Global** ties all q's through one shared low-order polynomial, which *cannot*
  independently jump basins at each q — it is forced to thread a **single smooth
  trajectory** through all the flat canyons at once, imposing smoothness as a constraint
  during the fit rather than hoping to recover it by regressing afterward. (Geometrically:
  2-stage drops a ball into each canyon separately → random floor positions; global draws
  one smooth curve that must pass through every canyon.)
- **PN anchors** go further: fixing the leading q-dependence analytically (`X1^(6/5)`, the
  `55/42` slope) *removes* much of the degenerate freedom — fewer knobs, shallower canyons,
  less to scatter. That is why `pn_anchored` needs only 9 constants and extrapolates cleanly.

---

Examples in this repo:
- 2-stage: `fit_scaling_wf_nu_hybrid_q_dep.py` (per-q `optimize_case_worker` → `fit_master`).
- global: `peaks_results/prototypes/global_joint_fit.py` (`wf_nu_hybrid_global`) and
  `fit_scaling_pn_anchored.py` (`pn_anchored`, 9 residual coeffs).
