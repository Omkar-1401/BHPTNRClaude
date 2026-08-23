# Summary: Creative PN-Loss Calibration of BHPT to NR (q=5, (2,2) mode)

## The calibration problem

Black hole perturbation theory (BHPT) waveforms are computed in the extreme-mass-ratio
limit. At moderate mass ratios like q=5, they must be rescaled to match numerical
relativity (NR) waveforms. Two independent scalings are needed at each moment in time:

- **alpha(t)**: amplitude rescaling of the complex waveform.
- **beta(t)**: time-stretch factor — BHPT time runs at a different rate than NR time.
  Integrating beta maps BHPT coordinate time to the NR time axis.

The calibration finds alpha(t) and beta(t) such that the rescaled BHPT waveform
minimizes the mismatch error mathcalE against the NR surrogate `NRHybSur3dq8`.
The BHPT surrogate is `BHPTNRSur1dq1e4` with `calibrated=False` (raw output).

## The PN-loss coordinate

Rather than parameterizing alpha and beta as explicit functions of coordinate time,
this model uses a physically motivated proxy coordinate `p_loss`, constructed from
the PN energy and angular momentum flux radiated by the BHPT waveform:

```
omega_gw      = |d/dt unwrap(arg(h_BHPT_22))|
x_pn          = (omega_gw / 2)^(2/3)          # PN expansion parameter
F_E(t)        = dE/dt evaluated at x_pn(t)     # 2PN energy flux (Moore et al.)
F_J(t)        = dJ/dt evaluated at x_pn(t)     # 2PN angular momentum flux
DeltaE(t)     = integral from t_merger to t of F_E dt'
DeltaJ(t)     = integral from t_merger to t of F_J dt'
Ehat(t)       = DeltaE(t) / |DeltaE(t_ref)|
Jhat(t)       = DeltaJ(t) / |DeltaJ(t_ref)|
p_loss(t)     = 0.5 * (Ehat + Jhat)
```

`p_loss` runs monotonically from large negative values deep in the inspiral up through
zero near merger. Using it as the independent variable makes the model automatically
sensitive to where in the inspiral-merger-ringdown sequence a given instant falls,
rather than to the raw coordinate time (which differs between BHPT and NR).

The 2PN flux expressions are used as an IMR coordinate all the way through merger and
ringdown. In that regime they should be understood as a physically motivated proxy,
not a controlled PN approximation.

## The logistic (sigmoid) switch

The central architectural choice is a single logistic function that smoothly blends
inspiral-regime and merger/ringdown-regime behavior:

```
S(p_loss) = 1 / (1 + exp(-(p_loss - p0) / w))
```

- `p0` is the switch center in p_loss — the PN-loss value where S = 0.5.
- `w` controls the width of the transition. Constrained to w >= 0.015 so the
  transition spans a physically meaningful interval and the waveform is not
  artificially sharp.

S goes from ~0 deep in the inspiral to ~1 well past merger. The transition from
1% to 99% of S occurs over the NR time interval approximately [-105, -41] M,
spanning about 64 M and centered near t = -65 M.

## The 10-parameter model

```
dE  = Ehat - Ehat(p0)
dJ  = Jhat - Jhat(p0)
dp  = p_loss - p0

alpha = alpha_i  +  S * (alpha_E * dE  +  alpha_J * dJ)
beta  = (1 - S) * (beta_i  +  beta_L * dp)  +  S * beta_r
```

**alpha** (amplitude):
- Inspiral value is the constant `alpha_i`.
- Past the switch, it grows/falls according to how much energy (`alpha_E`) and
  angular momentum (`alpha_J`) have been lost relative to the switch point.
- Continuity at the switch is automatic: when S=0, the correction term vanishes,
  so alpha equals alpha_i on both sides of the switch with no discontinuity and
  no continuity constraint consuming a degree of freedom.

**beta** (time-stretch):
- Inspiral side: `beta_i + beta_L * dp` — a slowly drifting linear function of
  the PN-loss coordinate rather than a constant.
- Merger/ringdown side: the constant `beta_r`.
- The two sides are blended continuously by S.

The time map uses beta to convert BHPT coordinate time to NR time:

```
tau(t_bhpt) = t0_nr + integral from t_anchor to t_bhpt of beta(t') dt'
```

where `t_anchor = -100 M` is a fixed anchor (not fitted) and `t0_nr` is a global
time shift. The full waveform model is then:

```
h_model(tau(t_bhpt)) = alpha * exp(i * phi0) * h_BHPT(t_bhpt)
```

The 10 free parameters are: `p0, w, alpha_i, alpha_E, alpha_J, beta_i, beta_r,
t0_nr, phi0, beta_L`.

## The key physics insight: inspiral beta drift

All predecessor models held beta constant during the inspiral, treating the
time-stretch as a single fixed rescaling. This produced a mismatch floor near
mathcalE ~ 1.2e-4 that could not be broken by adding more amplitude parameters.

The insight is that beta is not truly constant: the effective time-stretch between
BHPT and NR drifts slowly during the inspiral as the system loses energy and angular
momentum. Parameterizing this drift as a linear slope `beta_L` in the PN-loss
coordinate captures that evolution with a single new parameter. The accepted fit gives:

```
beta_L = 2.43e-3
```

This is small (a slow drift) but not negligible over the long inspiral. Releasing
this single degree of freedom halved the full-window mismatch error compared to the
best constant-beta model, while simultaneously allowing two parameters from the
predecessor Hermite construction to be dropped.

## Results and comparison

| Model                              | Parameters | Full-window mathcalE |
|------------------------------------|------------|----------------------|
| Predecessor (12-param Hermite)     | 12         | 1.198e-4             |
| This work (creative logistic)      | 10         | 6.037e-5             |

Local error breakdown for the accepted fit:

- Pre-transition inspiral: mathcalE = 4.07e-5
- Transition region: mathcalE = 1.03e-4
- Post-transition (merger/ringdown): mathcalE = 2.91e-4

The post-transition error is highest, as expected: the PN-loss coordinate is least
reliable in the ringdown where the 2PN flux expressions have no formal validity.

Common NR support coverage: 99.95% of the requested NR window [-5000, 100] M.

## Accepted parameter values

```
p0      = -0.93897417194
w       =  0.025391542087
alpha_i =  0.80865511874
alpha_E =  0.29223097028
alpha_J = -0.41306589417
beta_i  =  0.80796434842
beta_r  =  0.84590841417
t0_nr   = -83.024974702
phi0    =  1.6339824069   (radians)
beta_L  =  0.0024285785544
```

## Files

| File                              | Purpose                                      |
|-----------------------------------|----------------------------------------------|
| `fit_scaling_PN_opt_creative.py`  | Model definition, optimizer, mismatch eval   |
| `scaling_PN_opt_creative.md`      | Auto-generated technical reference card      |
| `NRBHP_time_dep_PN_opt_creative.py` | Diagnostic plots of waveform and parameters |
| `Moore_PN_exprs.py`               | 2PN energy/angular-momentum flux expressions |
| `Agentic_plots/`                  | Output PDFs from the plotting script         |
| `.cache/waveforms_q5_22.npz`      | Cached waveforms (loading takes ~1 min)      |

The NR reference directory `../gwAgentic_NRBHP/` is read-only. Its `AGENTS.md`
documents the predecessor 12-parameter Hermite fit and the environment setup for
the older `.venv`/`ut_codex` workflow (not used here; this workspace uses `ut_claude`).
