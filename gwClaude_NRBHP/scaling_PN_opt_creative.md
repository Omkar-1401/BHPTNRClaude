# Creative smooth PN-loss q=5 (2,2) scaling fit

This is a 10-parameter replacement for the 12-parameter smooth PN-loss opt
fit in `gwAgentic_NRBHP`. Instead of a cubic Hermite transition between an
independent constant branch and an independent linear branch, the creative
model uses a single logistic switch `S` in the PN-loss coordinate and
enforces continuity of `alpha` at the switch:

```python
S = 1 / (1 + exp(-(p_loss - p0) / w))
dE = Ehat - Ehat(p0)
dJ = Jhat - Jhat(p0)
dp = p_loss - p0
alpha = alpha_i + S * (alpha_E * dE + alpha_J * dJ)
beta  = (1 - S) * (beta_i + beta_L * dp) + S * beta_r
```

The key new ingredient is `beta_L`: a slow linear drift of the inspiral
time-stretch factor along the PN-loss coordinate. The predecessor models
held `beta` exactly constant before the transition; releasing that single
degree of freedom halves the achievable `mathcalE` while dropping two
parameters elsewhere (the post-branch `beta` slopes and the independent
post-branch edge constants).

The BHPT waveform is the raw surrogate output with `calibrated=False`; the
NR reference is `NRHybSur3dq8`.

```python
q = 5
nu = q / (1 + q)**2
nu = 0.138888888888889
t_bhpt_merger = 1.10958353616297e-07
t_nr_merger = -0.837744212698453
t_ref = -100
```

PN-loss coordinates (identical to the predecessor PN-opt fits):

```python
omega_gw = abs(d unwrap(arg(h_BHPT_22)) / dt_bhpt)
x_pn = (omega_gw / 2)**(2/3)
F_E = Moore_PN_exprs.dEdt(nu, x_pn)
F_J = Moore_PN_exprs.dJdt(1.0, x_pn, nu)
DeltaE_PN(t) = integral_from_merger_to_t F_E dt
DeltaJ_PN(t) = integral_from_merger_to_t F_J dt
Ehat = DeltaE_PN / abs(DeltaE_PN(t_ref))
Jhat = DeltaJ_PN / abs(DeltaJ_PN(t_ref))
p_loss = 0.5 * (Ehat + Jhat)
```

The 10 fitted parameters are:

```python
p0 = -0.93897417194
w = 0.025391542087
alpha_i = 0.80865511874
alpha_E = 0.29223097028
alpha_J = -0.41306589417
beta_i = 0.80796434842
beta_r = 0.84590841417
t0_nr = -83.024974702
phi0 = 1.6339824069
beta_L = 0.0024285785544
```

Derived switch quantities:

```python
Ehat(p0) = -0.955603796444244
Jhat(p0) = -0.922344547435756
t_bhpt_switch = -77.6454404606747
t_switch_nr = -64.7592859531905
transition_interval_nr = [-104.553497943226, -40.7887984369544]  # 0.01 < S < 0.99
DeltaE_PN(t_ref) = -0.0216330274770254
DeltaJ_PN(t_ref) = -0.0443145705120832
```

The time map anchors the integral of `beta` at the fixed BHPT time
`t_anchor = -100` (not fitted):

```python
tau(t_bhpt) = t0_nr + integral_from_t_anchor_to_t_bhpt beta(t') dt'
```

Phase rotation:

```python
phi0 = 1.6339824069  # radians, principal value
phi0_unwrapped = 1.6339824069
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(Ehat, Jhat) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- BHPT `calibrated` flag: `False`
- common NR support: `[-4999.9377442127, 97.7622557873019]`
- common-support coverage of requested NR window: `0.999549028450423`
- optimizer downsampled `mathcalE`: `6.00105327586805e-05`
- full-window `mathcalE`: `6.03677661081137e-05`
- pre-transition local `mathcalE`: `4.07053119542355e-05`
- transition local `mathcalE`: `0.000102969987276492`
- post-transition local `mathcalE`: `0.000290659075164843`
- alpha endpoint values `(pre_start, switch, final)`: `(0.80865511874, 0.80865511874, 2.16086379815786)`
- beta endpoint values `(pre_start, switch_left, switch_right, final)`: `(0.804379815007932, 0.80796434842, 0.84590841417, 0.84590841417)`
- source samples in smooth transition: `388`
- time and phase shifts optimized: yes (`t0_nr`, `phi0`)

The requested `mathcalE ~ 1e-4` target is reached.

Compared with the accepted 12-parameter smooth PN-loss opt fit
(`gwAgentic_NRBHP/scaling_PN_opt.md`, full-window `mathcalE = 1.19776e-4`),
this model uses 2 fewer parameters and roughly halves the error. The same
caveat applies: the 2PN flux expressions are used as an IMR coordinate
through merger-ringdown, where they should be interpreted as a physically
motivated proxy rather than a controlled PN approximation.
