# Smooth optimized PN-loss q=5 (2,2) scaling fit

This replaces the discontinuous constant-left PN-loss opt fit with a smooth
constant-to-linear transition. Before the transition, `alpha` and `beta` are
constant. Across the transition, a cubic Hermite interpolation matches the
constant branch value and the post-transition branch value and slope along
`p_loss`. After the transition, the fit is linear in the PN radiated-energy
and radiated-angular-momentum coordinates.

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

PN-loss coordinates:

```python
omega_gw = abs(d unwrap(arg(h_BHPT_22)) / dt_bhpt)
x_pn = (omega_gw / 2)**(2/3)
F_E = PN_exprs.dEdt(nu, x_pn)
F_J = PN_exprs.dJdt(1.0, x_pn, nu)
DeltaE_PN(t) = integral_from_merger_to_t F_E dt
DeltaJ_PN(t) = integral_from_merger_to_t F_J dt
Ehat = DeltaE_PN / abs(DeltaE_PN(t_ref))
Jhat = DeltaJ_PN / abs(DeltaJ_PN(t_ref))
p_loss = 0.5 * (Ehat + Jhat)
```

Optimized PN-loss transition:

```python
p_start = -0.958016464129905
p_width = 0.0571434141038535
p_end = -0.900873050026052
Ehat_start = -0.969761626758554
Jhat_start = -0.946271301501256
Ehat_end = -0.926441194339401
Jhat_end = -0.875304905712703
t_bhpt_start = -83.9313340832618
t_bhpt_end = -66.6733348896892
t_start_nr = -67.4567468745995
t_end_nr = -53.3976077583536
DeltaE_PN(t_ref) = -0.0216330274770254
DeltaJ_PN(t_ref) = -0.0443145705120832
```

For `p_loss <= p_start`, the pre-transition branch is constant:

```python
alpha_left = 0.809882562802371
beta_left = 0.804462279293606
```

For `p_loss >= p_end`:

```python
dE = Ehat - Ehat_end
dJ = Jhat - Jhat_end
alpha_right = 0.80104333470093 + (0.253649409866864) * dE + (-0.379302453018587) * dJ
beta_right = 0.826043284021798 + (0.00176893980239059) * dE + (0.00168122994106532) * dJ
```

For `p_start < p_loss < p_end`, use cubic Hermite interpolation in
`z = (p_loss - p_start) / p_width` between the constant values and the
right-branch edge values. The left endpoint slope is zero, and the right
endpoint slope is the derivative of the right branch along the PN-loss path.

The time map is the numerical integral of `beta` anchored at `p_start`:

```python
tau(t_bhpt) = t_start_nr + integral_from_t_bhpt_start_to_t_bhpt beta(t') dt'
```

Phase rotation:

```python
phi0 = 1.31389242237568  # radians, principal value
phi0_unwrapped = 1.31389242237568
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(Ehat, Jhat) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- common NR support: `[-4999.9377442127, 99.6622557873015]`
- common-support coverage of requested NR window: `0.999921570165291`
- optimizer downsampled `mathcalE`: `0.000119760021460885`
- full-window `mathcalE`: `0.000119776117482135`
- pre-transition local `mathcalE`: `0.000101608219852769`
- transition local `mathcalE`: `0.000173661295356629`
- post-transition local `mathcalE`: `0.0003250473369137`
- alpha endpoint values `(pre_start, start, end, final)`: `(0.809882562802371, 0.809882562802371, 0.80104333470093, 1.74688025680414)`
- beta endpoint values `(pre_start, start, end, final)`: `(0.804462279293606, 0.804462279293606, 0.826043284021798, 0.882070248639787)`
- source samples in smooth transition: `86`

The fit does not reach `mathcalE ~ 1e-4`, but remains below `1e-3`.
The main caveat remains that the 2PN flux expressions are used as an IMR
coordinate through merger-ringdown, where they should be interpreted as a
physically motivated proxy rather than a controlled PN approximation.
