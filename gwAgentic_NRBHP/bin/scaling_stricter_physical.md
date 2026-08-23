# Stricter physical q=5 (2,2) piecewise-linear scaling fit

This rerun fits the two-segment stricter ansatz directly in merger-centered
physical time variables. It does not reuse the previous stricter
`s_min`/`s_max` coefficients.
The optimizer ranks generic physical starts and then applies a Nelder-Mead
polish to the best starts.

The BHPT waveform is the raw surrogate output with `calibrated=False`; the
NR reference is `NRHybSur3dq8`.

```python
q = 5
nu = q / (1 + q)**2
nu = 0.138888888888889

t_bhpt_merger = 1.10958353616297e-07
t_nr_merger = -0.837744212698453
theta = nu * (t_bhpt - t_bhpt_merger)
Theta_nr = nu * (t_nr - t_nr_merger)
```

The optimized source split and mapped NR cutoff are:

```python
theta_cut = -26.9836124660536
t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # -194.282009644627
Theta_cut_nr = -21.7044312527108
t_cut_nr = t_nr_merger + Theta_cut_nr / nu  # -157.109649232216
```

For `theta <= theta_cut`:

```python
dtheta = theta - theta_cut
alpha_left(theta) = 0.809491123690498 + (2.38606720292391e-06) * dtheta
beta_left(theta) = 0.805342357756683 + (1.6279461154321e-06) * dtheta
dt_bhpt = t_bhpt - t_bhpt_cut
tau_left(t_bhpt) = t_cut_nr + 0.805342357756683 * dt_bhpt + 0.5 * (1.6279461154321e-06) * nu * dt_bhpt**2
```

For `theta > theta_cut`:

```python
dtheta = theta - theta_cut
alpha_right(theta) = 0.838861089010922 + (-0.00359701009371564) * dtheta
beta_right(theta) = 0.794716587155693 + (0.00156739106877002) * dtheta
dt_bhpt = t_bhpt - t_bhpt_cut
tau_right(t_bhpt) = t_cut_nr + 0.794716587155693 * dt_bhpt + 0.5 * (0.00156739106877002) * nu * dt_bhpt**2
```

Phase rotation:

```python
phi0 = 1.41699350768004  # radians
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- common NR support: `[-4999.9377442127, 98.8622557873005]`
- optimizer downsampled `mathcalE`: `0.000149354545638502`
- full-window `mathcalE`: `0.000149405634356619`
- `t <= t_cut_nr` local `mathcalE`: `2.84463818837085e-05`
- `t > t_cut_nr` local `mathcalE`: `0.000900380188218523`
- alpha endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.807496597525667, 0.809491123690498, 0.838861089010922, 0.684448435000053)`
- beta endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.803981549009056, 0.805342357756683, 0.794716587155693, 0.86200164015695)`

No extrapolated BHPT samples are used. The evaluated source samples are the
raw BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison
support, and the NR grid is then restricted to that common support.
