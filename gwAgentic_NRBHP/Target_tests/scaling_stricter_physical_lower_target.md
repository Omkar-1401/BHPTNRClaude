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
theta_cut = -32
t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # -230.399999889042
Theta_cut_nr = -26.2725355260141
t_cut_nr = t_nr_merger + Theta_cut_nr / nu  # -190
```

For `theta <= theta_cut`:

```python
dtheta = theta - theta_cut
alpha_left(theta) = 0.81 + (0) * dtheta
beta_left(theta) = 0.805 + (0) * dtheta
dt_bhpt = t_bhpt - t_bhpt_cut
tau_left(t_bhpt) = t_cut_nr + 0.805 * dt_bhpt + 0.5 * (0) * nu * dt_bhpt**2
```

For `theta > theta_cut`:

```python
dtheta = theta - theta_cut
alpha_right(theta) = 0.84 + (-0.0025) * dtheta
beta_right(theta) = 0.81 + (0.00105) * dtheta
dt_bhpt = t_bhpt - t_bhpt_cut
tau_right(t_bhpt) = t_cut_nr + 0.81 * dt_bhpt + 0.5 * (0.00105) * nu * dt_bhpt**2
```

Phase rotation:

```python
phi0 = 1.64551686110548  # radians
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- common NR support: `[-4999.9377442127, 98.2622557873019]`
- optimizer downsampled `mathcalE`: `0.00182222510592318`
- full-window `mathcalE`: `0.00182218952978797`
- `t <= t_cut_nr` local `mathcalE`: `0.00133151362538459`
- `t > t_cut_nr` local `mathcalE`: `0.00452030774890166`
- alpha endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.81, 0.81, 0.84, 0.720138888888744)`
- beta endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.805, 0.805, 0.81, 0.860341666666728)`

No extrapolated BHPT samples are used. The evaluated source samples are the
raw BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison
support, and the NR grid is then restricted to that common support.
