# Stricter physical q=8 (2,2) piecewise-linear scaling fit

This q=8 run uses the same two-segment physical ansatz as the q=5 stricter
physical run, but it does not use the q=5 generic start grid as the final
fitting procedure. The q=5 start grid is not generally applicable: at q=8 the
best q=5-style starting point had `mathcalE ~= 0.61` before polishing, so the
optimizer starts in the wrong basin and can spend a long time improving a bad
alignment.

For this run I first used a q=8-specific single-segment physical alignment,
then seeded the two-segment fit from that q=8 alignment and polished the best
two-segment starts with Nelder-Mead.

The BHPT waveform is the raw surrogate output with `calibrated=False`; the NR
reference is `NRHybSur3dq8`.

```python
q = 8
nu = q / (1 + q)**2
nu = 0.0987654320987654

t_bhpt_merger = 1.10958353616297e-07
t_nr_merger = -1.17751139808388
theta = nu * (t_bhpt - t_bhpt_merger)
Theta_nr = nu * (t_nr - t_nr_merger)
```

The optimized source split and mapped NR cutoff are:

```python
theta_cut = -15.5653865185185
t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # -157.599538389042
Theta_cut_nr = -13.4294443652282
t_cut_nr = t_nr_merger + Theta_cut_nr / nu  # -137.15063559602
```

For `theta <= theta_cut`:

```python
dtheta = theta - theta_cut
alpha_left(theta) = 0.869068069754992 + (-4.65638019271435e-06) * dtheta
beta_left(theta) = 0.867154334310366 + (-1.57959824396477e-06) * dtheta
dt_bhpt = t_bhpt - t_bhpt_cut
tau_left(t_bhpt) = t_cut_nr + 0.867154334310366 * dt_bhpt + 0.5 * (-1.57959824396477e-06) * nu * dt_bhpt**2
```

For `theta > theta_cut`:

```python
dtheta = theta - theta_cut
alpha_right(theta) = 0.90645270840308 + (-0.00649672047348657) * dtheta
beta_right(theta) = 0.876532587268604 + (-2.12458285604868e-06) * dtheta
dt_bhpt = t_bhpt - t_bhpt_cut
tau_right(t_bhpt) = t_cut_nr + 0.876532587268604 * dt_bhpt + 0.5 * (-2.12458285604868e-06) * nu * dt_bhpt**2
```

Phase rotation:

```python
phi0 = -1.04023499730181  # radians, principal value
phi0_unwrapped = -7.3234203044814
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Diagnostics:

- q: `8`
- mode: `(2, 2)`
- common NR support: `[-4999.97751139808, 99.8224886019161]`
- common-support coverage of requested NR window: `0.999960785082645`
- optimizer downsampled `mathcalE`: `0.000147830478355841`
- full-window `mathcalE`: `0.000147807818538109`
- `t <= t_cut_nr` local `mathcalE`: `4.56287177678566e-05`
- `t > t_cut_nr` local `mathcalE`: `0.000911564230012056`
- alpha endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.871645750064047, 0.869068069754992, 0.90645270840308, 0.732950464670049)`
- beta endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.868028768768286, 0.867154334310366, 0.876532587268604, 0.876475847888478)`

No extrapolated BHPT samples are used. The evaluated source samples are the raw
BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison support,
and the NR grid is then restricted to that common support.
