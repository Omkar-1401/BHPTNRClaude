# gw_remnant energy-driven model, GLOBAL joint fit (gwr_energy_global)

Same model as `gw_remnant_energy` form `mult`; the master polynomial
coefficients are optimised directly against all training waveforms at once
(Powell) instead of per-q fit -> independent regression.

## Model

```python
E(t) = gw_remnant Eoft            # radiated energy, units of M, E(t_start)=0
alpha(t, nu) = alpha_PP(nu) * (1 + alpha_E(nu) * E(t))
beta (t, nu) = beta_PP (nu) * (1 + beta_E (nu) * E(t))
tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta dt'
```

Degree 3 in nu; alpha_PP, beta_PP PP-anchored to 1 at nu=0; the
couplings unanchored (E_rad -> 0 carries the PP limit).  **14 free
coefficients total.**  phi0 analytic, t0_nr a per-q 1-D nuisance.

Training: 12 mass ratios sampled even in nu over [3, 8] --
[3.0, 3.26, 3.51, 3.9, 4.2, 4.6, 5.0, 5.44, 5.95, 6.59, 7.23, 8.0].  No q<3 NR data is used; q=2 enters only
through the nonlinear-excursion penalty (lam = 0).

## Why global

(beta_PP, P = beta_PP*beta_E) is a strongly degenerate pair: at q=2 correcting
either alone makes the mismatch WORSE (2.4e-2 -> 3.3e-2 / 9.4e-2) while
correcting both together gives 2.5e-3.  Independent regression of the two
cannot stay on that valley; a joint fit lands on it by construction.

## Results

| q | global | per-q -> regress |
|---:|---:|---:|
| 3 | 9.6973e-04 | 9.8160e-04 |
| 4 | 9.3356e-04 | 9.3486e-04 |
| 5 | 7.0539e-04 | 7.0037e-04 |
| 6 | 5.4613e-04 | 5.4434e-04 |
| 7 | 4.3121e-04 | 4.2076e-04 |
| 8 | 2.5577e-04 | 2.6189e-04 |
| **in-range median** | **6.2576e-04** | 6.2236e-04 |
| **in-range max** | **9.6973e-04** | 9.8160e-04 |

| q < 3 | global | per-q -> regress | BHPT input |
|---:|---:|---:|:---|
| 2.75 | 9.8731e-04 | 1.0062e-03 | in domain |
| 2.5 | 1.3958e-03 | 1.3153e-03 | in domain |
| 2.25 | 4.8641e-03 | 4.1433e-03 | **extrapolated** |
| 2 | 2.7615e-02 | 2.4270e-02 | **extrapolated** |

`BHPTNRSur1dq1e4` is declared valid only for q >= 2.5, so q=2.5 is the
honest low-q gate; 2.25 and 2.0 additionally extrapolate the surrogate itself.

## Coefficients

| parameter | anchor | c0 | c1 | c2 | c3 |
|:---|:---|---:|---:|---:|---:|
| alpha_PP | 1 | 1 | -1.09875825 | 1.008834148 | -14.08074462 |
| alpha_E | free | -42.05798514 | 613.8690999 | -3757.019062 | 8094.614811 |
| beta_PP | 1 | 1 | -1.296113674 | 0.8989295617 | -11.98426959 |
| beta_E | free | 17.2644016 | -435.6052424 | 3129.947984 | -6760.409711 |

Command: `python fit_scaling_gwr_energy_global.py --degree 3 --lam 0 --ntrain 12`
