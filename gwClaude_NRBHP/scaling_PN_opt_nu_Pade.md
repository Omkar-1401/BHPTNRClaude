# PN-loss q-dependent fit with PP-limit anchoring (nu / Pade bake-off)

Per-q model: the unchanged 10-parameter creative logistic-switch model
(reuses the 40-point per-q cache from `PN_opt_creative_q_dep_results/`).
Only the q-regression layer changes.  All physical parameters are anchored
to the test-mass (PP) limit at `1/q -> 0` (`nu -> 0`):

```
alpha_i, beta_i, beta_r -> 1     alpha_E, alpha_J, beta_L -> 0
```

`p0`, `w` carry no PP value (fit unanchored); `t0_nr`, `phi0` are alignment
nuisances, fit unanchored and re-optimised at evaluation.

## Model equations

```python
S = 1 / (1 + exp(-(p_loss - p0) / w))
dE = Ehat - Ehat(p0);  dJ = Jhat - Jhat(p0);  dp = p_loss - p0
alpha = alpha_i + S * (alpha_E * dE + alpha_J * dJ)
beta  = (1 - S) * (beta_i + beta_L * dp) + S * beta_r
tau(t) = t0_nr + integral_{t_anchor}^{t} beta(t') dt'
h_model(tau(t)) = alpha * exp(i*phi0) * h_BHPT(t)
```

Calibration grid: 40 q in [3, 8].  nu = q/(1+q)^2.
Fit settings: source stride `3`, NR stride `5`.

## Bake-off (all PP-anchored)

| form | config | in-range median | in-range max | q=2 extrap | poles in nu<=0.25 |
|:---|:---|---:|---:|---:|:---:|
| pade_nu | [1/1] in nu | 0.001372 | 0.01462 | 0.3691 | False |
| poly_nu | deg 4 in nu | 7.05e-05 | 0.0001676 | 0.06459 | False |
| cheb_1q | deg 4 in 1/q | 7.421e-05 | 0.0001722 | 50 | False |

**Selected form: `poly_nu`**  (config: deg 4 in nu)

## Selected fit: per-parameter representation

| parameter | PP anchor | representation |
|:---|:---:|:---|
| p0 | — | [-5.65155, 126.958, -1273.08, 5609.05, -9112.09] (powers of nu) |
| w | — | [4.8194, -125.239, 1206.79, -5110.86, 8085.06] (powers of nu) |
| alpha_i | 1 | 1 + nu*[-1.29495, -0.138821, 4.01136, -51.9588] |
| alpha_E | 0 | 0 + nu*[-2.42245, 86.997, -553.807, 1165.93] |
| alpha_J | 0 | 0 + nu*[3.39996, -123.393, 802.971, -1761.32] |
| beta_i | 1 | 1 + nu*[-0.873396, -9.29171, 63.2971, -165.211] |
| beta_r | 1 | 1 + nu*[-0.704084, -7.79317, 58.4364, -170.603] |
| t0_nr | — | [111.752, -6132.87, 67830.8, -322471, 571954] (powers of nu) |
| phi0 | — | [717.121, -25805.2, 337550, -1.86925e+06, 3.70339e+06] (powers of nu) |
| beta_L | 0 | 0 + nu*[0.207083, -4.71163, 33.4732, -68.1849] |

## Error summary (selected form)

| model | min | median | mean | max | q at max |
|:---|---:|---:|---:|---:|---:|
| selected master (q in [3,8]) | 4.94181e-05 | 7.05036e-05 | 8.03542e-05 | 0.000167605 | 3 |

## Extrapolation

| q | mathcalE | coverage |
|---:|---:|---:|
| 2 | 0.0645901 | 0.9958 |
| 2.5 | 0.00038702 | 0.9975 |

## Design notes

**Why nu and not 1/q.** At equal polynomial degree the two variables fit
the training range almost identically, but 1/q extrapolates catastrophically:
a degree-4 Chebyshev-in-1/q matches this model in-range (median ~7.4e-5) yet
produces an *invalid* model at q=2, because the unanchored p0, w blow up in the
1/q monomial/Chebyshev basis (the original PN_opt_creative_q_dep failure mode).
nu is bounded on [0, 0.25], saturates toward equal mass, and places q=2 a much
shorter extrapolation beyond the training edge, so the same degree that sharpens
the in-range fit does NOT degrade the q=2 extrapolation.

**Why both alpha_E and alpha_J are kept (not collapsed to one term).** The two
per-q amplitude coefficients are strongly anti-correlated across q (corr = -0.998),
and dE, dJ are ~0.999 collinear deep post-switch — which suggests a single term
might suffice.  A direct test (re-optimising a collapsed 9-parameter model with
alpha_E = alpha_J, i.e. a single slope alpha_p * dp) shows it does not: the error
grows from 1.99e-4 to 7.8e-4 at q=3 and from 5.9e-5 to 1.3e-4 at q=8.  Through the
switch transition dE and dJ are only ~0.76 collinear, so both carry real amplitude
information at the 1e-4 level.  The -0.998 correlation is a smooth linear locus
(each predicts the other), not redundancy; the true difficulty is *extrapolating
the split*, which the PP anchor (both -> 0 as nu -> 0) plus the bounded nu variable
control.

**Unanchored parameters.** p0, w, t0_nr, phi0 have no PP value and are fit as free
monomials in nu.  Their monomial coefficients are large and alternating (the usual
ill-conditioning of a monomial basis), but the fitted curves are smooth and the
physical-domain values are well-behaved; t0_nr and phi0 are re-optimised at
evaluation regardless.

## Per-q results (selected form, q in [3,8])

| q | nu | master mathcalE | coverage | p0 | w | alpha_i | beta_i | beta_r | beta_L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.1875 | 0.000167605 | 0.9982 | -0.8922 | 0.06652 | 0.7145 | 0.7226 | 0.7683 | 0.009559 |
| 3.12821 | 0.1836 | 0.000144637 | 0.9983 | -0.8962 | 0.06132 | 0.7234 | 0.7305 | 0.7759 | 0.008875 |
| 3.25641 | 0.1797 | 0.000129456 | 0.9984 | -0.9008 | 0.05708 | 0.7318 | 0.7379 | 0.7829 | 0.008211 |
| 3.38462 | 0.1761 | 0.000119865 | 0.9985 | -0.9057 | 0.0535 | 0.7397 | 0.7449 | 0.7895 | 0.007572 |
| 3.51282 | 0.1725 | 0.000112888 | 0.9986 | -0.9106 | 0.05032 | 0.7471 | 0.7515 | 0.7956 | 0.006962 |
| 3.64103 | 0.1690 | 0.000106803 | 0.9987 | -0.9153 | 0.04738 | 0.7541 | 0.7577 | 0.8013 | 0.006383 |
| 3.76923 | 0.1657 | 0.000100517 | 0.9988 | -0.9196 | 0.04457 | 0.7607 | 0.7636 | 0.8066 | 0.005837 |
| 3.89744 | 0.1625 | 9.38689e-05 | 0.9989 | -0.9235 | 0.04182 | 0.7669 | 0.7691 | 0.8116 | 0.005323 |
| 4.02564 | 0.1594 | 8.7245e-05 | 0.9990 | -0.9268 | 0.03909 | 0.7728 | 0.7744 | 0.8163 | 0.004842 |
| 4.15385 | 0.1564 | 8.11849e-05 | 0.9991 | -0.9297 | 0.03637 | 0.7784 | 0.7794 | 0.8208 | 0.004395 |
| 4.28205 | 0.1535 | 7.62808e-05 | 0.9992 | -0.9322 | 0.03366 | 0.7837 | 0.7842 | 0.825 | 0.003979 |
| 4.41026 | 0.1507 | 7.26973e-05 | 0.9992 | -0.9342 | 0.03098 | 0.7887 | 0.7888 | 0.829 | 0.003593 |
| 4.53846 | 0.1480 | 7.05342e-05 | 0.9993 | -0.9358 | 0.02836 | 0.7935 | 0.7932 | 0.8327 | 0.003238 |
| 4.66667 | 0.1453 | 6.93938e-05 | 0.9994 | -0.9372 | 0.02582 | 0.798 | 0.7974 | 0.8363 | 0.002911 |
| 4.79487 | 0.1428 | 6.90528e-05 | 0.9995 | -0.9382 | 0.02341 | 0.8023 | 0.8014 | 0.8398 | 0.00261 |
| 4.92308 | 0.1403 | 6.91137e-05 | 0.9995 | -0.939 | 0.02116 | 0.8065 | 0.8053 | 0.8431 | 0.002335 |
| 5.05128 | 0.1379 | 6.91861e-05 | 0.9996 | -0.9397 | 0.0191 | 0.8104 | 0.809 | 0.8462 | 0.002084 |
| 5.17949 | 0.1356 | 6.9009e-05 | 0.9996 | -0.9402 | 0.01726 | 0.8142 | 0.8126 | 0.8492 | 0.001856 |
| 5.30769 | 0.1334 | 6.81438e-05 | 0.9997 | -0.9407 | 0.01568 | 0.8178 | 0.8161 | 0.8521 | 0.001649 |
| 5.4359 | 0.1312 | 6.62768e-05 | 0.9997 | -0.9412 | 0.015 | 0.8213 | 0.8194 | 0.8549 | 0.001461 |
| 5.5641 | 0.1291 | 6.34432e-05 | 0.9998 | -0.9417 | 0.015 | 0.8247 | 0.8226 | 0.8575 | 0.001292 |
| 5.69231 | 0.1271 | 5.97693e-05 | 0.9998 | -0.9424 | 0.015 | 0.8279 | 0.8257 | 0.8601 | 0.00114 |
| 5.82051 | 0.1251 | 5.59515e-05 | 0.9999 | -0.9431 | 0.015 | 0.8309 | 0.8288 | 0.8626 | 0.001005 |
| 5.94872 | 0.1232 | 5.2973e-05 | 0.9999 | -0.944 | 0.015 | 0.8339 | 0.8317 | 0.8649 | 0.0008834 |
| 6.07692 | 0.1213 | 5.22816e-05 | 0.9999 | -0.9451 | 0.015 | 0.8367 | 0.8345 | 0.8672 | 0.0007761 |
| 6.20513 | 0.1195 | 5.51195e-05 | 0.9999 | -0.9464 | 0.015 | 0.8395 | 0.8372 | 0.8695 | 0.0006814 |
| 6.33333 | 0.1178 | 6.17021e-05 | 0.9999 | -0.9479 | 0.015 | 0.8421 | 0.8399 | 0.8716 | 0.0005985 |
| 6.46154 | 0.1161 | 7.0473e-05 | 1.0000 | -0.9497 | 0.01663 | 0.8447 | 0.8425 | 0.8737 | 0.0005265 |
| 6.58974 | 0.1144 | 7.89719e-05 | 0.9999 | -0.9517 | 0.01867 | 0.8472 | 0.845 | 0.8757 | 0.0004645 |
| 6.71795 | 0.1128 | 8.43156e-05 | 0.9999 | -0.954 | 0.02112 | 0.8495 | 0.8474 | 0.8777 | 0.0004117 |
| 6.84615 | 0.1112 | 8.42484e-05 | 0.9999 | -0.9566 | 0.02397 | 0.8518 | 0.8497 | 0.8796 | 0.0003675 |
| 6.97436 | 0.1097 | 7.78504e-05 | 0.9999 | -0.9596 | 0.02723 | 0.8541 | 0.852 | 0.8814 | 0.000331 |
| 7.10256 | 0.1082 | 6.6608e-05 | 0.9999 | -0.9628 | 0.0309 | 0.8562 | 0.8543 | 0.8832 | 0.0003017 |
| 7.23077 | 0.1067 | 5.54657e-05 | 0.9999 | -0.9663 | 0.03496 | 0.8583 | 0.8564 | 0.885 | 0.0002791 |
| 7.35897 | 0.1053 | 4.94181e-05 | 1.0000 | -0.9702 | 0.03942 | 0.8604 | 0.8586 | 0.8867 | 0.0002624 |
| 7.48718 | 0.1039 | 5.1335e-05 | 0.9999 | -0.9743 | 0.04426 | 0.8623 | 0.8606 | 0.8883 | 0.0002513 |
| 7.61538 | 0.1026 | 6.14509e-05 | 0.9999 | -0.9788 | 0.04949 | 0.8643 | 0.8626 | 0.8899 | 0.0002452 |
| 7.74359 | 0.1013 | 7.78693e-05 | 0.9999 | -0.9836 | 0.05508 | 0.8661 | 0.8646 | 0.8915 | 0.0002438 |
| 7.87179 | 0.1000 | 9.69427e-05 | 0.9999 | -0.9887 | 0.06104 | 0.8679 | 0.8665 | 0.893 | 0.0002466 |
| 8 | 0.0988 | 0.00011422 | 0.9999 | -0.9941 | 0.06734 | 0.8697 | 0.8684 | 0.8945 | 0.0002532 |
