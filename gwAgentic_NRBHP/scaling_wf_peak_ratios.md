# Waveform peak-ratio q-dependent scaling

This run follows the waveform-peak ratio construction suggested by
Section II A 3 of arXiv:2307.03155. The calibration uses the raw
BHPT `(2, 2)` mode and the nonspinning `NRHybSur3dq8` `(2, 2)` mode.
For each q, both waveforms are rotated so the merger-amplitude peak
has zero phase, and positive real-part peaks are matched by their
index counted backward from merger.

For every matched peak:

```python
nu = q / (1 + q)**2
s_bhpt = t_bhpt_peak - t_bhpt_merger
s_nr = t_nr_peak - t_nr_merger
theta = nu * s_bhpt
alpha_peak = h_nr_peak / h_bhpt_peak
beta_peak = s_nr / s_bhpt
```

The final model is indexed by peak number rather than by a global
polynomial in theta. For each peak index `k`, `alpha_peak(k, q)` and
`beta_peak(k, q)` are represented as follows:

```python
Q(q) = mass_norm
inside 3 <= q <= 8: PCHIP interpolation in Q(q)
outside the fitted interval: local edge polynomial extrapolation in Q(q)
edge_n = 12
edge_degree = 2
max_k = 38
```

For a target q, the BHPT peak locations determine the target theta
nodes. The predicted peak-index ratios are then interpolated in theta
with PCHIP to form continuous arrays:

```python
alpha(q, theta) = PCHIP_theta({theta_k(q), alpha_peak(k, q)})
beta(q, theta) = PCHIP_theta({theta_k(q), beta_peak(k, q)})
tau(t_bhpt) = t_nr_merger + beta(q, theta) * (t_bhpt - t_bhpt_merger)
h_model(tau) = alpha(q, theta) * exp(1j * phi0) * h_bhpt(t_bhpt)
```

The constant phase `phi0` is optimized analytically for each q
evaluation and is not part of the fitted q-dependent model.

## Fit Setup

- Calibration grid: `64` uniformly spaced q values from `3` to `8`.
- Total accepted peak-ratio samples: `2720`.
- Peak search window: `-5000 <= t - t_merger <= -20`.
- Peak prominence fraction: `0.01`.
- q coordinate for interpolation/extrapolation: `mass_norm`.
- Edge extrapolator: degree `2` polynomial using `12` nearest q nodes.
- Maximum retained peak index: `38`.
- Minimum target model peaks required: `6`.

This is the best low-cost checkpoint from the current run, not a fully
accepted q-dependent calibration. The quadratic edge extrapolator keeps
q=2 below the requested 1% level, but it is not uniformly reliable over
2 < q < 3. A separate lower-edge scan showed that the more conservative
linear edge rule behaves better at q=2.5 but leaves q=2 slightly above
1%.

## Error Summary

| range | min mathcalE | median mathcalE | mean mathcalE | max mathcalE |
|:---|---:|---:|---:|---:|
| 3 <= q <= 8 | 0.00027368896 | 0.0003578352 | 0.00085427999 | 0.032426115 |

Maximum in-range mathcalE occurs at q `3.1587302`.
Nearest grid point to q=3 has mathcalE `0.00045332244`.
Nearest grid point to q=8 has mathcalE `0.00041140009`.

## Extrapolation Diagnostics

These q values were not used in the 64-point q calibration grid.

| q | mathcalE | coverage | phi0 | model peaks | common support | status |
|---:|---:|---:|---:|---:|:---|:---|
| 2.5 | 0.24051092 | 0.99847062 | 2.0031523 | 38 | [-4999.8927, 92.307298] | ok |
| 2 | 0.0015452622 | 0.99805886 | -1.217621 | 38 | [-4999.9745, 90.125549] | ok |
| 1.5 | 50 | nan | nan | 0 | [nan, nan] | nonmonotonic_tau |
| 10 | 0.010675595 | 0.99996079 | -2.9305656 | 38 | [-4999.9019, 99.898088] | ok |

## Per-q Calibration Results

| q | mathcalE | coverage | phi0 | accepted peak samples | model peaks | common support |
|---:|---:|---:|---:|---:|---:|:---|
| 3 | 0.00045332244 | 0.99890198 | 1.713143 | 39 | 38 | [-4999.9613, 94.438698] |
| 3.0793651 | 0.00044777008 | 0.9989608 | -1.9427083 | 39 | 38 | [-4999.9007, 94.799327] |
| 3.1587302 | 0.032426115 | 0.99901963 | 0.56175856 | 38 | 38 | [-4999.9082, 95.091827] |
| 3.2380952 | 0.00042576958 | 0.99907845 | 2.9250252 | 40 | 38 | [-4999.9917, 95.308264] |
| 3.3174603 | 0.00043328995 | 0.99917649 | -1.0585892 | 40 | 38 | [-4999.9917, 95.808264] |
| 3.3968254 | 0.00041510718 | 0.99919609 | 1.0809125 | 40 | 38 | [-4999.9398, 95.960248] |
| 3.4761905 | 0.00040952945 | 0.99925492 | 3.1414891 | 40 | 38 | [-4999.955, 96.245012] |
| 3.5555556 | 0.00040039647 | 0.99929413 | -1.1894217 | 40 | 38 | [-4999.9471, 96.452926] |
| 3.6349206 | 0.00041131555 | 0.99935295 | 0.70567604 | 40 | 38 | [-4999.8471, 96.852926] |
| 3.7142857 | 0.0004077207 | 0.99941178 | 2.4630788 | 40 | 38 | [-4999.998, 97.001965] |
| 3.7936508 | 0.00039713098 | 0.99943138 | -2.1334424 | 40 | 38 | [-4999.8153, 97.284705] |
| 3.8730159 | 0.00040022469 | 0.99950981 | -0.51129176 | 40 | 38 | [-4999.8851, 97.614898] |
| 3.952381 | 0.0003928903 | 0.99956864 | 1.0103133 | 41 | 38 | [-4999.9851, 97.814898] |
| 4.031746 | 0.00039402985 | 0.99960785 | 2.4616576 | 41 | 38 | [-4999.9219, 98.078106] |
| 4.1111111 | 0.00038810416 | 0.99964707 | -2.4569249 | 41 | 38 | [-4999.9192, 98.28078] |
| 4.1904762 | 0.00037245127 | 0.99966667 | -1.1836611 | 41 | 38 | [-4999.964, 98.335953] |
| 4.2698413 | 0.00038762971 | 0.9997451 | 0.07276225 | 41 | 38 | [-4999.964, 98.735953] |
| 4.3492063 | 0.00038439233 | 0.99976471 | 1.2308423 | 41 | 38 | [-4999.9724, 98.827603] |
| 4.4285714 | 0.00037066084 | 0.99978432 | 2.3035589 | 41 | 38 | [-4999.9272, 98.972844] |
| 4.5079365 | 0.00036767477 | 0.99982353 | -2.9540983 | 41 | 38 | [-4999.9331, 99.16687] |
| 4.5873016 | 0.00036375605 | 0.99984314 | -1.987241 | 41 | 38 | [-4999.9331, 99.26687] |
| 4.6666667 | 0.0003700146 | 0.99988236 | -1.0661744 | 42 | 38 | [-4999.8886, 99.511362] |
| 4.7460317 | 0.00036543866 | 0.99994118 | -0.21992154 | 42 | 38 | [-4999.992, 99.708008] |
| 4.8253968 | 0.00037149262 | 0.99994118 | 0.58733312 | 42 | 38 | [-4999.8415, 99.858489] |
| 4.9047619 | 0.00036716784 | 0.99996079 | 1.3267234 | 42 | 38 | [-4999.9415, 99.858489] |
| 4.984127 | 0.00036176949 | 0.99994118 | 2.0063959 | 42 | 38 | [-4999.9377, 99.762256] |
| 5.0634921 | 0.00036448701 | 0.99994118 | 2.6479095 | 42 | 38 | [-4999.8685, 99.831511] |
| 5.1428571 | 0.0003565944 | 0.99994118 | -3.0577797 | 42 | 38 | [-4999.9489, 99.751095] |
| 5.2222222 | 0.00035206012 | 0.99996079 | -2.5235078 | 42 | 38 | [-4999.9489, 99.851095] |
| 5.3015873 | 0.00035073429 | 0.99996079 | -2.0334741 | 42 | 38 | [-4999.9609, 99.839145] |
| 5.3809524 | 0.00033725872 | 0.99994118 | -1.6082828 | 43 | 38 | [-4999.8191, 99.880898] |
| 5.4603175 | 0.00035311557 | 0.99992157 | -1.1856898 | 43 | 38 | [-4999.8081, 99.791949] |
| 5.5396825 | 0.00034792378 | 0.99994118 | -0.83270263 | 43 | 38 | [-4999.8081, 99.891949] |
| 5.6190476 | 0.00034910721 | 0.99994118 | -0.51975654 | 43 | 38 | [-4999.8321, 99.867856] |
| 5.6984127 | 0.00033491288 | 0.99992157 | -0.26498869 | 43 | 38 | [-4999.8899, 99.710106] |
| 5.7777778 | 0.00032765835 | 0.99994118 | -0.042520877 | 43 | 38 | [-4999.8819, 99.818068] |
| 5.8571429 | 0.00032257295 | 0.99994118 | 0.14428715 | 43 | 38 | [-4999.9819, 99.718068] |
| 5.9365079 | 0.0003207131 | 0.99992157 | 0.29184981 | 43 | 38 | [-4999.8967, 99.703288] |
| 6.015873 | 0.00032532508 | 0.99996079 | 0.41603504 | 43 | 38 | [-4999.9467, 99.853339] |
| 6.0952381 | 0.00031361097 | 0.99994118 | 0.47899074 | 43 | 38 | [-4999.9264, 99.773568] |
| 6.1746032 | 0.00030705557 | 0.99994118 | 0.51328885 | 43 | 38 | [-4999.8264, 99.873568] |
| 6.2539683 | 0.00029979073 | 0.99992157 | 0.51447401 | 44 | 38 | [-4999.8269, 99.773099] |
| 6.3333333 | 0.00029763684 | 0.99996079 | 0.48673093 | 44 | 38 | [-4999.9505, 99.849483] |
| 6.4126984 | 0.00029116881 | 0.99994118 | 0.41808494 | 44 | 38 | [-4999.9505, 99.749483] |
| 6.4920635 | 0.00029754202 | 0.99992157 | 0.33746517 | 44 | 38 | [-4999.8053, 99.794714] |
| 6.5714286 | 0.00028530744 | 0.99996079 | 0.19818351 | 44 | 38 | [-4999.9768, 99.823173] |
| 6.6507937 | 0.00027790188 | 0.99992157 | 0.036136888 | 44 | 38 | [-4999.8693, 99.730685] |
| 6.7301587 | 0.00029695096 | 0.99992157 | -0.12393806 | 44 | 38 | [-4999.8693, 99.730685] |
| 6.8095238 | 0.00027368896 | 0.99994118 | -0.36529725 | 44 | 38 | [-4999.9835, 99.71655] |
| 6.8888889 | 0.0002772577 | 0.99994118 | -0.60229154 | 44 | 38 | [-4999.9084, 99.791619] |
| 6.968254 | 0.00027713885 | 0.99996079 | -0.87473598 | 44 | 38 | [-4999.9084, 99.891619] |
| 7.047619 | 0.00028872477 | 0.99996079 | -1.1589357 | 44 | 38 | [-4999.9559, 99.844137] |
| 7.1269841 | 0.00028349046 | 0.99994118 | -1.4901878 | 45 | 38 | [-4999.8209, 99.87906] |
| 7.2063492 | 0.00031335107 | 0.99994118 | -1.8057134 | 45 | 38 | [-4999.9209, 99.77906] |
| 7.2857143 | 0.00031705637 | 0.99994118 | -2.1888738 | 45 | 38 | [-4999.8951, 99.804946] |
| 7.3650794 | 0.0003189905 | 0.99996079 | -2.5940109 | 45 | 38 | [-4999.9806, 99.819405] |
| 7.4444444 | 0.00032497312 | 0.99996079 | -3.0150352 | 45 | 38 | [-4999.9852, 99.814821] |
| 7.5238095 | 0.00033372768 | 0.99996079 | 2.8186943 | 45 | 38 | [-4999.9852, 99.814821] |
| 7.6031746 | 0.00034378024 | 0.99994118 | 2.3479003 | 45 | 38 | [-4999.9952, 99.704787] |
| 7.6825397 | 0.000359076 | 0.99996079 | 1.8526855 | 45 | 38 | [-4999.9148, 99.885243] |
| 7.7619048 | 0.00036794975 | 0.99996079 | 1.3314549 | 45 | 38 | [-4999.9148, 99.885243] |
| 7.8412698 | 0.00038462664 | 0.99994118 | 0.79097175 | 45 | 38 | [-4999.8426, 99.857411] |
| 7.9206349 | 0.00040409389 | 0.99994118 | 0.22886003 | 45 | 38 | [-4999.9775, 99.722489] |
| 8 | 0.00041140009 | 0.99996079 | -0.36618648 | 45 | 38 | [-4999.9775, 99.822489] |

## Caveats

- The q extrapolation is empirical. It is accurate at q=2 in this
  diagnostic, but the same construction is not reliable across the
  whole q < 3 interval. In particular, q=2.5 is a poor point for this
  quadratic-edge checkpoint.
- The largest in-range error is caused by a peak-matching pathology
  near q=3.1587: the direct per-q peak-ratio reconstruction is already
  bad there, so this is not only a q-interpolation error.
- The q=1.5 failure should not be interpreted as a normal waveform
  mismatch value; the extrapolated time map fails the model validity
  checks before producing an acceptable comparison.
- `beta_peak` is the peak-location ratio `s_nr_peak / s_bhpt_peak`.
  The continuous time map uses this as a local multiplicative map,
  not as a separately integrated `d tau / d t_bhpt`.
- Values in theta outside the retained peak-node range are clipped
  to the nearest endpoint before evaluating alpha and beta.
- The complete peak-index q tables are stored in
  `wf_peak_ratios_results/fit_results.json` rather than expanded in this
  markdown file.
