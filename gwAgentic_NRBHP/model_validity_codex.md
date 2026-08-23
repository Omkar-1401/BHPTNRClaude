# Q-dependent model validity summary

This file compares the q-dependent calibration models that currently have
master coefficient tables in this directory. Single-q calibration files are not
included because they do not define a q-extrapolatable model.

The in-range median and q=3 entries are the selected master-model
`\mathcal{E}` values already reported in each model's `scaling_*.md` file over
the calibration interval `3 <= q <= 8`. The q=2 and q=1.5 entries were
evaluated from the selected coefficient table in each markdown file.

| model | family | source markdown | median in-range `\mathcal{E}` | q=3 `\mathcal{E}` | q=2 extrapolated `\mathcal{E}` | q=1.5 extrapolated `\mathcal{E}` |
|:---|:---|:---|---:|---:|---:|---:|
| PN-opt q-dependent | PN-inspired | `scaling_PN_opt_q_dep.md` | 1.0690248e-4 | 1.3458304e-3 | 6.2246388e-1 | 1.2250947 |
| Physical-smooth q-dependent | physical | `scaling_physical_smooth_qdep.md` | 3.4149525e-4 | 7.1426112e-4 | 3.9984155 | 5.5427989 |
| Two-constant q-dependent | physical | `scaling_two_constant_qdep.md` | 1.7644883e-4 | 1.0377794e-3 | 3.7746323e-1 | 7.7625935e-1 |

## Extrapolation status

The q=2 and q=1.5 rows are formal extrapolations outside the calibrated
training interval. `NRHybSur3dq8` emits an out-of-bounds warning for these
evaluations.

| model | q=2 evaluation status | q=1.5 evaluation status |
|:---|:---|:---|
| PN-opt q-dependent | diagnostic fallback; coverage 0.393169 | diagnostic fallback; coverage 0.999959 |
| Physical-smooth q-dependent | diagnostic fallback; coverage 0.999941 | diagnostic fallback; coverage 0.999939 |
| Two-constant q-dependent | normal evaluator; coverage 0.984530 | normal evaluator; coverage 0.974198 |

For the diagnostic-fallback entries, the normal evaluator rejected the
extrapolated parameter vector before returning waveform arrays. The listed
`\mathcal{E}` values were computed only after verifying positive scaling,
monotonic `tau`, and nonzero common support. They should be treated as failure
diagnostics, not as accepted calibration errors.
