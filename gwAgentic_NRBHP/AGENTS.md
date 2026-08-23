# AGENTS.md

These instructions apply to the whole repository.

## Project Goal

Find simple, physically interpretable time-dependent scalings between a raw BHPT
waveform and the corresponding NR hybrid waveform.

The main objects of interest are:

- amplitude scaling: `alpha(t)`
- time/frequency scaling or time reparameterization: `beta(t)`

The goal is not just to find one constant prefactor. The project should identify
smooth functions of time that map the interpolated BHPT `(2, 2)` waveform toward
the NR hybrid counterpart while keeping the ansatz simple enough to interpret.

The immediate target is the nonspinning `q=5`, `(2, 2)` comparison in
`Untitled1.py`:

- BHPT source: `BHPTNRSurrogate/surrogates/BHPTNRSur1dq1e4.py`
- BHPT mode: `hbhpt[(2, 2)]`
- NR source: `gwsurrogate.LoadSurrogate("NRHybSur3dq8")`
- NR mode: `h_NRHybSur3dq8[(2, 2)]`
- Error target: `mathcalE < 1`

Use `calibrated=False` when deriving the raw BHPT-to-NR scaling. Use
`calibrated=True` only as a diagnostic or upper-bound reference, because that
already applies the surrogate's built-in NR calibration.

## Environment

Use the local virtual environment created at `.venv`. Run commands from the
repository root:

```bash
source .venv/bin/activate
mkdir -p .cache/matplotlib
export MPLCONFIGDIR=.cache/matplotlib
export LD_LIBRARY_PATH=/home/omkarnm1401/miniconda3/envs/ut_codex/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PWD/BHPTNRSurrogate/surrogates:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

The `LD_LIBRARY_PATH` export is required for `gwsurrogate` to find
`libcblas`/`libgslcblas` in the existing `ut_codex` conda environment. Without
it, `import gwsurrogate` can fail even though the Python package is visible.

Do not use more than 4 cores. If using multiprocessing, joblib, NumPy-backed
threading, or optimizer parallelism, keep workers and BLAS threads at or below 4.

## Reference Workflow

`Untitled1.py` is the starting reference. It:

1. Loads raw BHPT data with `bhptsur.generate_surrogate(q=5, calibrated=False)`.
2. Loads `NRHybSur3dq8` and evaluates a nonspinning NR waveform with
   `dt=0.1`, `f_low=5e-3`, and `t_start=-5000.1`.
3. Compares the complex `(2, 2)` modes with `mathcalE_error`.

Avoid hidden notebook state. Any new result should be reproducible from a script
run by `.venv/bin/python`.

## Error Metric

Use the NR waveform as the reference waveform:

```python
def mathcalE_error(h_ref, h_model):
    return np.sum(np.abs(h_ref - h_model) ** 2) / (2 * np.sum(np.abs(h_ref) ** 2))
```

This is equivalent to the expression in `Untitled1.py`, but easier to audit.
Always report:

- `q`
- mode
- time window
- BHPT `calibrated` flag
- functional forms and fitted parameters for `alpha(t)` and `beta(t)`
- whether any phase/time shifts were optimized
- final `mathcalE`

## Scaling Protocol

Keep the model simple and interpretable. Prefer low-dimensional, smooth
time-dependent scalings such as:

- time reparameterization: `t_bhpt -> tau(t)`, with
  `d tau / d t = beta(t)` or an equivalent monotonic time map
- amplitude scaling: `h_bhpt(t) -> alpha(t) * h_bhpt(t)`
- phase rotation: `h_bhpt -> exp(1j * phi0) * h_bhpt`

Reasonable first ansatz families include constants, low-order polynomials,
splines with few knots, or simple functions of normalized time such as
`x = (t - t_min) / (t_max - t_min)`. Constant scalings using quantities such as
`q`, `1/q`, symmetric mass ratio `eta = q / (1 + q)**2`, and
`mass_norm = 1 / (1 + 1/q)` are useful baselines, but they are not the final
target if time-dependent structure remains in the residual.

Constrain `beta(t)` so the BHPT time map remains monotonic. Avoid highly
oscillatory or high-capacity fits that reduce `mathcalE` by overfitting rather
than revealing a theoretical scaling.

When comparing waveforms:

1. Define `alpha(t)` and `beta(t)` or the equivalent monotonic time map
   `tau(t)`.
2. Apply the time map to the BHPT time coordinate and apply `alpha(t)` to the
   BHPT strain.
3. Restrict the NR times to the common support of the transformed BHPT time
   array.
4. Interpolate transformed BHPT onto the restricted NR time grid.
5. Compute `mathcalE` on the same complex mode samples.

Do not accept results that rely on extrapolated BHPT samples. The current
`Untitled1.py` interpolation path emits warnings about requested samples outside
the interpolation domain; fix this by masking to common support before
interpolation.

Also note that `Untitled1.py` plots `hbhpt[(2, 2)] * mass_norm`, but its final
`mathcalE` call currently interpolates unscaled BHPT strain. When reporting a
result, state the exact scaling used in the metric, not just the plot.

## Baseline Check

With the environment above, local imports and model loading were verified:

```bash
.venv/bin/python -c "import numpy, matplotlib, gwsurrogate, gwtools, sklearn, h5py, scipy"
.venv/bin/python -c "import gwsurrogate; gwsurrogate.LoadSurrogate('NRHybSur3dq8')"
```

Using the current `Untitled1.py` interpolation behavior for `q=5` gives:

- raw BHPT: `mathcalE ~= 1.40`
- time-scaled by `mass_norm`: `mathcalE ~= 1.36`
- time and amplitude scaled by `mass_norm`: `mathcalE ~= 1.11`

These constant-scaling baselines remain above the `mathcalE < 1` target and are
not final because they include interpolation-domain warnings. Use common-support
masking for accepted comparisons.

## Coding Guidelines

- Prefer small scripts and functions over notebook-only analysis.
- Do not hard-code `/home/omkarnm1401/UT Austin` inside new Python code; derive
  paths from `Path(__file__).resolve()` or the current working directory.
- Keep generated results in a clearly named results directory and include enough
  metadata to reproduce the run.
- Reuse local waveform utilities where practical, but keep the scaling logic
  explicit and easy to inspect.
- Avoid changing bundled surrogate data or vendored model code unless the user
  explicitly asks for that.
