"""Stage-2 prototype: fit alpha(x,nu), beta(x,nu); compare beta_abs vs beta_slope;
small degree bake-off. Anchored so correction ∝ nu (vanishes at nu->0)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
PER_Q = ROOT / "peaks_results" / "per_q"

cases = [np.load(p) for p in sorted(PER_Q.glob("peaks_q*.npz"))]
print(f"{len(cases)} q cases loaded")

# pool all peaks (broadcast scalar per-case values to peak count)
def pool(key):
    out = []
    for c in cases:
        v = c[key]
        n = len(c["x"])
        out.append(v if v.ndim else np.full(n, float(v)))
    return np.concatenate(out)
x = pool("x"); nu = pool("nu")
alpha = pool("alpha"); b_abs = pool("beta_abs"); b_slope = pool("beta_slope")
qall = pool("q")
print(f"{len(x)} pooled peaks; x in [{x.min():.3f},{x.max():.3f}], "
      f"nu in [{nu.min():.4f},{nu.max():.4f}]")


def design(x, nu, Mx, Nnu):
    """columns nu^n * x^m, n=1..Nnu, m=0..Mx  -> value = 1 + design @ coeffs."""
    cols = []
    for n in range(1, Nnu + 1):
        for m in range(0, Mx + 1):
            cols.append((nu ** n) * (x ** m))
    return np.vstack(cols).T


def fit_eval(target, Mx, Nnu):
    A = design(x, nu, Mx, Nnu)
    coeffs, *_ = np.linalg.lstsq(A, target - 1.0, rcond=None)
    pred = 1.0 + A @ coeffs
    resid = target - pred
    # per-q RMS
    per_q = []
    for c in cases:
        Aq = design(c["x"], c["nu"], Mx, Nnu)
        pq = 1.0 + Aq @ coeffs
        per_q.append(np.sqrt(np.mean((c[TKEY] - pq) ** 2)) if False else
                     np.sqrt(np.mean((_tq(c) - pq) ** 2)))
    return coeffs, np.sqrt(np.mean(resid ** 2)), np.max(np.abs(resid)), np.array(per_q)


# helper to pull the right target per case (set by TKEY global)
TKEY = "alpha"
def _tq(c):
    return c[TKEY]


print("\n=== degree bake-off (global RMS / max resid) ===")
print(f"{'target':>10} {'Mx':>3} {'Nnu':>4} {'nparam':>7} {'RMS':>10} {'maxabs':>10}")
for label, tgt, key in [("alpha", alpha, "alpha"),
                        ("beta_abs", b_abs, "beta_abs"),
                        ("beta_slope", b_slope, "beta_slope")]:
    TKEY = key
    for Mx, Nnu in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3)]:
        coeffs, rms, mx, per_q = fit_eval(tgt, Mx, Nnu)
        print(f"{label:>10} {Mx:3d} {Nnu:4d} {len(coeffs):7d} {rms:10.3e} {mx:10.3e}")
    print()

# pick a working degree and show per-q RMS spread for alpha + both betas
print("=== per-q RMS at Mx=3,Nnu=3 ===")
for label, tgt, key in [("alpha", alpha, "alpha"),
                        ("beta_abs", b_abs, "beta_abs"),
                        ("beta_slope", b_slope, "beta_slope")]:
    TKEY = key
    coeffs, rms, mx, per_q = fit_eval(tgt, 3, 3)
    print(f"{label:>10}: global RMS {rms:.3e}, per-q RMS median {np.median(per_q):.3e}, "
          f"max {per_q.max():.3e}")

# extrapolation sanity: value at nu->0 (should be ~1) and at q=2 (nu=0.2222)
print("\n=== extrapolation check (beta_abs, Mx=3,Nnu=3) ===")
TKEY = "beta_abs"
coeffs, *_ = np.linalg.lstsq(design(x, nu, 3, 3), b_abs - 1.0, rcond=None),
coeffs = coeffs[0]
for qc, xc in [(1e6, 0.10), (2.0, 0.10), (5.0, 0.10), (8.0, 0.10)]:
    nuc = qc / (1 + qc) ** 2
    val = 1.0 + design(np.array([xc]), np.array([nuc]), 3, 3) @ coeffs
    print(f"q={qc:>8.1f} nu={nuc:.5f} x={xc}: beta_abs={val[0]:.5f}")
