"""EXPLORATION pass 40 -- closed-form R2(x) in the pass-39 L4 leader (user, 2026-10-18: interpretability).

R2(x) = 2SF / 1SF (2,2) amplitude ratio (Schwarzschild, circular; WaSABI table of 22 points on x in [0.04, 0.16]).
Least-squares fit to the table (2026-10-18; max abs error 0.0020, rms 8e-4; relative < 1% for x < 0.14):
    R2_cf(x) = (55/21) x (1 - 6x) (1 + a x + b x^2),   a = 3.691, b = 30.255
55/21 = 2 x 55/42, the 1PN O(nu) amplitude coefficient (a free slope fits 2.73 +- 0.05); (1 - 6x) vanishes at the
Schwarzschild ISCO (a free power of it fits 1.003).
Three closed forms (user, 2026-10-18: "try all of it"), all fitted to the same 22-point table:
    O1  (55/21) x (1 - 6x) (1 + 3.691 x + 30.255 x^2)                 2 fitted; max abs err 0.0020
    O2  (55/21) x (1 - 6x) (1 + 4.768 x + 21.68 x^2), 4.768 from 2PN    1 fitted; 0.0039
    O3  R2_PN3.5(x) (1 - 6x) / (1 - 6.463 x + 7.656 x^2)                2 fitted; 0.0034
        R2_PN3.5 = 2 Re[N/H0]: N = nu-linear, H0 = test-mass part of H22 (Faye+ 2012 Eq. 6.5, verified)
Each on the pass-39 L4 model with its 18 coefficients FROZEN, in two ranges:
    <O>t  x clamped to the table range [0.04, 0.16] (exactly how the table is used): the fit error alone
    <O>n  x clamped to [0.005, 1/6]: the closed form's natural range (0 at and beyond the ISCO)
Usage: python p40_r2closed.py.  Writes exploration/p40/p40.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "L4"]
import p39_numlimit as L                        # noqa: E402
P6 = L.P6; A = L.A; PR = L.PR
Z = L.G.M.Z                                     # p23_s3: holds the R2 table (_RX, _RY) and the per-case cache _X
OUT = HERE / "p40"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
TH0 = np.array(R["theta"]); FREE0 = list(R["free"])
RX0, RY0 = Z._RX.copy(), Z._RY.copy()
_PI, _GE = np.pi, np.euler_gamma


def r2pn35(x):
    N = (55 / 42 * x - 1069 / 216 * x ** 2 + (34 * _PI / 21 - 24j) * x ** 2.5 + (-278185 / 33264 + 41 * _PI ** 2 / 96) * x ** 3
         + (-2495 * _PI / 378 + 14333j / 162) * x ** 3.5)
    H = (1 - 107 / 42 * x + 2 * _PI * x ** 1.5 - 2173 / 1512 * x ** 2 - 107 * _PI / 21 * x ** 2.5
         + (27027409 / 646800 - 856 / 105 * _GE + 428j * _PI / 105 + 2 * _PI ** 2 / 3 - 428 / 105 * np.log(16 * x)) * x ** 3
         - 2173 * _PI / 756 * x ** 3.5)
    return 2 * np.real(N / H)


FORMS = {"O1": lambda x: 55 / 21 * x * (1 - 6 * x) * (1 + 3.691 * x + 30.255 * x * x),
         "O2": lambda x: 55 / 21 * x * (1 - 6 * x) * (1 + 4.768 * x + 21.68 * x * x),
         "O3": lambda x: r2pn35(x) * (1 - 6 * x) / (1 - 6.463 * x + 7.656 * x * x)}


def use(variant):
    """Install R2 as a dense table that np.interp reproduces to < 1e-7 (the model clamps x to the table ends)."""
    if variant == "T":
        Z._RX, Z._RY = RX0, RY0
    else:
        lo, hi = (0.04, 0.16) if variant[-1] == "t" else (0.005, 1.0 / 6.0)
        xx = np.linspace(lo, hi, 4001); Z._RX, Z._RY = xx, FORMS[variant[:2]](xx)
    Z._X.clear()


def run(variant, th):
    use(variant)
    pool = mp.get_context("fork").Pool(96)
    sc = PR.score(pool, th)
    return pool, sc


def line(tag, sc):
    t, h = sc["T96"], sc["H270"]
    lq = [v for k, v in sc["lowq"].items() if k.startswith("2.5")]
    print(f"{tag}: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 median {h[1]:.4e} | q = 2.5 max {max(lq):.3e}", flush=True)


def main():
    P6.get = L.get; P6.ab = L.ab                # BEFORE any fork
    out = dict(forms={"O1": "(55/21) x (1-6x)(1 + 3.691 x + 30.255 x^2)", "O2": "(55/21) x (1-6x)(1 + 4.768 x + 21.68 x^2)", "O3": "R2_PN3.5 (1-6x)/(1 - 6.463 x + 7.656 x^2)"})
    pool, sc = run("T", TH0); pool.close()
    r = np.array([sc["T96_cases"][k] / R["T96_cases"][k] for k in R["T96_cases"]])
    print(f"table R2 reproduces L4: ratio {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    for v in ("O1t", "O1n", "O2t", "O2n", "O3t", "O3n"):
        pool, sc = run(v, TH0); pool.close()
        out[v] = dict(theta="L4 frozen", **sc); line(f"{v} (L4 frozen)", sc)
        (OUT / "p40.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
