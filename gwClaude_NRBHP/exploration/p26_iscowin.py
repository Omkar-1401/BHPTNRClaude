"""EXPLORATION pass 26 -- derived blend onset: the ppBHPT ISCO crossing (coefficient reduction).

The fitted onset s1 = t1a + t1b chi1 (V7-S3: -150 m1 at chi1 = -0.6, -44 m1 at +0.8) moves with spin the way the
primary's Kerr ISCO does.  Per case, s_isco = (first time x_pp >= x_isco(chi1)) - t_pk, in m1, from the ppBHPT waveform,
x_isco = (r_isco^1.5 + chi1)^(-2/3) (Kerr, prograde convention).  Variants on the pass-25 final model:
    I1:  s1 = s_isco + t1a          (t1b dropped; -1c)
    I0:  s1 = s_isco                (t1a, t1b dropped; -2c)
    J1:  s1 = s_isco + t1a + t1n (nu/nu5 - 1)   (pass 29; t1n stored in the unused kab slot; 20c)
    J2:  s1 = t1a + t1b chi1 + t1n (nu/nu5 - 1)  (pass 29 control; 21c)
The s1 range check (-150 <= s1 <= 30) is kept.  Refit of everything else free; full scoring.
Usage: python p26_iscowin.py I1|I0|J1|J2 [budget_min].  Writes exploration/p26/p26_<var>.json.
2026-10-08 fix: the first run set M.S1FUN AFTER forking the pool, so the workers never saw it ("I1" there was a constant
onset s1 = t1a, "I0" was invalid); kept as p26/p26_iscowin_forkbug.json.  Now one variant per run, hook set before the fork.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
sys.argv = [sys.argv[0], "35"]
import p25_nuexp as M                           # noqa: E402  (P6.ab = M.ab)
P6 = M.P6; A = M.A; PR = M.PR
OUT = HERE / "p26"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p25" / "p25_nuexp.json").read_text())["round2"]
TH0 = np.array(R["theta"]); FREE0 = list(R["free"])
I1A, I1B = M.NAMES.index("t1a"), M.NAMES.index("t1b")
_SI = {}


def x_isco(chi):
    z1 = 1 + (1 - chi ** 2) ** (1 / 3) * ((1 + chi) ** (1 / 3) + (1 - chi) ** (1 / 3))
    z2 = np.sqrt(3 * chi ** 2 + z1 ** 2)
    r = 3 + z2 - np.sign(chi) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return (r ** 1.5 + chi) ** (-2 / 3)


def s_isco(k):
    if k not in _SI:
        c = P6.get(k); t, x = c["case"]["t_bhpt"], c["case"]["x"]
        pre = t < c["tpk"]; i = int(np.argmax((x >= x_isco(k[1])) & pre))
        _SI[k] = float(t[i] - c["tpk"])
    return _SI[k]


def s1_I1(k, d):
    return s_isco(k) + d["t1a"]


def s1_J1(k, d):                    # pass 29: ISCO onset + offset + nu slope; the slope lives in the unused "kab" slot (tie = 0)
    nu = k[0] / (1 + k[0]) ** 2
    return s_isco(k) + d["t1a"] + d["kab"] * (nu / M.NU5 - 1)


def s1_J2(k, d):                    # pass 29: fitted linear onset + nu slope (control for J1)
    nu = k[0] / (1 + k[0]) ** 2
    return d["t1a"] + d["t1b"] * k[1] + d["kab"] * (nu / M.NU5 - 1)


def s1_I0(k, d):
    return s_isco(k)


def _sis(k):
    return s_isco(k)


def main():
    fn, drop = {"I1": (s1_I1, [I1B]), "I0": (s1_I0, [I1A, I1B]), "J1": (s1_J1, [I1B]), "J2": (s1_J2, [])}[VAR]
    IK = M.NAMES.index("kab")
    assert TH0[M.NAMES.index("tie")] == 0
    with mp.get_context("fork").Pool(96) as p0:          # s_isco per case, then into the parent cache before the real fork
        si = dict(zip(P6.T96, p0.map(_sis, P6.T96)))
    _SI.update(si)
    gap = float(np.median([TH0[I1A] + TH0[I1B] * k[1] - si[k] for k in P6.T96]))
    M.S1FUN = fn                        # BEFORE the fork
    pool = mp.get_context("fork").Pool(96)
    out = dict(variant=VAR, base_n=len(FREE0), base_T96=R["T96"], s_isco={f"{q:.6f}_{c:+.4f}": v for (q, c), v in si.items()})
    f0 = TH0.copy()
    for i in drop:
        f0[i] = 0.0
    if VAR in ("I1", "J1"):
        f0[I1A] = gap
    if VAR in ("J1", "J2"):
        f0[IK] = 0.0
    free = [j for j in FREE0 if j not in drop] + ([IK] if VAR in ("J1", "J2") else [])
    i2 = M.NAMES.index("t2a")
    if VAR == "I0" and max(si.values()) + 3 > f0[i2]:   # s2 must exceed s1 + 3 everywhere: free t2a only if needed
        f0[i2] = max(si.values()) + 10
        if i2 not in free:
            free.append(i2)
    print(f"{VAR}: s_isco range [{min(si.values()):.1f}, {max(si.values()):.1f}], s2 = {f0[i2]:.2f}, {len(free)}c", flush=True)
    e_chk = float(np.mean(pool.map(P6._fit, [(list(f0), k) for k in P6.T96])))
    print(f"{VAR}: start fit-mean {e_chk:.4e} (must be < 1)", flush=True)
    th, e = A.refit(pool, f0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out.update(n_coef=len(free), free=free, theta=list(map(float, th)), fit_mean=e, **sc)
    (OUT / f"p26_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
