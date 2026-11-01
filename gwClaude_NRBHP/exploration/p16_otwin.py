"""EXPLORATION pass 16 (B2) -- blend window scaled by the Ori-Thorne transition timescale.

Base = the pass-15 reduced model of V5 or V6 (p15/p15_<VAR>.json, round 2: zeroed coefficients and alpha-shape tie kept).
Window change: the fitted linear spin dependence (t1b, t2b) is replaced by
    s1 = t1a (L/L0)^p,   s2 = t2a (L/L0)^p,      L(q, chi1) = q^(1/5) tau_o(chi1) / (dtau/dt)(chi1)
the ppBHPT's own Ori-Thorne transition unit on the m1 clock (spin_dep_split/ot_join.ot_constants), L0 = L(5, 0);
p fitted (1 coefficient, start 1/3 from the pass-15 width-vs-L comparison).  So the window gains q dependence and loses
t1b, t2b: net -1 coefficient.  Usage: python p16_otwin.py V5|V6.  Writes exploration/p16/p16_<VAR>.json.
"""
from __future__ import annotations
import json, sys, time, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO / "spin_dep_split", REPO):
    sys.path.insert(0, str(p))
import p15_ablate as A                         # noqa: E402  (sets VAR from argv, P6.ab = A.ab)
import p6_m2_global as P6                      # noqa: E402
import fit_pn_anchored_spin as S               # noqa: E402
from ot_join import ot_constants               # noqa: E402

VAR = A.VAR
OUT = HERE / "p16"; OUT.mkdir(exist_ok=True)
NAMES = A.NAMES + ["p_win"]
IX = {n: i for i, n in enumerate(NAMES)}


def L_ot(q, chi1):
    c = ot_constants(chi1); return q ** 0.2 * c["tau_o"] / c["dtaudt"]


L0 = L_ot(5.0, 0.0)
_LC = {}


def ab(th, k, key):
    if len(th) == len(A.NAMES):                     # a pass-15 / V5 / V6 vector (no p_win): the linear window
        return A.ab(th, k, key)
    q, chi1 = k
    if k not in _LC:
        _LC[k] = L_ot(q, chi1) / L0
    f = _LC[k] ** th[IX["p_win"]]
    th2 = np.array(th[:len(A.NAMES)], float)
    # A.ab uses s_i = t_ia + t_ib chi1; emulate s_i = t_ia f with t_ib = t_ia (f - 1) / chi1, chi1 != 0
    if chi1 != 0.0:
        th2[A.IX["t1b"]] = th2[A.IX["t1a"]] * (f - 1) / chi1
        th2[A.IX["t2b"]] = th2[A.IX["t2a"]] * (f - 1) / chi1
    else:
        th2[A.IX["t1a"]] *= f; th2[A.IX["t2a"]] *= f
        th2[A.IX["t1b"]] = th2[A.IX["t2b"]] = 0.0
    return A.ab(th2, k, key)


P6.ab = ab


def main():
    pool = mp.get_context("fork").Pool(96)
    r2 = json.loads((HERE / "p15" / f"p15_{VAR}.json").read_text())["round2"]
    base = np.r_[r2["theta"], 1.0 / 3.0]
    zero = set(r2["zeroed"]) | {"t1b", "t2b"}
    free = [IX[n] for n in A.NAMES32 if n not in zero] + ([IX["kab"]] if r2["tie"] else []) + [IX["p_win"]]
    print(f"{VAR} base: pass-15 reduced {r2['n_coef']}c; here {len(free)}c (zeroed {sorted(zero)})", flush=True)
    th, e = A.refit(pool, base, free, 85, "otwin")
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    col = {f"{ch:+.1f}": float(np.median([v for (q, c), v in zip(P6.T96, e96) if c == ch])) for ch in S.CHIS}
    row = {f"{q:.3f}": float(np.median([v for (qq, c), v in zip(P6.T96, e96) if qq == q])) for q in S.TRAIN_Q}
    out = dict(variant=VAR, names=NAMES, theta=list(map(float, th)), zeroed=sorted(zero), tie=r2["tie"], n_coef=len(free),
               T96=[float(e96.mean()), float(np.median(e96)), float(e96.max())],
               H270=[float(eh.mean()), float(np.median(eh)), float(eh.max())], col_median=col, row_median=row,
               T96_cases={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in zip(P6.T96, e96)})
    (OUT / f"p16_{VAR}.json").write_text(json.dumps(out, indent=1))
    print(f"{VAR} OT window {len(free)}c, p = {th[IX['p_win']]:.3f}: T96 {e96.mean():.4e} / {np.median(e96):.4e} / "
          f"{e96.max():.4e} | H270 {eh.mean():.4e} / {np.median(eh):.4e} / {eh.max():.4e}", flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in col.items()), flush=True)
    print("   row medians: " + " ".join(f"{v:.2e}" for v in row.values()), flush=True)
    # gate L (user 2026-10-07): q = 2.5 < 1e-2 at every grid chi1; q = 2 reported.  Extrapolated coefficients, t0 only.
    LQ = [(q, c) for q in (2.5, 2.0) for c in S.CHIS]
    th32 = json.loads(A.SRC.read_text())["theta"]
    models = {f"{VAR} (32c)": (np.r_[th32, -0.93, 0.0, 0.0], 0),
              f"pass-15 reduced ({r2['n_coef']}c)": (np.r_[r2["theta"], 0.0], 0),
              f"pass-16 OT window ({len(free)}c)": (th, 1)}
    out["lowq"] = {}
    for name, (vec, use_ot) in models.items():
        el = pool.map(P6._full, [(list(vec if use_ot else vec[:len(A.NAMES)]), k) for k in LQ])
        out["lowq"][name] = {f"{q:.1f}_{c:+.1f}": float(v) for (q, c), v in zip(LQ, el)}
        for qq in (2.5, 2.0):
            v = [e for (q, c), e in zip(LQ, el) if q == qq]
            print(f"   low q {name:26s} q = {qq}: max {max(v):.3e} median {np.median(v):.3e}  "
                  + " ".join(f"{e:.1e}" for e in v), flush=True)
    (OUT / f"p16_{VAR}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
