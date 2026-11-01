"""EXPLORATION pass 15 -- coefficient ablation of V5 / V6 (user, 2026-10-07: "don't use a crazy number of coefficients").

Model = p14_v6.py (V5: w = 1, V6: w = nu/nu5), with one extra option: tie the alpha-shape amplitude to the beta one,
B = kab * A  (pass 12 measured B ~ -0.93 A), replacing pb0, pb1, pb2 by one coefficient.
Full parameter vector (34) = the 32 V5/V6 names + kab + tie flag (0/1, never fitted).

Round 1: for each candidate (one coefficient, or a named group), fix it at 0 (or tie), refit the rest from the
           converged V5/V6 solution with a short budget, record the fit-stride T96 mean.
Round 2: drop jointly every candidate whose single cost was < CUT, refit longer, score T96 and H270 at full resolution.
Usage: python p15_ablate.py V5|V6.   Writes exploration/p15/p15_<VAR>.json (rewritten after every candidate).
"""
from __future__ import annotations
import json, sys, time, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline as _CS

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import p6_m2_global as P6                     # noqa: E402
import fit_pn_anchored_spin as S              # noqa: E402
import p3_rd as P3                            # noqa: E402
from p14_v6 import flux13                     # noqa: E402

VAR = sys.argv[1] if len(sys.argv) > 1 else "V6"
OUT = HERE / "p15"; OUT.mkdir(exist_ok=True)
SRC = {"V5": HERE / "p13" / "p13_V5.json", "V6": HERE / "p14" / "p14_V6.json"}[VAR]
NAMES32 = json.loads(SRC.read_text())["names"]
NAMES = NAMES32 + ["kab", "tie"]
IX = {n: i for i, n in enumerate(NAMES)}
CUT = 0.03                     # round-2 joint drop: single-drop cost below 3 % of the T96 fit mean
STEP_MIN = 3.5                 # refit budget per round-1 candidate (minutes)

_SH = json.loads((HERE / "p12" / "p12_shapes.json").read_text())
_PB = _CS(_SH["knots_beta"], _SH["phi_beta"], bc_type="natural")
_PA = _CS(_SH["knots_alpha"], _SH["phi_alpha"], bc_type="natural")
_KB, _KA = (_SH["knots_beta"][0], _SH["knots_beta"][-1]), (_SH["knots_alpha"][0], _SH["knots_alpha"][-1])


def ab(th, k, key):
    q, chi1 = k; c = P6.get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    t = case[pre + "t_bhpt"]
    d = dict(zip(NAMES, th))
    aF, bF = flux13(np.asarray(th[:13]), q, chi1, case, key)
    s1 = d["t1a"] + d["t1b"] * chi1; s2 = d["t2a"] + d["t2b"] * chi1
    if not (-150 <= s1 <= 30 and s1 + 3 <= s2 <= 150) or np.any(aF <= 0) or np.any(bF <= 0):
        return None, None
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    br, kap = c["anc"]
    s = t - c["tpk"]
    W = P3.smooth5((s - s1) / (s2 - s1))
    wv = nu / (5.0 / 36.0) if VAR == "V6" else 1.0
    lbm = np.log(br) + wv * (d["r0"] + d["r1"] * chi1 + d["r2"] * nu + d["r3"] * chi1 ** 2)
    lam = (1.2 * np.log(X1) + wv * (d["l0"] + d["l1"] * chi1 + d["l2"] * nu + d["l3"] * chi1 ** 2)
           + kap * br * np.clip(s - P6.S_RING, 0, None))
    la = (1 - W) * np.log(aF) + W * lam
    lb = (1 - W) * np.log(bF) + W * lbm
    A = d["pa0"] + d["pa1"] * chi1 + d["pa2"] * chi1 ** 2 + d["pa3"] * nu
    B = d["kab"] * A if d["tie"] > 0.5 else d["pb0"] + d["pb1"] * chi1 + d["pb2"] * chi1 ** 2
    lb = lb + wv * A * _PB(np.clip(s, *_KB))
    la = la + wv * B * _PA(np.clip(s, *_KA))
    return np.exp(la), np.exp(lb)


P6.ab = ab

# round-1 candidates: name -> (coefficients fixed at 0, tie?)
CANDS = {n: ([n], False) for n in
         ["flux_c1", "flux_A1", "flux_s1", "flux_s2", "flux_b2", "flux_b3", "flux_s3", "flux_s4",
          "t1b", "t2b", "r1", "r2", "r3", "l1", "l2", "l3", "pa1", "pa2", "pa3", "pb1", "pb2"]}
CANDS["window_spin(t1b,t2b)"] = (["t1b", "t2b"], False)
CANDS["tie_alpha_shape(pb->kab)"] = (["pb0", "pb1", "pb2"], True)
CANDS["V3_insp(b2,b3,s3,s4)"] = (["flux_b2", "flux_b3", "flux_s3", "flux_s4"], False)


class Stop(Exception):
    pass


def refit(pool, full0, free, budget_min, tag):
    full0 = np.array(full0, float); free = list(free)
    st = dict(n=0, best=np.inf, bx=full0[free].copy(), t=time.time())

    def expand(x):
        f = full0.copy(); f[free] = x; return f

    def obj(x):
        v = float(np.mean(pool.map(P6._fit, [(list(expand(x)), k) for k in P6.T96])))
        st["n"] += 1
        if v < st["best"]:
            st["best"], st["bx"] = v, np.array(x, float)
        if time.time() - st["t"] > budget_min * 60:
            raise Stop
        return v
    try:
        minimize(obj, full0[free], method="Powell", options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 40})
    except Stop:
        pass
    print(f"  [{tag}] {st['n']} evals {(time.time()-st['t'])/60:.1f} min  fit-mean {st['best']:.4e}", flush=True)
    return expand(st["bx"]), st["best"]


def main():
    pool = mp.get_context("fork").Pool(96)
    th32 = json.loads(SRC.read_text())["theta"]
    base = np.r_[th32, -0.93, 0.0]
    e_base = float(np.mean(pool.map(P6._fit, [(list(base), k) for k in P6.T96])))
    print(f"{VAR} full (32c) fit-stride T96 mean {e_base:.4e}", flush=True)
    out = dict(variant=VAR, base_fit_mean=e_base, round1={}, CUT=CUT)
    fit_ix = [IX[n] for n in NAMES32]
    for cname, (zero, tie) in CANDS.items():
        f0 = base.copy()
        for n in zero:
            f0[IX[n]] = 0.0
        free = [i for i in fit_ix if NAMES[i] not in zero]
        if tie:
            f0[IX["tie"]] = 1.0; free = free + [IX["kab"]]
        th, e = refit(pool, f0, free, STEP_MIN, cname)
        out["round1"][cname] = dict(zeroed=zero, tie=tie, n_coef=len(free), fit_mean=e, cost=e / e_base - 1)
        print(f"{cname:28s} {len(free):2d}c  cost {100*(e/e_base-1):+6.2f} %", flush=True)
        (OUT / f"p15_{VAR}.json").write_text(json.dumps(out, indent=1))
    # round 2: joint drop of the single coefficients (not groups) that each cost < CUT
    cheap = [c for c, r in out["round1"].items() if len(r["zeroed"]) == 1 and not r["tie"] and r["cost"] < CUT]
    tie = out["round1"]["tie_alpha_shape(pb->kab)"]["cost"] < CUT
    zero = sorted(set(cheap) | ({"pb0", "pb1", "pb2"} if tie else set()))
    f0 = base.copy()
    for n in zero:
        f0[IX[n]] = 0.0
    free = [i for i in fit_ix if NAMES[i] not in zero]
    if tie:
        f0[IX["tie"]] = 1.0; free.append(IX["kab"])
    left = 110 - (time.time() - T_START) / 60 - 8
    th, e = refit(pool, f0, free, max(left, 10), "joint")
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    col = {f"{ch:+.1f}": float(np.median([e for (q, c), e in zip(P6.T96, e96) if c == ch])) for ch in S.CHIS}
    out["round2"] = dict(zeroed=zero, tie=tie, n_coef=len(free), theta=list(map(float, th)), names=NAMES,
                         T96=[float(e96.mean()), float(np.median(e96)), float(e96.max())],
                         H270=[float(eh.mean()), float(np.median(eh)), float(eh.max())], col_median=col)
    (OUT / f"p15_{VAR}.json").write_text(json.dumps(out, indent=1))
    print(f"{VAR} reduced {len(free)}c (zeroed {zero}, tie {tie}): T96 {e96.mean():.4e} / {np.median(e96):.4e} / "
          f"{e96.max():.4e} | H270 {eh.mean():.4e} / {np.median(eh):.4e} / {eh.max():.4e}", flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in col.items()), flush=True)


T_START = time.time()
if __name__ == "__main__":
    main()
