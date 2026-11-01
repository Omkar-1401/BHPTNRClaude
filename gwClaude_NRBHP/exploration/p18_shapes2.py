"""EXPLORATION pass 18 -- V6r + the leading pass-17 residual shapes as fixed basis functions (D2, second round).

Base = V6r (pass 15 round 2) or the pass-16 OT-window model (whichever argv[2] names).  Added, for each chosen component j
of the pass-17 PCA (beta: dlnB, alpha: dlnA):
    ln beta  += w (e0 + e1 chi1 + e2 chi1^2) psi_beta_j(s),   ln alpha += w (f0 + f1 chi1 + f2 chi1^2) psi_alpha_j(s)
w = nu/nu5 (test-mass limit kept), psi = natural cubic spline through the component (0 beyond its knots), s = t - t_peak,pp.
Start: least squares of the per-case pass-17 amplitudes on (w, w chi1, w chi1^2).  Then a T96 refit of everything that was
free in the base plus the new amplitudes; T96, H270 and the low-q scorecard (gate L) at full resolution.
Usage: python p18_shapes2.py V6 p15|p16 NB NA     (NB, NA = number of beta / alpha components, e.g. 1 1).
Writes exploration/p18/p18_<base>_<NB><NA>.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p16_otwin as P16                         # noqa: E402
A = P16.A; P6 = P16.P6
import fit_pn_anchored_spin as S                # noqa: E402

BASE = sys.argv[2]; NB, NA = int(sys.argv[3]), int(sys.argv[4])
BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 80.0
OUT = HERE / "p18"; OUT.mkdir(exist_ok=True)
if BASE == "p15":
    R = json.loads((HERE / "p15" / f"p15_{P16.VAR}.json").read_text())["round2"]
    BASEV = np.array(R["theta"]); ZERO = set(R["zeroed"]); TIE = R["tie"]
    FREE0 = [A.IX[n] for n in A.NAMES32 if n not in ZERO] + ([A.IX["kab"]] if TIE else [])
else:
    R = json.loads((HERE / "p16" / f"p16_{P16.VAR}.json").read_text())
    BASEV = np.array(R["theta"]); ZERO = set(R["zeroed"]); TIE = R["tie"]
    FREE0 = [P16.IX[n] for n in A.NAMES32 if n not in ZERO] + ([P16.IX["kab"]] if TIE else []) + [P16.IX["p_win"]]
NB0 = len(BASEV)
PC = json.loads((HERE / "p17" / f"p17_{BASE if BASE == 'p15' else 'p15'}.json").read_text())
CHI = np.array([r["chi1"] for r in PC["rows"]]); QQ = np.array([r["q"] for r in PC["rows"]])
SH = []                          # (field, spline, (lo, hi), per-case amplitudes)
for field, nc in (("dlnB", NB), ("dlnA", NA)):
    d = PC["pca"][field]
    for j in range(nc):
        kn = np.array(d["knots"], float); comp = np.array(d["comps"][j])
        full = np.array(PC["knots"], float); pad = full[full > kn[-1]]     # zero-padded to +60 m1, as pass 12
        kn, comp = np.r_[kn, pad], np.r_[comp, np.zeros(len(pad))]
        SH.append((field, CubicSpline(kn, comp, bc_type="natural"), (kn[0], kn[-1]), np.array(d["amps"][j])))


def ab(th, k, key):
    a, b = P16.ab(th[:NB0], k, key)
    if a is None:
        return a, b
    q, chi1 = k; c = P6.get(k); pre = "fit_" if key == "fit" else ""
    s = c["case"][pre + "t_bhpt"] - c["tpk"]
    wv = (q / (1 + q) ** 2) / (5.0 / 36.0)
    la, lb = np.log(a), np.log(b)
    for j, (field, sp, (lo, hi), _) in enumerate(SH):
        e0, e1, e2 = th[NB0 + 3 * j: NB0 + 3 * j + 3]
        y = wv * (e0 + e1 * chi1 + e2 * chi1 ** 2) * sp(np.clip(s, lo, hi))
        if field == "dlnB":
            lb = lb + y
        else:
            la = la + y
    return np.exp(la), np.exp(lb)


P6.ab = ab


def start_amps():
    w = (QQ / (1 + QQ) ** 2) / (5.0 / 36.0)
    M = np.column_stack([w, w * CHI, w * CHI ** 2])
    return np.concatenate([np.linalg.lstsq(M, amps, rcond=None)[0] for (_, _, _, amps) in SH])


def main():
    t_start = time.time()
    pool = mp.get_context("fork").Pool(96)
    x0 = np.r_[BASEV, start_amps()]
    free = FREE0 + list(range(NB0, len(x0)))
    e_b = float(np.mean(pool.map(P6._fit, [(list(np.r_[BASEV, np.zeros(len(x0) - NB0)]), k) for k in P6.T96])))
    e_s = float(np.mean(pool.map(P6._fit, [(list(x0), k) for k in P6.T96])))
    print(f"{BASE} + {NB} beta / {NA} alpha shapes: {len(free)}c.  Base fit-mean {e_b:.4e}; with the LSQ start {e_s:.4e}", flush=True)
    th, e = A.refit(pool, x0, free, BUDGET, f"shapes{NB}{NA}")
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    LQ = [(q, c) for q in (2.5, 2.0) for c in S.CHIS]
    el = np.array(pool.map(P6._full, [(list(th), k) for k in LQ]))
    col = {f"{ch:+.1f}": float(np.median([v for (q, c), v in zip(P6.T96, e96) if c == ch])) for ch in S.CHIS}
    row = {f"{q:.3f}": float(np.median([v for (qq, c), v in zip(P6.T96, e96) if qq == q])) for q in S.TRAIN_Q}
    out = dict(base=BASE, NB=NB, NA=NA, n_coef=len(free), theta=list(map(float, th)), free=free,
               T96=[float(e96.mean()), float(np.median(e96)), float(e96.max())],
               H270=[float(eh.mean()), float(np.median(eh)), float(eh.max())], col_median=col, row_median=row,
               lowq={f"{q:.1f}_{c:+.1f}": float(v) for (q, c), v in zip(LQ, el)},
               T96_cases={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in zip(P6.T96, e96)})
    (OUT / f"p18_{BASE}_{NB}{NA}.json").write_text(json.dumps(out, indent=1))
    print(f"{len(free)}c: T96 {e96.mean():.4e} / {np.median(e96):.4e} / {e96.max():.4e} | "
          f"H270 {eh.mean():.4e} / {np.median(eh):.4e} / {eh.max():.4e}", flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in col.items()), flush=True)
    print("   row medians: " + " ".join(f"{v:.2e}" for v in row.values()), flush=True)
    for qq in (2.5, 2.0):
        v = [x for (q, c), x in zip(LQ, el) if q == qq]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}  " + " ".join(f"{x:.1e}" for x in v), flush=True)
    print(f"   wall {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
