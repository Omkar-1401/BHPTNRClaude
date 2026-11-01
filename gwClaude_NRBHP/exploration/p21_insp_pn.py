"""EXPLORATION pass 21 -- 27c leader (pass 18) + PN-ordered inspiral terms, applied before the blend (factor 1 - W).

Passes 19-20: the +0.8 column and the q = 3, 8 rows lose in the inspiral; per-case inspiral terms halve the error and are
smooth in q.  Global form (every term x nu, so it vanishes at nu -> 0):
  A (5):  ln beta  += nu (g1 x + g2 chi1 x^1.5 + g3 chi1^2 x^2) (1 - W),   ln alpha += nu (h1 + h2 chi1) x (1 - W)
  B (7):  A + nu^2 (g4 x) in ln beta + nu^2 (h3 x) in ln alpha
x = flux-branch x (x_scale(q) min(x_pp, XCLIP)); W = the leader's blend.  Refit of all leader coefficients + new terms
on T96; T96, H270, low q at full resolution.  Usage: python p21_insp_pn.py A|B [budget_min].  Writes p21/p21_<form>.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FORM = sys.argv[1]; BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
sys.argv = [sys.argv[0], "V6", "p15", "1", "1"]
import p18_shapes2 as L                         # noqa: E402
import p3_rd as P3                              # noqa: E402
import fit_pn_anchored_spin as S                # noqa: E402
P6 = L.P6; A = L.A
OUT = HERE / "p21"; OUT.mkdir(exist_ok=True)
LEAD = json.loads((HERE / "p18" / "p18_p15_11.json").read_text())
TH0 = np.array(LEAD["theta"]); FREE0 = list(LEAD["free"]); NL = len(TH0)
NEW = {"A": ["g1", "g2", "g3", "h1", "h2"], "B": ["g1", "g2", "g3", "h1", "h2", "g4", "h3"]}[FORM]
_X = {}


def ab(th, k, key):
    a, b = L.ab(th[:NL], k, key)
    if a is None:
        return a, b
    q, chi1 = k; c = P6.get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    if (k, key) not in _X:
        if len(_X) > 400:
            _X.clear()
        _X[(k, key)] = S.PC.x_scale(q) * np.minimum(case[pre + "x"], S.XCLIP)
    x = _X[(k, key)]
    d = dict(zip(A.NAMES, th[:len(A.NAMES)])); g = dict(zip(NEW, th[NL:]))
    s = case[pre + "t_bhpt"] - c["tpk"]
    s1 = d["t1a"] + d["t1b"] * chi1; s2 = d["t2a"] + d["t2b"] * chi1
    w = 1 - P3.smooth5((s - s1) / (s2 - s1))
    nu = q / (1 + q) ** 2
    db = nu * (g["g1"] * x + g["g2"] * chi1 * x ** 1.5 + g["g3"] * chi1 ** 2 * x ** 2) + nu ** 2 * g.get("g4", 0.0) * x
    da = nu * (g["h1"] + g["h2"] * chi1) * x + nu ** 2 * g.get("h3", 0.0) * x
    return a * np.exp(da * w), b * np.exp(db * w)


P6.ab = ab


def main():
    t_start = time.time()
    pool = mp.get_context("fork").Pool(96)
    x0 = np.r_[TH0, np.zeros(len(NEW))]
    free = FREE0 + list(range(NL, len(x0)))
    print(f"form {FORM}: {len(free)}c (leader 27c + {len(NEW)})", flush=True)
    th, e = A.refit(pool, x0, free, BUDGET, f"insp{FORM}")
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    LQ = [(q, c) for q in (2.5, 2.0) for c in S.CHIS]
    el = np.array(pool.map(P6._full, [(list(th), k) for k in LQ]))
    col = {f"{ch:+.1f}": float(np.median([v for (q, c), v in zip(P6.T96, e96) if c == ch])) for ch in S.CHIS}
    row = {f"{q:.3f}": float(np.median([v for (qq, c), v in zip(P6.T96, e96) if qq == q])) for q in S.TRAIN_Q}
    out = dict(form=FORM, new=NEW, n_coef=len(free), theta=list(map(float, th)), free=free,
               T96=[float(e96.mean()), float(np.median(e96)), float(e96.max())],
               H270=[float(eh.mean()), float(np.median(eh)), float(eh.max())], col_median=col, row_median=row,
               lowq={f"{q:.1f}_{c:+.1f}": float(v) for (q, c), v in zip(LQ, el)},
               T96_cases={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in zip(P6.T96, e96)})
    (OUT / f"p21_{FORM}.json").write_text(json.dumps(out, indent=1))
    print(f"{len(free)}c: T96 {e96.mean():.4e} / {np.median(e96):.4e} / {e96.max():.4e} | "
          f"H270 {eh.mean():.4e} / {np.median(eh):.4e} / {eh.max():.4e}", flush=True)
    print("   new: " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(NEW, th[NL:])), flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in col.items()), flush=True)
    print("   row medians: " + " ".join(f"{v:.2e}" for v in row.values()), flush=True)
    for qq in (2.5, 2.0):
        v = [x for (q, c), x in zip(LQ, el) if q == qq]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}  " + " ".join(f"{x:.1e}" for x in v), flush=True)
    print(f"   wall {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
