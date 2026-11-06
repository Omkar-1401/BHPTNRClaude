"""Re-fit only the two failed Fig. 10 points of split B (2026-10-15): chi1 = -0.5, -0.4 at t_end = -100 came out ~4x
above t_end = -50 (optimiser stuck; run 1 started every window from the installed coefficients only).  Here each is also
started from its own t_end = -50 refit, as fig10_ourmodels.py chains.  Every other point is copied from run 1.
-> results/diag_AB/tend_split_A_slim_T-2.json (run 1 unchanged), tend_split_B_b0_T-3.json (2 points replaced)"""
import json
from multiprocessing import get_context
import diag_split_AB as G

R1 = json.loads((G.OUTR / "tend_run1.json").read_text())["results"]
FIX = [(-0.5, -100.0), (-0.4, -100.0)]
Dm = G.models(); name = "split_B_b0_T-3"; d = Dm[name]
for k in G.Q4 + list(G.X.T96):
    G.M.CASES[k] = G.M.prepare(*k)
G.set_win(d["window"]); G.MOD["m"] = d
G.RUN1.update({(r["q"], r["chi1"], r["t_end"]): r for r in R1[name]})
prev = {c: next(r["params"] for r in R1[name] if r["chi1"] == c and r["t_end"] == -50.0) for c, _ in FIX}
with get_context("fork").Pool(len(FIX)) as pool:
    new = pool.map(G.work_tend, [((4.0, c), te, prev[c]) for c, te in FIX])
rows = [r for r in R1[name] if (r["chi1"], r["t_end"]) not in FIX] + new
for r in new:
    old = G.RUN1[(4.0, r["chi1"], r["t_end"])]["E_refit"]
    print(f"chi1 {r['chi1']:+.1f} t_end {r['t_end']:.0f}: run 1 {old:.3e} -> {r['E_refit']:.3e} ({r['seconds']}s)", flush=True)
for n, rr in ((name, rows), ("split_A_slim_T-2", R1["split_A_slim_T-2"])):
    (G.OUTR / f"tend_{n}.json").write_text(json.dumps(dict(T_START=G.T_START, results={n: rr})))
print("wrote per-model tend files", flush=True)
