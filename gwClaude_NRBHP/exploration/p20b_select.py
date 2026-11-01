"""EXPLORATION pass 20b (login arithmetic): greedy choice of regression terms for the pass-20 per-case inspiral corrections.
Per-case functions  dlnB(x) = d0 + d1 x^1.5 + d2 x^2,  dlnA(x) = d3 + d4 x  on x in [0.07, 0.22]; basis f(x) g(nu, chi1),
g in {nu, nu^2} x {1, chi1, chi1^2, chi1^3} (vanishes at nu -> 0).  Forward selection on the rms of the function values."""
import json, itertools, numpy as np
R = json.load(open('p20/p20_insp_floor.json'))
q = np.array([r['q'] for r in R]); c = np.array([r['chi1'] for r in R]); P = np.array([r['p'] for r in R])
nu = q / (1 + q) ** 2; X = np.linspace(0.07, 0.22, 16)
G = {f'{nn}*chi^{k}': (nu ** p) * c ** k for (nn, p) in (('nu', 1), ('nu2', 2)) for k in range(4)}
for field, F, fx in (('beta', P[:, 0:1] + P[:, 1:2] * X ** 1.5 + P[:, 2:3] * X ** 2, {'1': X * 0 + 1, 'x^1.5': X ** 1.5, 'x^2': X ** 2}),
                     ('alpha', P[:, 3:4] + P[:, 4:5] * X, {'1': X * 0 + 1, 'x': X})):
    y = F.ravel(); cands = {f'{a}|{b}': (np.outer(gv, fv)).ravel() for a, fv in fx.items() for b, gv in G.items()}
    chosen = []; print(field, 'rms of target', np.sqrt(np.mean(y ** 2)).round(5))
    for step in range(8):
        best = None
        for k, v in cands.items():
            if k in chosen: continue
            M = np.column_stack([cands[j] for j in chosen] + [v]); co, *_ = np.linalg.lstsq(M, y, rcond=None)
            r = np.sqrt(np.mean((y - M @ co) ** 2))
            if best is None or r < best[0]: best = (r, k)
        chosen.append(best[1]); print(f'   +{best[1]:16s} rms {best[0]:.5f}')
