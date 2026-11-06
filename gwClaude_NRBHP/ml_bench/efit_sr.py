"""Paradigm E, functional route: closed forms for ln alpha and ln(beta / X1) in (u, nu, chi1) found by PySR, then their
constants refitted through E on T96 exactly as the hand forms are (efit.train: same loss, same t0 handling).

1. data: the trained E-fitted MLP (models/efit_mlp_T96.pt) evaluated on the T96 cases at every 3rd point of efit.S_C
   (symbolic distillation of the black box; PySR cannot evaluate E itself -- one loss evaluation is a full waveform map);
2. PySR per output: + - * /, square cube sqrt exp; maxsize --maxsize; --minutes per search;
3. candidates: per output the lowest-loss expression at complexity <= 12, 20, 28, maxsize, paired by rank (<= 4 pairs);
   each pair's constants refitted through E on T96 for --refit-minutes;
4. choice: lowest T96 mean E after the refit (the training objective, as for the hand forms); V300 scored for every
   candidate to show the complexity / accuracy trade-off, never used for the choice.

    python efit.py sr --minutes 12 --refit-minutes 5 --procs 120     -> results/efit_sr_T96.json, models/efit_sr_T96.pt
"""
from __future__ import annotations
import json, time
from multiprocessing import get_context

import numpy as np
import torch

import common as C
import efit as F

SUB = np.arange(0, len(F.S_C), 3)


def _raw(p):                               # nn_direct.nu_feat inverse -> nu, chi1
    nu = (p[:, 0] + 1) / 2 * (3 / 16 - 8 / 81) + 8 / 81; chi = p[:, 1] * 0.7 + 0.1
    return nu, chi


class SRModel(torch.nn.Module):
    """two sympytorch modules (PySR .pytorch()) -> the efit model interface (B, len(S_C), 2)"""
    def __init__(self, fa, fb):
        super().__init__(); self.fa, self.fb = fa.double(), fb.double()

    def forward(self, p):
        nu, chi = _raw(p); B, L = len(p), len(F.U_C)
        u = torch.tensor(F.U_C)[None, :].expand(B, L).reshape(-1)
        X = torch.stack([u, nu[:, None].expand(B, L).reshape(-1), chi[:, None].expand(B, L).reshape(-1)], -1)
        ya = self.fa(X); yb = self.fb(X)
        ya = ya.expand(B * L) if ya.dim() == 0 or ya.numel() == 1 else ya
        yb = yb.expand(B * L) if yb.dim() == 0 or yb.numel() == 1 else yb
        return torch.stack([ya.reshape(B, L), yb.reshape(B, L)], -1)


def candidates(m, maxsize):
    eq = m.equations_; out = []
    for cmax in (12, 20, 28, maxsize):
        sel = eq[eq["complexity"] <= cmax]
        if len(sel):
            out.append(int(sel["loss"].idxmin()))
    return list(dict.fromkeys(out))


def main(a):
    log = lambda s: print(s, flush=True); torch.set_num_threads(a.threads)
    keys = list(C.T96)
    mlp = F.MLP(); mlp.load_state_dict(torch.load(F.MOD / f"efit_{a.src}.pt"))
    tag = "sr_T96" + ("" if a.src == "mlp_T96" else "_from_" + a.src)
    with torch.no_grad():
        Y = mlp(F.pfeat(keys))[:, SUB].numpy()                      # (96, n, 2)
    nu, chi = _raw(F.pfeat(keys).numpy())
    X = np.column_stack([np.tile(F.U_C[SUB], len(keys)), np.repeat(nu, len(SUB)), np.repeat(chi, len(SUB))])
    from pysr import PySRRegressor                                  # Julia starts here (after numpy / torch)
    srs, out = [], dict(n_rows=len(X), outputs=[])
    for j, name in enumerate(("ln_alpha", "ln_beta_over_X1")):
        t0 = time.time()
        m = PySRRegressor(niterations=100000, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["square", "cube", "sqrt", "exp"], maxsize=a.maxsize,
                          populations=3 * a.procs // 4, procs=a.procs, parallelism="multiprocessing",
                          model_selection="accuracy", variable_names=["u", "nu", "chi"], random_state=20261031 + j,
                          verbosity=0, progress=False, temp_equation_file=True, timeout_in_seconds=a.minutes * 60)
        m.fit(X, Y[..., j].reshape(-1)); srs.append(m)
        rel = m.equations_["loss"].values / np.var(Y[..., j])
        log(f"PySR {name}: {len(m.equations_)} on the front, {time.time() - t0:.0f} s; best rel. MSE {rel.min():.2e}")
        out["outputs"].append(dict(name=name, front=[dict(complexity=int(r.complexity), rel_mse=float(r.loss / np.var(Y[..., j])),
                                                          expr=str(r.sympy_format)) for r in m.equations_.itertuples()]))
    ca, cb = candidates(srs[0], a.maxsize), candidates(srs[1], a.maxsize)
    pairs = [(ca[min(i, len(ca) - 1)], cb[min(i, len(cb) - 1)]) for i in range(max(len(ca), len(cb)))]
    pairs = list(dict.fromkeys(pairs)); log(f"candidate pairs {pairs}")
    best, res = None, []
    with get_context("fork").Pool(a.nproc) as pool:
        for ia, ib in pairs:
            model = SRModel(srs[0].pytorch(index=ia), srs[1].pytorch(index=ib))
            e_dist = F.evaluate(model, pool, {"T96": keys})["T96"]
            info = F.train(model, keys, a.refit_minutes, pool, log, lr=a.lr, tag=f"sr[{ia},{ib}]")
            ev = F.evaluate(model, pool, {"T96": keys})["T96"]
            cx = int(srs[0].equations_.loc[ia, "complexity"] + srs[1].equations_.loc[ib, "complexity"])
            n_const = sum(p.numel() for p in model.parameters())
            # V300 for EVERY candidate: the complexity / accuracy trade-off (reported only; the choice below uses T96)
            r = dict(pair=[ia, ib], complexity=cx, n_constants=n_const, T96_distilled=e_dist["mean"], T96=ev, steps=info["steps"],
                     V300=F.evaluate(model, pool, {"V300": list(C.V300)})["V300"],
                     expr_alpha=str(srs[0].equations_.loc[ia, "sympy_format"]), expr_beta=str(srs[1].equations_.loc[ib, "sympy_format"]))
            log(f"pair {ia},{ib}: complexity {cx}, {n_const} constants; T96 mean E distilled {e_dist['mean']:.3e} -> refit {C.fmt(ev)}"
                f"  V300 {C.fmt(r['V300'])}")
            res.append(r)
            if best is None or ev["mean"] < best[0]:
                best = (ev["mean"], r, model)
        _, r, model = best
        torch.save(model.state_dict(), F.MOD / f"efit_{tag}.pt")
        aV, bV = F.ab_numpy(model, list(C.V300))                  # for plots (the model needs the PySR objects to rebuild)
        np.savez_compressed(F.RES / f"efit_{tag}_abV300.npz", alpha=aV.astype(np.float32), beta=bV.astype(np.float32), s=F.S_C)
        # constants after the refit, as text (the refitted expressions)
        r["constants_refit"] = [float(x) for p in model.parameters() for x in p.detach().reshape(-1)]
    log(f"SR chosen pair {r['pair']} (complexity {r['complexity']}): T96 {C.fmt(r['T96'])}  V300 {C.fmt(r['V300'])}")
    out.update(src=a.src, candidates=res, chosen=r)
    (F.RES / f"efit_{tag}.json").write_text(json.dumps(out, indent=1))
