"""Paradigm C follow-up for the power-weighted pointwise MLP (the best NN variant, chosen on validation E):
  (a) the saved model (trained on 572 pool cases) on the extrapolation sets of extrap.py (same points as pass 40);
  (b) the same architecture / loss / budget trained on EXACTLY T96 (pass 40's training set; all 96 used for training,
      the checkpoint is chosen on 8 of the TRAINING cases -- no held-out data), scored on V300 and the extrapolation sets.

    python nn_extra.py --minutes 25 --nproc 128      -> results/nn_mlp_extra.json, models/nn_mlp_w_T96.pt
"""
from __future__ import annotations
import argparse, json, time
from multiprocessing import get_context

import numpy as np
import common as C
import extrap as X
import nn_direct as N
import torch                                                   # noqa: E402  (after numpy)

RES = C.HERE / "results"; MOD = C.HERE / "models"


def predict(model, d, keys):
    dd = dict(d, keys=np.asarray(keys, float))
    return N.curves_from(model, "mlp", dd, np.arange(len(keys)), None)


def score(pool, keys, A, P):
    return np.array(pool.map(X._score_ml, [("direct", tuple(k), A[i], P[i]) for i, k in enumerate(keys)], chunksize=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--nproc", type=int, default=128); ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    log = lambda s: print(s, flush=True)
    S = X.sets(); out = {}
    d = N.load()
    # power weights from the training cases (as nn_direct)
    _, _, pw = N.pp_features(d["keys"][d["tr"]], a.nproc); W = pw.mean(0); W = W / W.mean()
    with get_context("fork").Pool(a.nproc) as pool:
        # (a) saved 572-case model
        m = N.MLP(); m.load_state_dict(torch.load(MOD / "nn_mlp_w.pt")); m.eval()
        for name, ks in S.items():
            A, P = predict(m, d, ks); e = score(pool, ks, A, P)
            out[f"mlpw|572|{name}"] = dict(C.stats(e), cases=e.tolist()); log(f"mlp-w N=572 {name:6s} {C.fmt(C.stats(e))}")
        # (b) train on exactly T96
        keys = d["keys"]; idx = {tuple(k): i for i, k in enumerate(map(tuple, keys))}
        tr = np.array([idx[tuple(k)] for k in C.T96])
        A_, P_ = d["A"], d["P"]
        mA, mP = A_[tr].mean(0), P_[tr].mean(0); sA, sP = (A_[tr] - mA).std(), (P_[tr] - mP).std()
        _, _, pw96 = N.pp_features(keys[tr], a.nproc); W96 = pw96.mean(0); W96 = W96 / W96.mean()
        d96 = dict(d, tr=tr, va=tr[:8], mA=mA, mP=mP, sA=sA, sP=sP)       # va only feeds the internal checkpoint log
        t = time.time()
        model, steps, secs = N.train("mlp", "w", d96, a.minutes, dict(w=W96, seq=None), log)
        torch.save(model.state_dict(), MOD / "nn_mlp_w_T96.pt")
        out["mlpw|T96|train_s"] = secs
        for name, ks in dict(S, V300=list(C.V300)).items():
            A, P = predict(model, d96, ks); e = score(pool, ks, A, P)
            out[f"mlpw|T96|{name}"] = dict(C.stats(e), cases=e.tolist()); log(f"mlp-w N=T96 {name:6s} {C.fmt(C.stats(e))}")
    (RES / "nn_mlp_extra.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
