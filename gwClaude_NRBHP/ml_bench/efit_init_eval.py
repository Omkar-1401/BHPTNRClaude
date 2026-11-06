"""V300 of each efit_lin method at its INITIAL state (structure + values from the regression on model-free T96 alpha,
beta, before any E refinement) -> results/efit_lin_init.json.  Shows what the E refinement adds out of sample."""
import json
from multiprocessing import get_context
import numpy as np, torch
import common as C, efit as F, efit_lin as L

if __name__ == "__main__":
    torch.set_num_threads(16)
    keys = list(C.T96); X = F.pfeat(keys).numpy(); Y = L.init_targets(keys); out = {}
    with get_context("fork").Pool(96) as pool:
        for m in ("knn", "rbf", "gpr", "rf", "et", "xgb", "sparse"):
            mod, _ = L.build(m, X, Y); ev = F.evaluate(mod, pool, {"V300": list(C.V300)})["V300"]
            out[m] = ev; print(f"{m} init V300 {C.fmt(ev)}", flush=True)
    (F.RES / "efit_lin_init.json").write_text(json.dumps(out, indent=1))
