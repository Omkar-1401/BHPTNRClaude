"""Re-score saved E-fitted networks (models/efit_<arch>_T96.pt) on T96 and V300 with common.score_ab -> results json.
Needed once: the first efit.py run lost its json to a key clash (models were saved before)."""
import json, re, sys
from multiprocessing import get_context
import numpy as np, torch
import common as C, efit as F

def main(log_path):
    txt = open(log_path).read()
    for arch in ("mlp", "gru"):
        f = F.MOD / f"efit_{arch}_T96.pt"
        if not f.exists():
            continue
        m = F.MLP() if arch == "mlp" else F.GRU(); m.load_state_dict(torch.load(f))
        steps = [int(x) for x in re.findall(rf"^{arch} step +(\d+)", txt, re.M)]
        secs = [float(x) for x in re.findall(rf"^{arch} step +\d+ +(\d+)s", txt, re.M)]
        hist = [(int(a), float(b), float(c)) for a, b, c in re.findall(rf"^{arch} step +(\d+) +(\d+)s +mean E (\S+)", txt, re.M)]
        with get_context("fork").Pool(96) as pool:
            ev = F.evaluate(m, pool, {"train": list(C.T96), "V300": list(C.V300)})
        npar = sum(p.numel() for p in m.parameters())
        print(arch, "T96", C.fmt(ev["train"]), "V300", C.fmt(ev["V300"]), flush=True)
        (F.RES / f"efit_{arch}_T96.json").write_text(json.dumps(dict(arch=arch, train_set="T96", n_params=npar,
            steps=max(steps) if steps else None, train_s=max(secs) if secs else None, hist=hist, **ev), indent=1))

if __name__ == "__main__":
    torch.set_num_threads(16); main(sys.argv[1])
