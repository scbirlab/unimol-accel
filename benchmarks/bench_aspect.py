#!/usr/bin/env python
"""Offline end-to-end benchmark through scbirlab/aspect: ("smiles", ["conformers", "unimol"]).

Random-weight checkpoint in a temp dir; Aspect's Hugging Face download is redirected to it, so nothing is
downloaded and nothing outside the temp dir is written. Conformer generation (not accelerated) is timed
separately; original / accelerated runs are interleaved. Requires: pip install aspect-data[unimol]
    python benchmarks/bench_aspect.py --n 48 --batch-size 32 [--batch-size 8 ...] [--json out.json]
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from mols import SMILES
from _common import make_model, make_dict
import unimol_accel as ua


def build_hub(root):
    d = make_dict()
    with open(os.path.join(root, "mol.dict.txt"), "w") as f:
        f.write("".join(f"{s} {c}\n" for s, c in zip(d.symbols, d.count) if s != "[MASK]"))
    torch.save(
        {"model": make_model(seed=3).state_dict()}, os.path.join(root, "mol_pre_all_h_220816.pt")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, action="append")
    ap.add_argument("--json")
    a = ap.parse_args()
    sizes = a.batch_size or [32]
    import huggingface_hub
    import aspect.transform.unimol as au
    from aspect.transform.conformers import Conformers

    out_json = {"n": a.n, "threads": torch.get_num_threads(), "runs": {}}
    with tempfile.TemporaryDirectory() as hub:
        build_hub(hub)
        huggingface_hub.hf_hub_download = lambda repo, name, **kw: os.path.join(hub, name)
        t = time.perf_counter()
        rows = Conformers()({"smiles": SMILES[: a.n]}, "smiles")
        t_conf = time.perf_counter() - t
        print(
            f"molecules {a.n}  threads {torch.get_num_threads()}  conformer generation (not accelerated): {t_conf:.2f} s ({t_conf / a.n * 1e3:.0f} ms/molecule)"
        )
        for bs in sizes:

            def embed(accel, bs=bs):
                (ua.enable if accel else ua.disable)()
                au._load_unimol_cached.cache_clear()
                fn = au.UniMolEmbedding(conformers="aspect", batch_size=bs, device="cpu")
                fn({"c": rows[:1]}, "c")
                t = time.perf_counter()
                out = fn({"c": rows}, "c")
                return time.perf_counter() - t, out

            embed(False)
            embed(True)
            res = {"original": [], "accel": []}
            for _ in range(a.repeats):
                for k, acc in (("original", False), ("accel", True)):
                    dt, out = embed(acc)
                    res[k].append(dt)
                    res[k + "_out"] = out
            mo, ma = statistics.median(res["original"]), statistics.median(res["accel"])
            err = float(
                np.linalg.norm(res["accel_out"] - res["original_out"])
                / max(np.linalg.norm(res["original_out"]), 1e-30)
            )
            print(
                f"batch_size={bs:3d}  embedding: original {mo / a.n * 1e3:6.0f} ms/mol  accel {ma / a.n * 1e3:6.0f} ms/mol  {mo / ma:.2f}x   rel err {err:.1e}"
            )
            out_json["runs"][bs] = {
                "original_s": res["original"],
                "accel_s": res["accel"],
                "rel_error": err,
            }
        ua.enable()
    out_json["t_conformers_s"] = t_conf
    if a.json:
        json.dump(out_json, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
