"""Throughput (ms/molecule) of original vs unimol_accel across batch sizes: random weights, offline.
    python benchmarks/sweep_batch_size.py
Original Uni-Mol is already fastest at small batches; accel mainly removes the padding penalty of large ones."""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import torch
from _um_common import make_model, make_samples
import unimol_accel as ua

torch.manual_seed(0)
m = make_model().eval()
S = make_samples(48)


def run(bs):
    for i in range(0, len(S), bs):
        batch, _ = m.batch_collate_fn(S[i : i + bs])
        with torch.no_grad():
            m(**batch, return_repr=True)


for bs in (4, 8, 16, 32, 48):
    r = {}
    for _k, sw in (("orig", ua.disable), ("accel", ua.enable)):
        sw()
        run(bs)
    ts = {"orig": [], "accel": []}
    for _ in range(3):
        for k, sw in (("orig", ua.disable), ("accel", ua.enable)):
            sw()
            t = time.perf_counter()
            run(bs)
            ts[k].append(time.perf_counter() - t)
    o, a = statistics.median(ts["orig"]), statistics.median(ts["accel"])
    print(
        f"batch_size={bs:2d}  orig {o / 48 * 1e3:5.0f} ms/mol  accel {a / 48 * 1e3:5.0f} ms/mol  {o / a:.2f}x",
        flush=True,
    )
