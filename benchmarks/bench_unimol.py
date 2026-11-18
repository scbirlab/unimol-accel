#!/usr/bin/env python
"""Safe, local benchmark: original unimol_tools vs unimol_accel (random weights).

No network access, no downloads, no writes except the optional --json file.
    python benchmarks/bench_unimol.py --batch 32 --repeats 5 [--device cuda] [--json out.json]
Configurations are interleaved (A B A B ...) and medians reported, to be robust to drift/noise.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import torch
from _common import make_model, make_samples
import unimol_accel as ua


def timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    m = make_model().to(dev).eval()
    batch, _ = m.batch_collate_fn(make_samples(a.batch))
    batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    lens = (batch["src_tokens"] != m.padding_idx).sum(1)

    def infer():
        with torch.no_grad():
            m(**batch, return_repr=True)

    def train():
        m.train()
        m.zero_grad()
        m(**batch).float().pow(2).mean().backward()
        m.eval()

    res = {"inference": {"original": [], "accel": []}, "train_step": {"original": [], "accel": []}}
    for name, fn in (("inference", infer), ("train_step", train)):
        ua.disable()
        fn()
        ua.enable()
        fn()  # warm-up both
        for _ in range(a.repeats):
            for label, sw in (("original", ua.disable), ("accel", ua.enable)):
                sw()
                res[name][label].append(timed(fn, dev))
    ua.enable()
    out = {
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "threads": torch.get_num_threads(),
            "device": dev,
            "gpu": torch.cuda.get_device_name() if dev == "cuda" else None,
            "torch": torch.__version__,
        },
        "batch": a.batch,
        "token_len": {
            "min": int(lens.min()),
            "max": int(lens.max()),
            "mean": float(lens.float().mean()),
        },
        "status": ua.status()["patches"],
    }
    for k, v in res.items():
        mo, ma = statistics.median(v["original"]), statistics.median(v["accel"])
        out[k] = {"original_s_median": mo, "accel_s_median": ma, "speedup": mo / ma, "raw": v}
        print(
            f"{k:11s} original {mo * 1e3:8.0f} ms   accel {ma * 1e3:8.0f} ms   speedup {mo / ma:.2f}x  (median of {a.repeats})"
        )
    print("NOTE: results are specific to this machine, batch composition and thread count.")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
