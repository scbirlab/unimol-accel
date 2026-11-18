"""Fit the length-bucketing cost model to *this* machine.

    python benchmarks/calibrate_cost_model.py [--threads N] [--repeats R]

Times the original (unpatched) Uni-Mol forward on random-weight batches of ``n`` molecules padded to ``L`` tokens.
1. ``t ≈ a·n·L + b·n·L²`` (least squares) gives ``pair_weight = b/a``.
2. The per-group overhead is measured *directly*: the same ``n`` molecules at the same ``L`` run once as one group
   and again split into k groups; the extra time per added group, in units of ``a·L``, is ``group_overhead``.
   (A regression term for it is not identifiable from whole-batch timings: it fits to ~0 and then over-splits.)
``unimol_accel.CostModel`` uses the same shape, ``n·(L + pair_weight·L²) + group_overhead·L``. The script prints the
fit quality and the environment variables that apply the result. Nothing is written to disk. Fits differ between CPUs,
thread counts and torch builds: that is the point, and why the shipped constants (1/16, 48) are only defaults.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import numpy as np
import torch
from _um_common import make_model

import unimol_accel as ua


def nnls(A, y, iters=20000):
    """Tiny projected-gradient NNLS (3 unknowns; avoids a scipy dependency)."""
    A = A / A.max(axis=0)
    scale = np.linalg.norm(A, 2) ** 2
    x = np.zeros(A.shape[1])
    for _ in range(iters):
        x = np.maximum(0.0, x - (A.T @ (A @ x - y)) / scale)
    return x


def median_time(fn, repeats):
    ts = []
    for _ in range(repeats + 1):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts[1:])[len(ts[1:]) // 2]


def group_overhead_at(m, vocab, L, a_, repeats, n=32, k=4):
    """Extra time per added group (in units of a*L) when n molecules at length L are split into k groups."""
    tok = torch.randint(4, vocab - 1, (n, L))
    dist, coord = torch.rand(n, L, L), torch.randn(n, L, 3)
    edge = torch.randint(0, vocab * vocab, (n, L, L))

    def run(lo, hi):
        with torch.no_grad():
            m(tok[lo:hi], dist[lo:hi], coord[lo:hi], edge[lo:hi], return_repr=True)

    t1 = median_time(lambda: run(0, n), repeats)
    tk = median_time(lambda: [run(i * n // k, (i + 1) * n // k) for i in range(k)], repeats)
    extra = (tk - t1) / (k - 1) / (a_ * L)
    print(
        f"L={L:3d}: 1 group {t1 * 1e3:7.1f} ms, {k} groups {tk * 1e3:7.1f} ms -> overhead {extra:.0f} (units of a*L)"
    )
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--threads", type=int, default=0, help="torch threads (default: torch's default)"
    )
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    ua.disable()  # time the original forward
    m = make_model().eval()
    vocab = m.embed_tokens.num_embeddings
    rows, times = [], []
    for n in (1, 2, 4, 8, 16, 32):
        for L in (16, 32, 48, 64, 96, 128):
            tok = torch.randint(4, vocab - 1, (n, L))
            dist = torch.rand(n, L, L)
            coord = torch.randn(n, L, 3)
            edge = torch.randint(0, vocab * vocab, (n, L, L))
            ts = []
            for _ in range(a.repeats + 1):
                t = time.perf_counter()
                with torch.no_grad():
                    m(tok, dist, coord, edge, return_repr=True)
                ts.append(time.perf_counter() - t)
            rows.append((n * L, n * L * L, L))
            times.append(sorted(ts[1:])[len(ts[1:]) // 2])
            print(f"n={n:2d} L={L:3d}  {times[-1] * 1e3:8.1f} ms", flush=True)
    A, y = np.array(rows, float), np.array(times)
    A2 = A[:, :2]
    coef = nnls(A2, y) / A2.max(axis=0)  # nnls() works on column-normalised A
    pred = A2 @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    a_, b_ = coef
    print(
        f"\nfit: t = {a_:.3e}*n*L + {b_:.3e}*n*L^2    R^2 = {r2:.3f}   threads={torch.get_num_threads()}"
    )
    if a_ <= 0:
        print(
            "linear term fitted to zero: the fit is not usable on this machine; keep the defaults"
        )
        return 1
    pw = b_ / a_

    # direct per-group overhead: one group of n versus k groups of n/k at the same L
    extras = [group_overhead_at(m, vocab, L, a_, a.repeats) for L in (24, 48, 80)]
    go = max(0.0, sorted(extras)[len(extras) // 2])
    d = ua._patches.CostModel()
    print(
        f"\npair_weight    = {pw:.4f}   (default {d.pair_weight:.4f} = 1/{1 / d.pair_weight:.0f})"
    )
    print(
        f"group_overhead = {go:.1f}     (default {d.group_overhead:.0f}; noisy: compare with cost_model_sensitivity.py)"
    )
    print("\nApply with:")
    print(f"  export UNIMOL_ACCEL_COST_PAIR={pw:.4f} UNIMOL_ACCEL_COST_GROUP={go:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
