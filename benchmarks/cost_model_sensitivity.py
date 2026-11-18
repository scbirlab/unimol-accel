"""Does the choice of cost-model constants matter in practice?  (random weights, offline, in-process interleaved timing)

    python benchmarks/cost_model_sensitivity.py [--batch 32] [--repeats 3]

For several token-length distributions, times batches under: the original forward, the shipped defaults, and
alternative constants (e.g. the fit printed by ``calibrate_cost_model.py``). Reports median ms per molecule and the
planned groups, so a change of constants that alters nothing in practice is visible as such.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import torch
from _um_common import make_model

import unimol_accel as ua

DISTS = {
    "uniform 12-80": lambda g, n: torch.randint(12, 81, (n,), generator=g),
    "bimodal 14-24 / 60-80": lambda g, n: torch.where(
        torch.rand(n, generator=g) < 0.7,
        torch.randint(14, 25, (n,), generator=g),
        torch.randint(60, 81, (n,), generator=g),
    ),
    "long tail": lambda g, n: (12 + (torch.rand(n, generator=g) ** 3) * 110).long(),
    "narrow 28-36": lambda g, n: torch.randint(28, 37, (n,), generator=g),
}
SETTINGS = {
    "defaults (pair 1/16, group 48)": {},
    "fit-like (pair 0.056, group 0)": {
        "UNIMOL_ACCEL_COST_PAIR": "0.056",
        "UNIMOL_ACCEL_COST_GROUP": "0",
    },
    "heavier overhead (group 150)": {"UNIMOL_ACCEL_COST_GROUP": "150"},
}


def make_batch(m, lengths):
    n, L = len(lengths), int(max(lengths))
    vocab = m.embed_tokens.num_embeddings
    tok = torch.randint(4, vocab - 1, (n, L))
    for i, l in enumerate(lengths.tolist()):
        tok[i, l:] = m.padding_idx
    return (
        tok,
        torch.rand(n, L, L),
        torch.randn(n, L, 3),
        torch.randint(0, vocab * vocab, (n, L, L)),
    )


def timed(m, b):
    t = time.perf_counter()
    with torch.no_grad():
        m(*b, return_repr=True)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(0)
    m = make_model().eval()
    g = torch.Generator().manual_seed(1)
    keys = [k for s in SETTINGS.values() for k in s]
    for name, draw in DISTS.items():
        lengths = draw(g, a.batch)
        b = make_batch(m, lengths)
        ts = {"original": []}
        ts.update({k: [] for k in SETTINGS})
        groups = {}
        for r in range(a.repeats + 1):
            ua.disable()
            t0 = timed(m, b)
            ua.enable()
            for label, env in SETTINGS.items():
                for k in keys:
                    os.environ.pop(k, None)
                os.environ.update(env)
                groups[label] = len(ua._patches.plan_buckets(lengths.tolist()))
                t = timed(m, b)
                if r:
                    ts[label].append(t)
            if r:
                ts["original"].append(t0)
        for k in keys:
            os.environ.pop(k, None)
        o = statistics.median(ts["original"])
        print(f"\n{name}  (lengths {int(lengths.min())}-{int(lengths.max())}, batch {a.batch})")
        print(f"  {'original':34s} {o / a.batch * 1e3:6.0f} ms/mol")
        for label in SETTINGS:
            t = statistics.median(ts[label])
            print(
                f"  {label:34s} {t / a.batch * 1e3:6.0f} ms/mol  {o / t:4.2f}x  groups={groups[label]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
