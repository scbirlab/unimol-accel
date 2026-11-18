"""Worker: runs in a fresh subprocess, prints one ``RESULT {json}`` line."""

from __future__ import annotations

import json
import statistics
import sys
import time
import warnings


def _sync(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def _no_dropout(model):
    import torch.nn as nn

    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0
        if hasattr(mod, "emb_dropout"):
            mod.emb_dropout = 0.0


def main() -> int:
    # third-party import-time deprecation noise (e.g. torch.jit.script via e3nn) would flood CI logs
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = json.loads(sys.argv[1])
    import torch

    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg.get("seed", 0))
    import unimol_accel as ua
    from ._models import make_model, make_samples

    dev = cfg["device"]
    model = make_model().to(dev)
    batch, _ = model.batch_collate_fn(make_samples(cfg["batch"]))
    batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    lens = (batch["src_tokens"] != model.padding_idx).sum(1)
    out = {
        "token_len": {
            "min": int(lens.min()),
            "max": int(lens.max()),
            "mean": float(lens.float().mean()),
        }
    }

    def cls(o):
        return o["cls_repr"] if isinstance(o, dict) else o

    def infer():
        model.eval()
        with torch.no_grad():
            return cls(model(**batch, return_repr=True)).detach().clone()

    def train():
        model.train()
        model.zero_grad()
        model(**batch).float().pow(2).mean().backward()
        model.eval()

    kind = cfg["kind"]
    if kind == "grad-check":
        import os

        _no_dropout(model)

        def grads_of(enable):
            (ua.enable if enable else ua.disable)()
            model.train()
            model.zero_grad()
            model(**batch).float().pow(2).sum().backward()
            return torch.cat(
                [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            ).clone()

        ref = grads_of(False)
        errs = {}
        for label, flag in (("default", "0"), ("train_bucketing", "1")):
            os.environ["UNIMOL_ACCEL_BUCKET_TRAIN"] = flag
            errs[label] = float((grads_of(True) - ref).norm() / ref.norm())
        err = max(errs.values())
        out.update(rel_error=err, rel_error_by_mode=errs, tolerance=1e-4, passed=err < 1e-4)
        print("RESULT " + json.dumps(out))
        return 0

    fn = infer if kind == "inference" else train
    if kind == "train-bucketed":
        # opt-in mode: bucketing in training changes dropout-mask grouping (see README)
        import os

        os.environ["UNIMOL_ACCEL_BUCKET_TRAIN"] = "1"
    ua.disable()
    ref = fn()
    ua.enable()
    new = fn()  # warm-up both paths
    times = {"original": [], "accel": []}
    for _ in range(cfg["repeats"]):
        for label, sw in (("original", ua.disable), ("accel", ua.enable)):
            sw()
            _sync(torch, dev)
            t = time.perf_counter()
            fn()
            _sync(torch, dev)
            times[label].append(time.perf_counter() - t)
    ua.enable()
    out.update(
        original_s=times["original"],
        accel_s=times["accel"],
        speedup=statistics.median(times["original"]) / statistics.median(times["accel"]),
    )
    if kind == "inference":
        err = float((new - ref).norm() / ref.norm())
        out.update(rel_error=err, tolerance=1e-5, passed=err < 1e-5)
    print("RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
