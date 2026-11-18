"""Safe, self-contained benchmark runner for unimol_accel.

    python -m unimol_accel.bench                    # quick preset, writes ./unimol_accel_bench/
    python -m unimol_accel.bench --preset standard
    python -m unimol_accel.bench --dry-run          # show the plan, run nothing

What it does / does not do
--------------------------
* Runs every configuration in a **fresh subprocess** with a per-run timeout and a global time budget.
* Uses a small built-in molecule panel and **random weights**: no network access, no downloads, no external data.
* Writes only into ``--output`` (results JSON + a self-contained HTML report). It never modifies your installs.
* Original and accelerated code are timed **interleaved in the same process** (A B A B ...), medians reported.
* Also checks equivalence (outputs, and gradients with dropout off); the exit status is non-zero if it fails.
* Touches the GPU only with ``--device cuda`` (and only if available).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = 1
PACKAGE = "unimol_accel"
DIST = "unimol-accel"

PRESETS = {
    "quick": dict(
        kinds=["inference", "train", "train-bucketed", "grad-check"],
        batch=[8, 32],
        train_batch=[8, 32],
        threads=["all"],
        repeats=3,
        check_batch=8,
    ),
    "standard": dict(
        kinds=["inference", "train", "train-bucketed", "grad-check"],
        batch=[4, 8, 16, 32, 48],
        train_batch=[8, 32],
        threads=["1", "all"],
        repeats=5,
        check_batch=8,
    ),
    "ci": dict(
        kinds=["inference", "train", "train-bucketed", "grad-check"],
        batch=[16, 32],
        train_batch=[16],
        threads=["all"],
        repeats=3,
        check_batch=8,
    ),
}


def host_info(device: str = "cpu") -> dict:
    info = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch

        info.update(
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            cuda=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name()
            if device == "cuda" and torch.cuda.is_available()
            else None,
        )
    except Exception as exc:  # pragma: no cover
        info["torch"] = f"unavailable: {exc!r}"
    for dist in ("unimol-tools", "unimol-accel", "numpy", "rdkit"):
        try:
            from importlib.metadata import version

            info[dist] = version(dist)
        except Exception:
            info[dist] = None
    return info


def _threads(spec, avail):
    out = []
    for s in spec:
        n = avail if str(s) == "all" else int(s)
        n = max(1, min(n, avail))
        if n not in out:
            out.append(n)
    return out


def build_plan(a, avail):
    p = dict(PRESETS[a.preset])
    for k in ("kinds", "batch", "threads", "repeats"):
        v = getattr(a, k, None)
        if v:
            p[k] = v
    plan = []
    for kind in p["kinds"]:
        batches = (
            p["train_batch"]
            if kind in ("train", "train-bucketed") and "train_batch" in p and not a.batch
            else p["batch"]
        )
        if kind == "grad-check":
            batches = [p.get("check_batch", 4)]
        for threads in _threads(p["threads"], avail):
            for b in batches:
                plan.append(
                    dict(
                        kind=kind,
                        batch=b,
                        threads=threads,
                        device=a.device,
                        repeats=p["repeats"],
                        seed=0,
                    )
                )
    return plan


def run_one(cfg, timeout):
    t0 = time.perf_counter()
    cmd = [sys.executable, "-W", "ignore", "-m", f"{PACKAGE}.bench._worker", json.dumps(cfg)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**cfg, "status": "timeout", "wall_s": time.perf_counter() - t0}
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("RESULT ")), None)
    if r.returncode != 0 or line is None:
        tail = (r.stderr or r.stdout)[-600:]
        if r.returncode in (-9, 137):
            tail = "killed (probably out of memory); use a smaller --batch. " + tail
        return {**cfg, "status": "error", "error": tail, "wall_s": time.perf_counter() - t0}
    res = json.loads(line[len("RESULT ") :])
    return {**cfg, **res, "status": "ok", "wall_s": time.perf_counter() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog=f"python -m {PACKAGE}.bench", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--preset", choices=PRESETS, default="quick")
    ap.add_argument(
        "--kinds", nargs="+", choices=["inference", "train", "train-bucketed", "grad-check"]
    )
    ap.add_argument("--batch", nargs="+", type=int)
    ap.add_argument("--threads", nargs="+", help="e.g. 1 2 all")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--name", help="label for this machine in the report (default: hostname)")
    ap.add_argument("--output", default=f"{PACKAGE}_bench", help="output directory (created)")
    ap.add_argument("--max-minutes", type=float, default=20.0, help="global time budget")
    ap.add_argument("--run-timeout", type=float, default=900.0, help="per-run timeout in seconds")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("CUDA requested but not available", file=sys.stderr)
            return 2
    if not a.dry_run:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            print(
                "The benchmark builds molecules/conformers from SMILES and needs RDKit: "
                f'pip install "{DIST}[bench]"  (or: pip install rdkit)',
                file=sys.stderr,
            )
            return 2
    avail = os.cpu_count() or 1
    plan = build_plan(a, avail)
    print(
        f"{PACKAGE} benchmark: {len(plan)} configuration(s), preset={a.preset}, device={a.device}"
    )
    for i, c in enumerate(plan, 1):
        print(
            f"  [{i}] {c['kind']:10s} batch={c['batch']:<3d} threads={c['threads']:<2d} repeats={c['repeats']}"
        )
    if a.dry_run:
        print("dry run: nothing executed")
        return 0

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or platform.node() or "machine"
    deadline = time.perf_counter() + a.max_minutes * 60
    runs, skipped = [], 0
    for i, c in enumerate(plan, 1):
        if time.perf_counter() > deadline:
            skipped += 1
            continue
        r = run_one(c, min(a.run_timeout, max(10.0, deadline - time.perf_counter())))
        runs.append(r)
        if r["status"] == "ok":
            msg = (
                f"speedup {r['speedup']:.2f}x"
                if "speedup" in r
                else f"rel err {r.get('rel_error', float('nan')):.1e}"
            )
            print(f"  [{i}] {c['kind']:10s} batch={c['batch']:<3d} {msg}")
        else:
            print(
                f"  [{i}] {c['kind']:10s} batch={c['batch']:<3d} {r['status']}: {str(r.get('error', ''))[-120:]}"
            )
    if skipped:
        print(f"time budget reached: {skipped} configuration(s) skipped")

    equivalence = [
        {
            "key": f"{r['kind']}/b{r['batch']}/t{r['threads']}",
            "passed": bool(r.get("passed")),
            "rel_error": r.get("rel_error"),
            "tolerance": r.get("tolerance"),
        }
        for r in runs
        if r["status"] == "ok" and "passed" in r
    ]
    doc = {
        "schema": SCHEMA,
        "package": PACKAGE,
        "name": name,
        "device": a.device,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": host_info(a.device),
        "preset": a.preset,
        "runs": runs,
        "equivalence": equivalence,
    }
    try:
        mod = __import__(PACKAGE)
        doc["status"] = mod.status()
    except Exception as exc:  # pragma: no cover
        doc["status"] = {"error": repr(exc)}
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    (out / f"results-{safe}.json").write_text(json.dumps(doc, indent=2, default=str))
    from .report import write_report

    html = write_report([doc], out / f"report-{safe}.html", title=f"{PACKAGE} benchmark: {name}")
    print(f"wrote {out / f'results-{safe}.json'} and {html}")
    bad = [e for e in equivalence if not e["passed"]]
    failed = [r for r in runs if r["status"] != "ok"]
    if bad:
        print("EQUIVALENCE FAILURE:", bad, file=sys.stderr)
    return 1 if bad or failed else 0


if __name__ == "__main__":
    sys.exit(main())
