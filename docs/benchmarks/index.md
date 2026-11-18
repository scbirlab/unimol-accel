# Benchmarks

## Run it yourself, safely

```bash
python -m unimol_accel.bench                    # quick preset, writes ./unimol_accel_bench/report-<host>.html
python -m unimol_accel.bench --dry-run          # show exactly what would run
python -m unimol_accel.bench --preset standard  # batch-size and thread sweeps
python -m unimol_accel.bench --device cuda      # also time a GPU, if present
```

Each configuration runs in a **fresh subprocess** with a timeout and a global time budget (`--max-minutes`). It uses a
built-in molecule panel and **random weights**: no network access and no downloads. It writes only into `--output`
(a results JSON and a self-contained HTML report) and never modifies your installs. Original and accelerated code are
timed **interleaved in the same process** (A B A B …) and medians reported, and equivalence is checked (outputs, and
gradients with dropout off): the exit status is non-zero if it fails. A configuration that is killed (for example by the
out-of-memory killer) is reported, not hidden.

Other scripts (`benchmarks/`): `bench_unimol.py` (direct), `sweep_batch_size.py`, `bench_aspect.py` (through Aspect).

## Report from the development sandbox

2-vCPU shared cloud VM (Intel Xeon @ 2.3 GHz), PyTorch 2.15, `--preset quick`:

| kind | batch | original → accel | speed-up |
| --- | ---: | --- | ---: |
| inference | 8 | see report | 1.04× |
| inference | 32 | see report | 1.68× |
| training step | 8 | see report | 1.26× |
| training step | 32 | see report | 1.71× |
| gradient check (dropout off) | 8 | relative error 1.2e-11 | pass |

*Noise:* this VM is shared and `quick` uses 3 repeats. An earlier identical run of the same code gave 1.22× / 1.91× / 1.01× / 1.71× for the same rows, so differences of ±20 % between runs are within noise; use `--preset standard` (5 repeats, sweeps) on your own machine.

```{raw} html
<p><a href="../_static/report-sandbox-2vcpu.html">Open the full HTML report (sandbox, 2 vCPU)</a></p>
```

CI additionally builds a combined report across runners (`docs.yml`, published at `benchmarks/report/` once the owner
enables GitHub Pages). Shared CI runners are noisy: treat them as a regression signal, not as performance claims.
