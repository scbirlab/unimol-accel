# unimol-accel

<p align="left">
<a href="https://github.com/scbirlab/unimol-accel/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/scbirlab/unimol-accel/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://scbirlab.github.io/unimol-accel/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-blue"></a>
<a href="https://scbirlab.github.io/unimol-accel/benchmarks/report/"><img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-live%20report-informational"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.14-blue">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>


Drop-in **CPU** acceleration for [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) (`unimol_tools`), activated by one import. Pure PyTorch, no compiled code, no change to weights, checkpoints or `state_dict` keys. Works under [scbirlab/aspect](https://github.com/scbirlab/aspect) (`unimol` featuriser) without editing Aspect.

```python
import unimol_accel  # before or after unimol_tools; patches are applied on import
import unimol_tools  # ordinary code, unchanged

unimol_accel.status()  # which patches are active, and why any is not
unimol_accel.disable()  # restore the original methods
```
`UNIMOL_ACCEL_DISABLE=1` prevents automatic activation.

## Use with Aspect (or any program you don't want to edit)

```bash
pip install unimol-accel "aspect-data[unimol]"
python -m unimol_accel -m aspect.cli_module.cli featurize compounds.csv \
    --features "smiles:conformers:unimol@um" --output features.parquet
python -m unimol_accel --also atom_jepa_accel script.py          # activate both accel packages
python -m unimol_accel -m pytest path/to/aspect/test             # run Aspect's own tests with accel on
```
Both Aspect input modes go through the same patched `UniMolModel.forward`: pre-generated conformers (`conformers="aspect"`, `UniMolRepr.get_repr({"atoms":…, "coordinates":…})`) and SMILES (`conformers="unimol"`, Uni-Mol's own RDKit conformer step). The Aspect-in-the-loop tests in `tests/integration/` cover both, plus multiple conformers per molecule spanning batches and missing/invalid rows.

## What it does

| Patch | Effect | Exact? |
|---|---|---|
| Length-bucketed sub-batching inside `UniMolModel.forward` | Molecules in a batch are grouped by true token length (optimal DP partition), each group runs at its own padded length, outputs return in the original order. Padding dominates pair-bias cost (L² logits per head per layer). Falls back to the original path when it would not pay. | Yes: per-sample maths is independent (LayerNorm only) |
| Skip the unused encoder tail | `delta_pair_repr`/norm losses are computed by the encoder but discarded by `UniMolModel.forward` | Yes |
| Lean pair-bias attention (`baddbmm`, no clone) | Fewer passes over the (B·H, L, L) logits | Bit-identical in fp32 |

Each patch is guarded by a hash of the upstream source. **Verified against:** `unimol_tools` 0.1.6 (PyPI) and Uni-Mol repo commit `90f52c4` (0.1.3.post1). Anything else keeps the original code and `status()` says why. Outputs of both known `UniMolModel.forward` shapes are handled (a bare `cls_repr` tensor in ≥ 0.1.6, a dict in older versions, plus atomic representations).

Switches: `UNIMOL_ACCEL_BUCKET=0` (no bucketing), `UNIMOL_ACCEL_BUCKET_TRAIN=0` (none in training), `UNIMOL_ACCEL_BUCKET_MIN_BATCH` (default 8).

**Training note.** Bucketing is on in training. Gradients match the original (tested with dropout off), but dropout masks are drawn per sub-batch, so runs are not bitwise reproducible against the unpatched model under the same seed.

## Measured results — read the second table before the first

Hardware: **2 vCPU cloud VM (Intel Xeon @ 2.3 GHz), 7 GB RAM**, PyTorch 2.15.0, random weights, 50 drug-like molecules (token length 17–78). Interleaved runs, medians. Not generalisable to other machines.

**Same batch size, original vs accel (batch of 32):** inference 1778 → 865 ms (**2.06×**), training step 6790 → 3980 ms (**1.71×**).

**But the original is already fast at small batches.** ms per molecule, inference, 48 molecules:

| batch size | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|
| original | 36 | 37 | 41 | 52 | 75 |
| unimol-accel | 36 | 36 | 30 | 33 | 28 |

The speedup is therefore mostly the removal of the padding penalty that large ragged batches carry. If you already use small batches (4–8) you gain ≈1.0–1.1×; best accel vs best original here is **≈1.3×** (36 → 28 ms/mol). Through Aspect (`unimol`, pre-generated conformers, 48 molecules): 43/42/54/76 ms/mol original vs 42/39/31/29 accel at batch 4/8/32/48 (1.02×/1.09×/1.75×/2.62×); Aspect's default `batch_size=32` gives 1.75×. A random-composition sweep (B = 8–48) never ran slower than the original (min 0.98×, i.e. no split chosen) and gained 1.3–2.8× on ragged batches.

**Amdahl.** Conformer generation (RDKit ETKDG + MMFF) is not accelerated and cost ≈27 ms/molecule here, comparable to the accelerated embedding. In Aspect, `conformers(n_threads=…)` or caching conformers is the next lever.

Run-to-run noise on this shared VM was ±10–15 %.

## Negative results

- A fused C++ pair-bias attention kernel (forward, backward, hash dropout) was **slower than eager PyTorch** (0.2–0.7×) even after row-blocking and templated head size (head_dim = 8 is too small to beat MKL batched GEMM); not shipped, kept in `experiments/native_pair_attention/`.
- Lean attention + tail skip alone are within noise end to end (0.98–1.04×).

## Limits of what was verified

- **Real pretrained weights could not be downloaded** in the environment where this was developed (Hugging Face blocked), so all equivalence tests use random weights saved in the real on-disk format and exercise the real `UniMolRepr`/Aspect code. The maths is exact, so real weights are expected to behave the same, but please run `python -m unimol_accel -m pytest path/to/aspect/test` on a machine with the weights cached.
- Many-core CPUs and CUDA were not measured.

## Tests and benchmarks

```bash
pip install -e .[test]            # plus: pip install "aspect-data[unimol]" for tests/integration
pytest tests                       # 23 tests: outputs, atomic reprs, gradients, idempotence, hash guard, Aspect (both input modes)
python benchmarks/bench_unimol.py --batch 32 --repeats 5 [--device cuda] [--json out.json]
python benchmarks/sweep_batch_size.py
python benchmarks/bench_aspect.py --n 48 --batch-size 8 --batch-size 32
```
Benchmarks use random weights, make no network access and write only the optional JSON.

## Benchmark it yourself, offline

```bash
python -m unimol_accel.bench               # quick preset, ~2-5 min; writes ./unimol_accel_bench/report-<host>.html
python -m unimol_accel.bench --dry-run     # show exactly what would run
python -m unimol_accel.bench --preset standard [--device cuda]
```
Each configuration runs in a fresh subprocess with a timeout and a global time budget; random weights, no network, writes only into
`--output`; original and accelerated code are timed interleaved, and the exit status is non-zero if outputs or gradients disagree.

## Documentation, contributing, CI

**Docs:** <https://scbirlab.github.io/unimol-accel/> · [Optimisations](https://scbirlab.github.io/unimol-accel/optimisations.html) · [Aspect usage](https://scbirlab.github.io/unimol-accel/aspect.html) · [Compatibility](https://scbirlab.github.io/unimol-accel/compatibility.html) ·
**Benchmarks:** [methodology and results](https://scbirlab.github.io/unimol-accel/benchmarks/index.html) · [live multi-runner report (CI, latest main)](https://scbirlab.github.io/unimol-accel/benchmarks/report/)
(these URLs work once the owner enables GitHub Pages and `docs.yml` has run).

- Docs (Sphinx, pydata theme, modelled on [scbirlab/chemprop-accel](https://github.com/scbirlab/chemprop-accel)): `docs/`;
  build with `pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html`. It builds warning-free here; publishing to
  GitHub Pages is done by `.github/workflows/docs.yml` once the owner enables Pages.
- [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), [CHANGELOG.md](CHANGELOG.md), [SECURITY.md](SECURITY.md).
- CI (`.github/workflows/ci.yml`, `docs.yml`): lint, Python 3.10-3.14 matrix (+ non-blocking 3.15), an upstream-`main` canary, Aspect integration, macOS/Windows/arm,
  packaging, and a multi-runner benchmark report. **These workflows have been linted with actionlint but not run**: this repository was
  prepared read-only and nothing has been pushed. Locally everything was run on Python 3.13 only.
