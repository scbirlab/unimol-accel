# Contributing to unimol-accel

Thanks for helping! This project wraps a few internals of Uni-Mol with faster equivalents, so the bar for correctness is
high and every performance claim must come with a measurement. This guide is for humans; automated agents should also
read [`AGENTS.md`](https://github.com/scbirlab/unimol-accel/blob/main/AGENTS.md).

## Quick start

```bash
git clone https://github.com/scbirlab/unimol-accel && cd unimol-accel
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU build is enough
pip install "unimol-tools>=0.1.6" pytest ruff
pip install -e ".[test]"
pytest -q                                   # unit tests
pip install "aspect-data[unimol]"            # optional: also runs tests/integration (Python >= 3.11)
ruff check . && ruff format .
pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html
```

No test, benchmark or tool in this repository needs network access or downloads (pretrained weights are *not*
required: tests build random-weight models in the real on-disk formats).

## Repository map

| Path | What lives there |
| --- | --- |
| `src/unimol_accel/_patches.py` | the guarded, reversible, idempotent patches, the bucket planner and the output merge |
| `src/unimol_accel/__init__.py` | `enable` / `disable` / `status`, auto-enable on import |
| `src/unimol_accel/__main__.py` | `python -m unimol_accel [--also PKG] -m MODULE ...` runner for programs you cannot edit |
| `src/unimol_accel/bench/` | benchmark worker, CLI orchestrator, stdlib-only HTML report |
| `benchmarks/` | direct scripts: `bench_unimol.py`, `sweep_batch_size.py`, `bench_aspect.py` |
| `experiments/` | rejected ideas kept as evidence (native C++ attention) |
| `tests/` | equivalence tests; `tests/integration/` runs real Aspect with random weights |
| `docs/` | Sphinx (pydata theme) sources |
| `.github/workflows/` | `ci.yml` (lint, version matrix, Aspect integration), `docs.yml` (docs + benchmark report) |

## Ground rules

1. **Equivalence first.** A change must keep original-vs-accelerated outputs and gradients equal (to the tolerance
   documented in `AGENTS.md`) and checkpoints / `state_dict` interchangeable.
2. **Measure before you optimise, and keep negative results.** Use `python -m unimol_accel.bench` and the scripts in
   `benchmarks/`. Interleave repeats, report medians, state the CPU, compare to the best original configuration as well
   as the same one. If an idea does not help, add it to `docs/optimisations.md` under "Negative results".
3. **Fail safe.** An upstream version or function whose source hash is not verified must run the original code. Never
   patch more than the targeted functions; never change global PyTorch state at import time.
4. **No hidden cost or side effect at import.** No compilation, network or large allocations on `import unimol_accel`.
5. **No network in tests/benchmarks.** Mock the Hugging Face download (see `tests/integration/conftest.py`).

## Supporting a new Uni-Mol release

1. Install the release and compute the hashes of the patched functions (`unimol_accel` prints them in `status()["patches"]`
   when a hash is unverified, or use `inspect.getsource` + SHA-256 as in the `KNOWN_HASHES` comment).
2. For each new hash, **read the diff of that function** against the previous verified source and confirm the
   replacement still has identical semantics; only then add the hash to `KNOWN_HASHES`.
3. Extend the CI matrix, update `docs/compatibility.md` and `CHANGELOG.md`, and rerun the Aspect integration tests.

## Pull requests

- Small, focused PRs; conventional-commit style titles (`fix:`, `feat:`, `perf:`, `docs:`).
- CI must be green: lint, version matrix, Aspect integration, docs build.
- Update `CHANGELOG.md` under *Unreleased*. Performance claims need `unimol_accel.bench` output (JSON or table, with CPU).

## Releasing (maintainers)

1. Update `CHANGELOG.md` and `__version__` in `src/unimol_accel/__init__.py`; merge to `main`.
2. Build with `python -m build` and `twine check dist/*` (the CI `package` job does this) and publish as you normally do.
3. Docs and the benchmark report redeploy from `main` (`docs.yml`).

## Code of conduct

See [`CODE_OF_CONDUCT.md`](https://github.com/scbirlab/unimol-accel/blob/main/CODE_OF_CONDUCT.md). Security issues:
[`SECURITY.md`](https://github.com/scbirlab/unimol-accel/blob/main/SECURITY.md).
