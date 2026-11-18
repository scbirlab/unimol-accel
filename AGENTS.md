# AGENTS.md — instructions for AI coding agents

Project: **unimol-accel** — `import unimol_accel` makes Uni-Mol (`unimol_tools`) faster on CPU by wrapping three methods (length-bucketed sub-batching in `UniMolModel.forward`, skipping the unused encoder tail, a leaner pair-bias attention). Pure PyTorch. Human-oriented docs are in `README.md` and `CONTRIBUTING.md`; this file is the
machine-oriented summary. Keep it in sync with the code.

## Commands (run from the repo root)

```bash
pip install -e ".[test]"                    # plus the upstream package and torch
pytest -q                                   # unit tests (tests/integration needs: pip install "aspect-data[unimol]")
ruff check . && ruff format --check .
sphinx-build -W -b html docs docs/_build/html
python -m unimol_accel.bench --preset quick --output /tmp/b        # end-to-end speed + equivalence (offline)
python -m unimol_accel status                      # what is active
```

Prerequisites: Python >= 3.10, `torch>=2.1`, `unimol-tools`. Verified against `unimol_tools` 0.1.6 (PyPI) and the Uni-Mol repository at commit `90f52c4` (0.1.3.post1).

## Architecture in one paragraph

`_patches.py` defines `Patch` (owner class, attribute, factory, original) and applies each patch only if the SHA-256
prefix of the upstream function source is in `KNOWN_HASHES`. Three patches: `SelfMultiheadAttention.forward`
(`baddbmm`, no clone), `TransformerEncoderWithPair.forward` (skips the tail only while the `_SKIP_TAIL` context
variable is set), and `UniMolModel.forward` (sets `_SKIP_TAIL`, plans length buckets with `plan_buckets`, runs the
original forward per group, and `_merge_outputs` restores the caller's order for tensors, lists and dicts). The
bucketing wrapper falls back to the original path when the plan would not save at least 10 % of padded cost.

## Invariants — do not break these

- Per-sample results equal the original up to float rounding; bucketing never changes per-sample maths (Uni-Mol uses
  LayerNorm only; never introduce batch-statistics layers into the bucketed path).
- Output order and structure of `UniMolModel.forward` are preserved for every `return_repr` / `return_atomic_reprs`
  combination of every verified version (tensor in unimol_tools >= 0.1.6, dict before).
- Unknown upstream source hash => the **original** code runs and `status()` says why.
- `enable()` is idempotent, `disable()` restores the original function objects, parameters and `state_dict` are never
  touched, importing never changes threads, seeds or dtype.
- No network access anywhere (tests, benchmarks, runner).

## Things agents commonly get wrong here

- Don't add a hash to `KNOWN_HASHES` without reading the diff of that upstream function and re-verifying the patch.
- Compare against the **best** original batch size, not only the same batch size: the original is already fast at
  small batches (see `docs/optimisations.md`). Never present the large-batch speedup as the headline.
- Don't claim a speedup without `python -m unimol_accel.bench` (or `benchmarks/`) numbers on a stated CPU; shared CI
  runners are a regression signal only. Never fabricate or extrapolate timings (all published numbers: 2-vCPU VM).
- Real pretrained weights were not available when this was written; equivalence tests use random weights in the real
  on-disk format. Keep it that way for CI (no downloads).
- Don't run `pytest` while timing.
- Negative results belong in `docs/optimisations.md` (don't delete them).
- Read-only default: don't push, tag, publish or write to any remote service without the maintainer's instruction.

## Style

Ruff (`line-length = 100`), type hints on public API, NumPy-style docstrings, English (British spellings fine).
Commit messages: short imperative subject (`perf: ...`, `fix: ...`).
