# Changelog

## Unreleased

- **Fixed:** the decision to accept a bucketing split compared the plan *without* per-group overhead, inconsistent with the
  planner's own objective (a 16-molecule case looked like a 27 % saving; the full model says 6 % and now runs unsplit).
  `worth_splitting()` uses the full model. Invalid cost-model environment values (NaN, inf, negative, `min_saving` outside
  [0, 1), `min_group` < 1, non-numbers) are rejected with a warning and replaced by the default.
- **Changed (default):** length bucketing is no longer applied in training unless `UNIMOL_ACCEL_BUCKET_TRAIN=1`. By default
  training keeps the original dropout-mask order, so seeded runs match the original; the opt-in path still gives the
  larger training speed-up but changes the trajectory (no model-quality benchmark yet).
- **Added:** `CostModel` and environment variables for the bucketing constants; `status()["bucketing"]`;
  `benchmarks/calibrate_cost_model.py` and `benchmarks/cost_model_sensitivity.py`; exhaustive optimality test for the planner.
- **Tests:** the training test no longer ends in an uncompared backward pass. It now checks that bucketing really splits the
  batch (encoder-call count), that default training is unsplit, that gradients match in both modes, and that seeded dropout
  outputs match by default.
- **Benchmarks:** new `train-bucketed` kind; the built-in gradient check covers both training modes; docs report regenerated.
- Fixed: benchmark jobs in `docs.yml` failed with "SMILES input needs RDKit". New `[bench]` extra (RDKit), installed by the
  workflow; the benchmark now checks for RDKit up front and exits with an install hint; import-time `FutureWarning` noise is
  hidden in worker processes.
- README: badges and links to the docs, benchmark methodology and live report.
- Verified against `unimol_tools` 0.1.6 (PyPI) and the Uni-Mol repository at commit `90f52c4` (0.1.3.post1).
- `python -m unimol_accel ...` runner to activate the package for programs that cannot be edited (for example the
  `scbirlab/aspect` CLI), with `--also` to combine both accel packages.
- Aspect integration tests (`tests/integration/`) with random-weight checkpoints and a mocked Hugging Face hub.
- `python -m unimol_accel.bench`: offline benchmark runner (fresh subprocess per configuration, time budget, equivalence
  gate) and a self-contained HTML report.
- Documentation site (Sphinx), CI matrix, CONTRIBUTING, AGENTS.md.

## 0.1.0

- Initial release.
