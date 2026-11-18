# Compatibility and safety

**Verified against:** `unimol_tools` 0.1.6 (PyPI) and the Uni-Mol repository at commit `90f52c4` (0.1.3.post1).

Each replaced function is identified by a SHA-256 prefix of its exact source. If the installed Uni-Mol differs, that patch
is **not applied** and the original code runs; `unimol_accel.status()` reports the unverified hash. This hash list is the
safety mechanism: do not add a hash without reading the upstream diff and re-verifying the replacement (see
{doc}`contributing`).

| Guarantee | |
| --- | --- |
| Parameters, buffers, `state_dict` keys | unchanged; checkpoints are interchangeable with and without the package |
| Importing | no network, no compilation, no change to threads, seeds or dtype |
| `disable()` | restores the original function objects (and, for Atom-JEPA, every encoder it touched) |
| Unknown upstream source | original code runs |
| Network | none used by the package, its tests or its benchmarks |

## Output shapes handled

`UniMolModel.forward` returns a bare `cls_repr` tensor in `unimol_tools >= 0.1.6` and a dict in older versions, plus a
dict with atomic symbols/coordinates/representations when `return_atomic_reprs=True`. The bucketing wrapper merges
tensors, lists and dicts back into the caller's molecule order and, for any other structure, recomputes with the
original path rather than guessing.

## Environment switches

| Variable | Effect |
| --- | --- |
| `UNIMOL_ACCEL_DISABLE=1` | don't activate on import |
| `UNIMOL_ACCEL_BUCKET=0` | no length bucketing |
| `UNIMOL_ACCEL_BUCKET_TRAIN=1` | **opt-in:** bucket in training too (default: off; dropout masks are then drawn per sub-batch and trajectories differ from the original under the same seed) |
| `UNIMOL_ACCEL_BUCKET_MIN_BATCH` | smallest batch that is bucketed (default 8) |
| `UNIMOL_ACCEL_COST_PAIR` | cost model: weight of the L² term (default 0.0625) |
| `UNIMOL_ACCEL_COST_GROUP` | cost model: per-group overhead in units of L (default 48) |
| `UNIMOL_ACCEL_COST_MIN_GROUP` | smallest group except the longest (default 4) |
| `UNIMOL_ACCEL_COST_MIN_SAVING` | only bucket if the plan saves at least this fraction of padded cost (default 0.10) |
