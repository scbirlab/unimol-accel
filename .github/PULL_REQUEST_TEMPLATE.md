## What and why

## Checklist
- [ ] `ruff check . && ruff format --check .` pass
- [ ] `pytest` passes locally (and `tests/integration` if `aspect-data[unimol]` is installed)
- [ ] New/changed behaviour has tests (equivalence vs the original Uni-Mol code where relevant)
- [ ] Performance claims are backed by `python -m unimol_accel.bench` output (attach JSON / table, state the CPU)
- [ ] Docs / CHANGELOG updated
- [ ] If an upstream function hash changed: `KNOWN_HASHES` updated **and** the replacement re-verified
