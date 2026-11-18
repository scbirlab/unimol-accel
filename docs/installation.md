# Installation

```bash
pip install torch unimol-tools>=0.1.6     # your usual way; CPU or CUDA build both fine
pip install unimol-accel                          # pure-Python wheel; no compiler needed
```

For development: `pip install -e ".[test]"` from a checkout (see {doc}`contributing`).

Importing the package never compiles anything, never touches the network and never fails: if the upstream version is
not one of the verified ones, the original code runs and `unimol_accel.status()` says why.

| | Supported / tested |
| --- | --- |
| Uni-Mol | `unimol_tools` 0.1.6 (PyPI) and the Uni-Mol repository at commit `90f52c4` (0.1.3.post1) |
| Python | >= 3.10; developed and measured on 3.13; CI matrix 3.10–3.14 plus non-blocking 3.15 (workflow written but not yet run) |
| PyTorch | developed and measured with 2.15 (CPU); older versions are untested |
| GPU | not accelerated; see {doc}`optimisations` |
