# Security policy

`unimol-accel` runs in-process Python that monkey-patches a small number of Uni-Mol functions (pure Python, no native code).

- Report suspected vulnerabilities privately through GitHub's *Security → Report a vulnerability* form, not in
  public issues.
- Patches are applied only to functions whose exact source hash was verified; otherwise the original code runs.
- `python -m unimol_accel ...` executes the program you name, in the same process, and does nothing else.
- The benchmark tool and tests perform no network access and write only to their `--output` / temporary directories.
- Only the latest minor release receives fixes.
