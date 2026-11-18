"""Run any Python program with unimol_accel active, without editing it.

    python -m unimol_accel status
    python -m unimol_accel -m aspect.cli_module.cli featurize ...        # e.g. the scbirlab/aspect CLI
    python -m unimol_accel [--also atom_jepa_accel] script.py [args...]

`--also PKG` (repeatable) activates another accel package first. The wrapped program sees the
original `sys.argv` (without the wrapper arguments). Nothing else is changed: the same process,
environment and working directory.
"""

from __future__ import annotations

import importlib
import json
import runpy
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    packages = ["unimol_accel"]
    while argv and argv[0] == "--also":
        if len(argv) < 2:
            print("--also needs a package name", file=sys.stderr)
            return 2
        packages.append(argv[1])
        argv = argv[2:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    mods = [
        importlib.import_module(p) for p in packages
    ]  # importing activates (unless *_ACCEL_DISABLE=1)
    if argv[0] == "status":
        for m in mods:
            print(json.dumps(m.status(), indent=2, default=str))
        return 0
    if argv[0] == "-m":
        if len(argv) < 2:
            print("-m needs a module name", file=sys.stderr)
            return 2
        sys.argv = [argv[1]] + argv[2:]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = argv
        runpy.run_path(argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
