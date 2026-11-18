"""unimol_accel -- drop-in CPU acceleration for Uni-Mol (``unimol_tools``).

    import unimol_accel          # activates the verified optimisations
    import unimol_tools          # ordinary code, unchanged

Set ``UNIMOL_ACCEL_DISABLE=1`` to prevent automatic activation.
"""

from __future__ import annotations

import os
from typing import Dict, List

__version__ = "0.1.0"

_active: List = []
_enabled = False


def enable() -> Dict:
    """Apply all verified patches (idempotent). Returns :func:`status`."""
    global _enabled, _active
    from . import _patches as P

    if not _active:
        try:
            _active = P.build_patches()
        except Exception as exc:  # unimol_tools missing or restructured
            _active = []
            return {"enabled": False, "reason": f"cannot import unimol_tools: {exc!r}"}
    for p in _active:
        p.apply()
    _enabled = any(p.applied for p in _active)
    return status()


def disable() -> Dict:
    """Restore the original methods (trained parameters are never touched)."""
    global _enabled
    for p in _active:
        p.revert()
    _enabled = False
    return status()


def status() -> Dict:
    import torch

    try:
        import unimol_tools  # noqa: F401
        from importlib.metadata import version

        ut = version("unimol-tools")
    except Exception:
        ut = None
    return {
        "enabled": _enabled,
        "unimol_accel": __version__,
        "torch": torch.__version__,
        "unimol_tools": ut,
        "patches": {
            p.name: ("active" if p.applied else f"inactive: {p.reason or 'not applied'}")
            for p in _active
        },
    }


if os.environ.get("UNIMOL_ACCEL_DISABLE", "0") != "1":
    enable()
