"""The ``python -m unimol_accel`` runner and the benchmark CLI (no models needed)."""

import json
import os
import subprocess
import sys
import textwrap


ENV = {**os.environ, "PYTHONWARNINGS": "ignore"}


def run(*args, **kw):
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, env=ENV, timeout=300, **kw
    )


def test_status_subcommand():
    r = run("-m", "unimol_accel", "status")
    assert r.returncode == 0, r.stderr[-500:]
    assert json.loads(r.stdout[r.stdout.index("{") :])["unimol_accel"]


def test_runs_script_with_original_argv(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent("""
        import sys, unimol_accel
        print("ARGV", sys.argv[1:], unimol_accel.status()["enabled"])
    """)
    )
    r = run("-m", "unimol_accel", str(script), "a", "--b")
    assert r.returncode == 0, r.stderr[-500:]
    assert "ARGV ['a', '--b']" in r.stdout


def test_runs_module_with_dash_m():
    r = run("-m", "unimol_accel", "-m", "json.tool", "--help")
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_unknown_also_package_fails_cleanly():
    r = run("-m", "unimol_accel", "--also", "no_such_package_xyz", "status")
    assert r.returncode != 0


def test_disable_env_prevents_activation(tmp_path):
    script = tmp_path / "child.py"
    script.write_text("import unimol_accel; print('ENABLED', unimol_accel.status()['enabled'])")
    r = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**ENV, "UNIMOL_ACCEL_DISABLE": "1", "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert "ENABLED False" in r.stdout, r.stderr[-500:]


def test_bench_dry_run_runs_nothing(tmp_path):
    r = run("-m", "unimol_accel.bench", "--dry-run", "--output", str(tmp_path / "o"))
    assert r.returncode == 0 and "dry run" in r.stdout
    assert not (tmp_path / "o").exists()


def test_report_renders_from_synthetic_results(tmp_path):
    from unimol_accel.bench.report import render, write_report

    doc = {
        "name": "synthetic",
        "device": "cpu",
        "preset": "quick",
        "created": "2026-01-01",
        "host": {"cpu": "x"},
        "status": {"patches": {"p": "active"}},
        "runs": [
            {
                "kind": "inference",
                "batch": 8,
                "threads": 2,
                "status": "ok",
                "original_s": [2.0, 2.2],
                "accel_s": [1.0, 1.1],
                "speedup": 2.0,
                "rel_error": 1e-8,
                "passed": True,
            },
            {"kind": "train", "batch": 8, "threads": 2, "status": "error", "error": "killed"},
        ],
        "equivalence": [{"key": "k", "passed": True}],
    }
    html = render([doc])
    assert "2.00" in html and "All equivalence checks passed" in html and "killed" in html
    bad = dict(doc, equivalence=[{"key": "k", "passed": False}])
    assert "EQUIVALENCE FAILURES" in render([bad])
    assert write_report([doc], tmp_path / "r.html").read_text().startswith("<!doctype html>")


def test_bench_without_rdkit_fails_with_install_hint(monkeypatch, capsys):
    """CI regression: a bare ``pip install -e .`` has no RDKit; say so instead of failing every run."""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "rdkit":
            raise ImportError("no rdkit")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    from importlib import import_module

    mod = import_module("unimol_accel.bench.__main__")
    assert mod.main(["--preset", "ci", "--output", "/tmp/_x_bench"]) == 2
    assert "RDKit" in capsys.readouterr().err
