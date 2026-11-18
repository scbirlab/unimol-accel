"""Standalone HTML report from results JSON files (stdlib only; no torch needed).

python -m unimol_accel.bench.report results-*.json -o report.html --title "..."
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import statistics
from pathlib import Path

CHECK, CROSS = "\u2713", "\u2717"
TIMES = "\u00d7"
CSS = """
body{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1d2330}
h1{font-size:1.5rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #d0d5dd;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;font-size:.9rem} th,td{padding:.3rem .5rem;border-bottom:1px solid #e4e7ec;text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left} th{background:#f5f6f8}
.note{background:#fff8e1;border-left:4px solid #f0b429;padding:.5rem .8rem;margin:1rem 0}
.bad{color:#b42318;font-weight:600} .ok{color:#067647} .muted{color:#667085;font-size:.85rem}
svg text{font:12px system-ui,sans-serif;fill:#1d2330}
"""


def _err(x):
    return "\u2013" if x is None else f"{x:.1e}"


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def _bars(items, width=900):
    """items: [(label, speedup)] -> inline SVG horizontal bars with a 1.0x reference line."""
    if not items:
        return ""
    top = max(max(v for _, v in items), 1.2)
    row, left = 22, 260
    h = row * len(items) + 24
    x1 = left + (width - left - 60) / top
    parts = [f'<svg width="{width}" height="{h}" role="img" aria-label="speed-up bars">']
    for i, (label, v) in enumerate(items):
        y = 8 + i * row
        w = max(1.0, (width - left - 60) * v / top)
        parts.append(f'<text x="0" y="{y + 12}">{html.escape(label)}</text>')
        parts.append(
            f'<rect x="{left}" y="{y}" width="{w:.1f}" height="14" fill="{"#2e7d6b" if v >= 1.0 else "#b42318"}" rx="2"/>'
        )
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + 12}">{v:.2f}×</text>')
    parts.append(
        f'<line x1="{x1:.1f}" y1="0" x2="{x1:.1f}" y2="{h - 16}" stroke="#667085" stroke-dasharray="4 3"/>'
    )
    parts.append(f'<text x="{x1 - 14:.1f}" y="{h - 3}">1.0× = no change</text></svg>')
    return "".join(parts)


def _paired_range(orig, acc):
    pairs = [o / a for o, a in zip(orig, acc) if a > 0]
    return (min(pairs), max(pairs)) if pairs else (float("nan"), float("nan"))


def _section(doc):
    host = doc.get("host", {})
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in host.items()
        if v is not None
    )
    runs = doc.get("runs", [])
    bars, body = [], []
    for r in runs:
        label = f"{r['kind']} · batch {r['batch']} · {r['threads']} thr"
        if r.get("status") != "ok":
            body.append(
                f"<tr><td>{html.escape(r['kind'])}</td><td>batch {r['batch']}, {r['threads']} thr</td>"
                f"<td colspan='5' class='bad'>{html.escape(r.get('status', ''))}: "
                f"{html.escape(str(r.get('error', ''))[-160:])}</td></tr>"
            )
            continue
        if "speedup" in r:
            o, a = r["original_s"], r["accel_s"]
            lo, hi = _paired_range(o, a)
            bars.append((label, r["speedup"]))
            body.append(
                f"<tr><td>{r['kind']}</td><td>batch {r['batch']}, {r['threads']} thr</td>"
                f"<td>{_med(o) * 1e3:,.0f}</td><td>{_med(a) * 1e3:,.0f}</td><td><b>{r['speedup']:.2f}×</b></td>"
                f"<td class='muted'>{lo:.2f}–{hi:.2f}× (n={len(o)})</td>"
                f"<td class='{'ok' if r.get('passed', True) else 'bad'}'>{_err(r.get('rel_error'))}</td></tr>"
            )
        else:
            ok = r.get("passed")
            body.append(
                f"<tr><td>{r['kind']}</td><td>batch {r['batch']}, {r['threads']} thr</td><td colspan='3'>"
                f"gradients with dropout off</td><td class='muted'>tolerance {r.get('tolerance', 0):.0e}</td>"
                f"<td class='{'ok' if ok else 'bad'}'>{_err(r.get('rel_error'))} {CHECK if ok else CROSS}</td></tr>"
            )
    status = doc.get("status", {})
    patches = status.get("patches", {})
    pt = "".join(
        f"<li><code>{html.escape(str(k))}</code>: {html.escape(str(v))}</li>"
        for k, v in patches.items()
    )
    return (
        f"<h2>{html.escape(doc.get('name', 'machine'))} <span class='muted'>({html.escape(doc.get('device', 'cpu'))}, "
        f"preset {html.escape(str(doc.get('preset', '')))}, {html.escape(str(doc.get('created', '')))})</span></h2>"
        f"<h3>Speed-up (original time / accelerated time, medians)</h3>{_bars(bars)}"
        "<table><tr><th>kind</th><th>configuration</th><th>original ms</th><th>accel ms</th><th>speed-up</th>"
        f"<th>paired range</th><th>rel. error</th></tr>{''.join(body)}</table>"
        f"<h3>Patches</h3><ul>{pt or '<li>none reported</li>'}</ul>"
        f"<details><summary>Host</summary><table>{rows}</table></details>"
    )


def render(docs, title="unimol_accel benchmark report"):
    bad = [
        (d.get("name"), e["key"]) for d in docs for e in d.get("equivalence", []) if not e["passed"]
    ]
    banner = (
        ("<p class='bad'>EQUIVALENCE FAILURES: " + html.escape(str(bad)) + "</p>")
        if bad
        else "<p class='ok'>All equivalence checks passed.</p>"
    )
    note = (
        "<div class='note'><b>Read this as a measurement on the machine(s) listed, not a promise.</b> Speed-ups depend on "
        "CPU, cores, batch composition and size. Shared CI runners are noisy: use these as a regression signal. "
        "Random weights; the arithmetic is the same as with trained weights.</div>"
    )
    return (
        f"<!doctype html><html lang='en'><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style>"
        f"<h1>{html.escape(title)}</h1><p class='muted'>generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC</p>"
        f"{banner}{note}{''.join(_section(d) for d in docs)}</html>"
    )


def write_report(docs, path, title="unimol_accel benchmark report"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(docs, title), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="results JSON files")
    ap.add_argument("-o", "--output", default="report.html")
    ap.add_argument("--title", default="unimol_accel benchmark report")
    a = ap.parse_args(argv)
    docs = [json.loads(Path(p).read_text()) for p in a.inputs]
    print(write_report(docs, a.output, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
