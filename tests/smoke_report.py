"""Write the quick-test results as summary.json and a self-contained report.html.

To rebuild the report of a saved run: python tests/smoke_report.py tests/runs/<run>
"""

import csv
import html
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys

PACKAGES = ["jax", "jaxlib", "numpyro", "pymc", "pytensor", "arviz", "lanfactory", "ssm-simulators", "numpy", "pandas"]
# Lines printed by the libraries on every run that carry no information
NOISE = re.compile(r"passing through (identity|transform)|wandb not available|^passing 1$")
LOG_TAIL = 30
MAX_ROWS = 50  # rows shown from a script's own output CSV (posterior summaries are shown in full)


def environment(root):
    """Python, package versions, git commit and container image of this run."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None

    def git(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                                  timeout=20).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "container": os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_uncommitted_changes": None if status is None else len(status.splitlines()),
        "packages": versions,
    }


def write_summary(out_root, results, meta):
    path = os.path.join(out_root, "summary.json")
    with open(path, "w") as f:
        json.dump({**meta, "results": results}, f, indent=2)
    return path


# --- HTML ---------------------------------------------------------------------

CSS = """
:root { --bg:#f7f7f5; --panel:#ffffff; --text:#1d1d1b; --muted:#6b6b66; --border:#e2e1dc;
        --pass:#1a7f37; --pass-bg:#e6f4ea; --fail:#c62828; --fail-bg:#fdecea; --skip:#6b6b66; --skip-bg:#efefec;
        --code-bg:#f1f0ec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#161615; --panel:#1f1f1d; --text:#ecebe6; --muted:#a3a29c; --border:#34332f;
        --pass:#5cc97b; --pass-bg:#1b3324; --fail:#ff7b72; --fail-bg:#3d1f1d; --skip:#a3a29c; --skip-bg:#2a2a27;
        --code-bg:#262624; } }
:root[data-theme="dark"] { --bg:#161615; --panel:#1f1f1d; --text:#ecebe6; --muted:#a3a29c; --border:#34332f;
        --pass:#5cc97b; --pass-bg:#1b3324; --fail:#ff7b72; --fail-bg:#3d1f1d; --skip:#a3a29c; --skip-bg:#2a2a27;
        --code-bg:#262624; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1000px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 12px; }
h3 { font-size:16px; margin:0; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.muted { color:var(--muted); } .small { font-size:13px; }
.verdict { display:inline-block; margin:16px 0 4px; padding:8px 14px; border-radius:8px; font-weight:600; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; margin:12px 0; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
th { color:var(--muted); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; text-align:right; }
.badge { display:inline-block; min-width:48px; text-align:center; padding:1px 8px; border-radius:999px;
         font-size:12px; font-weight:700; letter-spacing:.03em; }
.PASS { color:var(--pass); background:var(--pass-bg); } .FAIL { color:var(--fail); background:var(--fail-bg); }
.SKIP { color:var(--skip); background:var(--skip-bg); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }
pre { background:var(--code-bg); padding:12px; border-radius:8px; overflow-x:auto; margin:8px 0 0; }
details { margin-top:12px; } summary { cursor:pointer; color:var(--muted); }
h4 { font-size:14px; margin:20px 0 6px; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; margin:0; font-size:14px; }
dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
"""


def _esc(x):
    return html.escape(str(x))


def _fmt(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    return f"{number:.4g}"


def _is_num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _table(header, rows):
    """HTML table; numeric cells are formatted and right-aligned, other cells escaped."""
    head = "".join(f"<th>{_esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f'<td class="num">{_fmt(c)}</td>' if _is_num(c) else f"<td>{_esc(c)}</td>"
                                    for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _csv_table(path, max_rows=None):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "<p class='muted'>empty file</p>"
    return _table(rows[0], rows[1:] if max_rows is None else rows[1:max_rows + 1])


def _fit_block(run_dir, fit):
    """Full posterior summary (every parameter, every ArviZ column) of one fit."""
    path = os.path.join(run_dir, fit["summary"])
    terms = (" · likelihood terms: " + " + ".join(fit["terms"])) if fit["terms"] else ""
    note = f"<p class='muted small'>{_esc(fit['note'])}</p>" if fit.get("note") else ""
    table = _csv_table(path) if os.path.isfile(path) else "<p class='muted'>summary file missing</p>"
    return f"<h4>{_esc(fit['label'])}<span class='muted small'>{_esc(terms)}</span></h4>{note}{table}"


def _log_tail(path):
    if not os.path.isfile(path):
        return ""
    lines = [l.rstrip() for l in open(path, errors="replace") if l.strip() and not NOISE.search(l)]
    return "\n".join(lines[-LOG_TAIL:])


def write_html(out_root, results, meta):
    counts = {s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "SKIP")}
    verdict = "FAIL" if counts["FAIL"] else "PASS"
    verdict_text = (f"{counts['FAIL']} of {len(results)} scripts failed" if counts["FAIL"]
                    else f"All {counts['PASS']} scripts ran" + (f" ({counts['SKIP']} skipped)" if counts["SKIP"] else ""))

    rows = "".join(
        f"<tr><td><a href='#{_esc(r['name'])}'>{_esc(r['name'])}</a></td>"
        f"<td><span class='badge {r['status']}'>{r['status']}</span></td>"
        f"<td class='num'>{r['seconds']:.0f}</td><td>{_esc(r['message'])}</td></tr>" for r in results)
    overview = ("<div class='scroll'><table><thead><tr><th>Script</th><th>Status</th><th>Time (s)</th>"
                f"<th>Result</th></tr></thead><tbody>{rows}</tbody></table></div>")

    sections = []
    for r in results:
        run_dir = os.path.join(out_root, r["name"])
        previews = [_fit_block(run_dir, fit) for fit in r.get("fits", [])]
        for out in r["outputs"]:
            if out.endswith(".csv"):
                previews.append(f"<h4>Script output <code>{_esc(out)}</code></h4>"
                                f"{_csv_table(os.path.join(run_dir, out), MAX_ROWS)}")
        tail = _log_tail(os.path.join(out_root, r["log"]))
        log = (f"<details{' open' if r['status'] == 'FAIL' else ''}><summary>Last lines of "
               f"<code>{_esc(r['log'])}</code></summary><pre>{_esc(tail)}</pre></details>") if tail else ""
        sections.append(
            f"<section class='panel' id='{_esc(r['name'])}'>"
            f"<h3><span class='badge {r['status']}'>{r['status']}</span>{_esc(r['name'])}"
            f"<span class='muted small'><code>{_esc(r['script'])}</code> · {r['seconds']:.0f}s</span></h3>"
            + (f"<p class='small'>{_esc(r['message'])}</p>" if r["status"] != "PASS" else "")
            + f"{''.join(previews)}{log}</section>")

    env = meta["environment"]
    pkgs = ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)
    git = env["git_commit"] or "unknown"
    if env["git_uncommitted_changes"]:
        git += f" (+{env['git_uncommitted_changes']} uncommitted changes)"
    s = meta["settings"]
    details = (
        f"<dl><dt>Started</dt><dd>{_esc(meta['started'])}</dd>"
        f"<dt>Wall time</dt><dd>{meta['wall_seconds']:.0f} s</dd>"
        f"<dt>Settings</dt><dd>NUTS {s['tune']} tuning + {s['draws']} draws per chain; simulations: grid cell 0, "
        f"1 dataset; hierarchical: {s['n_subj'] or 'all'} subjects</dd>"
        f"<dt>Git commit</dt><dd>{_esc(git)}</dd>"
        f"<dt>Container</dt><dd>{_esc(env['container'] or 'none')}</dd>"
        f"<dt>Python</dt><dd>{_esc(env['python'])} on {_esc(env['platform'])}</dd>"
        f"<dt>Packages</dt><dd>{_esc(pkgs)}</dd></dl>")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPN Quick Test</title><style>{CSS}</style></head>
<body><main>
<h1>OPN quick test</h1>
<p class="muted">Every analysis script run end to end with tiny settings. This checks that the code runs;
the estimates below are not meaningful. For each script, the full posterior summary of every fit is shown
in the order the script runs them (for the simulations: fit 1, 2, 3 = the <code>_nd</code>, <code>_d</code>,
<code>_f</code> columns of its output CSV), followed by the output file itself. Parameters are on the scale
they are sampled on (unbounded for the Weibull and hierarchical models).</p>
<div class="verdict {verdict}">{_esc(verdict_text)}</div>
<div class="panel">{overview}</div>
<div class="panel">{details}</div>
<h2>Scripts</h2>
{''.join(sections)}
</main></body></html>
"""
    path = os.path.join(out_root, "report.html")
    with open(path, "w") as f:
        f.write(page)
    return path


if __name__ == "__main__":
    # Rebuild report.html of a saved run: python tests/smoke_report.py tests/runs/<run>
    run_dir = sys.argv[1]
    with open(os.path.join(run_dir, "summary.json")) as f:
        saved = json.load(f)
    results = saved.pop("results")
    print(write_html(run_dir, results, saved))
