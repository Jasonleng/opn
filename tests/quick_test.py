"""Quick end-to-end check that every analysis script in the repository runs.

Each script is executed for real (same imports, networks, model code and
output writing) but with tiny settings so the whole check takes minutes:

  * simulation scripts: grid cell 0, one simulated dataset (--n-reps 1)
  * hierarchical fits (paper and demo): first 3 subjects only (--n-subjects 3)
  * NUTS shortened to 20 draws and 20 tuning steps per chain

Each run is saved in its own folder, tests/runs/<date-time>/ (results/ is never
touched): per script its outputs, log and the full posterior summary of every
fit it made, plus summary.json and report.html, a page to open in a browser.
The estimates are meaningless at these settings; only pass/fail matters.

Usage (from the repository root, inside the container):
    apptainer exec --bind "$PWD" opn.sif make smoke
    apptainer exec --bind "$PWD" opn.sif python tests/quick_test.py --only ddm weibull
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import smoke_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scripts that need the (non-public) task data are skipped when it is absent
DATA_FILE = os.path.join(ROOT, "data", "hddm_all.csv")
NEEDS_DATA = {"hierarchical"}

# name -> (script, function(run_dir, opts) -> command-line args)
SCRIPTS = {
    "hierarchical": ("scripts/hierarchical/fit_hierarchical_angle.py",
                     lambda run_dir, opts: ["--n-subjects", str(opts.n_subj), "--out", os.path.join(run_dir, "posterior.nc")]),
    "hierarchical_demo": ("demo/fit_demo.py",
                          lambda run_dir, opts: ["--n-subjects", str(opts.n_subj), "--out-dir", run_dir]),
    "angle": ("scripts/simulation/recover_angle.py",
              lambda run_dir, opts: ["0", "--n-reps", "1", "--out-dir", run_dir]),
    "angle_outlier": ("scripts/simulation/recover_angle_outlier.py",
                      lambda run_dir, opts: ["0", "--n-reps", "1", "--out-dir", run_dir]),
    "ddm": ("scripts/simulation/recover_ddm.py",
            lambda run_dir, opts: ["0", "--n-reps", "1", "--out-dir", run_dir]),
    "weibull": ("scripts/simulation/recover_weibull.py",
                lambda run_dir, opts: ["0", "--n-reps", "1", "--out-dir", run_dir]),
}

# Runs inside the child process (cwd = the script's run folder): shorten NUTS, record
# the posterior summary of every fit, then run the script as __main__.
# Nothing here may import pymc.sampling.jax: that loads pytensor's JAX backend, which
# fixes jax_enable_x64 from floatX before the script has set floatX to float32.
BOOTSTRAP = r"""
import json, os, runpy, sys
import arviz, arviz.stats
from numpyro.infer import MCMC

draws, tune = int(os.environ["QT_DRAWS"]), int(os.environ["QT_TUNE"])
_mcmc_init = MCMC.__init__
def quick_mcmc_init(self, *args, **kwargs):
    # keep the script's num_chains: PyMC has already drawn one init point per chain
    kwargs.update(num_samples=draws, num_warmup=tune, progress_bar=False)
    _mcmc_init(self, *args, **kwargs)
MCMC.__init__ = quick_mcmc_init

fits = []
def record(stem, label, terms, table, note=""):
    path = stem + "_summary.csv"
    table.to_csv(path, index_label="parameter")
    fits.append({"label": label, "terms": terms, "summary": path, "note": note})
    with open("fits.json", "w") as fits_file:
        json.dump(fits, fits_file, indent=2)

# The scripts call pm.summary(trace) after every fit. pm.summary is arviz.stats.summary,
# copied when pymc is imported, so patching it here records each fit's full table.
_summary = arviz.stats.summary
def recording_summary(data, *args, **kwargs):
    table = _summary(data, *args, **kwargs)
    try:
        terms = [potential.name for potential in sys.modules["pymc"].modelcontext(None).potentials]
    except Exception:
        terms = []
    record(f"fit_{len(fits) + 1}", f"Fit {len(fits) + 1}", terms, table)
    return table
arviz.summary = arviz.stats.summary = recording_summary

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

# Scripts that save their posterior without summarising it (the paper's hierarchical fit):
# summarise each .nc, leaving out trial-level variables (more than 50 values per draw)
for name in sorted(file for file in os.listdir(".") if file.endswith(".nc") and not fits):
    posterior = arviz.from_netcdf(name).posterior
    values_per_draw = {var: posterior[var].size // (posterior.sizes["chain"] * posterior.sizes["draw"])
                       for var in posterior.data_vars}
    kept_vars = [var for var, n_values in values_per_draw.items() if n_values <= 50]
    left_out = sorted(set(values_per_draw) - set(kept_vars))
    note = f"Trial-level variables left out: {', '.join(left_out)}." if left_out else ""
    record(name[:-3], name, [], _summary(posterior, var_names=kept_vars), note)
"""

def run_one(name, out_root, cache_dir, opts):
    """Run one script; return its result record (status PASS / FAIL / SKIP)."""
    script, make_args = SCRIPTS[name]
    run_dir = os.path.join(out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")

    def result(status, seconds, message):
        outputs = sorted(file for file in os.listdir(run_dir)
                         if file.endswith((".csv", ".nc")) and not file.endswith("_summary.csv"))
        fits_file = os.path.join(run_dir, "fits.json")
        fits = json.load(open(fits_file)) if os.path.isfile(fits_file) else []
        return {"name": name, "script": script, "status": status, "seconds": round(seconds, 1),
                "message": message, "outputs": outputs, "fits": fits,
                "log": os.path.relpath(log_path, out_root)}

    if name in NEEDS_DATA and not os.path.isfile(DATA_FILE):
        return result("SKIP", 0.0, f"no {os.path.relpath(DATA_FILE, ROOT)}")

    env = dict(os.environ,
               QT_DRAWS=str(opts.draws), QT_TUNE=str(opts.tune),
               MPLBACKEND="Agg",
               PYTENSOR_FLAGS=f"base_compiledir={os.path.join(cache_dir, name)}",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    command = [sys.executable, "-c", BOOTSTRAP, os.path.join(ROOT, script)] + make_args(run_dir, opts)

    start = time.time()
    with open(log_path, "w") as log:
        try:
            process = subprocess.run(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=opts.timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            exit_code = None
    elapsed = time.time() - start

    if exit_code is None:
        return result("FAIL", elapsed, f"timed out after {opts.timeout}s")
    if exit_code != 0:
        lines = open(log_path).read().strip().splitlines()
        errors = [line for line in lines if re.match(r"^\w+(Error|Exception|Interrupt)\b", line)]
        reason = (errors or lines or ["(no output)"])[-1]
        return result("FAIL", elapsed, f"exit code {exit_code}: {reason}")

    passed = result("PASS", elapsed, "")
    if not passed["outputs"]:
        return {**passed, "status": "FAIL", "message": "finished but wrote no .csv/.nc output"}
    return {**passed, "message": "wrote " + ", ".join(passed["outputs"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME", choices=list(SCRIPTS),
                        help="run only these: " + ", ".join(SCRIPTS))
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--tune", type=int, default=20)
    parser.add_argument("--n-subj", type=int, default=3,
                        help="subjects kept for the hierarchical fit (0 = all)")
    parser.add_argument("--jobs", type=int, default=len(SCRIPTS), help="scripts run in parallel")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per script")
    parser.add_argument("--out", default=None,
                        help="folder for outputs, logs and report (default: tests/runs/<date-time>)")
    opts = parser.parse_args()

    selected = opts.only or list(SCRIPTS)
    started = time.localtime()
    out_root = opts.out or os.path.join(ROOT, "tests", "runs", time.strftime("%Y%m%d-%H%M%S", started))
    os.makedirs(out_root, exist_ok=True)
    cache_dir = tempfile.mkdtemp(prefix="opn_quicktest_pytensor_")  # compiled C code, not worth keeping

    print(f"Running {len(selected)} script(s), saving to {out_root}\n", flush=True)
    results = []
    wall_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=opts.jobs) as pool:
        futures = [pool.submit(run_one, name, out_root, cache_dir, opts) for name in selected]
        for future in concurrent.futures.as_completed(futures):
            script_result = future.result()
            results.append(script_result)
            print(f"  [{script_result['status']}] {script_result['name']:<18} {script_result['seconds']:6.0f}s  "
                  f"{script_result['message']}", flush=True)
    results.sort(key=lambda script_result: list(SCRIPTS).index(script_result["name"]))

    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z", started),
            "wall_seconds": round(time.time() - wall_start, 1),
            "settings": {"draws": opts.draws, "tune": opts.tune, "n_subj": opts.n_subj},
            "environment": smoke_report.environment(ROOT)}
    smoke_report.write_summary(out_root, results, meta)
    report = smoke_report.write_html(out_root, results, meta)

    failed = [script_result for script_result in results if script_result["status"] == "FAIL"]
    skipped = sum(script_result["status"] == "SKIP" for script_result in results)
    print(f"\n{len(results) - len(failed) - skipped}/{len(results)} passed" + (f", {skipped} skipped" if skipped else ""))
    for script_result in failed:
        print(f"  see {os.path.join(out_root, script_result['log'])}")
    print(f"\nReport: {report}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
