# Repository for omission paper

This repository includes code and data to generate figures for our paper "The perils of omitting omission when modeling evidence accumulation".

Models are fitted with PyMC + NumPyro using neural-network likelihoods: a LAN for choices and response times, and an omission network (OPN) for the probability of missing the response deadline. You can use this repository to

- reproduce the parameter-recovery simulations and the hierarchical fit to the task data,
- fit a hierarchical model with omissions to your own dataset, defined in a short Python script ([Fitting a new dataset](#fitting-a-new-dataset)),
- use the trained networks to add an omission likelihood to your own PyMC models.

New to hierarchical fitting with omissions? The notebook `demo/fit_demo.ipynb` is a guided tour: it fits a simulated dataset step by step, explains each choice, and shows how to check and read the results ([Notebook](#notebook)).

## Quick start

All commands are run from the repository root.

**1. Get the code.**

This version of the code is on the `v2` branch (`main` holds the original analysis code):

```bash
git clone --branch v2 https://github.com/Jasonleng/opn.git
cd opn
```

If you already have a clone, switch to it with `git fetch origin && git checkout v2`.

The task data (`data/hddm_all.csv`) is not part of the public repository; the simulations and the library do not need it.

**2. Build the software container** (one-off, ~700 MB image, ~5–15 min). The image pins every package version, so results do not depend on what is installed on your machine.

```bash
# Linux with Apptainer (add --fakeroot if you are not root, e.g. on a cluster login node)
apptainer build opn.sif container/opn.def

# macOS, or any machine with Docker but no Apptainer: builds an x86-64 Linux image
container/build_with_docker.sh
```

Build once and copy `opn.sif` to your cluster (e.g. with `scp`). Without containers, see [Plain Python environment](#plain-python-environment).

**3. Check that everything works.**

```bash
apptainer exec --bind "$PWD" opn.sif make test    # library tests, ~1 min
apptainer exec --bind "$PWD" opn.sif make smoke   # runs every analysis script with tiny settings, ~5 min
```

Both print a pass/fail summary. `make smoke` also saves the run to `tests/runs/<date-time>/` with a `report.html` to open in a browser (see [Tests](#tests)).

**4. Run an analysis.**

```bash
apptainer run --bind "$PWD" opn.sif scripts/simulation/recover_ddm.py 0
```

`apptainer run opn.sif <script> [args]` runs `python <script> [args]` inside the container; `--bind "$PWD"` makes the repository visible to it.

## Running the analyses

Every script has `--help`. Inputs (networks, data) are found relative to the repository, so scripts can be started from any folder.

### Parameter recovery simulations

Each simulation script takes a job index that selects one cell of its grid of true parameters. For that cell it simulates `--n-reps` datasets, fits each one with and without the omission likelihood, and writes `output_<jobid>.csv`.

| Script | Model | Deadline | Job indices | Default `--n-reps` | Default output folder |
|---|---|---|---|---|---|
| `recover_angle.py` | angle | 1.5 s | 0–76 | 20 | `results/simulation/varying_deadline/ddl_1.5/angle/` |
| `recover_angle_outlier.py` | angle with 5% outlier trials, fitted with an outlier mixture | 1.25 s | 0–35 | 20 | `results/simulation/angle/p005_l/` |
| `recover_ddm.py` | DDM (also fits the LAN to the untruncated data) | 1.25 s | 0–120 | 20 | `results/simulation/ddm/` |
| `recover_weibull.py` | Weibull | 1.25 s | 0–26 | 30 | `results/simulation/weibull/p000_3_s/` |

```bash
S="apptainer run --bind $PWD opn.sif"
$S scripts/simulation/recover_ddm.py 0                          # one grid cell, all repetitions
$S scripts/simulation/recover_ddm.py 0 --n-reps 2 --out-dir tmp # small test run elsewhere
```

One grid cell runs 40–90 full NUTS fits (`--n-reps` datasets × 2–3 fits each) and can take hours on a CPU, so a full grid is meant to run as a cluster job array, one task per cell. With SLURM:

```bash
sbatch --array=0-120 --cpus-per-task=2 --mem=4G --time=24:00:00 \
    --wrap 'apptainer run --bind "$PWD" opn.sif scripts/simulation/recover_ddm.py $SLURM_ARRAY_TASK_ID'
```

Use the job-index range from the table; the resources are a starting point, adjust them to your cluster.

**Output columns.** One row per simulated dataset:

| Column | Meaning |
|---|---|
| `<param>_true` | Parameter value used to simulate |
| `<param>_pred_nd` | Posterior mean when fitting only the responses before the deadline (LAN alone, omissions ignored) |
| `<param>_pred_d` | Posterior mean when also modelling omissions (LAN + omission network) |
| `<param>_pred_f` | DDM only: LAN fitted to the full data without a deadline |
| `omission_rate` | Fraction of simulated trials at or beyond the deadline |
| `r_hat_nd`, `r_hat_d` | Largest R-hat of the respective fit (outlier and Weibull scripts) |

`p_pred_*` in `recover_angle_outlier.py` is the posterior mean of the unbounded outlier parameter; the outlier probability is `0.1 * sigmoid(p)`. In `recover_weibull.py` the parameters are sampled on an unbounded scale and the reported values map that posterior mean back into each parameter's range.

### Hierarchical fits

```bash
apptainer run --bind "$PWD" opn.sif scripts/hierarchical/fit_hierarchical_angle.py --n-subjects 5   # the paper's fit
apptainer run --bind "$PWD" opn.sif demo/fit_demo.py                                               # demo / template for any dataset
apptainer exec --bind "$PWD" opn.sif jupyter lab --no-browser --port 8888                           # then open demo/fit_demo.ipynb: the demo step by step
```

See [Hierarchical fitting](#hierarchical-fitting) for fitting a new dataset, the demo, and the paper's model.

## Using the networks in your own models

The `opn` package (in `src/`) turns the trained networks into log-likelihood terms for PyMC. A minimal DDM fit that models both the responses and the omissions:

```python
import pytensor
pytensor.config.floatX = "float32"  # must come first: the networks are float32

import pymc as pm
import pytensor.tensor as pt
from pymc.sampling import jax as pymc_jax

from opn.likelihoods import make_lan_logp, make_omission_logp
from opn.networks import load_network
from opn.ops import make_summed_logp_op
from opn.simulate import simulate, split_by_deadline

# Likelihood terms; parameters are passed in the order v, a, z, t (then the deadline for omissions)
lan = make_summed_logp_op(make_lan_logp(load_network("ddm", "lan")), n_params=4)
omission = make_summed_logp_op(make_omission_logp(load_network("ddm", "cpn")), n_params=5)

data = simulate("ddm", [1.0, 1.2, 0.5, 0.3], n_samples=1000)   # replace with your (rt, choice) array
responded, omitted = split_by_deadline(data, deadline=1.25)

with pm.Model():
    v = pm.Uniform("v", -3, 3)
    a = pm.Uniform("a", 0.3, 3)
    t = pm.Uniform("t", 0.1, 2)
    z = pt.constant(0.5)
    pm.Potential("responses", lan(responded, v, a, z, t))
    pm.Potential("omissions", omission(omitted, v, a, z, t, pt.constant(1.25)))
    idata = pymc_jax.sample_numpyro_nuts(chains=2, draws=500, tune=500)
```

Run your own code with `src/` on the import path:

```bash
apptainer exec --bind "$PWD" --env PYTHONPATH="$PWD/src" opn.sif python my_analysis.py
```

(in a plain Python environment, `pip install -e .` once instead).

Things to know:

- **Available networks:** `load_network(model, kind)` with model `"angle"`, `"ddm"` or `"weibull"` and kind `"lan"` (choice/RT) or `"cpn"` (omission). Parameter order follows ssm-simulators: `ssms.config.model_config[model]["params"]`. The networks are only accurate within the parameter ranges they were trained on; keep your priors inside those ranges.
- **Omission network output:** the angle and DDM omission networks output log P(response before the deadline); the Weibull network outputs log P(omission). Pass `network="opn"` to the omission functions for Weibull.
- **Outliers:** `outlier=True` adds a uniform contaminant mixture; the extra last parameter is mapped to a probability in (0, 0.1).
- **Per-trial parameters** (regressions, hierarchical models): see [Hierarchical fitting](#hierarchical-fitting).
- **Sampling:** use `pymc.sampling.jax.sample_numpyro_nuts`, which differentiates the networks with JAX directly.

## Hierarchical fitting

In a hierarchical model every participant has their own parameters, drawn from a group distribution whose mean and spread are estimated too. Participants with few trials borrow strength from the group, and condition effects can differ between participants.

### Fitting a new dataset

If you are new to this, work through the demo notebook first ([Notebook](#notebook)): it runs the steps below on a small simulated dataset and shows what good results and convergence look like.

A fit is defined in a short Python script. `demo/fit_demo.py` is a complete one: copy it and edit its definition section. The library (`opn.hierarchical`) checks the data, builds the model for a hierarchical DDM, angle or Weibull model, samples, saves the results and reports on convergence.

**1. Prepare the data** as a CSV file with one row per trial (column names are up to you):

| Content | Format |
|---|---|
| Participant | Any identifier |
| Response time | Seconds. Empty (or at/after the deadline) for trials without a response |
| Choice | 0/1 or −1/+1 (which option or boundary was chosen); empty for omissions |
| Omission flag (optional) | true/false, 1/0 or yes/no; for omissions that still have an RT recorded |
| Conditions (optional) | Numeric columns are used as they are (e.g. −1/+1 or a centred covariate). A text column with two levels is coded −1/+1 in alphabetical order. Recode factors with more levels as numeric contrast columns |

A trial counts as omitted if its RT is empty or at least the deadline, or if the omission flag is set. `demo/demo_data.csv` is an example.

**2. Write the fit script.** Copy the template, e.g. `cp demo/fit_demo.py scripts/hierarchical/fit_my_study.py`, and edit its definition section:

```python
DATA_FILE = "data/my_study.csv"
DEADLINE = 1.5                      # seconds; no response before the deadline = omission
COLUMNS = dict(subject="participant", rt="rt", response="choice")
# omitted="missed" would add an optional true/false column marking omissions

SPEC = hier.ModelSpec(
    model="ddm",                     # "ddm", "angle" or "weibull"
    fixed={"z": 0.5},                # unbiased starting point
    regressions={"v": ["difficulty"], "a": ["speed_instruction"]},
    # bounds={"t": (0.05, 1.0)},     # narrower parameter ranges than the simulator's
)

SAMPLING = dict(chains=4, draws=1000, tune=1000, target_accept=0.95, seed=1)
OUT_DIR = "results/hierarchical/my_study"
```

The rest of the script reads the data, applies these definitions and calls `hier.fit`. The options:

| Where | Option | Meaning |
|---|---|---|
| `COLUMNS` (passed to `prepare_data`) | `subject`, `rt`, `response`, `omitted` | Names of your columns; `omitted` is optional |
| `ModelSpec` | `model` | `"ddm"` (parameters `v`, `a`, `z`, `t`), `"angle"` (+ `theta`, collapse) or `"weibull"` (+ `alpha`, `beta`) |
| | `fixed` | Parameters held at a value on their own scale instead of estimated |
| | `regressions` | Condition effects: `{parameter: [condition columns]}`. They act on the unbounded scale (see below) |
| | `bounds` | Parameter ranges; defaults are the simulator's (`ssms.config.model_config[model]["param_bounds"]`). Stay within them: the networks are only accurate where they were trained |
| | `random_slopes` | `True` (default): condition effects vary by participant; `False`: one effect for everyone |
| | `lapse` | `True` (default): estimate a lapse probability (0–5 %, per participant) |
| `SAMPLING` (passed to `fit`) | `chains`, `draws`, `tune`, `target_accept`, `seed` | NUTS settings. `fit` runs the chains in parallel, one CPU each |

Mistakes are reported before any sampling: missing columns, responses not coded 0/1 or −1/+1, text conditions with more than two levels, unknown parameter names, or a parameter that is both fixed and given condition effects.

**3. Try it on a few participants** with short chains, to check that the data are read as intended. The script first prints how many trials and omissions it found and how text conditions were coded:

```bash
S="apptainer run --bind $PWD opn.sif"
$S scripts/hierarchical/fit_my_study.py --n-subjects 3 --draws 200 --tune 200 --out-dir tmp/check
```

**4. Run the full fit**, typically on the cluster:

```bash
sbatch --cpus-per-task=4 --mem=8G --time=48:00:00 \
    --wrap 'apptainer run --bind "$PWD" opn.sif scripts/hierarchical/fit_my_study.py'
```

Request one CPU per chain. Run time grows with the number of trials and parameters; the resources above are a starting point (estimate the run time from a fit of a few participants with full-length chains).

**5. Read the results** in `OUT_DIR` (or `--out-dir`):

| File | Contents |
|---|---|
| `posterior.nc` | Full posterior (`arviz.from_netcdf`) |
| `summary.csv` | Posterior mean, sd, 94 % HDI, MCSE, ESS and R-hat of every parameter |
| `fit_info.json` | Numbers of participants, trials and omissions, participant order, coding of text conditions, parameter ranges, the model specification, sampling settings, convergence diagnostics |

The script ends with the group-level estimates and the diagnostics, and prints a warning for each problem it finds:

- **R-hat above 1.01** (chains disagree) or **ESS below 400** (too few effective draws): run longer chains (more `tune`, `draws`).
- **Divergent transitions** (the sampler struggled somewhere): raise `target_accept` (e.g. 0.95 → 0.99). A handful out of thousands of draws can be harmless; check that they are not concentrated in one region (`arviz.plot_pair(idata, var_names=[...], divergences=True)`) and that the estimates do not change with a higher `target_accept`.

Parameter names in the posterior (for each estimated parameter `p`, e.g. `v`):

| Name | Meaning |
|---|---|
| `p_mu_natural`, `p_subj_natural` | Group and participant values **on the parameter's own scale**, with all conditions at 0 |
| `p_mu`, `p_sigma` | Group mean and spread on the unbounded sampling scale |
| `p_subj` | Participant values on the unbounded scale (`p_mu + p_sigma * p_offset`) |
| `p_<condition>_mu`, `p_<condition>_subj` | Condition effect: group and per participant, unbounded scale |
| `p_lapse_*` | Lapse probability, same pattern |

Participants appear in the order of their first row in the data (also listed in `fit_info.json`). A parameter on a given trial is `low + (high − low) · sigmoid(p_subj + Σ condition · effect)`, with `(low, high)` its range from `fit_info.json`. So the drift of a participant in one condition is:

```python
import arviz as az, json
from scipy.special import expit

post = az.from_netcdf("results/hierarchical/my_study/posterior.nc").posterior
low, high = json.load(open("results/hierarchical/my_study/fit_info.json"))["parameter_bounds"]["v"]
v_hard = low + (high - low) * expit(post["v_subj"] + 1 * post["v_difficulty_subj"])   # hard = +1
print(v_hard.mean(("chain", "draw")))       # posterior mean per participant
```

**What the model assumes.** Group means have `Normal(0, 1)` priors on the unbounded scale (spreading over most of each parameter's range), group spreads `HalfNormal(0.5)`, condition effects `Normal(0, 1)`; participant values are non-centred, which samples well with few trials per participant. On a lapse trial (probability `p_lapse`) the response is random with an RT uniform before the deadline; otherwise it follows the model. Answered trials are scored by the LAN mixed with this lapse, and omitted trials by the omission network, with probability (1 − `p_lapse`) × P(omission).

**Extending the model.** `hier.fit` is `build_model` + `sample` + saving. To add something the options do not cover (another likelihood term, a covariate on the group level, different priors), build the model, add to it, and sample:

```python
model = hier.build_model(data, SPEC)           # a pymc.Model
with model:
    ...                                         # extra variables or pm.Potential terms
idata = hier.sample(model, chains=4, draws=1000, tune=1000, target_accept=0.95)
print(hier.summarize(idata))
```

For a different structure altogether, `opn.hierarchical.build_model` shows how the likelihood terms are assembled from `make_trialwise_lan_logp`, `make_trialwise_omission_logp` and `make_trialwise_logp_op`, and `scripts/hierarchical/fit_hierarchical_angle.py` is the paper's hand-built model.

### Demo

`demo/` contains a complete worked example: `demo_data.csv` (6 simulated participants, 200 trials each, an easy/hard condition, 19 % omissions at a 1.5 s deadline, formatted like an experiment export), the true parameters (`demo_truth.csv`), the fit script `fit_demo.py`, and `check_recovery.py`, which compares the fit with the truth.

```bash
apptainer exec --bind "$PWD" opn.sif make demo
```

runs the fit (4 chains × 1000 tuning + 1000 draws) and the comparison, writing to `demo/output/`. It took 36 minutes here (4 CPUs, under x86 emulation on an Apple-silicon Mac; expect it to be faster on a Linux machine). What to expect:

- R-hat 1.00 and ESS above 900 for every parameter.
- Drift rates per participant and condition track the truth closely (correlation 0.89 for easy, 0.97 for hard trials). Boundary and non-decision time less so (0.68 and 0.45): with only 6 participants, their estimates are pulled toward the group mean. The group lapse probability comes out at 0.025 (true participant values 0.014–0.029).
- About 26 of the 30 true participant values fall inside the 94 % intervals.
- The run ends with a warning about a few divergent transitions (2 here). They are harmless in this case: they are isolated rather than concentrated in one region, and the estimates are the same with `target_accept` 0.9, 0.95 and 0.99. This is the check to do with your own data (see step 5).

`demo/make_demo_data.py` regenerates the data (with a fixed seed); edit it to try other designs.

#### Notebook

`demo/fit_demo.ipynb` walks through the same fit step by step, with plots: the data and its omissions, `prepare_data` and `ModelSpec`, the fit, convergence checks (trace and pair plots), group-level estimates and the difficulty effect, participant estimates against the truth, and a posterior predictive check of omission rates and RTs. It reuses the fit in `demo/output/` from `make demo` if there is one, otherwise it fits (set `REFIT = True` in it to fit again). The container includes JupyterLab:

```bash
apptainer exec --bind "$PWD" opn.sif jupyter lab --no-browser --port 8888   # then open the printed http://localhost:8888/... URL
apptainer exec --bind "$PWD" opn.sif make notebook                          # or: run it without a browser, saving the outputs in the notebook
```

On a cluster, start JupyterLab on a compute node and forward the port to your computer first (`ssh -L 8888:localhost:8888 <node>`). Jupyter is also in `requirements.txt`, so the plain Python environment (and VS Code with it as the kernel) works too. Running the x86 container under emulation on an Apple-silicon Mac does not work for notebooks: the kernel hangs as soon as PyTensor starts its compiler; use the plain Python environment there.

### The paper's fit

`scripts/hierarchical/fit_hierarchical_angle.py` fits the angle model to the task data:

- **Subject level:** each of `v`, `a`, `z`, `t`, `theta` (collapse) and `p_outlier` (lapse) has a group mean `mu_*`, a group spread `sigma_*` and subject values `*_subj ~ Normal(mu, sigma)`.
- **Conditions:** `v`, `a` and `theta` also depend on reward, punishment, their interaction and congruency; `z` on congruency. Each effect has a group mean (e.g. `v_catRewLevel`), a spread and subject values (`v_catRewLevel_subj`).
- **Likelihood:** the LAN scores choice and RT of trials answered within 1.25 s (mixed with a uniform lapse of probability `p_outlier`); the omission network scores whether each trial was omitted.
- **Sampling:** NumPyro NUTS, 2 chains of 1000 tuning + 1000 draws.

**Data** (`--data`, default `data/hddm_all.csv`): one row per trial.

| Column | Content |
|---|---|
| `subj_idx` | Subject identifier |
| `rt` | Response time in seconds (any value for omitted trials) |
| `response` | Choice, coded 0/1 |
| `oe` | `TRUE` if the trial was omitted; trials with `rt` > 1.25 s are also treated as omitted |
| `catRewLevel`, `catPunLevel`, `catCong` | Two-level condition labels, coded −1/+1 by the order they first appear in the file (in `hddm_all.csv`: Low Reward, Low Penalty and incongruent are −1) |

**Running it.** Start with a few subjects to check the setup, then fit everyone on the cluster:

```bash
S="apptainer run --bind $PWD opn.sif"
$S scripts/hierarchical/fit_hierarchical_angle.py --n-subjects 5 --out results/hierarchical/five_subjects.nc

sbatch --cpus-per-task=2 --mem=16G --time=72:00:00 \
    --wrap 'apptainer run --bind "$PWD" opn.sif scripts/hierarchical/fit_hierarchical_angle.py'
```

Run time grows with the number of trials (the full data has about 170,000), so give the job a generous time limit; the resources above are a starting point.

**Reading the results.** The output (`--out`, default `results/hierarchical/test_new_pind.nc`) is an ArviZ `InferenceData` file. All parameters are on the unbounded sampling scale; the network maps them into these ranges:

| `v` | `a` | `z` | `t` | `theta` | `p_outlier` |
|---|---|---|---|---|---|
| −3 to 3 | 0.2 to 2.5 | 0.1 to 0.9 | 0.01 to 0.5 | 0 to 1.2 | 0 to 0.05 |

```python
import arviz as az
from scipy.special import expit

idata = az.from_netcdf("results/hierarchical/test_new_pind.nc")
print(az.summary(idata, var_names=["~_trial"], filter_vars="like"))   # all but the per-trial values

def to_range(x, low, high):                    # unbounded value -> parameter range
    return expit(x) * (high - low) + low

a_subj = to_range(idata.posterior["a_subj"], 0.2, 2.5)     # boundary per subject, conditions at 0
a_trial = to_range(idata.posterior["a_trial2"], 0.2, 2.5)  # boundary on every trial, all effects included
```

Subjects are numbered 0, 1, … in the order they first appear in the data (coordinate `id`). The per-trial variables end in `_trial1` (trials answered within the deadline) or `_trial2` (all trials). Condition effects act on the unbounded scale, so compare conditions through the per-trial values or by transforming `subject value ± effect`.

**Adapting it to another dataset** with the same design: rename your columns to the ones above (or edit the recoding block after `pd.read_csv`), and change `DEADLINE = 1.25` near the top of the script (it sets which trials count as omissions, the deadline given to the omission network and the lapse density). For a different design or model, use [Fitting a new dataset](#fitting-a-new-dataset).

## Library reference (`src/opn`)

| Module | Contents |
|---|---|
| `opn.paths` | Repository folders (`NETWORK_DIR`, `DATA_DIR`, `RESULTS_DIR`) |
| `opn.networks` | Registry of the trained networks per model; `load_network`, `network_files`, `input_dim` |
| `opn.likelihoods` | JAX log-likelihoods: summed over trials (`make_lan_logp`, `make_omission_logp`) or per trial with bounded parameters (`make_trialwise_lan_logp`, `make_trialwise_omission_logp`) |
| `opn.ops` | PyTensor Ops wrapping those functions for PyMC (`make_summed_logp_op`, `make_trialwise_logp_op`) |
| `opn.simulate` | `simulate`, `contaminate` (outlier trials), `split_by_deadline` |
| `opn.hierarchical` | Hierarchical fits of any dataset: `prepare_data` (checks and codes a trial table), `ModelSpec`, `fit` (build, sample, save, report), and its steps `build_model`, `sample`, `summarize` |
| `opn.cli` | Command-line options and `output_<jobid>.csv` writing for the simulation scripts |

## Tests

- `make test` runs pytest on `tests/test_*.py`: the likelihood formulas against direct NumPy computations, the Ops' values and gradients against JAX, the PyMC JAX backend, the simulation helpers, the CLI, loading of every network, `test_hierarchical.py` (data checks and coding, specification errors, models for all three SSMs evaluating identically in PyMC and JAX, and `fit` saving its results), and `test_reference.py`, which checks that the library reproduces log-likelihoods and gradients recorded from the original analysis code (`tests/data/reference_loglik.json`).
- `make demo` fits the demo dataset and checks parameter recovery (see [Demo](#demo)).
- `make notebook` executes `demo/fit_demo.ipynb` and saves its outputs in place (see [Demo](#demo)).
- `make smoke` (`tests/quick_test.py`) runs every analysis script for real with tiny settings (one simulated dataset; 3 participants for the paper's hierarchical fit and for the demo fit (`demo/fit_demo.py`); 20 tuning + 20 draws per chain) and prints PASS/FAIL per script, with the error and log path on failure. `python tests/quick_test.py --only ddm weibull` runs a subset; the hierarchical script is tested only if `data/hddm_all.csv` is present.

Each `make smoke` run is saved in `tests/runs/<date-time>/` (not tracked by git; `results/` is never touched):

| File | Contents |
|---|---|
| `report.html` | Self-contained page: overall verdict, status and run time per script, environment (git commit, container, package versions), and per script the full posterior summary of every fit (mean, sd, HDI, MCSE, ESS, R-hat for every parameter), its output file and the end of its log |
| `summary.json` | The same results in machine-readable form |
| `<script>/` | The script's output files, `fit_<n>_summary.csv` per fit (`posterior_summary.csv` for the hierarchical fit), and full `log.txt` |

Open the report with `open tests/runs/<run>/report.html` (macOS) or `xdg-open ...` (Linux), or copy the folder from the cluster and open it locally; the page needs no internet access. `python tests/smoke_report.py tests/runs/<run>` rebuilds a report from its `summary.json`.

Run both after changing the library or a script.

## Plain Python environment

Without containers, Python 3.11 is required (lanfactory needs >3.10, ssm-simulators needs <3.12):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make test
```

The versions are pinned to early 2024 on purpose: the code uses `from jax.config import config` (removed in jax 0.4.25) and `pm.ConstantData` (removed in newer PyMC). `requirements.txt` and `container/opn.def` contain the same pins; change both together.

## Repository layout

```
.
├── container/
│   ├── opn.def                Apptainer definition for the pinned software environment
│   └── build_with_docker.sh   Builds opn.sif via Docker (macOS / no Apptainer)
├── requirements.txt           Same pins for a plain Python 3.11 environment
├── pyproject.toml, Makefile   Package metadata and pytest config; `make test`, `make smoke`
├── src/opn/                   Shared library used by all scripts
├── scripts/
│   ├── simulation/            Parameter-recovery simulations (one job = one grid cell)
│   └── hierarchical/          The paper's hierarchical fit (put your own fit scripts here)
├── demo/                      Demo dataset, fit script (template for new data), recovery check (`make demo`)
│                              and a step-by-step notebook (fit_demo.ipynb)
├── networks/
│   ├── jax/<model>/{lan,cpn}/ Trained LAN / omission networks (lanfactory, used by the code)
│   └── onnx/                  ONNX exports of the omission networks
├── data/hddm_all.csv          Trial-level task data (not tracked by git)
├── results/
│   ├── simulation/            Parameter-recovery outputs (output_<jobid>.csv)
│   └── hierarchical/          Posterior samples from the hierarchical fit (.nc)
└── tests/                     Library tests (test_*.py) and quick_test.py for the scripts
```

### Where things used to be

| Old location | New location |
|---|---|
| `SimulationCode/runModel.py` | `scripts/simulation/recover_angle.py` |
| `SimulationCode/runModel_p.py` | `scripts/simulation/recover_angle_outlier.py` |
| `SimulationCode/runModelDDM.py` | `scripts/simulation/recover_ddm.py` |
| `SimulationCode/runModelWeibull_sig.py` | `scripts/simulation/recover_weibull.py` |
| `Hierachical/runModel.py` | `scripts/hierarchical/fit_hierarchical_angle.py` |
| `Hierachical/hddm_all.csv` | `data/hddm_all.csv` |
| `network/` | `networks/jax/` |
| `Networks/` | `networks/onnx/` |
| `SimulationData/results_DDM/` | `results/simulation/ddm/` |
| `SimulationData/results_angle/` | `results/simulation/angle/` |
| `SimulationData/results_weibull/` | `results/simulation/weibull/` |
| `SimulationData/varying_ddl/` | `results/simulation/varying_deadline/` |
