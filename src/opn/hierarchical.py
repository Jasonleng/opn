"""Hierarchical (multi-subject) fits of a dataset with LAN + omission-network likelihoods.

Typical use (demo/fit_demo.py is a complete script to copy):

    data = prepare_data(pd.read_csv("my_data.csv"), deadline=1.5, subject="participant",
                        rt="rt", response="choice", predictors=["difficulty"])
    spec = ModelSpec("ddm", fixed={"z": 0.5}, regressions={"v": ["difficulty"]})
    idata = fit(data, spec, out_dir="results/hierarchical/my_study", chains=4, draws=1000, tune=1000)

`fit` builds the model, samples, saves and reports; `build_model` and `sample` are the
steps it uses, for models extended by hand.

Model. Every free parameter of the SSM (and the lapse probability) has a group mean
`<p>_mu` and spread `<p>_sigma` on an unbounded scale and subject values
`<p>_subj = <p>_mu + <p>_sigma * <p>_offset` (non-centred). Condition effects
`<p>_<x>_mu` (and per subject `<p>_<x>_subj` with random slopes) add `x * effect` per
trial. The per-trial value is mapped into the parameter's bounds by a scaled sigmoid.
`<p>_subj_natural` / `<p>_mu_natural` give subject / group values on the parameter's
own scale with all predictors at 0.

Likelihood: on a lapse trial (probability p_lapse) the response is random and its RT
uniform before the deadline; otherwise it follows the SSM. Answered trials therefore
have density (1 - p_lapse) * LAN + p_lapse / (2 * deadline), and omitted trials
probability (1 - p_lapse) * P(omission), from the omission network.
"""

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import ssms

from opn import set_cpu_devices
from opn.likelihoods import make_trialwise_lan_logp, make_trialwise_omission_logp
from opn.networks import OMISSION_NETWORK_OUTPUT, load_network
from opn.ops import make_trialwise_logp_op

LAPSE = "p_lapse"
LAPSE_BOUNDS = (0.0, 0.05)
TRUE_STRINGS = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_STRINGS = {"false", "f", "no", "n", "0", "0.0", ""}


# --- Data --------------------------------------------------------------------------

@dataclass
class TrialData:
    """A dataset in the form the model needs (see `prepare_data`)."""
    trials: pd.DataFrame          # subject (0..n-1), rt, response (-1/+1), omitted, predictors
    subjects: list                # original subject labels, index = subject number
    predictors: list
    coding: dict                  # predictor -> {level: code} for non-numeric predictors
    deadline: float

    @property
    def n_omitted(self):
        return int(self.trials["omitted"].sum())


def _as_bool(series, name):
    if series.dtype == bool:
        return series.to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy() != 0
    text = series.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - TRUE_STRINGS - FALSE_STRINGS)
    if unknown:
        raise ValueError(f"column {name!r}: cannot read {unknown[:5]} as true/false")
    return text.isin(TRUE_STRINGS).to_numpy()


def prepare_data(df, deadline, subject="subject", rt="rt", response="response", omitted=None, predictors=()):
    """Check a trial table and code it for `build_model`.

    df: one row per trial. Column names are given by the keyword arguments.
    - rt: response time in seconds. A trial counts as omitted if rt is missing or
      rt >= deadline, or if the optional `omitted` column is true.
    - response: choice of answered trials, coded 0/1 or -1/+1.
    - predictors: condition columns. Numeric columns are used as they are (centre or
      code them as you like, e.g. -1/+1); a text column with two levels is coded -1/+1
      in sorted order of the levels. Other columns need recoding first.
    Subjects are numbered in order of first appearance.
    """
    predictors = list(predictors)
    needed = [subject, rt, response] + ([omitted] if omitted else []) + predictors
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found in the data: {missing}; available: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError("the data has no rows")

    rts = pd.to_numeric(df[rt], errors="coerce").to_numpy(dtype=float)
    is_omitted = ~np.isfinite(rts) | (rts >= deadline)
    if omitted:
        is_omitted |= _as_bool(df[omitted], omitted)

    answered = ~is_omitted
    if np.any(rts[answered] <= 0):
        raise ValueError(f"column {rt!r}: answered trials need rt > 0 (seconds)")
    resp = pd.to_numeric(df[response], errors="coerce").to_numpy(dtype=float)
    values = set(np.unique(resp[answered]))
    if values <= {0.0, 1.0}:
        resp = np.where(resp == 1, 1.0, -1.0)
    elif not values <= {-1.0, 1.0}:
        raise ValueError(f"column {response!r}: answered trials must be coded 0/1 or -1/+1, found {sorted(values)[:5]}")

    subject_codes, subject_labels = pd.factorize(df[subject], sort=False)
    if np.any(subject_codes < 0):
        raise ValueError(f"column {subject!r} has missing values")

    trials = pd.DataFrame({"subject": subject_codes, "rt": np.where(answered, rts, np.nan),
                           "response": np.where(answered, resp, np.nan), "omitted": is_omitted})
    coding = {}
    for x in predictors:
        col = df[x]
        if col.isna().any():
            raise ValueError(f"predictor {x!r} has missing values")
        if pd.api.types.is_numeric_dtype(col) and col.dtype != bool:
            trials[x] = col.to_numpy(dtype=float)
            continue
        levels = sorted(col.astype(str).unique())
        if len(levels) != 2:
            raise ValueError(f"predictor {x!r} has {len(levels)} levels {levels[:5]}; text predictors need exactly "
                             "two levels. Code it as numeric column(s) instead (e.g. one -1/+1 column per contrast).")
        coding[x] = {levels[0]: -1.0, levels[1]: 1.0}
        trials[x] = col.astype(str).map(coding[x]).to_numpy(dtype=float)

    return TrialData(trials, [str(s) for s in subject_labels], predictors, coding, float(deadline))


def first_subjects(df, subject, n):
    """Rows of the first n subjects (in order of appearance); all rows if n is 0."""
    if not n:
        return df
    keep = pd.unique(df[subject])[:n]
    return df[df[subject].isin(keep)].reset_index(drop=True)


# --- Model -------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """What to fit.

    model: "ddm", "angle" or "weibull".
    fixed: parameters held at a value on their own scale, e.g. {"z": 0.5}.
    regressions: condition effects per parameter, e.g. {"v": ["difficulty"]}.
    bounds: replace the default range (the simulator's) of a parameter, e.g. {"t": (0.05, 1.0)}.
    random_slopes: condition effects vary by subject (otherwise one effect for everyone).
    lapse: include a lapse probability (0-5 %, per subject).
    """
    model: str = "ddm"
    fixed: dict = field(default_factory=dict)
    regressions: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    random_slopes: bool = True
    lapse: bool = True

    @property
    def params(self):
        return list(ssms.config.model_config[self.model]["params"])

    @property
    def predictors(self):
        return sorted({x for xs in self.regressions.values() for x in xs})

    @property
    def free(self):
        """Parameters estimated hierarchically, in order (the lapse last)."""
        return [p for p in self.params if p not in self.fixed] + ([LAPSE] if self.lapse else [])

    def param_bounds(self):
        low, high = ssms.config.model_config[self.model]["param_bounds"]
        bounds = {p: (float(lo), float(hi)) for p, lo, hi in zip(self.params, low, high)}
        bounds.update({p: tuple(map(float, b)) for p, b in self.bounds.items()})
        bounds[LAPSE] = LAPSE_BOUNDS
        return {p: b for p, b in bounds.items() if p in self.free}

    def validate(self, predictors=None):
        if self.model not in OMISSION_NETWORK_OUTPUT:
            raise ValueError(f"model must be one of {sorted(OMISSION_NETWORK_OUTPUT)}, got {self.model!r}")
        params = set(self.params)
        for what, keys in (("fixed", self.fixed), ("regressions", self.regressions), ("bounds", self.bounds)):
            unknown = sorted(set(keys) - params)
            if unknown:
                raise ValueError(f"{what}: unknown parameter(s) {unknown}; {self.model} has {self.params}")
        both = sorted(set(self.fixed) & set(self.regressions))
        if both:
            raise ValueError(f"parameter(s) {both} cannot be both fixed and have condition effects")
        for p, (lo, hi) in self.bounds.items():
            if not lo < hi:
                raise ValueError(f"bounds for {p!r}: lower must be below upper, got {(lo, hi)}")
        for p, value in self.fixed.items():
            if not np.isscalar(value):
                raise ValueError(f"fixed value of {p!r} must be a number")
        if predictors is not None:
            missing = sorted(set(self.predictors) - set(predictors))
            if missing:
                raise ValueError(f"regressions use predictor(s) {missing} that are not in the prepared data")


def _sigmoid_range(x, low, high):
    return low + (high - low) * pm.math.sigmoid(x)


def build_model(data, spec):
    """PyMC model of `data` (from `prepare_data`) as described by `spec` (a ModelSpec)."""
    if pytensor.config.floatX != "float32":
        raise RuntimeError('set pytensor.config.floatX = "float32" before building the model '
                           "(the networks are float32)")
    spec.validate(data.predictors)
    params, free, bounds = spec.params, spec.free, spec.param_bounds()

    names = params + [LAPSE]
    is_trialwise = {p: p in free for p in names}
    lan = make_trialwise_logp_op(
        make_trialwise_lan_logp(load_network(spec.model, "lan"), names, bounds,
                                lapse_density=1 / (2 * data.deadline)),   # uniform over (rt < deadline) x 2 choices
        params_is_reg=[is_trialwise[p] for p in names])
    # A lapse never produces an omission: P(omitted) = (1 - p_lapse) * P(omission), which is the
    # outlier mixture with contaminant RTs uniform on [0, deadline]
    omission = make_trialwise_logp_op(
        make_trialwise_omission_logp(load_network(spec.model, "cpn"), params + ["deadline", LAPSE], bounds,
                                     network=OMISSION_NETWORK_OUTPUT[spec.model], outlier=True,
                                     rt_max=data.deadline),
        params_is_reg=[is_trialwise[p] for p in params] + [False, is_trialwise[LAPSE]])

    trials = data.trials
    s_all = trials["subject"].to_numpy()
    answered = np.flatnonzero(~trials["omitted"].to_numpy())
    fixed = {**{p: np.float32(v) for p, v in spec.fixed.items()}, **({} if spec.lapse else {LAPSE: np.float32(0)})}

    with pm.Model(coords={"subject": data.subjects}) as model:
        per_trial = {}
        for p in free:
            mu = pm.Normal(f"{p}_mu", 0, 1)
            sigma = pm.HalfNormal(f"{p}_sigma", 0.5)
            offset = pm.Normal(f"{p}_offset", 0, 1, dims="subject")
            subj = pm.Deterministic(f"{p}_subj", mu + sigma * offset, dims="subject")
            low, high = bounds[p]
            pm.Deterministic(f"{p}_mu_natural", _sigmoid_range(mu, low, high))
            pm.Deterministic(f"{p}_subj_natural", _sigmoid_range(subj, low, high), dims="subject")

            value = subj[s_all]
            for x in spec.regressions.get(p, []):
                effect = pm.Normal(f"{p}_{x}_mu", 0, 1)
                if spec.random_slopes:
                    e_sigma = pm.HalfNormal(f"{p}_{x}_sigma", 0.5)
                    e_offset = pm.Normal(f"{p}_{x}_offset", 0, 1, dims="subject")
                    effect = pm.Deterministic(f"{p}_{x}_subj", effect + e_sigma * e_offset, dims="subject")[s_all]
                value = value + trials[x].to_numpy(dtype=np.float32) * effect
            per_trial[p] = value

        responses = trials.loc[answered, ["rt", "response"]].to_numpy(dtype=np.float32)
        pm.Potential("responses", lan(responses, *[per_trial[p][answered] if p in per_trial else fixed[p]
                                                    for p in names]))
        pm.Potential("omissions", omission(trials["omitted"].to_numpy(dtype=np.float32),
                                           *[per_trial[p] if p in per_trial else fixed[p] for p in params],
                                           np.float32(data.deadline),
                                           per_trial[LAPSE] if LAPSE in per_trial else fixed[LAPSE]))
    return model


# --- Sampling and results ----------------------------------------------------------------

def sample(model, chains=4, draws=1000, tune=1000, target_accept=0.9, seed=None):
    """NUTS with NumPyro (chains run in parallel if JAX has one device per chain)."""
    from pymc.sampling import jax as pmj
    with model:
        return pmj.sample_numpyro_nuts(chains=chains, draws=draws, tune=tune, target_accept=target_accept,
                                       random_seed=seed)


def summarize(idata):
    """ArviZ summary of every parameter except the non-centring offsets."""
    import arviz as az
    return az.summary(idata, var_names=["~_offset"], filter_vars="like")


def diagnostics(summary, idata):
    """Largest R-hat, smallest ESS and number of divergent transitions."""
    return {"max_r_hat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "min_ess_tail": float(summary["ess_tail"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum())}


# --- Whole pipeline ---------------------------------------------------------------------

def fit(data, spec, out_dir, chains=4, draws=1000, tune=1000, target_accept=0.9, seed=None):
    """Fit `spec` to `data`, save the results in out_dir and print a report.

    Writes posterior.nc (the full posterior), summary.csv (every parameter except the
    non-centring offsets) and fit_info.json (data overview, subject order, predictor
    coding, parameter ranges, model, sampling settings, diagnostics). Prints the data
    overview, the group-level estimates and a warning for each convergence problem.
    Returns the InferenceData.
    """
    n = len(data.trials)
    print(f"Data: {len(data.subjects)} subjects, {n} trials, {data.n_omitted} omitted "
          f"({data.n_omitted / n:.1%}) at deadline {data.deadline} s")
    for x, levels in data.coding.items():
        print(f"  predictor {x}: " + ", ".join(f"{level} = {code:+g}" for level, code in levels.items()))
    print(f"Model: {spec.model}; estimated per subject: {', '.join(spec.free)}"
          + (f"; fixed: {spec.fixed}" if spec.fixed else "")
          + (f"; condition effects: {spec.regressions}" if spec.regressions else ""))

    sampling = {"chains": chains, "draws": draws, "tune": tune, "target_accept": target_accept, "seed": seed}
    set_cpu_devices(chains)          # run the chains in parallel, if JAX has not started yet
    model = build_model(data, spec)
    import jax
    if jax.device_count() < chains:
        print(f"Note: JAX has {jax.device_count()} device(s), so the {chains} chains run one after another. "
              "Call opn.set_cpu_devices(chains) at the top of the script to run them in parallel.")
    idata = sample(model, **sampling)

    os.makedirs(out_dir, exist_ok=True)
    idata.to_netcdf(os.path.join(out_dir, "posterior.nc"))
    summary = summarize(idata)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index_label="parameter")
    diag = diagnostics(summary, idata)
    info = {"model": asdict(spec), "deadline": data.deadline, "n_subjects": len(data.subjects), "n_trials": n,
            "n_omitted": data.n_omitted, "subjects": data.subjects, "predictor_coding": data.coding,
            "parameter_bounds": spec.param_bounds(), "sampling": sampling, "diagnostics": diag}
    with open(os.path.join(out_dir, "fit_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    effects = [f"{p}_{x}_mu" for p, xs in spec.regressions.items() for x in xs]
    group = summary.loc[[i for i in summary.index if i.endswith("_mu_natural") or i in effects]]
    print("\nGroup-level estimates (natural scale; condition effects on the unbounded scale):")
    print(group[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())
    print(f"\nDiagnostics: max R-hat {diag['max_r_hat']:.3f}, min bulk ESS {diag['min_ess_bulk']:.0f}, "
          f"min tail ESS {diag['min_ess_tail']:.0f}, divergences {diag['divergences']}")
    for problem in convergence_problems(diag):
        print(f"WARNING: {problem}")
    print(f"\nSaved to {out_dir}")
    return idata


def convergence_problems(diag):
    """Readable descriptions of the problems in a `diagnostics` result (empty if none)."""
    problems = []
    if diag["max_r_hat"] > 1.01:
        problems.append("R-hat above 1.01: the chains disagree; run longer chains (more tune/draws)")
    if min(diag["min_ess_bulk"], diag["min_ess_tail"]) < 400:
        problems.append("ESS below 400: too few effective draws; run longer chains (more draws)")
    if diag["divergences"]:
        problems.append(f"{diag['divergences']} divergent transitions: the sampler struggled somewhere; raise "
                        "target_accept (e.g. 0.95 -> 0.99) and check that the estimates do not change")
    return problems
