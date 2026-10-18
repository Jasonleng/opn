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


def _as_bool(series, column_name):
    if series.dtype == bool:
        return series.to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy() != 0
    normalized_text = series.fillna("").astype(str).str.strip().str.lower()
    unreadable_values = sorted(set(normalized_text) - TRUE_STRINGS - FALSE_STRINGS)
    if unreadable_values:
        raise ValueError(f"column {column_name!r}: cannot read {unreadable_values[:5]} as true/false")
    return normalized_text.isin(TRUE_STRINGS).to_numpy()


def prepare_data(trial_table, deadline, subject="subject", rt="rt", response="response", omitted=None,
                 predictors=()):
    """Check a trial table and code it for `build_model`.

    trial_table: one row per trial. Column names are given by the keyword arguments.
    - rt: response time in seconds. A trial counts as omitted if rt is missing or
      rt >= deadline, or if the optional `omitted` column is true.
    - response: choice of answered trials, coded 0/1 or -1/+1.
    - predictors: condition columns. Numeric columns are used as they are (centre or
      code them as you like, e.g. -1/+1); a text column with two levels is coded -1/+1
      in sorted order of the levels. Other columns need recoding first.
    Subjects are numbered in order of first appearance.
    """
    predictors = list(predictors)
    required_columns = [subject, rt, response] + ([omitted] if omitted else []) + predictors
    missing_columns = [column for column in required_columns if column not in trial_table.columns]
    if missing_columns:
        raise ValueError(f"columns not found in the data: {missing_columns}; "
                         f"available: {list(trial_table.columns)}")
    if len(trial_table) == 0:
        raise ValueError("the data has no rows")

    rt_values = pd.to_numeric(trial_table[rt], errors="coerce").to_numpy(dtype=float)
    is_omitted = ~np.isfinite(rt_values) | (rt_values >= deadline)
    if omitted:
        is_omitted |= _as_bool(trial_table[omitted], omitted)

    answered = ~is_omitted
    if np.any(rt_values[answered] <= 0):
        raise ValueError(f"column {rt!r}: answered trials need rt > 0 (seconds)")
    response_values = pd.to_numeric(trial_table[response], errors="coerce").to_numpy(dtype=float)
    response_codes = set(np.unique(response_values[answered]))
    if response_codes <= {0.0, 1.0}:
        response_values = np.where(response_values == 1, 1.0, -1.0)
    elif not response_codes <= {-1.0, 1.0}:
        raise ValueError(f"column {response!r}: answered trials must be coded 0/1 or -1/+1, "
                         f"found {sorted(response_codes)[:5]}")

    subject_codes, subject_labels = pd.factorize(trial_table[subject], sort=False)
    if np.any(subject_codes < 0):
        raise ValueError(f"column {subject!r} has missing values")

    trials = pd.DataFrame({"subject": subject_codes, "rt": np.where(answered, rt_values, np.nan),
                           "response": np.where(answered, response_values, np.nan), "omitted": is_omitted})
    coding = {}
    for predictor in predictors:
        predictor_column = trial_table[predictor]
        if predictor_column.isna().any():
            raise ValueError(f"predictor {predictor!r} has missing values")
        if pd.api.types.is_numeric_dtype(predictor_column) and predictor_column.dtype != bool:
            trials[predictor] = predictor_column.to_numpy(dtype=float)
            continue
        levels = sorted(predictor_column.astype(str).unique())
        if len(levels) != 2:
            raise ValueError(f"predictor {predictor!r} has {len(levels)} levels {levels[:5]}; text predictors "
                             "need exactly two levels. Code it as numeric column(s) instead (e.g. one -1/+1 "
                             "column per contrast).")
        coding[predictor] = {levels[0]: -1.0, levels[1]: 1.0}
        trials[predictor] = predictor_column.astype(str).map(coding[predictor]).to_numpy(dtype=float)

    return TrialData(trials, [str(label) for label in subject_labels], predictors, coding, float(deadline))


def first_subjects(trial_table, subject, n_subjects):
    """Rows of the first n_subjects subjects (in order of appearance); all rows if n_subjects is 0."""
    if not n_subjects:
        return trial_table
    kept_subjects = pd.unique(trial_table[subject])[:n_subjects]
    return trial_table[trial_table[subject].isin(kept_subjects)].reset_index(drop=True)


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
        return sorted({predictor for predictor_list in self.regressions.values() for predictor in predictor_list})

    @property
    def free_params(self):
        """Parameters estimated hierarchically, in order (the lapse last)."""
        return [param for param in self.params if param not in self.fixed] + ([LAPSE] if self.lapse else [])

    def param_bounds(self):
        lower_bounds, upper_bounds = ssms.config.model_config[self.model]["param_bounds"]
        bounds = {param: (float(lower), float(upper))
                  for param, lower, upper in zip(self.params, lower_bounds, upper_bounds)}
        bounds.update({param: tuple(map(float, custom_bounds)) for param, custom_bounds in self.bounds.items()})
        bounds[LAPSE] = LAPSE_BOUNDS
        return {param: param_range for param, param_range in bounds.items() if param in self.free_params}

    def validate(self, predictors=None):
        if self.model not in OMISSION_NETWORK_OUTPUT:
            raise ValueError(f"model must be one of {sorted(OMISSION_NETWORK_OUTPUT)}, got {self.model!r}")
        known_params = set(self.params)
        for option_name, option_params in (("fixed", self.fixed), ("regressions", self.regressions),
                                           ("bounds", self.bounds)):
            unknown_params = sorted(set(option_params) - known_params)
            if unknown_params:
                raise ValueError(f"{option_name}: unknown parameter(s) {unknown_params}; "
                                 f"{self.model} has {self.params}")
        fixed_and_regressed = sorted(set(self.fixed) & set(self.regressions))
        if fixed_and_regressed:
            raise ValueError(f"parameter(s) {fixed_and_regressed} cannot be both fixed and have condition effects")
        for param, (lower, upper) in self.bounds.items():
            if not lower < upper:
                raise ValueError(f"bounds for {param!r}: lower must be below upper, got {(lower, upper)}")
        for param, fixed_value in self.fixed.items():
            if not np.isscalar(fixed_value):
                raise ValueError(f"fixed value of {param!r} must be a number")
        if predictors is not None:
            missing_predictors = sorted(set(self.predictors) - set(predictors))
            if missing_predictors:
                raise ValueError(f"regressions use predictor(s) {missing_predictors} that are not in the "
                                 "prepared data")


def _sigmoid_range(unbounded_value, low, high):
    return low + (high - low) * pm.math.sigmoid(unbounded_value)


def build_model(data, spec):
    """PyMC model of `data` (from `prepare_data`) as described by `spec` (a ModelSpec)."""
    if pytensor.config.floatX != "float32":
        raise RuntimeError('set pytensor.config.floatX = "float32" before building the model '
                           "(the networks are float32)")
    spec.validate(data.predictors)
    params, free_params, bounds = spec.params, spec.free_params, spec.param_bounds()

    lan_param_names = params + [LAPSE]
    varies_by_trial = {param: param in free_params for param in lan_param_names}
    lan_logp_op = make_trialwise_logp_op(
        make_trialwise_lan_logp(load_network(spec.model, "lan"), lan_param_names, bounds,
                                lapse_density=1 / (2 * data.deadline)),   # uniform over (rt < deadline) x 2 choices
        varies_by_trial=[varies_by_trial[param] for param in lan_param_names])
    # A lapse never produces an omission: P(omitted) = (1 - p_lapse) * P(omission), which is the
    # outlier mixture with contaminant RTs uniform on [0, deadline]
    omission_logp_op = make_trialwise_logp_op(
        make_trialwise_omission_logp(load_network(spec.model, "cpn"), params + ["deadline", LAPSE], bounds,
                                     network=OMISSION_NETWORK_OUTPUT[spec.model], outlier=True,
                                     rt_max=data.deadline),
        varies_by_trial=[varies_by_trial[param] for param in params] + [False, varies_by_trial[LAPSE]])

    trials = data.trials
    subject_of_trial = trials["subject"].to_numpy()
    answered_trials = np.flatnonzero(~trials["omitted"].to_numpy())
    fixed_values = {**{param: np.float32(value) for param, value in spec.fixed.items()},
                    **({} if spec.lapse else {LAPSE: np.float32(0)})}

    with pm.Model(coords={"subject": data.subjects}) as model:
        trial_values = {}
        for param in free_params:
            group_mean = pm.Normal(f"{param}_mu", 0, 1)
            group_sd = pm.HalfNormal(f"{param}_sigma", 0.5)
            subject_offset = pm.Normal(f"{param}_offset", 0, 1, dims="subject")
            subject_values = pm.Deterministic(f"{param}_subj", group_mean + group_sd * subject_offset,
                                              dims="subject")
            low, high = bounds[param]
            pm.Deterministic(f"{param}_mu_natural", _sigmoid_range(group_mean, low, high))
            pm.Deterministic(f"{param}_subj_natural", _sigmoid_range(subject_values, low, high), dims="subject")

            trial_value = subject_values[subject_of_trial]
            for predictor in spec.regressions.get(param, []):
                effect = pm.Normal(f"{param}_{predictor}_mu", 0, 1)
                if spec.random_slopes:
                    effect_sd = pm.HalfNormal(f"{param}_{predictor}_sigma", 0.5)
                    effect_offset = pm.Normal(f"{param}_{predictor}_offset", 0, 1, dims="subject")
                    effect = pm.Deterministic(f"{param}_{predictor}_subj", effect + effect_sd * effect_offset,
                                              dims="subject")[subject_of_trial]
                trial_value = trial_value + trials[predictor].to_numpy(dtype=np.float32) * effect
            trial_values[param] = trial_value

        response_data = trials.loc[answered_trials, ["rt", "response"]].to_numpy(dtype=np.float32)
        pm.Potential("responses", lan_logp_op(
            response_data, *[trial_values[param][answered_trials] if param in trial_values else fixed_values[param]
                             for param in lan_param_names]))
        pm.Potential("omissions", omission_logp_op(
            trials["omitted"].to_numpy(dtype=np.float32),
            *[trial_values[param] if param in trial_values else fixed_values[param] for param in params],
            np.float32(data.deadline),
            trial_values[LAPSE] if LAPSE in trial_values else fixed_values[LAPSE]))
    return model


# --- Sampling and results ----------------------------------------------------------------

def sample(model, chains=4, draws=1000, tune=1000, target_accept=0.9, seed=None):
    """NUTS with NumPyro (chains run in parallel if JAX has one device per chain)."""
    from pymc.sampling import jax as pymc_jax
    with model:
        return pymc_jax.sample_numpyro_nuts(chains=chains, draws=draws, tune=tune, target_accept=target_accept,
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
    n_trials = len(data.trials)
    print(f"Data: {len(data.subjects)} subjects, {n_trials} trials, {data.n_omitted} omitted "
          f"({data.n_omitted / n_trials:.1%}) at deadline {data.deadline} s")
    for predictor, level_codes in data.coding.items():
        print(f"  predictor {predictor}: " + ", ".join(f"{level} = {code:+g}" for level, code in level_codes.items()))
    print(f"Model: {spec.model}; estimated per subject: {', '.join(spec.free_params)}"
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
    diagnostic_values = diagnostics(summary, idata)
    fit_info = {"model": asdict(spec), "deadline": data.deadline, "n_subjects": len(data.subjects),
                "n_trials": n_trials, "n_omitted": data.n_omitted, "subjects": data.subjects,
                "predictor_coding": data.coding, "parameter_bounds": spec.param_bounds(), "sampling": sampling,
                "diagnostics": diagnostic_values}
    with open(os.path.join(out_dir, "fit_info.json"), "w") as info_file:
        json.dump(fit_info, info_file, indent=2)

    effect_names = [f"{param}_{predictor}_mu" for param, predictor_list in spec.regressions.items()
                    for predictor in predictor_list]
    group_estimates = summary.loc[[name for name in summary.index
                                   if name.endswith("_mu_natural") or name in effect_names]]
    print("\nGroup-level estimates (natural scale; condition effects on the unbounded scale):")
    print(group_estimates[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())
    print(f"\nDiagnostics: max R-hat {diagnostic_values['max_r_hat']:.3f}, "
          f"min bulk ESS {diagnostic_values['min_ess_bulk']:.0f}, "
          f"min tail ESS {diagnostic_values['min_ess_tail']:.0f}, divergences {diagnostic_values['divergences']}")
    for problem in convergence_problems(diagnostic_values):
        print(f"WARNING: {problem}")
    print(f"\nSaved to {out_dir}")
    return idata


def convergence_problems(diagnostic_values):
    """Readable descriptions of the problems in a `diagnostics` result (empty if none)."""
    problems = []
    if diagnostic_values["max_r_hat"] > 1.01:
        problems.append("R-hat above 1.01: the chains disagree; run longer chains (more tune/draws)")
    if min(diagnostic_values["min_ess_bulk"], diagnostic_values["min_ess_tail"]) < 400:
        problems.append("ESS below 400: too few effective draws; run longer chains (more draws)")
    if diagnostic_values["divergences"]:
        problems.append(f"{diagnostic_values['divergences']} divergent transitions: the sampler struggled "
                        "somewhere; raise target_accept (e.g. 0.95 -> 0.99) and check that the estimates do "
                        "not change")
    return problems
