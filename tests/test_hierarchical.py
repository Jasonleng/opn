"""opn.hierarchical: data checks and coding, the model specification, the model it builds, and fit()."""

import json
import os

import numpy as np
import pandas as pd
import pytensor
import pytest
from pymc.sampling.jax import get_jaxified_logp

from opn import hierarchical as hier


def trial_table(**overrides):
    trials = pd.DataFrame({"id": ["s2", "s2", "s1", "s1", "s3", "s3"],
                           "rt": [0.5, np.nan, 0.7, 1.4, 0.6, 0.9],
                           "resp": [1, np.nan, 0, 1, 0, 1],
                           "cond": ["hard", "easy", "easy", "hard", "hard", "easy"],
                           "load": [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]})
    return trials.assign(**overrides)


def prepare(trials=None, **options):
    options = {"subject": "id", "rt": "rt", "response": "resp", **options}
    return hier.prepare_data(trial_table() if trials is None else trials, 1.25, **options)


# --- prepare_data -------------------------------------------------------------------------

def test_omissions_from_missing_or_late_rt():
    data = prepare()
    assert data.trials["omitted"].tolist() == [False, True, False, True, False, False]
    assert data.n_omitted == 2
    assert data.trials["rt"].isna().tolist() == data.trials["omitted"].tolist()


def test_omitted_column_with_text_values():
    data = prepare(trial_table(missed=["FALSE", "FALSE", "TRUE", "false", "no", "0"]), omitted="missed")
    assert data.trials["omitted"].tolist() == [False, True, True, True, False, False]


def test_unreadable_omitted_column_is_rejected():
    with pytest.raises(ValueError, match="true/false"):
        prepare(trial_table(missed=["x"] * 6), omitted="missed")


def test_response_coding():
    assert prepare().trials["response"].dropna().tolist() == [1, -1, -1, 1]
    signed = trial_table(resp=[1, np.nan, -1, 1, -1, 1])
    assert prepare(signed).trials["response"].dropna().tolist() == [1, -1, -1, 1]
    with pytest.raises(ValueError, match="0/1 or -1/\\+1"):
        prepare(trial_table(resp=[2, np.nan, 0, 1, 0, 1]))


def test_subjects_in_order_of_appearance():
    data = prepare()
    assert data.subjects == ["s2", "s1", "s3"]
    assert data.trials["subject"].tolist() == [0, 0, 1, 1, 2, 2]


def test_predictor_coding():
    data = prepare(predictors=["cond", "load"])
    assert data.coding == {"cond": {"easy": -1.0, "hard": 1.0}}
    assert data.trials["cond"].tolist() == [1, -1, -1, 1, 1, -1]
    assert data.trials["load"].tolist() == [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]


def test_predictor_with_three_levels_is_rejected():
    with pytest.raises(ValueError, match="exactly two levels"):
        prepare(trial_table(cond=["a", "b", "c", "a", "b", "c"]), predictors=["cond"])


def test_missing_column_is_reported():
    with pytest.raises(ValueError, match="not found"):
        prepare(subject="participant")


def test_first_subjects():
    assert hier.first_subjects(trial_table(), "id", 2)["id"].tolist() == ["s2", "s2", "s1", "s1"]
    assert len(hier.first_subjects(trial_table(), "id", 0)) == 6


# --- ModelSpec ----------------------------------------------------------------------------

def test_spec_free_parameters_and_bounds():
    spec = hier.ModelSpec("ddm", fixed={"z": 0.5}, bounds={"t": (0.05, 1.0)})
    assert spec.free_params == ["v", "a", "t", "p_lapse"]
    assert spec.param_bounds()["t"] == (0.05, 1.0)
    assert spec.param_bounds()["p_lapse"] == hier.LAPSE_BOUNDS
    assert "z" not in spec.param_bounds()
    assert hier.ModelSpec("ddm", lapse=False).free_params == ["v", "a", "z", "t"]


@pytest.mark.parametrize("spec,message", [
    (hier.ModelSpec("lba"), "model must be"),
    (hier.ModelSpec("ddm", fixed={"theta": 0.1}), "unknown parameter"),
    (hier.ModelSpec("ddm", fixed={"v": 1.0}, regressions={"v": ["cond"]}), "both fixed"),
    (hier.ModelSpec("ddm", bounds={"a": (2.0, 1.0)}), "lower must be below upper"),
])
def test_spec_validation(spec, message):
    with pytest.raises(ValueError, match=message):
        spec.validate()


def test_regression_on_unknown_predictor():
    with pytest.raises(ValueError, match="not in the prepared data"):
        hier.ModelSpec("ddm", regressions={"v": ["cond"]}).validate(predictors=[])


# --- build_model --------------------------------------------------------------------------

def simulated_table(model_name, param_values, n_subjects=2, n_trials=40):
    from opn.simulate import simulate
    subject_tables = []
    for subject_index in range(n_subjects):
        simulated = simulate(model_name, param_values, n_trials, random_state=subject_index)
        subject_tables.append(pd.DataFrame({"subject": f"s{subject_index}", "rt": simulated[:, 0],
                                            "response": simulated[:, 1],
                                            "cond": np.tile(["a", "b"], n_trials // 2)}))
    return pd.concat(subject_tables, ignore_index=True)


@pytest.mark.parametrize("model_name,param_values,spec_options", [
    ("ddm", [1.0, 1.2, 0.5, 0.3], {"fixed": {"z": 0.5}, "regressions": {"v": ["cond"]}}),
    ("angle", [1.0, 1.2, 0.5, 0.3, 0.3], {"regressions": {"a": ["cond"]}, "random_slopes": False}),
    ("weibull", [1.0, 1.2, 0.5, 0.3, 2.0, 2.0], {"fixed": {"z": 0.5}, "lapse": False}),
])
def test_model_evaluates_and_matches_jax(model_name, param_values, spec_options):
    spec = hier.ModelSpec(model_name, **spec_options)
    data = hier.prepare_data(simulated_table(model_name, param_values), 1.0, predictors=spec.predictors)
    assert 0 < data.n_omitted < len(data.trials), "test data should contain both kinds of trials"
    model = hier.build_model(data, spec)

    free_rv_names = {random_variable.name for random_variable in model.free_RVs}
    for param in spec.free_params:
        assert {f"{param}_mu", f"{param}_sigma", f"{param}_offset"} <= free_rv_names
    assert not any(name.startswith(f"{param}_") for param in spec.fixed for name in free_rv_names)

    initial_point = model.initial_point()
    logp = float(model.compile_logp()(initial_point))
    assert np.isfinite(logp)
    jax_logp = float(get_jaxified_logp(model)([initial_point[value_var.name] for value_var in model.value_vars]))
    assert jax_logp == pytest.approx(logp, rel=1e-4)

    bounds = spec.param_bounds()
    for param in spec.free_params:   # a group mean of 0 on the unbounded scale is the middle of the range
        natural_value = model[f"{param}_mu_natural"].eval({model[f"{param}_mu"]: np.float32(0)})
        assert float(natural_value) == pytest.approx(sum(bounds[param]) / 2, abs=1e-6)


def test_build_model_requires_float32():
    data = hier.prepare_data(simulated_table("ddm", [1.0, 1.2, 0.5, 0.3]), 1.0)
    pytensor.config.floatX = "float64"
    try:
        with pytest.raises(RuntimeError, match="float32"):
            hier.build_model(data, hier.ModelSpec("ddm"))
    finally:
        pytensor.config.floatX = "float32"


# --- fit ----------------------------------------------------------------------------------

def test_fit_saves_results(tmp_path):
    """The whole pipeline on a tiny problem (results not meaningful)."""
    spec = hier.ModelSpec("ddm", fixed={"z": 0.5}, regressions={"v": ["cond"]})
    data = hier.prepare_data(simulated_table("ddm", [1.0, 1.2, 0.5, 0.3]), 1.0, predictors=spec.predictors)
    idata = hier.fit(data, spec, str(tmp_path), chains=1, draws=10, tune=10, seed=0)
    assert idata.posterior.sizes["draw"] == 10
    assert {"posterior.nc", "summary.csv", "fit_info.json"} <= set(os.listdir(tmp_path))
    fit_info = json.load(open(tmp_path / "fit_info.json"))
    assert fit_info["subjects"] == ["s0", "s1"] and fit_info["model"]["regressions"] == {"v": ["cond"]}
    assert fit_info["predictor_coding"] == {"cond": {"a": -1.0, "b": 1.0}}
    summary = pd.read_csv(tmp_path / "summary.csv", index_col=0)
    assert "v_mu_natural" in summary.index and not any("_offset" in name for name in summary.index)


def test_convergence_problems():
    healthy = {"max_r_hat": 1.0, "min_ess_bulk": 900, "min_ess_tail": 900, "divergences": 0}
    assert hier.convergence_problems(healthy) == []
    problematic = {"max_r_hat": 1.2, "min_ess_bulk": 50, "min_ess_tail": 900, "divergences": 3}
    assert len(hier.convergence_problems(problematic)) == 3
