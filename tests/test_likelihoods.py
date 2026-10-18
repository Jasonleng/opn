"""The likelihood formulas, checked against direct NumPy computations with a stand-in network."""

import numpy as np
import pytest
from scipy.special import expit

from opn import likelihoods

DATA = np.array([[0.5, 1.0], [0.8, -1.0], [1.1, 1.0], [0.4, -1.0]], dtype=np.float32)
OMITTED = np.array([1, 0, 1, 0, 0], dtype=np.float32)
PARAMS = [np.float32(value) for value in (0.4, 1.2, 0.5, 0.3)]


def fake_log_output(network_input):
    return -0.5 * np.sum(np.asarray(network_input, dtype=np.float64) ** 2, axis=-1)


def expected_outlier_probability(unbounded_weight):
    return 0.1 * expit(unbounded_weight)


def test_scaled_sigmoid_maps_into_bounds():
    assert float(likelihoods.scaled_sigmoid(0.0, 0.2, 2.5)) == pytest.approx(1.35)
    assert 0.2 < float(likelihoods.scaled_sigmoid(-30.0, 0.2, 2.5)) < 0.2001
    assert 2.4999 < float(likelihoods.scaled_sigmoid(30.0, 0.2, 2.5)) <= 2.5


def test_transform_params_only_touches_bounded_names():
    transformed = likelihoods.transform_params([0.0, 0.7], ["a", "z"], {"a": (0.2, 2.5)})
    assert float(transformed[0]) == pytest.approx(1.35)
    assert transformed[1] == 0.7


def test_omission_probability_network_types():
    network_output = np.log(0.8)
    assert float(likelihoods.omission_probability(network_output, "cpn")) == pytest.approx(0.2)
    assert float(likelihoods.omission_probability(network_output, "opn")) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        likelihoods.omission_probability(network_output, "lan")


def test_lan_logp(fake_forward):
    expected = sum(fake_log_output(np.r_[PARAMS, trial]) for trial in DATA)
    actual = likelihoods.make_lan_logp(fake_forward)(DATA, *PARAMS)
    assert float(actual) == pytest.approx(expected, rel=1e-5)


def test_lan_logp_with_outlier_mixture(fake_forward):
    unbounded_weight = np.float32(-1.0)
    p_outlier = expected_outlier_probability(unbounded_weight)
    expected = sum(np.log(p_outlier * 0.1 + (1 - p_outlier) * np.exp(fake_log_output(np.r_[PARAMS, trial])))
                   for trial in DATA)
    actual = likelihoods.make_lan_logp(fake_forward, outlier=True)(DATA, *PARAMS, unbounded_weight)
    assert float(actual) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("network", ["cpn", "opn"])
def test_omission_logp(fake_forward, network):
    params = PARAMS + [np.float32(1.25)]
    network_probability = np.exp(fake_log_output(params))
    p_omission = 1 - network_probability if network == "cpn" else network_probability
    expected = OMITTED.sum() * np.log(p_omission)
    actual = likelihoods.make_omission_logp(fake_forward, network=network)(OMITTED, *params)
    assert float(actual) == pytest.approx(expected, rel=1e-5)


def test_omission_logp_with_outlier_mixture(fake_forward):
    deadline, unbounded_weight = np.float32(1.25), np.float32(-1.0)
    p_outlier = expected_outlier_probability(unbounded_weight)
    p_omission = 1 - np.exp(fake_log_output(PARAMS + [deadline]))
    expected = OMITTED.sum() * np.log((1 - p_outlier) * p_omission + p_outlier * (1 - deadline / 5))
    actual = likelihoods.make_omission_logp(fake_forward, outlier=True)(OMITTED, *PARAMS, deadline, unbounded_weight)
    assert float(actual) == pytest.approx(expected, rel=1e-5)


def test_trialwise_lan_logp(fake_forward):
    names, bounds = ["v", "a", "p_outlier"], {"a": (0.2, 2.5), "p_outlier": (0, 0.05)}
    unbounded_params = [np.float32(0.3), np.float32(0.0), np.float32(0.5)]
    a_natural, p_lapse = 1.35, 0.05 * expit(0.5)
    expected = np.log(np.exp(fake_log_output([0.3, a_natural, *DATA[0]])) * (1 - p_lapse) + p_lapse / 2.5)
    actual = likelihoods.make_trialwise_lan_logp(fake_forward, names, bounds, lapse_density=1 / 2.5)(
        DATA[0], *unbounded_params)
    assert float(actual) == pytest.approx(expected, rel=1e-5)


def test_trialwise_omission_logp_is_zero_for_responded_trials(fake_forward):
    logp = likelihoods.make_trialwise_omission_logp(fake_forward, ["v", "deadline"], {})
    assert float(logp(np.float32(0), np.float32(0.3), np.float32(1.25))) == 0.0
    expected = np.log(1 - np.exp(fake_log_output([0.3, 1.25])))
    assert float(logp(np.float32(1), np.float32(0.3), np.float32(1.25))) == pytest.approx(expected, rel=1e-5)


def test_trialwise_omission_lapse_within_deadline(fake_forward):
    """rt_max = deadline: a lapse always responds in time, so P(omitted) = (1 - p) * P(omission)."""
    names = ["v", "deadline", "p_lapse"]
    logp = likelihoods.make_trialwise_omission_logp(fake_forward, names, {}, outlier=True, rt_max=1.25)
    p_lapse = 0.03
    expected = np.log((1 - p_lapse) * (1 - np.exp(fake_log_output([0.3, 1.25]))))
    actual = logp(np.float32(1), np.float32(0.3), np.float32(1.25), np.float32(p_lapse))
    assert float(actual) == pytest.approx(expected, rel=1e-5)


def test_trialwise_omission_logp_with_outlier(fake_forward):
    names = ["v", "deadline", "p_outlier"]
    logp = likelihoods.make_trialwise_omission_logp(fake_forward, names, {}, network="opn", outlier=True)
    p_outlier = 0.02
    expected = np.log((1 - p_outlier) * np.exp(fake_log_output([0.3, 1.25])) + p_outlier * (1 - 1.25 / 5))
    actual = logp(np.float32(1), np.float32(0.3), np.float32(1.25), np.float32(p_outlier))
    assert float(actual) == pytest.approx(expected, rel=1e-5)
