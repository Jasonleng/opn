"""The likelihood formulas, checked against direct NumPy computations with a stand-in network."""

import numpy as np
import pytest
from scipy.special import expit

from opn import likelihoods as lk

DATA = np.array([[0.5, 1.0], [0.8, -1.0], [1.1, 1.0], [0.4, -1.0]], dtype=np.float32)
OMITTED = np.array([1, 0, 1, 0, 0], dtype=np.float32)
PARAMS = [np.float32(x) for x in (0.4, 1.2, 0.5, 0.3)]


def fake_log_output(x):
    return -0.5 * np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1)


def outlier_p(x):
    return 0.1 * expit(x)


def test_scaled_sigmoid_maps_into_bounds():
    assert float(lk.scaled_sigmoid(0.0, 0.2, 2.5)) == pytest.approx(1.35)
    assert 0.2 < float(lk.scaled_sigmoid(-30.0, 0.2, 2.5)) < 0.2001
    assert 2.4999 < float(lk.scaled_sigmoid(30.0, 0.2, 2.5)) <= 2.5


def test_transform_params_only_touches_bounded_names():
    out = lk.transform_params([0.0, 0.7], ["a", "z"], {"a": (0.2, 2.5)})
    assert float(out[0]) == pytest.approx(1.35)
    assert out[1] == 0.7


def test_omission_probability_network_types():
    net_out = np.log(0.8)
    assert float(lk.omission_probability(net_out, "cpn")) == pytest.approx(0.2)
    assert float(lk.omission_probability(net_out, "opn")) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        lk.omission_probability(net_out, "lan")


def test_lan_logp(fake_forward):
    expected = sum(fake_log_output(np.r_[PARAMS, row]) for row in DATA)
    got = lk.make_lan_logp(fake_forward)(DATA, *PARAMS)
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_lan_logp_with_outlier_mixture(fake_forward):
    raw_p = np.float32(-1.0)
    p = outlier_p(raw_p)
    expected = sum(np.log(p * 0.1 + (1 - p) * np.exp(fake_log_output(np.r_[PARAMS, row]))) for row in DATA)
    got = lk.make_lan_logp(fake_forward, outlier=True)(DATA, *PARAMS, raw_p)
    assert float(got) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("network", ["cpn", "opn"])
def test_omission_logp(fake_forward, network):
    params = PARAMS + [np.float32(1.25)]
    net = np.exp(fake_log_output(params))
    p_om = 1 - net if network == "cpn" else net
    expected = OMITTED.sum() * np.log(p_om)
    got = lk.make_omission_logp(fake_forward, network=network)(OMITTED, *params)
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_omission_logp_with_outlier_mixture(fake_forward):
    deadline, raw_p = np.float32(1.25), np.float32(-1.0)
    p = outlier_p(raw_p)
    p_om = 1 - np.exp(fake_log_output(PARAMS + [deadline]))
    expected = OMITTED.sum() * np.log((1 - p) * p_om + p * (1 - deadline / 5))
    got = lk.make_omission_logp(fake_forward, outlier=True)(OMITTED, *PARAMS, deadline, raw_p)
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_trialwise_lan_logp(fake_forward):
    names, bounds = ["v", "a", "p_outlier"], {"a": (0.2, 2.5), "p_outlier": (0, 0.05)}
    raw = [np.float32(0.3), np.float32(0.0), np.float32(0.5)]
    a, p = 1.35, 0.05 * expit(0.5)
    expected = np.log(np.exp(fake_log_output([0.3, a, *DATA[0]])) * (1 - p) + p / 2.5)
    got = lk.make_trialwise_lan_logp(fake_forward, names, bounds, lapse_density=1 / 2.5)(DATA[0], *raw)
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_trialwise_omission_logp_is_zero_for_responded_trials(fake_forward):
    logp = lk.make_trialwise_omission_logp(fake_forward, ["v", "deadline"], {})
    assert float(logp(np.float32(0), np.float32(0.3), np.float32(1.25))) == 0.0
    expected = np.log(1 - np.exp(fake_log_output([0.3, 1.25])))
    assert float(logp(np.float32(1), np.float32(0.3), np.float32(1.25))) == pytest.approx(expected, rel=1e-5)


def test_trialwise_omission_lapse_within_deadline(fake_forward):
    """rt_max = deadline: a lapse always responds in time, so P(omitted) = (1 - p) * P(omission)."""
    names = ["v", "deadline", "p_lapse"]
    logp = lk.make_trialwise_omission_logp(fake_forward, names, {}, outlier=True, rt_max=1.25)
    p = 0.03
    expected = np.log((1 - p) * (1 - np.exp(fake_log_output([0.3, 1.25]))))
    got = logp(np.float32(1), np.float32(0.3), np.float32(1.25), np.float32(p))
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_trialwise_omission_logp_with_outlier(fake_forward):
    names = ["v", "deadline", "p_outlier"]
    logp = lk.make_trialwise_omission_logp(fake_forward, names, {}, network="opn", outlier=True)
    p = 0.02
    expected = np.log((1 - p) * np.exp(fake_log_output([0.3, 1.25])) + p * (1 - 1.25 / 5))
    got = logp(np.float32(1), np.float32(0.3), np.float32(1.25), np.float32(p))
    assert float(got) == pytest.approx(expected, rel=1e-5)
