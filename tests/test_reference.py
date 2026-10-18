"""The library reproduces the original analysis code.

tests/data/reference_loglik.json holds log-likelihoods and gradients computed with
the NetworkLike classes that were copied into each script before the shared
library existed (SimulationCode/*.py at commit eedff16 and Hierachical/runModel.py),
using the real trained networks. Each case here rebuilds the same likelihood from
`opn` and must match.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opn import likelihoods
from opn.networks import load_network

REFERENCE = json.load(open(os.path.join(os.path.dirname(__file__), "data", "reference_loglik.json")))
DATA = np.asarray(REFERENCE["inputs"]["data"], dtype=np.float32)
OMITTED = np.asarray(REFERENCE["inputs"]["omission"], dtype=np.float32)
RTOL = 1e-5

SUMMED = {  # case -> (model, outlier, n LAN params, n omission params)
    "angle": ("angle", False, 5, 6),
    "ddm": ("ddm", False, 4, 5),
    "angle_outlier": ("angle", True, 6, 7),
}


@pytest.mark.parametrize("case", sorted(SUMMED))
def test_summed_likelihoods_match_original(case):
    model, outlier, n_lan_params, n_omission_params = SUMMED[case]
    reference = REFERENCE["cases"][case]
    lan_logp = likelihoods.make_lan_logp(load_network(model, "lan"), outlier=outlier)
    omission_logp = likelihoods.make_omission_logp(load_network(model, "cpn"), outlier=outlier)
    lan_params = [np.float32(value) for value in reference["lan_params"]]
    omission_params = [np.float32(value) for value in reference["cpn_params"]]

    np.testing.assert_allclose(lan_logp(DATA, *lan_params), reference["lan_logp"], rtol=RTOL)
    np.testing.assert_allclose(jax.grad(lan_logp, argnums=range(1, 1 + n_lan_params))(DATA, *lan_params),
                               reference["lan_grad"], rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(omission_logp(OMITTED, *omission_params), reference["cpn_logp"], rtol=RTOL)
    np.testing.assert_allclose(
        jax.grad(omission_logp, argnums=range(1, 1 + n_omission_params))(OMITTED, *omission_params),
        reference["cpn_grad"], rtol=RTOL, atol=1e-5)


def _check_trialwise(lan_logp, omission_logp, reference, lan_varies_by_trial, omission_varies_by_trial, omitted):
    lan_params = [np.asarray(value, dtype=np.float32) for value in reference["lan_params"]]
    omission_params = [np.asarray(value, dtype=np.float32) for value in reference["cpn_params"]]
    for logp, data, params, varies_by_trial, network_kind in (
            (lan_logp, DATA, lan_params, lan_varies_by_trial, "lan"),
            (omission_logp, omitted, omission_params, omission_varies_by_trial, "cpn")):
        batched_logp = jax.vmap(logp, in_axes=[0] + [0 if param_varies else None for param_varies in varies_by_trial])
        value, pullback = jax.vjp(batched_logp, data, *params)
        np.testing.assert_allclose(value, reference[f"{network_kind}_logp"], rtol=RTOL)
        for actual_vjp, expected_vjp in zip(pullback(jnp.ones_like(value))[1:], reference[f"{network_kind}_vjp"]):
            np.testing.assert_allclose(actual_vjp, expected_vjp, rtol=RTOL, atol=1e-5)


def test_weibull_likelihoods_match_original():
    bounds = {"v": (-3, 3), "a": (0.2, 2.5), "t": (0.01, 0.5), "alpha": (0.3, 4.5), "beta": (0.3, 4.5)}
    lan_logp = likelihoods.make_trialwise_lan_logp(load_network("weibull", "lan"),
                                                   ["v", "a", "z", "t", "alpha", "beta", "p_outlier"], bounds,
                                                   lapse_density=likelihoods.OUTLIER_DENSITY)
    omission_logp = likelihoods.make_trialwise_omission_logp(
        load_network("weibull", "cpn"), ["v", "a", "z", "t", "alpha", "beta", "deadline", "p_outlier"], bounds,
        network="opn", outlier=True)
    _check_trialwise(lan_logp, omission_logp, REFERENCE["cases"]["weibull"], [False] * 7, [False] * 8, OMITTED)


def test_hierarchical_likelihoods_match_original():
    bounds = {"v": (-3, 3), "a": (0.2, 2.5), "z": (0.1, 0.9), "t": (0.01, 0.5), "theta": (0, 1.2)}
    lan_logp = likelihoods.make_trialwise_lan_logp(load_network("angle", "lan"),
                                                   ["v", "a", "z", "t", "theta", "p_outlier"],
                                                   {**bounds, "p_outlier": (0, 0.05)}, lapse_density=1 / 2.5)
    omission_logp = likelihoods.make_trialwise_omission_logp(load_network("angle", "cpn"),
                                                             ["v", "a", "z", "t", "theta", "deadline"], bounds)
    _check_trialwise(lan_logp, omission_logp, REFERENCE["cases"]["hierarchical"], [True] * 6, [True] * 5 + [False],
                     OMITTED.astype(bool))
