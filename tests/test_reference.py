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

from opn import likelihoods as lk
from opn.networks import load_network

REF = json.load(open(os.path.join(os.path.dirname(__file__), "data", "reference_loglik.json")))
DATA = np.asarray(REF["inputs"]["data"], dtype=np.float32)
OMITTED = np.asarray(REF["inputs"]["omission"], dtype=np.float32)
RTOL = 1e-5

SUMMED = {  # case -> (model, outlier, n LAN params, n omission params)
    "angle": ("angle", False, 5, 6),
    "ddm": ("ddm", False, 4, 5),
    "angle_outlier": ("angle", True, 6, 7),
}


@pytest.mark.parametrize("case", sorted(SUMMED))
def test_summed_likelihoods_match_original(case):
    model, outlier, n_lan, n_cpn = SUMMED[case]
    ref = REF["cases"][case]
    lan = lk.make_lan_logp(load_network(model, "lan"), outlier=outlier)
    cpn = lk.make_omission_logp(load_network(model, "cpn"), outlier=outlier)
    lp = [np.float32(p) for p in ref["lan_params"]]
    cp = [np.float32(p) for p in ref["cpn_params"]]

    np.testing.assert_allclose(lan(DATA, *lp), ref["lan_logp"], rtol=RTOL)
    np.testing.assert_allclose(jax.grad(lan, argnums=range(1, 1 + n_lan))(DATA, *lp), ref["lan_grad"], rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(cpn(OMITTED, *cp), ref["cpn_logp"], rtol=RTOL)
    np.testing.assert_allclose(jax.grad(cpn, argnums=range(1, 1 + n_cpn))(OMITTED, *cp), ref["cpn_grad"], rtol=RTOL, atol=1e-5)


def _check_trialwise(lan, cpn, ref, lan_reg, cpn_reg, omitted):
    lp = [np.asarray(p, dtype=np.float32) for p in ref["lan_params"]]
    cp = [np.asarray(p, dtype=np.float32) for p in ref["cpn_params"]]
    for logp, data, params, is_reg, key in ((lan, DATA, lp, lan_reg, "lan"), (cpn, omitted, cp, cpn_reg, "cpn")):
        vmapped = jax.vmap(logp, in_axes=[0] + [0 if r else None for r in is_reg])
        value, vjp_fn = jax.vjp(vmapped, data, *params)
        np.testing.assert_allclose(value, ref[f"{key}_logp"], rtol=RTOL)
        for got, exp in zip(vjp_fn(jnp.ones_like(value))[1:], ref[f"{key}_vjp"]):
            np.testing.assert_allclose(got, exp, rtol=RTOL, atol=1e-5)


def test_weibull_likelihoods_match_original():
    bounds = {"v": (-3, 3), "a": (0.2, 2.5), "t": (0.01, 0.5), "alpha": (0.3, 4.5), "beta": (0.3, 4.5)}
    lan = lk.make_trialwise_lan_logp(load_network("weibull", "lan"),
                                     ["v", "a", "z", "t", "alpha", "beta", "p_outlier"], bounds,
                                     lapse_density=lk.OUTLIER_DENSITY)
    cpn = lk.make_trialwise_omission_logp(load_network("weibull", "cpn"),
                                          ["v", "a", "z", "t", "alpha", "beta", "deadline", "p_outlier"], bounds,
                                          network="opn", outlier=True)
    _check_trialwise(lan, cpn, REF["cases"]["weibull"], [False] * 7, [False] * 8, OMITTED)


def test_hierarchical_likelihoods_match_original():
    bounds = {"v": (-3, 3), "a": (0.2, 2.5), "z": (0.1, 0.9), "t": (0.01, 0.5), "theta": (0, 1.2)}
    lan = lk.make_trialwise_lan_logp(load_network("angle", "lan"), ["v", "a", "z", "t", "theta", "p_outlier"],
                                     {**bounds, "p_outlier": (0, 0.05)}, lapse_density=1 / 2.5)
    cpn = lk.make_trialwise_omission_logp(load_network("angle", "cpn"),
                                          ["v", "a", "z", "t", "theta", "deadline"], bounds)
    _check_trialwise(lan, cpn, REF["cases"]["hierarchical"], [True] * 6, [True] * 5 + [False], OMITTED.astype(bool))
