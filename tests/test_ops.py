"""The PyTensor Ops: values and gradients match JAX, and PyMC's JAX backend sees the JAX function."""

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.sampling.jax import get_jaxified_logp

from opn.likelihoods import make_lan_logp, make_trialwise_lan_logp
from opn.ops import make_summed_logp_op, make_trialwise_logp_op

DATA = np.array([[0.5, 1.0], [0.8, -1.0], [1.1, 1.0]], dtype=np.float32)
PARAMS = [np.float32(value) for value in (0.4, 1.2, 0.5)]


def test_summed_op_value_and_gradient(fake_forward):
    logp = make_lan_logp(fake_forward)
    logp_op = make_summed_logp_op(logp, n_params=3)
    param_vars = [pt.scalar(f"param{index}") for index in range(3)]
    logp_output = logp_op(DATA, *param_vars)
    compiled = pytensor.function(param_vars, [logp_output] + pytensor.grad(logp_output, param_vars))
    value, *gradients = compiled(*PARAMS)
    assert value == pytest.approx(float(logp(DATA, *PARAMS)), rel=1e-6)
    expected = jax.grad(logp, argnums=(1, 2, 3))(DATA, *PARAMS)
    np.testing.assert_allclose(gradients, expected, rtol=1e-5)


def test_trialwise_op_value_and_vjp(fake_forward):
    names = ["v", "a", "p_outlier"]
    logp = make_trialwise_lan_logp(fake_forward, names, {"a": (0.2, 2.5), "p_outlier": (0, 0.05)}, 0.4)
    # v varies per trial, the other parameters are shared
    logp_op = make_trialwise_logp_op(logp, varies_by_trial=[True, False, False])
    v_values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    v_var, a_var, p_var = pt.vector("v"), pt.scalar("a"), pt.scalar("p")
    logp_output = logp_op(DATA, v_var, a_var, p_var)
    compiled = pytensor.function([v_var, a_var, p_var],
                                 [logp_output] + pytensor.grad(logp_output.sum(), [v_var, a_var, p_var]))
    value, grad_v, grad_a, grad_p = compiled(v_values, PARAMS[1], PARAMS[2])

    batched_logp = jax.vmap(logp, in_axes=(0, 0, None, None))
    np.testing.assert_allclose(value, batched_logp(DATA, v_values, PARAMS[1], PARAMS[2]), rtol=1e-6)
    expected = jax.grad(lambda *params: batched_logp(DATA, *params).sum(), argnums=(0, 1, 2))(
        v_values, PARAMS[1], PARAMS[2])
    for actual_gradient, expected_gradient in zip((grad_v, grad_a, grad_p), expected):
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5)


def test_pymc_jax_backend_uses_jax_function(fake_forward):
    logp = make_lan_logp(fake_forward)
    logp_op = make_summed_logp_op(logp, n_params=2)
    with pm.Model() as model:
        v = pm.Normal("v", 0, 1)
        a = pm.Normal("a", 1, 1)
        pm.Potential("likelihood", logp_op(DATA[:, :2], v, a))
    point = {"v": np.float32(0.3), "a": np.float32(1.1)}
    expected = model.compile_logp()(point)
    jax_logp = get_jaxified_logp(model)
    actual = jax_logp([jnp.asarray(point["v"]), jnp.asarray(point["a"])])
    assert float(actual) == pytest.approx(float(expected), rel=1e-5)
