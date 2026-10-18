"""PyTensor Ops wrapping the JAX log-likelihoods, for use in PyMC models.

Each Op evaluates the jitted JAX function in PyTensor's own backends and, when a
model is sampled with the JAX backend (`pymc.sampling.jax`), is replaced by the
plain JAX function so NumPyro can differentiate it directly.

PyTensor's JAX backend is imported only when an Op is built, because importing it
fixes `jax_enable_x64` from `pytensor.config.floatX`; scripts set floatX first.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from jax import grad, jit, vjp, vmap
from pytensor.graph import Apply, Op

_jax_registered = False


def _register_jax_dispatch():
    global _jax_registered
    if _jax_registered:
        return
    from pytensor.link.jax.dispatch import jax_funcify

    @jax_funcify.register(SummedLogpOp)
    @jax_funcify.register(TrialwiseLogpOp)
    def _logp_op_dispatch(op, **kwargs):
        return op.logp_nojit

    _jax_registered = True


def _as_list(outputs):
    return outputs if isinstance(outputs, (list, tuple)) else [outputs]


# --- Summed family: scalar output, gradient via jax.grad ---------------------

class SummedLogpOp(Op):
    """``op(data, *params)`` -> scalar summed log-likelihood."""

    def __init__(self, logp, n_params):
        self.logp_nojit = logp
        self.logp = jit(logp)
        self.grad_op = SummedLogpGradOp(jit(grad(logp, argnums=range(1, 1 + n_params))))

    def make_node(self, data, *params):
        inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(param) for param in params]
        return Apply(self, inputs, [pt.scalar()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.asarray(self.logp(*inputs), dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_grads):
        param_gradients = _as_list(self.grad_op(*inputs))
        return [pytensor.gradient.grad_not_implemented(self, 0, inputs[0])] + [
            output_grads[0] * param_gradient for param_gradient in param_gradients]


class SummedLogpGradOp(Op):
    """Gradient of a `SummedLogpOp` with respect to its parameters."""

    def __init__(self, logp_grad):
        self.logp_grad = logp_grad

    def make_node(self, data, *params):
        inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(param) for param in params]
        return Apply(self, inputs, [input_variable.type() for input_variable in inputs[1:]])

    def perform(self, node, inputs, outputs):
        for param_index, param_gradient in enumerate(self.logp_grad(inputs[0], *inputs[1:])):
            outputs[param_index][0] = np.asarray(param_gradient, dtype=node.outputs[param_index].dtype)


def make_summed_logp_op(logp, n_params):
    """Wrap a summed log-likelihood (`likelihoods.make_lan_logp` / `make_omission_logp`)."""
    _register_jax_dispatch()
    return SummedLogpOp(logp, n_params)


# --- Trialwise family: one value per trial, gradient via jax.vjp -------------

class TrialwiseLogpOp(Op):
    """``op(data, *params)`` -> vector of per-trial log-likelihoods."""

    def __init__(self, logp, varies_by_trial):
        in_axes = [0] + [0 if param_varies else None for param_varies in varies_by_trial]
        batched_logp = vmap(logp, in_axes=in_axes)

        def batched_logp_vjp(data, *params, output_gradient):
            _, pullback = vjp(batched_logp, data, *params)
            return pullback(output_gradient)[1:]

        self.logp_nojit = batched_logp
        self.logp = jit(batched_logp)
        self.vjp_op = TrialwiseLogpVJPOp(jit(batched_logp_vjp))

    def make_node(self, data, *params):
        inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(param) for param in params]
        return Apply(self, inputs, [pt.vector()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.asarray(self.logp(*inputs), dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        param_gradients = _as_list(self.vjp_op(inputs[0], *inputs[1:], output_gradient=output_gradients[0]))
        return [pytensor.gradient.grad_not_implemented(self, 0, inputs[0])] + param_gradients


class TrialwiseLogpVJPOp(Op):
    """Vector-Jacobian product of a `TrialwiseLogpOp` with respect to its parameters."""

    def __init__(self, logp_vjp):
        self.logp_vjp = logp_vjp

    def make_node(self, data, *params, output_gradient):
        inputs = ([pt.as_tensor_variable(data)] + [pt.as_tensor_variable(param) for param in params]
                  + [pt.as_tensor_variable(output_gradient)])
        return Apply(self, inputs, [input_variable.type() for input_variable in inputs[1:-1]])

    def perform(self, node, inputs, outputs):
        param_gradients = self.logp_vjp(inputs[0], *inputs[1:-1], output_gradient=inputs[-1])
        for param_index, param_gradient in enumerate(param_gradients):
            outputs[param_index][0] = np.asarray(param_gradient, dtype=node.outputs[param_index].dtype)


def make_trialwise_logp_op(logp, varies_by_trial):
    """Wrap a trialwise log-likelihood; `varies_by_trial[i]` marks parameters that vary per trial."""
    _register_jax_dispatch()
    return TrialwiseLogpOp(logp, varies_by_trial)
