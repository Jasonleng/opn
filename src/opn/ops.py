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
        inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(p) for p in params]
        return Apply(self, inputs, [pt.scalar()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.asarray(self.logp(*inputs), dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_grads):
        results = _as_list(self.grad_op(*inputs))
        return [pytensor.gradient.grad_not_implemented(self, 0, inputs[0])] + [
            output_grads[0] * result for result in results]


class SummedLogpGradOp(Op):
    """Gradient of a `SummedLogpOp` with respect to its parameters."""

    def __init__(self, logp_grad):
        self.logp_grad = logp_grad

    def make_node(self, data, *params):
        inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(p) for p in params]
        return Apply(self, inputs, [inp.type() for inp in inputs[1:]])

    def perform(self, node, inputs, outputs):
        for i, result in enumerate(self.logp_grad(inputs[0], *inputs[1:])):
            outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)


def make_summed_logp_op(logp, n_params):
    """Wrap a summed log-likelihood (`likelihoods.make_lan_logp` / `make_omission_logp`)."""
    _register_jax_dispatch()
    return SummedLogpOp(logp, n_params)


# --- Trialwise family: one value per trial, gradient via jax.vjp -------------

class TrialwiseLogpOp(Op):
    """``op(data, *params)`` -> vector of per-trial log-likelihoods."""

    def __init__(self, logp, params_is_reg):
        in_axes = [0] + [0 if is_reg else None for is_reg in params_is_reg]
        vmap_logp = vmap(logp, in_axes=in_axes)

        def vmap_logp_vjp(data, *params, gz):
            _, vjp_fn = vjp(vmap_logp, data, *params)
            return vjp_fn(gz)[1:]

        self.logp_nojit = vmap_logp
        self.logp = jit(vmap_logp)
        self.vjp_op = TrialwiseLogpVJPOp(jit(vmap_logp_vjp))

    def make_node(self, data, *params):
        inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(p) for p in params]
        return Apply(self, inputs, [pt.vector()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.asarray(self.logp(*inputs), dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        results = _as_list(self.vjp_op(inputs[0], *inputs[1:], gz=output_gradients[0]))
        return [pytensor.gradient.grad_not_implemented(self, 0, inputs[0])] + results


class TrialwiseLogpVJPOp(Op):
    """Vector-Jacobian product of a `TrialwiseLogpOp` with respect to its parameters."""

    def __init__(self, logp_vjp):
        self.logp_vjp = logp_vjp

    def make_node(self, data, *params, gz):
        inputs = ([pt.as_tensor_variable(data)] + [pt.as_tensor_variable(p) for p in params]
                  + [pt.as_tensor_variable(gz)])
        return Apply(self, inputs, [inp.type() for inp in inputs[1:-1]])

    def perform(self, node, inputs, outputs):
        for i, result in enumerate(self.logp_vjp(inputs[0], *inputs[1:-1], gz=inputs[-1])):
            outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)


def make_trialwise_logp_op(logp, params_is_reg):
    """Wrap a trialwise log-likelihood; `params_is_reg[i]` marks parameters that vary per trial."""
    _register_jax_dispatch()
    return TrialwiseLogpOp(logp, params_is_reg)
