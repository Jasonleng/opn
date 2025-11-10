import ssms
import torch
import pickle
import numpy as np
import os
import sys
import lanfactory
from copy import deepcopy
import pandas as pd
from os import PathLike
from typing import Callable, Tuple
import pytensor 
pytensor.config.floatX = "float32"
import pytensor.tensor as pt
import jax.numpy as jnp
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from jax import grad, jit
from numpy.typing import ArrayLike

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]

import pymc as pm
from pytensor.tensor.random.op import RandomVariable

import warnings 
warnings.filterwarnings('ignore')

class NetworkLike:
    @classmethod
    def make_logp_jax_funcs(
        cls,
        model = None,
        n_params: int | None = None,
        kind: str = 'lan',
    ) -> Tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc,]:
        """Makes a jax log likelihood function from flax network forward pass.
        Args:
            model: A path or url to the ONNX model, or an ONNX Model object
            already loaded.
            compile: Whether to use jit in jax to compile the model.
        Returns: A triple of jax or Python functions. The first calculates the
            forward pass, the second calculates the gradient, and the third is
            the forward-pass that's not jitted.
        """
        if kind == 'lan':
            def logp_lan(data: np.ndarray, *dist_params) -> ArrayLike:
                """
                Computes the sum of the log-likelihoods given data and arbitrary
                numbers of parameters assuming the trial by trial likelihoods
                are derived from a LAN.
                Args:
                    data: response time with sign indicating direction.
                    dist_params: a list of parameters used in the likelihood computation.
                Returns:
                    The sum of log-likelihoods.
                """

                # Makes a matrix to feed to the LAN model
                params_matrix = jnp.repeat(
                    jnp.stack(dist_params).reshape(1, -1), axis=0, repeats=data.shape[0]
                )

                # Set 'v' parameters depending on condition
                # params_matrix = params_matrix.at[:, 0].set(params_matrix[:, 0] * data[:, 2])

                # Stack parameters and data to have full input
                input_matrix = jnp.hstack([params_matrix, data[:, :2]])

                ssm_ll = jnp.exp(model(input_matrix))


                # Include lapse distribution (uniform) into (rt,choice) likelihood
                
                full_ll = ssm_ll

                # Network forward and sum
                return jnp.sum(
                    jnp.squeeze(jnp.log(full_ll))
                )

            logp_grad_lan = grad(logp_lan, argnums=range(1, 1 + n_params))
            return jit(logp_lan), jit(logp_grad_lan), logp_lan

        elif kind == 'cpn':
            def logp_cpn(data: np.ndarray, *dist_params) -> ArrayLike:
                """
                Computes the sum of the log-likelihoods given data and arbitrary
                numbers of parameters assuming the trial-by-trial likelihood derive for a CPN.
                Args:
                    data: response time with sign indicating direction.
                    dist_params: a list of parameters used in the likelihood computation.
                Returns:
                    The sum of log-likelihoods.
                """

                # Makes a matrix to feed to the LAN model
                # n_nogo_go_condition = jnp.sum(data > 0)
                # n_nogo_nogo_condition = jnp.sum(data < 0)
                n_omission = jnp.sum(data>0)
                n_total = jnp.sum(data>=0)

                params_matrix  = jnp.stack(dist_params).reshape(1, -1)

                # AF-TODO Bugfix here !
                # dist_params_nogo = jnp.stack(dist_params).reshape(1, -1)
                # dist_params_nogo = dist_params_nogo.at[0].set((-1) * dist_params_nogo[0])

                net_in = params_matrix

                net_out = jnp.squeeze(model(net_in))

                # Include lapse distribution (uniform) into omission likelihood
                # dist_params[-1]: outlier
                # dist_params[-2]: deadline (in second)

                out = jnp.log(1 - jnp.exp(net_out) + 1e-64) * n_omission

                return out

            logp_grad_cpn = grad(logp_cpn, argnums=range(1, 1 + n_params))
            return jit(logp_cpn), jit(logp_grad_cpn), logp_cpn

    @staticmethod
    def make_jax_logp_ops(
        logp: LogLikeFunc,
        logp_grad: LogLikeGrad,
        logp_nojit: LogLikeFunc,
    ) -> LogLikeFunc:
        """Wraps the JAX functions and its gradient in Pytensor Ops.
        Args:
            logp: A JAX function that represents the feed-forward operation of the
                LAN network.
            logp_grad: The derivative of the above function.
            logp_nojit: A Jax function
        Returns:
            An pytensor op that wraps the feed-forward operation and can be used with
            pytensor.grad.
        """

        class LogpOp(Op):
            """Wraps a JAX function in an pytensor Op."""

            def make_node(self, data, *dist_params):
                inputs = [
                    pt.as_tensor_variable(data),
                ] + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]

                outputs = [pt.scalar()]

                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, output_storage):
                """Performs the Apply node.
                Args:
                    inputs: This is a list of data from which the values stored in
                        output_storage are to be computed using non-symbolic language.
                    output_storage: This is a list of storage cells where the output
                        is to be stored. A storage cell is a one-element list. It is
                        forbidden to change the length of the list(s) contained in
                        output_storage. There is one storage cell for each output of
                        the Op.
                """
                result = logp(*inputs)
                output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

            def grad(self, inputs, output_grads):
                results = lan_logp_grad_op(*inputs)
                output_gradient = output_grads[0]
                return [
                    pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
                ] + [output_gradient * result for result in results]

        class LogpGradOp(Op):
            """Wraps the gradient opearation of a jax function in an pytensor op."""

            def make_node(self, data, *dist_params):
                inputs = [
                    pt.as_tensor_variable(data),
                ] + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
                outputs = [inp.type() for inp in inputs[1:]]

                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, outputs):
                results = logp_grad(inputs[0], *inputs[1:])

                for i, result in enumerate(results):
                    outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)

        lan_logp_op = LogpOp()
        lan_logp_grad_op = LogpGradOp()

        # Unwraps the JAX function for sampling with JAX backend.
        @jax_funcify.register(LogpOp) # Can fail in notebooks
        def logp_op_dispatch(op, **kwargs):  # pylint: disable=W0612,W0613
            return logp_nojit

        return lan_logp_op

# Loaded Net

model_config = ssms.config.model_config['angle']
try:
    jax_infer_lan = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/ddm/lan/e91014e8b9a011eeafa5a0423f3e9be0_lan_ddm__network_config.pickle",
        train=False,
    )
    
    forward_pass_lan, forward_pass_jitted_lan = jax_infer_lan.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 2,
        state="../network/ddm/lan/e91014e8b9a011eeafa5a0423f3e9be0_lan_ddm__train_state.jax",
        add_jitted=True,
    )
    
    # Loaded Net
    jax_infer_cpn = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/ddm/cpn/0ea678a4b93911ee8857a0423f3e9a40_cpn_ddm__network_config.pickle",
        train=False,
    )
    
    forward_pass_cpn, forward_pass_jitted_cpn = jax_infer_cpn.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 1,
        state="../network/ddm/cpn/0ea678a4b93911ee8857a0423f3e9a40_cpn_ddm__train_state.jax",
        add_jitted=True,
    )
except:
    jax_infer_lan = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/ddm/lan/e91014e8b9a011eeafa5a0423f3e9be0_lan_ddm__network_config.pickle",
        train=False,
    )
    
    forward_pass_lan, forward_pass_jitted_lan = jax_infer_lan.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 2,
        state="../network/ddm/lan/e91014e8b9a011eeafa5a0423f3e9be0_lan_ddm__train_state.jax",
        add_jitted=True,
    )
    
    # Loaded Net
    jax_infer_cpn = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/ddm/cpn/0ea678a4b93911ee8857a0423f3e9a40_cpn_ddm__network_config.pickle",
        train=False,
    )
    
    forward_pass_cpn, forward_pass_jitted_cpn = jax_infer_cpn.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 1,
        state="../network/ddm/cpn/0ea678a4b93911ee8857a0423f3e9a40_cpn_ddm__train_state.jax",
        add_jitted=True,
    )    


# Instantiate LAN logp functions
lan_logp_jitted, lan_logp_grad_jitted, lan_logp = NetworkLike.make_logp_jax_funcs(model = forward_pass_lan,
                                                                                  n_params = 4,
                                                                                  kind = "lan")

# Turn into logp op
lan_logp_op = NetworkLike.make_jax_logp_ops(
                                logp = lan_logp_jitted,
                                logp_grad = lan_logp_grad_jitted,
                                logp_nojit = lan_logp)

# Instantiate CPN logp functions
cpn_logp_jitted, cpn_logp_grad_jitted, cpn_logp = NetworkLike.make_logp_jax_funcs(model = forward_pass_cpn,
                                                                                  n_params = 5,
                                                                                  kind = "cpn")

# Turn into logp op
cpn_logp_op = NetworkLike.make_jax_logp_ops(
                                logp = cpn_logp_jitted,
                                logp_grad = cpn_logp_grad_jitted,
                                logp_nojit = cpn_logp)

from pymc.sampling import jax as pmj

# Just to keep the blog-post pretty automatically
import warnings 
warnings.filterwarnings('ignore')

jobid = int(sys.argv[1])
a_list  = np.linspace(0.5,1.5,11)
v_list  = np.linspace(0.5,1.5,11)
a_index = jobid%11
v_index = jobid//11
# a_list = [1.5]
# theta_list = [0.95]
# Test parameters:
# v, z, t, deadline = 2, 0.5, 0.3, 1.25

a_true = []
v_true = []

a_pred_nd = []
v_pred_nd = []

a_pred_d = []
v_pred_d = []

a_pred_f = []
v_pred_f = []

omission_list = []

for i in range(20):
    v, a, z, t, deadline = v_list[v_index], a_list[a_index], 0.5, 0.3, 1.25
    a_true.append(a)
    v_true.append(v)

    # Comparison simulator run
    sim_out = ssms.basic_simulators.simulator.simulator(
        model='ddm', theta=[v, a, z, t], n_samples=1000
    )
    data = np.hstack([sim_out['rts'], sim_out['choices']]).astype(np.float32)
    data_commission = data[data[:,0]<deadline,:]
    data_omission = data[:,0]>=deadline
    data_omission = data_omission.astype(np.float32)
    omission_rate = np.mean(data_omission)

    print(omission_rate)

    omission_list.append(omission_rate)
    
    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -3.0, 3.0)
        a1 = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        deadline = pt.constant(1.25)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data_commission, v1, a1, z, t))
        #pm.Potential("omission", cpn_logp_op(data_omission, v1, a1, z, t, theta,deadline))
    
        ddm_blog_traces_numpyro_nd = pmj.sample_numpyro_nuts(
                chains=2, draws=1000, tune=2000,target_accept=0.95,progressbar=False
                )
        post_summary = pm.summary(ddm_blog_traces_numpyro_nd)

    a_pred_nd.append(post_summary.loc['a','mean'])
    v_pred_nd.append(post_summary.loc['v','mean'])
    
    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -3.0, 3.0)
        a1 = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        deadline = pt.constant(1.25)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data_commission, v1, a1, z, t))
        pm.Potential("omission", cpn_logp_op(data_omission, v1, a1, z, t,deadline))
    
        ddm_blog_traces_numpyro_d = pmj.sample_numpyro_nuts(
                chains=2, draws=1000, tune=2000,target_accept=0.95,progressbar=False
                )
        post_summary = pm.summary(ddm_blog_traces_numpyro_d)     
    a_pred_d.append(post_summary.loc['a','mean'])
    v_pred_d.append(post_summary.loc['v','mean'])

    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -3.0, 3.0)
        a1 = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        deadline = pt.constant(1.25)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data, v1, a1, z, t))
    
        ddm_blog_traces_numpyro_f = pmj.sample_numpyro_nuts(
                chains=2, draws=1000, tune=2000,target_accept=0.95,progressbar=False
                )
        post_summary = pm.summary(ddm_blog_traces_numpyro_f)     
    a_pred_f.append(post_summary.loc['a','mean'])
    v_pred_f.append(post_summary.loc['v','mean'])

    sys.stdout.flush()
    


percentile_list = pd.DataFrame(
    {'a_true': a_true,
     'v_true': v_true,
     'a_pred_nd': a_pred_nd,
     'v_pred_nd': v_pred_nd,
     'a_pred_d': a_pred_d,
     'v_pred_d': v_pred_d,
     'a_pred_f': a_pred_f,
     'v_pred_f': v_pred_f,
     'omission_rate': omission_list
    })

percentile_list.to_csv('results_DDM/output_'+sys.argv[1]+'.csv',index=False)
