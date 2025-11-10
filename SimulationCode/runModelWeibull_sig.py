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
from jax import grad, jit, vmap
from numpy.typing import ArrayLike
from jax.scipy.special import expit


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
        params_is_reg: list[bool],
        list_params: list,
        model = None,
        n_params: int | None = None,
        bounds = None,
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
                
                
                transformed_params = []
                for i in range(len(dist_params)):
                    if list_params[i] in bounds.keys():
                        transformed_params.append(expit(dist_params[i]) * (bounds[list_params[i]][1] - bounds[list_params[i]][0]) + bounds[list_params[i]][0])
                    else:
                        transformed_params.append(dist_params[i])
                input_matrix = jnp.concatenate((jnp.array(transformed_params[:-1]), data))

                ll = jnp.multiply(jnp.exp(model(input_matrix)),1-transformed_params[-1]) + transformed_params[-1] * 1/10

                # Network forward and sum
                return jnp.sum(
                    jnp.squeeze(jnp.log(ll))
                )
            # The vectorization of the logp function
            vmap_logp_lan = vmap(
                logp_lan,
                in_axes=[0] + [0 if is_regression else None for is_regression in params_is_reg],
            )
            # logp_grad_lan = grad(logp_lan, argnums=range(1, 1 + n_params))
            # return jit(logp_lan), jit(logp_grad_lan), logp_lan
            
            def vjp_vmap_logp_lan(
                data: np.ndarray, *dist_params: list[float | ArrayLike], gz: ArrayLike
            ) -> list[ArrayLike]:
                """Compute the VJP of the log-likelihood function.

                Parameters
                ----------
                data
                    A two-column numpy array with response time and response.
                dist_params
                    A list of parameters used in the likelihood computation.
                gz
                    The value of vmap_logp at which the VJP is evaluated, typically is just
                    vmap_logp(data, *dist_params)

                Returns
                -------
                list[ArrayLike]
                    The VJP of the log-likelihood function computed at gz.
                """
                _, vjp_fn = vjp(vmap_logp_lan, data, *dist_params)
                return vjp_fn(gz)[1:]

            return jit(vmap_logp_lan), jit(vjp_vmap_logp_lan), vmap_logp_lan

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
                # n_omission = jnp.sum(data>0)
                # n_total = jnp.sum(data>=0)
                transformed_params = []

                n_omission = jnp.sum(data>0)
                
                for i in range(len(dist_params)):
                    if list_params[i] in bounds.keys():
                        transformed_params.append(expit(dist_params[i]) * (bounds[list_params[i]][1] - bounds[list_params[i]][0]) + bounds[list_params[i]][0])
                    else:
                        transformed_params.append(dist_params[i])
                        
                params_matrix  = jnp.array(transformed_params[:-1])

                # AF-TODO Bugfix here !
                # dist_params_nogo = jnp.stack(dist_params).reshape(1, -1)
                # dist_params_nogo = dist_params_nogo.at[0].set((-1) * dist_params_nogo[0])

                net_out = jnp.squeeze(model(params_matrix))

                # Include lapse distribution (uniform) into omission likelihood
                # dist_params[-1]: outlier
                # dist_params[-2]: deadline (in second)

                out = jnp.log((1 - transformed_params[-1]) * (jnp.exp(net_out) + 1e-64) + transformed_params[-1] * (1 - transformed_params[-2]/5)) * n_omission
                
                return out
            vmap_logp_cpn = vmap(
                logp_cpn,
                in_axes=[0] + [0 if is_regression else None for is_regression in params_is_reg],
            )
            def vjp_vmap_logp_cpn(
                data: np.ndarray, *dist_params: list[float | ArrayLike], gz: ArrayLike
            ) -> list[ArrayLike]:
                """Compute the VJP of the log-likelihood function.

                Parameters
                ----------
                data
                    A two-column numpy array with response time and response.
                dist_params
                    A list of parameters used in the likelihood computation.
                gz
                    The value of vmap_logp at which the VJP is evaluated, typically is just
                    vmap_logp(data, *dist_params)

                Returns
                -------
                list[ArrayLike]
                    The VJP of the log-likelihood function computed at gz.
                """
                _, vjp_fn = vjp(vmap_logp_cpn, data, *dist_params)
                return vjp_fn(gz)[1:]

            return jit(vmap_logp_cpn), jit(vjp_vmap_logp_cpn), vmap_logp_cpn

    @staticmethod

    def make_jax_logp_ops(
        logp: LogLikeFunc,
        logp_vjp: LogLikeGrad,
        logp_nojit: LogLikeFunc,
    ) -> Op:
        """Wrap the JAX functions and its gradient in pytensor Ops.

        Parameters
        ----------
        logp
            A JAX function that represents the feed-forward operation of the LAN
            network.
        logp_vjp
            The Jax function that calculates the VJP of the logp function.
        logp_nojit
            The non-jit version of logp.

        Returns
        -------
        Op
            An pytensor op that wraps the feed-forward operation and can be used with
            pytensor.grad.
        """

        class LANLogpOp(Op):  # pylint: disable=W0223
            """Wraps a JAX function in an pytensor Op."""

            def make_node(self, data, *dist_params):
                """Take the inputs to the Op and puts them in a list.

                Also specifies the output types in a list, then feed them to the Apply node.

                Parameters
                ----------
                data
                    A two-column numpy array with response time and response.
                dist_params
                    A list of parameters used in the likelihood computation. The parameters
                    can be both scalars and arrays.
                """
                inputs = [
                    pt.as_tensor_variable(data),
                ] + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]

                outputs = [pt.vector()]

                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, output_storage):
                """Perform the Apply node.

                Parameters
                ----------
                inputs
                    This is a list of data from which the values stored in
                    output_storage are to be computed using non-symbolic language.
                output_storage
                    This is a list of storage cells where the output
                    is to be stored. A storage cell is a one-element list. It is
                    forbidden to change the length of the list(s) contained in
                    output_storage. There is one storage cell for each output of
                    the Op.
                """
                result = logp(*inputs)
                output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

            def grad(self, inputs, output_gradients):
                """Perform the pytensor.grad() operation.

                Parameters
                ----------
                inputs
                    The same as the inputs produced in `make_node`.
                output_gradients
                    Holds the results of the perform `perform` method.

                Notes
                -----
                    It should output the VJP of the Op. In other words, if this `Op`
                    outputs `y`, and the gradient at `y` is grad(x), the required output
                    is y*grad(x).
                """
                results = lan_logp_vjp_op(inputs[0], *inputs[1:], gz=output_gradients[0])
                output = [
                    pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
                ] + results

                return output

        class LANLogpVJPOp(Op):  # pylint: disable=W0223
            """Wraps the VJP operation of a jax function in an pytensor op."""

            def make_node(self, data, *dist_params, gz):
                """Take the inputs to the Op and puts them in a list.

                Also specifies the output types in a list, then feed them to the Apply node.

                Parameters
                ----------
                data:
                    A two-column numpy array with response time and response.
                dist_params:
                    A list of parameters used in the likelihood computation.
                """
                inputs = (
                    [
                        pt.as_tensor_variable(data),
                    ]
                    + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
                    + [pt.as_tensor_variable(gz)]
                )
                outputs = [inp.type() for inp in inputs[1:-1]]

                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, outputs):
                """Perform the Apply node.

                Parameters
                ----------
                inputs
                    This is a list of data from which the values stored in
                    `output_storage` are to be computed using non-symbolic language.
                output_storage
                    This is a list of storage cells where the output
                    is to be stored. A storage cell is a one-element list. It is
                    forbidden to change the length of the list(s) contained in
                    output_storage. There is one storage cell for each output of
                    the Op.
                """
                results = logp_vjp(inputs[0], *inputs[1:-1], gz=inputs[-1])

                for i, result in enumerate(results):
                    outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)

        lan_logp_op = LANLogpOp()
        lan_logp_vjp_op = LANLogpVJPOp()

        # Unwraps the JAX function for sampling with JAX backend.
        @jax_funcify.register(LANLogpOp)
        def logp_op_dispatch(op, **kwargs):  # pylint: disable=W0612,W0613
            return logp_nojit

        return lan_logp_op
    
model_config = ssms.config.model_config['weibull']
try:
    jax_infer_lan = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/weibull/lan/2b8c70363dff11ef8680a0423f3e9b42_lan_weibull_network_config.pickle",
        train=False,
    )

    forward_pass_lan, forward_pass_jitted_lan = jax_infer_lan.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 2,
        state="../network/weibull/lan/2b8c70363dff11ef8680a0423f3e9b42_lan_weibull__train_state.jax",
        add_jitted=True,
    )

    # Loaded Net
    jax_infer_cpn = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/weibull/cpn/cd5cafb23ec111efa0bca0423f3e9b5e_opn_weibull_deadline_network_config.pickle",
        train=False,
    )

    forward_pass_cpn, forward_pass_jitted_cpn = jax_infer_cpn.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 1,
        state="../network/weibull/cpn/cd5cafb23ec111efa0bca0423f3e9b5e_cpn_weibull_deadline__train_state.jax",
        add_jitted=True,
    )
except:
    jax_infer_lan = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/weibull/lan/2b8c70363dff11ef8680a0423f3e9b42_lan_weibull_network_config.pickle",
        train=False,
    )

    forward_pass_lan, forward_pass_jitted_lan = jax_infer_lan.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 2,
        state="../network/weibull/lan/2b8c70363dff11ef8680a0423f3e9b42_lan_weibull__train_state.jax",
        add_jitted=True,
    )

    # Loaded Net
    jax_infer_cpn = lanfactory.trainers.MLPJaxFactory(
        network_config="../network/weibull/cpn/cd5cafb23ec111efa0bca0423f3e9b5e_opn_weibull_deadline_network_config.pickle",
        train=False,
    )

    forward_pass_cpn, forward_pass_jitted_cpn = jax_infer_cpn.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 1,
        state="../network/weibull/cpn/cd5cafb23ec111efa0bca0423f3e9b5e_cpn_weibull_deadline__train_state.jax",
        add_jitted=True,
    )

# Instantiate LAN logp functions
lan_logp_jitted, lan_logp_vjp_jitted, lan_logp = NetworkLike.make_logp_jax_funcs(model = forward_pass_lan,
                                                                                  n_params = 7,
                                                                                  kind = "lan",
                                                                                  list_params = ['v','a','z','t','alpha','beta','p_outlier'],
                                                                                  bounds = {'v':(-3,3),
                                                                                            'a':(0.2,2.5),
                                                                                            't':(0.01,0.5),
                                                                                            'alpha':(0.3,4.5),
                                                                                            'beta':(0.3,4.5)},
                                                                                  params_is_reg=[False]*7)

# Turn into logp op
lan_logp_op = NetworkLike.make_jax_logp_ops(
                                logp = lan_logp_jitted,
                                logp_vjp = lan_logp_vjp_jitted,
                                logp_nojit = lan_logp)

# Instantiate CPN logp functions
cpn_logp_jitted, cpn_logp_vjp_jitted, cpn_logp = NetworkLike.make_logp_jax_funcs(model = forward_pass_cpn,
                                                                                 n_params = 8,
                                                                                 list_params = ['v','a','z','t','alpha','beta','deadline','p_outlier'],
                                                                                 bounds = {'v':(-3,3),
                                                                                            'a':(0.2,2.5),
                                                                                            't':(0.01,0.5),
                                                                                            'alpha':(0.3,4.5),
                                                                                            'beta':(0.3,4.5)},
                                                                                 kind = "cpn",
                                                                                 params_is_reg=[False]*8)

# Turn into logp op
cpn_logp_op = NetworkLike.make_jax_logp_ops(
                                logp = cpn_logp_jitted,
                                logp_vjp = cpn_logp_vjp_jitted,
                                logp_nojit = cpn_logp)


from pymc.sampling import jax as pmj

# Just to keep the blog-post pretty automatically
import warnings 
warnings.filterwarnings('ignore')

jobid = int(sys.argv[1])
a_list  = np.linspace(1,1.2,3)
alpha_list  = np.linspace(0.5,1.5,3)
beta_list = np.linspace(1,2,3)
a_index = jobid//9
sub_id = jobid%9
alpha_index = sub_id%3
beta_index = sub_id//3
# a_list = [1.5]
# theta_list = [0.95]
# Test parameters:
# v, z, t, deadline = 2, 0.5, 0.3, 1.25
p = 0

a_true = []
alpha_true = []
beta_true = []

a_pred_nd = []
alpha_pred_nd = []
beta_pred_nd = []
v_pred_nd = []
p_pred_nd = []

a_pred_d = []
alpha_pred_d = []
beta_pred_d = []
v_pred_d = []
p_pred_d = []

r_hat_nd = []
r_hat_d = []

omission_list = []

for i in range(30):
    v, a, z, t, alpha, beta, deadline = 1.3, a_list[a_index], 0.5, 0.3, alpha_list[alpha_index], beta_list[beta_index],1.25
    a_true.append(a)
    alpha_true.append(alpha)
    beta_true.append(beta)

    # Comparison simulator run
    sim_out = ssms.basic_simulators.simulator.simulator(
        model='weibull', theta=[v, a, z, t, alpha,beta], n_samples=2000
    )
    rt_twoside = np.random.uniform(-5,5,2000)
    choice = 2 * (rt_twoside >= 0) - 1
    rt = np.abs(rt_twoside)
    outlier = np.zeros((2000,2))
    outlier[:,0] = rt
    outlier[:,1] = choice
    arr = np.arange(0,2000,1)
    np.random.shuffle(arr)
    mask = arr < p * 2000
    data = np.hstack([sim_out['rts'], sim_out['choices']]).astype(np.float32)
    data[mask,:] = outlier[mask,:]
    data_commission = data[data[:,0]<deadline,:]
    data_omission = data[:,0]>=deadline
    data_omission = data_omission.astype(np.float32)
    omission_rate = np.mean(data_omission)
    print(i)
    print(omission_rate)

    omission_list.append(omission_rate)
    
    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -5,5)
        a1 = pm.Uniform("a", -5,5)
        z1 = pt.constant(0.5)
        t1 = pm.Uniform("t", -5,5)
        alpha1 = pm.Uniform("alpha", -5,5)
        beta1 = pm.Uniform("beta", -5,5)
        deadline = pt.constant(1.25)
        p_outlier1 = pt.constant(0)#pm.Uniform("p_outlier", -5,5)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data_commission, v1, a1, z1, t1, alpha1,beta1, p_outlier1))
        #pm.Potential("omission", cpn_logp_op(data_omission, v1, a1, z, t, theta,deadline))   
        ddm_blog_traces_numpyro_nd = pmj.sample_numpyro_nuts(
                chains=2, draws=1000, tune=3000,target_accept=0.99,progressbar=False)
        post_summary = pm.summary(ddm_blog_traces_numpyro_nd)                 

    v_pred_nd.append(1/(1+np.exp(-post_summary.loc['v','mean'])) * 6 - 3)
    a_pred_nd.append(1/(1+np.exp(-post_summary.loc['a','mean'])) * 2.3 + 0.2)
    alpha_pred_nd.append(1/(1+np.exp(-post_summary.loc['alpha','mean'])) * 4.2 + 0.3)
    beta_pred_nd.append(1/(1+np.exp(-post_summary.loc['beta','mean'])) * 4.2 + 0.3)
    p_pred_nd.append(0)#1/(1+np.exp(-post_summary.loc['p_outlier','mean'])) * 0.1) 
    r_hat_nd.append(post_summary['r_hat'].max())
    
    
    
    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -5,5)
        a1 = pm.Uniform("a", -5,5)
        z1 = pt.constant(0.5)
        t1 = pm.Uniform("t", -5,5)
        alpha1 = pm.Uniform("alpha", -5,5)
        beta1 = pm.Uniform("beta", -5,5)
        deadline = pt.constant(1.25)
        p_outlier1 = pt.constant(0)#pm.Uniform("p_outlier", -5,5)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data_commission, v1, a1, z1, t1, alpha1,beta1,p_outlier1))
        pm.Potential("omission", cpn_logp_op(data_omission, v1, a1, z1, t1, alpha1,beta1,deadline,p_outlier1))

        ddm_blog_traces_numpyro = pmj.sample_numpyro_nuts(
                chains=2, draws=1000, tune=3000,target_accept=0.99,progressbar=False)
        post_summary = pm.summary(ddm_blog_traces_numpyro)

    v_pred_d.append(1/(1+np.exp(-post_summary.loc['v','mean'])) * 6 - 3)
    a_pred_d.append(1/(1+np.exp(-post_summary.loc['a','mean'])) * 2.3 + 0.2)
    alpha_pred_d.append(1/(1+np.exp(-post_summary.loc['alpha','mean'])) * 4.2 + 0.3)
    beta_pred_d.append(1/(1+np.exp(-post_summary.loc['beta','mean'])) * 4.2 + 0.3)
    p_pred_d.append(0)#1/(1+np.exp(-post_summary.loc['p_outlier','mean'])) * 0.1) 
    r_hat_d.append(post_summary['r_hat'].max())

    sys.stdout.flush()
    
percentile_list = pd.DataFrame(
    {'a_true': a_true,
     'alpha_true': alpha_true,
     'beta_true': beta_true,
     'a_pred_nd': a_pred_nd,
     'v_pred_nd': v_pred_nd,
     'alpha_pred_nd': alpha_pred_nd,
     'beta_pred_nd': beta_pred_nd,
     'p_pred_nd':p_pred_nd,
     'a_pred_d': a_pred_d,
     'v_pred_d': v_pred_d,
     'alpha_pred_d': alpha_pred_d,
     'beta_pred_d': beta_pred_d,
     'p_pred_d':p_pred_d,
     'omission_rate': omission_list,
     'r_hat_nd':r_hat_nd,
     'r_hat_d':r_hat_d
    })

percentile_list.to_csv('results_weibull/p000_3_s/output_'+sys.argv[1]+'.csv',index=False)
