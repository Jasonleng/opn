"""Parameter recovery for the Weibull model (deadline 1.25 s).

For each simulated dataset the model is fitted with the LAN alone and with the
LAN plus the omission network. Parameters are sampled on an unbounded scale and
mapped into `bounds`.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytensor

pytensor.config.floatX = "float32"  # the networks are float32; set before any likelihood Op is built
import pymc as pm
import pytensor.tensor as pt
from pymc.sampling import jax as pmj

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from opn.cli import save_results, simulation_parser
from opn.likelihoods import OUTLIER_DENSITY, make_trialwise_lan_logp, make_trialwise_omission_logp
from opn.networks import load_network
from opn.ops import make_trialwise_logp_op
from opn.paths import RESULTS_DIR
from opn.simulate import contaminate, simulate, split_by_deadline

warnings.filterwarnings('ignore')

args = simulation_parser(
    "Parameter recovery for the Weibull model (deadline 1.25 s).",
    n_cells=27, n_reps=30,
    out_dir=os.path.join(RESULTS_DIR, "simulation", "weibull", "p000_3_s"),
).parse_args()

bounds = {'v': (-3, 3), 'a': (0.2, 2.5), 't': (0.01, 0.5), 'alpha': (0.3, 4.5), 'beta': (0.3, 4.5)}
lan_logp_op = make_trialwise_logp_op(
    make_trialwise_lan_logp(load_network("weibull", "lan"),
                            ['v', 'a', 'z', 't', 'alpha', 'beta', 'p_outlier'], bounds,
                            lapse_density=OUTLIER_DENSITY),
    params_is_reg=[False] * 7)
# The Weibull omission network outputs log P(omission) ("opn")
cpn_logp_op = make_trialwise_logp_op(
    make_trialwise_omission_logp(load_network("weibull", "cpn"),
                                 ['v', 'a', 'z', 't', 'alpha', 'beta', 'deadline', 'p_outlier'], bounds,
                                 network="opn", outlier=True),
    params_is_reg=[False] * 8)

jobid = args.jobid
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

for i in range(args.n_reps):
    v, a, z, t, alpha, beta, deadline = 1.3, a_list[a_index], 0.5, 0.3, alpha_list[alpha_index], beta_list[beta_index],1.25
    a_true.append(a)
    alpha_true.append(alpha)
    beta_true.append(beta)

    data = contaminate(simulate('weibull', [v, a, z, t, alpha,beta], n_samples=2000), p)
    data_commission, data_omission = split_by_deadline(data, deadline)
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

save_results(percentile_list, args.out_dir, jobid)
