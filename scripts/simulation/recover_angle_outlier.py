"""Parameter recovery for the angle model with 5% outlier trials (deadline 1.25 s).

For each simulated dataset the model is fitted with the LAN alone and with the LAN plus the omission network,
both including an outlier mixture.
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
from opn.likelihoods import make_lan_logp, make_omission_logp
from opn.networks import load_network
from opn.ops import make_summed_logp_op
from opn.paths import RESULTS_DIR
from opn.simulate import contaminate, simulate, split_by_deadline

warnings.filterwarnings('ignore')

args = simulation_parser(
    "Parameter recovery for the angle model with 5% outlier trials (deadline 1.25 s).",
    n_cells=36, n_reps=20,
    out_dir=os.path.join(RESULTS_DIR, "simulation", "angle", "p005_l"),
).parse_args()

# LAN for (rt, choice) of responses before the deadline; omission network for omitted trials
lan_logp_op = make_summed_logp_op(make_lan_logp(load_network("angle", "lan"), outlier=True), n_params=6)
cpn_logp_op = make_summed_logp_op(make_omission_logp(load_network("angle", "cpn"), outlier=True), n_params=7)

jobid = args.jobid
a_list  = np.linspace(1,2,6)
theta_list  = np.linspace(0.5,1,6)
a_index = jobid%6
theta_index = jobid//6
p = 0.05
# a_list = [1.5]
# theta_list = [0.95]
# Test parameters:
# v, z, t, deadline = 2, 0.5, 0.3, 1.25

a_true = []
theta_true = []

a_pred_nd = []
theta_pred_nd = []
p_pred_nd = []

a_pred_d = []
theta_pred_d = []
p_pred_d = []

omission_list = []

r_hat_nd = []
r_hat_d  = []

for i in range(args.n_reps):
    v, a, z, t, theta, deadline = 1.3, a_list[a_index], 0.5, 0.3, theta_list[theta_index], 1.25
    a_true.append(a)
    theta_true.append(theta)

    data = contaminate(simulate('angle', [v, a, z, t, theta], n_samples=1000), p)
    data_commission, data_omission = split_by_deadline(data, deadline)
    omission_rate = np.mean(data_omission)
    print(i)
    print(omission_rate)

    
    
    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -3.0, 3.0)
        a1 = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        theta =pm.Uniform("theta", 0.1, 1.3)
        deadline = pt.constant(1.25)
        p_outlier = pm.Uniform("p_outlier", -5, 5)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data_commission, v1, a1, z, t, theta, p_outlier))
        #pm.Potential("omission", cpn_logp_op(data_omission, v1, a1, z, t, theta,deadline))
        # invalid_nd = True
        
        ddm_blog_traces_numpyro_d = pmj.sample_numpyro_nuts(
                chains=4, draws=1000, tune=2000,target_accept=0.9,chain_method='parallel',progressbar=False
                )
        post_summary_nd = pm.summary(ddm_blog_traces_numpyro_d)
    

    with pm.Model() as ddm:
        # Define simple Uniform priors
        v1 = pm.Uniform("v", -3.0, 3.0)
        a1 = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        theta =pm.Uniform("theta", 0.1, 1.3)
        deadline = pt.constant(1.25)
        p_outlier = pm.Uniform("p_outlier", -5, 5)#p_outlier = pm.Uniform("p_outlier", 0.0, 0.1)
        # t = pm.Uniform("t", 0.0, 2.0)
        # theta = pm.Uniform("theta",0,1.3)
        # ddl = pt.constant(1.25)
        
    
        pm.Potential("choice_rt", lan_logp_op(data_commission, v1, a1, z, t, theta,p_outlier))
        pm.Potential("omission", cpn_logp_op(data_omission, v1, a1, z, t, theta,deadline,p_outlier))
        

        # invalid_d = True
  
        ddm_blog_traces_numpyro_nd = pmj.sample_numpyro_nuts(
                chains=4, draws=1000, tune=4000,target_accept=0.9,chain_method='parallel',progressbar=False
                )
        post_summary_d = pm.summary(ddm_blog_traces_numpyro_nd)
            
    a_pred_nd.append(post_summary_nd.loc['a','mean'])
    theta_pred_nd.append(post_summary_nd.loc['theta','mean'])
    p_pred_nd.append(post_summary_nd.loc['p_outlier','mean'])
    a_pred_d.append(post_summary_d.loc['a','mean'])
    theta_pred_d.append(post_summary_d.loc['theta','mean'])
    p_pred_d.append(post_summary_d.loc['p_outlier','mean'])
    r_hat_nd.append(post_summary_nd['r_hat'].max())
    r_hat_d.append(post_summary_d['r_hat'].max())
    omission_list.append(omission_rate)
    sys.stdout.flush()


percentile_list = pd.DataFrame(
    {'a_true': a_true,
     'theta_true': theta_true,
     'a_pred_nd': a_pred_nd,
     'theta_pred_nd': theta_pred_nd,
     'p_pred_nd': p_pred_nd,
     'a_pred_d': a_pred_d,
     'theta_pred_d': theta_pred_d,
     'p_pred_d': p_pred_d,
     'omission_rate': omission_list,
     'r_hat_d':r_hat_d,
     'r_hat_nd':r_hat_nd
    })

save_results(percentile_list, args.out_dir, jobid)
