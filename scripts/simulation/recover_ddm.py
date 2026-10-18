"""Parameter recovery for the DDM (deadline 1.25 s).

For each simulated dataset the model is fitted with the LAN alone, with the LAN plus the omission
network, and with the LAN on the full data (no deadline).
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
from opn.simulate import simulate, split_by_deadline

warnings.filterwarnings('ignore')

args = simulation_parser(
    "Parameter recovery for the DDM (deadline 1.25 s).",
    n_cells=121, n_reps=20,
    out_dir=os.path.join(RESULTS_DIR, "simulation", "ddm"),
).parse_args()

# LAN for (rt, choice) of responses before the deadline; omission network for omitted trials
lan_logp_op = make_summed_logp_op(make_lan_logp(load_network("ddm", "lan")), n_params=4)
cpn_logp_op = make_summed_logp_op(make_omission_logp(load_network("ddm", "cpn")), n_params=5)

jobid = args.jobid
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

for i in range(args.n_reps):
    v, a, z, t, deadline = v_list[v_index], a_list[a_index], 0.5, 0.3, 1.25
    a_true.append(a)
    v_true.append(v)

    data = simulate('ddm', [v, a, z, t], n_samples=1000)
    data_commission, data_omission = split_by_deadline(data, deadline)
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

save_results(percentile_list, args.out_dir, jobid)
