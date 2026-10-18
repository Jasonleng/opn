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
from pymc.sampling import jax as pymc_jax

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
omission_logp_op = make_summed_logp_op(make_omission_logp(load_network("angle", "cpn"), outlier=True), n_params=7)

DEADLINE = 1.25
OUTLIER_FRACTION = 0.05   # fraction of simulated trials replaced by contaminants

# The job index selects one cell of the 6 x 6 grid of true boundary and collapse values
jobid = args.jobid
a_grid = np.linspace(1,2,6)
theta_grid = np.linspace(0.5,1,6)
true_v, true_a, true_z, true_t, true_theta = 1.3, a_grid[jobid % 6], 0.5, 0.3, theta_grid[jobid // 6]

# Output columns: _nd = LAN alone, _d = LAN + omission network; p = unbounded outlier weight
results = {column: [] for column in ['a_true', 'theta_true', 'a_pred_nd', 'theta_pred_nd', 'p_pred_nd',
                                     'a_pred_d', 'theta_pred_d', 'p_pred_d', 'omission_rate',
                                     'r_hat_d', 'r_hat_nd']}

for rep in range(args.n_reps):
    results['a_true'].append(true_a)
    results['theta_true'].append(true_theta)

    data = contaminate(simulate('angle', [true_v, true_a, true_z, true_t, true_theta], n_samples=1000),
                       OUTLIER_FRACTION)
    responded, omitted = split_by_deadline(data, DEADLINE)
    omission_rate = np.mean(omitted)
    print(rep)
    print(omission_rate)

    with pm.Model():
        # Uniform priors
        v = pm.Uniform("v", -3.0, 3.0)
        a = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        theta = pm.Uniform("theta", 0.1, 1.3)
        p_outlier = pm.Uniform("p_outlier", -5, 5)   # unbounded; outlier probability = 0.1 * sigmoid(p_outlier)

        pm.Potential("choice_rt", lan_logp_op(responded, v, a, z, t, theta, p_outlier))

        idata_lan_only = pymc_jax.sample_numpyro_nuts(
                chains=4, draws=1000, tune=2000,target_accept=0.9,chain_method='parallel',progressbar=False
                )
        summary_lan_only = pm.summary(idata_lan_only)

    with pm.Model():
        # Uniform priors
        v = pm.Uniform("v", -3.0, 3.0)
        a = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        theta = pm.Uniform("theta", 0.1, 1.3)
        deadline = pt.constant(DEADLINE)
        p_outlier = pm.Uniform("p_outlier", -5, 5)   # unbounded; outlier probability = 0.1 * sigmoid(p_outlier)

        pm.Potential("choice_rt", lan_logp_op(responded, v, a, z, t, theta, p_outlier))
        pm.Potential("omission", omission_logp_op(omitted, v, a, z, t, theta, deadline, p_outlier))

        idata_with_omissions = pymc_jax.sample_numpyro_nuts(
                chains=4, draws=1000, tune=4000,target_accept=0.9,chain_method='parallel',progressbar=False
                )
        summary_with_omissions = pm.summary(idata_with_omissions)

    results['a_pred_nd'].append(summary_lan_only.loc['a','mean'])
    results['theta_pred_nd'].append(summary_lan_only.loc['theta','mean'])
    results['p_pred_nd'].append(summary_lan_only.loc['p_outlier','mean'])
    results['a_pred_d'].append(summary_with_omissions.loc['a','mean'])
    results['theta_pred_d'].append(summary_with_omissions.loc['theta','mean'])
    results['p_pred_d'].append(summary_with_omissions.loc['p_outlier','mean'])
    results['r_hat_nd'].append(summary_lan_only['r_hat'].max())
    results['r_hat_d'].append(summary_with_omissions['r_hat'].max())
    results['omission_rate'].append(omission_rate)
    sys.stdout.flush()


save_results(pd.DataFrame(results), args.out_dir, jobid)
