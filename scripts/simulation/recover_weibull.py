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
from pymc.sampling import jax as pymc_jax

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
    varies_by_trial=[False] * 7)
# The Weibull omission network outputs log P(omission) ("opn")
omission_logp_op = make_trialwise_logp_op(
    make_trialwise_omission_logp(load_network("weibull", "cpn"),
                                 ['v', 'a', 'z', 't', 'alpha', 'beta', 'deadline', 'p_outlier'], bounds,
                                 network="opn", outlier=True),
    varies_by_trial=[False] * 8)


def posterior_mean_in_range(summary, name):
    """Posterior mean of an unbounded parameter, mapped into its range in `bounds`."""
    low, high = bounds[name]
    return 1/(1+np.exp(-summary.loc[name,'mean'])) * (high - low) + low


DEADLINE = 1.25
OUTLIER_FRACTION = 0   # no contaminant trials; the fits fix the outlier probability at 0 too

# The job index selects one cell of the 3 x 3 x 3 grid of true boundary, alpha and beta values
jobid = args.jobid
a_grid = np.linspace(1,1.2,3)
alpha_grid = np.linspace(0.5,1.5,3)
beta_grid = np.linspace(1,2,3)
alpha_beta_cell = jobid % 9
true_v, true_a, true_z, true_t = 1.3, a_grid[jobid // 9], 0.5, 0.3
true_alpha, true_beta = alpha_grid[alpha_beta_cell % 3], beta_grid[alpha_beta_cell // 3]

# Output columns: _nd = LAN alone, _d = LAN + omission network
results = {column: [] for column in ['a_true', 'alpha_true', 'beta_true',
                                     'a_pred_nd', 'v_pred_nd', 'alpha_pred_nd', 'beta_pred_nd', 'p_pred_nd',
                                     'a_pred_d', 'v_pred_d', 'alpha_pred_d', 'beta_pred_d', 'p_pred_d',
                                     'omission_rate', 'r_hat_nd', 'r_hat_d']}

for rep in range(args.n_reps):
    results['a_true'].append(true_a)
    results['alpha_true'].append(true_alpha)
    results['beta_true'].append(true_beta)

    data = contaminate(simulate('weibull', [true_v, true_a, true_z, true_t, true_alpha, true_beta], n_samples=2000),
                       OUTLIER_FRACTION)
    responded, omitted = split_by_deadline(data, DEADLINE)
    omission_rate = np.mean(omitted)
    print(rep)
    print(omission_rate)

    results['omission_rate'].append(omission_rate)

    with pm.Model():
        # Uniform priors on the unbounded scale
        v = pm.Uniform("v", -5,5)
        a = pm.Uniform("a", -5,5)
        z = pt.constant(0.5)
        t = pm.Uniform("t", -5,5)
        alpha = pm.Uniform("alpha", -5,5)
        beta = pm.Uniform("beta", -5,5)
        p_outlier = pt.constant(0)

        pm.Potential("choice_rt", lan_logp_op(responded, v, a, z, t, alpha, beta, p_outlier))
        idata_lan_only = pymc_jax.sample_numpyro_nuts(
                chains=2, draws=1000, tune=3000,target_accept=0.99,progressbar=False)
        summary_lan_only = pm.summary(idata_lan_only)

    results['v_pred_nd'].append(posterior_mean_in_range(summary_lan_only, 'v'))
    results['a_pred_nd'].append(posterior_mean_in_range(summary_lan_only, 'a'))
    results['alpha_pred_nd'].append(posterior_mean_in_range(summary_lan_only, 'alpha'))
    results['beta_pred_nd'].append(posterior_mean_in_range(summary_lan_only, 'beta'))
    results['p_pred_nd'].append(0)   # outlier probability fixed at 0
    results['r_hat_nd'].append(summary_lan_only['r_hat'].max())

    with pm.Model():
        # Uniform priors on the unbounded scale
        v = pm.Uniform("v", -5,5)
        a = pm.Uniform("a", -5,5)
        z = pt.constant(0.5)
        t = pm.Uniform("t", -5,5)
        alpha = pm.Uniform("alpha", -5,5)
        beta = pm.Uniform("beta", -5,5)
        deadline = pt.constant(DEADLINE)
        p_outlier = pt.constant(0)

        pm.Potential("choice_rt", lan_logp_op(responded, v, a, z, t, alpha, beta, p_outlier))
        pm.Potential("omission", omission_logp_op(omitted, v, a, z, t, alpha, beta, deadline, p_outlier))

        idata_with_omissions = pymc_jax.sample_numpyro_nuts(
                chains=2, draws=1000, tune=3000,target_accept=0.99,progressbar=False)
        summary_with_omissions = pm.summary(idata_with_omissions)

    results['v_pred_d'].append(posterior_mean_in_range(summary_with_omissions, 'v'))
    results['a_pred_d'].append(posterior_mean_in_range(summary_with_omissions, 'a'))
    results['alpha_pred_d'].append(posterior_mean_in_range(summary_with_omissions, 'alpha'))
    results['beta_pred_d'].append(posterior_mean_in_range(summary_with_omissions, 'beta'))
    results['p_pred_d'].append(0)   # outlier probability fixed at 0
    results['r_hat_d'].append(summary_with_omissions['r_hat'].max())

    sys.stdout.flush()

save_results(pd.DataFrame(results), args.out_dir, jobid)
