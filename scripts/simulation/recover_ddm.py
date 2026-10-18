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
from pymc.sampling import jax as pymc_jax

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
omission_logp_op = make_summed_logp_op(make_omission_logp(load_network("ddm", "cpn")), n_params=5)

DEADLINE = 1.25

# The job index selects one cell of the 11 x 11 grid of true boundary and drift values
jobid = args.jobid
a_grid = np.linspace(0.5,1.5,11)
v_grid = np.linspace(0.5,1.5,11)
true_v, true_a, true_z, true_t = v_grid[jobid // 11], a_grid[jobid % 11], 0.5, 0.3

# Output columns: _nd = LAN alone, _d = LAN + omission network, _f = LAN on the full data
results = {column: [] for column in ['a_true', 'v_true', 'a_pred_nd', 'v_pred_nd', 'a_pred_d', 'v_pred_d',
                                     'a_pred_f', 'v_pred_f', 'omission_rate']}

for rep in range(args.n_reps):
    results['a_true'].append(true_a)
    results['v_true'].append(true_v)

    data = simulate('ddm', [true_v, true_a, true_z, true_t], n_samples=1000)
    responded, omitted = split_by_deadline(data, DEADLINE)
    omission_rate = np.mean(omitted)

    print(omission_rate)

    results['omission_rate'].append(omission_rate)

    with pm.Model():
        # Uniform priors
        v = pm.Uniform("v", -3.0, 3.0)
        a = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)

        pm.Potential("choice_rt", lan_logp_op(responded, v, a, z, t))

        idata_lan_only = pymc_jax.sample_numpyro_nuts(
                chains=2, draws=1000, tune=2000,target_accept=0.95,progressbar=False
                )
        summary_lan_only = pm.summary(idata_lan_only)

    results['a_pred_nd'].append(summary_lan_only.loc['a','mean'])
    results['v_pred_nd'].append(summary_lan_only.loc['v','mean'])

    with pm.Model():
        # Uniform priors
        v = pm.Uniform("v", -3.0, 3.0)
        a = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)
        deadline = pt.constant(DEADLINE)

        pm.Potential("choice_rt", lan_logp_op(responded, v, a, z, t))
        pm.Potential("omission", omission_logp_op(omitted, v, a, z, t, deadline))

        idata_with_omissions = pymc_jax.sample_numpyro_nuts(
                chains=2, draws=1000, tune=2000,target_accept=0.95,progressbar=False
                )
        summary_with_omissions = pm.summary(idata_with_omissions)
    results['a_pred_d'].append(summary_with_omissions.loc['a','mean'])
    results['v_pred_d'].append(summary_with_omissions.loc['v','mean'])

    with pm.Model():
        # Uniform priors
        v = pm.Uniform("v", -3.0, 3.0)
        a = pm.Uniform("a", 0.3, 3)
        z = pt.constant(0.5)
        t = pm.Uniform("t", 0.1, 2.0)

        pm.Potential("choice_rt", lan_logp_op(data, v, a, z, t))

        idata_full_data = pymc_jax.sample_numpyro_nuts(
                chains=2, draws=1000, tune=2000,target_accept=0.95,progressbar=False
                )
        summary_full_data = pm.summary(idata_full_data)
    results['a_pred_f'].append(summary_full_data.loc['a','mean'])
    results['v_pred_f'].append(summary_full_data.loc['v','mean'])

    sys.stdout.flush()


save_results(pd.DataFrame(results), args.out_dir, jobid)
