"""Hierarchical angle-model fit to the task data.

Choices and RTs of responses are scored with the LAN (with a uniform lapse), the
omission of each trial with the omission network. Drift, boundary, starting point,
non-decision time, collapse and lapse vary by subject; v, a and theta also depend
on reward, punishment (and their interaction) and congruency, z on congruency.
Trials with RT above DEADLINE (1.25 s) are treated as omissions. Parameters are
sampled on an unbounded scale and mapped into `bounds`.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytensor

pytensor.config.floatX = "float32"  # the networks are float32; set before any likelihood Op is built
import pymc as pm
from jax.config import config as jax_config
from pymc.sampling import jax as pymc_jax

jax_config.update("jax_enable_x64", False)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from opn.likelihoods import make_trialwise_lan_logp, make_trialwise_omission_logp
from opn.networks import load_network
from opn.ops import make_trialwise_logp_op
from opn.paths import DATA_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="Hierarchical angle-model fit (LAN for choice/RT + omission network) to the task data.")
parser.add_argument("--data", default=os.path.join(DATA_DIR, "hddm_all.csv"), help="trial-level data (default: %(default)s)")
parser.add_argument("--n-subjects", type=int, default=0, help="fit only the first N subjects; 0 = all (default: %(default)s)")
parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "hierarchical", "test_new_pind.nc"), help="posterior InferenceData (default: %(default)s)")
args = parser.parse_args()

DEADLINE = 1.25  # seconds; responses at or after it count as omissions

bounds = {'v': (-3, 3), 'a': (0.2, 2.5), 'z': (0.1, 0.9), 't': (0.01, 0.5), 'theta': (0, 1.2)}
lan_logp_op = make_trialwise_logp_op(
    make_trialwise_lan_logp(load_network("angle", "lan"),
                            ['v', 'a', 'z', 't', 'theta', 'p_outlier'], {**bounds, 'p_outlier': (0, 0.05)},
                            lapse_density=1 / (2 * DEADLINE)),   # uniform over (rt < deadline) x 2 choices
    varies_by_trial=[True] * 6)
omission_logp_op = make_trialwise_logp_op(
    make_trialwise_omission_logp(load_network("angle", "cpn"),
                                 ['v', 'a', 'z', 't', 'theta', 'deadline'], bounds),
    varies_by_trial=[True] * 5 + [False])

trials = pd.read_csv(args.data)

if args.n_subjects > 0:
    trials = trials[trials['subj_idx'].isin(trials['subj_idx'].unique()[:args.n_subjects])].reset_index(drop=True)

# Subjects and two-level conditions are numbered in order of first appearance; conditions and
# responses are then coded -1/+1
trials['subject'] = trials['subj_idx'].factorize()[0]

trials['catRewLevel'] = trials['catRewLevel'].factorize()[0]

trials['catPunLevel'] = trials['catPunLevel'].factorize()[0]

trials['catCong'] = trials['catCong'].factorize()[0]

trials['oe'].mask(trials['rt'] > DEADLINE, True, inplace=True)

trials['response'] = trials['response'] * 2 - 1

trials['catRewLevel'] = trials['catRewLevel'] * 2 - 1

trials['catPunLevel'] = trials['catPunLevel'] * 2 - 1

trials['reward_x_punishment'] = trials['catRewLevel'] * trials['catPunLevel']

trials['catCong'] = trials['catCong'] * 2 - 1

answered_trials = trials.loc[trials['oe'] == False]

coords = {
    "id": trials.subject.unique(),  # actual group names
    "observation1": np.arange(answered_trials.shape[0]),  # trials answered before the deadline
    "observation2": np.arange(trials.shape[0])            # all trials
}

# Python names describe each variable; the PyMC names in quotes are kept as they are because they
# are the names in the saved posterior. In those, "_trial1" / "1" refers to the answered trials and
# "_trial2" / "2" to all trials; "catRewLevel" is reward, "catPunLevel" punishment and "cong"
# congruency.
with pm.Model(coords=coords) as hierarchical_model:
    # Hyperpriors
    v_mean = pm.Normal('mu_v',mu=0,sigma=1)
    v_sd = pm.HalfCauchy('sigma_v',beta=0.2)
    v_subj = pm.Normal('v_subj',mu=v_mean,sigma=v_sd,dims='id')

    v_reward_mean = pm.Normal('v_catRewLevel',mu=0,sigma=1)
    v_reward_sd = pm.HalfCauchy('sigma_v_catRewLevel',beta=0.2)
    v_reward_subj = pm.Normal('v_catRewLevel_subj',mu=v_reward_mean,sigma=v_reward_sd,dims='id')

    a_reward_mean = pm.Normal('a_catRewLevel',mu=0,sigma=1)
    a_reward_sd = pm.HalfCauchy('sigma_a_catRewLevel',beta=0.2)
    a_reward_subj = pm.Normal('a_catRewLevel_subj',mu=a_reward_mean,sigma=a_reward_sd,dims='id')

    theta_reward_mean = pm.Normal('theta_catRewLevel',mu=0,sigma=1)
    theta_reward_sd = pm.HalfCauchy('sigma_theta_catRewLevel',beta=0.2)
    theta_reward_subj = pm.Normal('theta_catRewLevel_subj',mu=theta_reward_mean,sigma=theta_reward_sd,dims='id')

    v_reward_x_punishment_mean = pm.Normal('v_catRewLevelcatPunLevel',mu=0,sigma=0.5)
    v_reward_x_punishment_sd = pm.HalfCauchy('sigma_v_catRewLevelPunLevel',beta=0.2)
    v_reward_x_punishment_subj = pm.Normal('v_catRewLevelcatPunLevel_subj',mu=v_reward_x_punishment_mean,
                                           sigma=v_reward_x_punishment_sd,dims='id')
    a_reward_x_punishment_mean = pm.Normal('a_catRewLevelcatPunLevel',mu=0,sigma=0.5)
    a_reward_x_punishment_sd = pm.HalfCauchy('sigma_a_catRewLevelcatPunLevel',beta=0.2)
    a_reward_x_punishment_subj = pm.Normal('a_catRewLevelcatPunLevel_subj',mu=a_reward_x_punishment_mean,
                                           sigma=a_reward_x_punishment_sd,dims='id')

    theta_reward_x_punishment_mean = pm.Normal('theta_catRewLevelcatPunLevel',mu=0,sigma=0.5)
    theta_reward_x_punishment_sd = pm.HalfCauchy('sigma_theta_catRewLevelcatPunLevel',beta=0.2)
    theta_reward_x_punishment_subj = pm.Normal('theta_catRewLevelcatPunLevel_subj',
                                               mu=theta_reward_x_punishment_mean,
                                               sigma=theta_reward_x_punishment_sd,dims='id')

    v_congruency_mean = pm.Normal('v_cong',mu=0,sigma=1)
    v_congruency_sd = pm.HalfCauchy('sigma_v_cong',beta=0.2)
    v_congruency_subj = pm.Normal('v_cong_subj',mu=v_congruency_mean,sigma=v_congruency_sd,dims='id')

    a_congruency_mean = pm.Normal('a_cong',mu=0,sigma=1)
    a_congruency_sd = pm.HalfCauchy('sigma_a_cong',beta=0.2)
    a_congruency_subj = pm.Normal('a_cong_subj',mu=a_congruency_mean,sigma=a_congruency_sd,dims='id')

    theta_congruency_mean = pm.Normal('theta_cong',mu=0,sigma=1)
    theta_congruency_sd = pm.HalfCauchy('sigma_theta_cong',beta=0.2)
    theta_congruency_subj = pm.Normal('theta_cong_subj',mu=theta_congruency_mean,sigma=theta_congruency_sd,dims='id')

    z_congruency_mean = pm.Normal('z_cong',mu=0,sigma=1)
    z_congruency_sd = pm.HalfCauchy('sigma_z_cong',beta=0.2)
    z_congruency_subj = pm.Normal('z_cong_subj',mu=z_congruency_mean,sigma=z_congruency_sd,dims='id')

    v_punishment_mean = pm.Normal('v_catPunLevel',mu=0,sigma=1)
    v_punishment_sd = pm.HalfCauchy('sigma_v_catPunLevel',beta=0.2)
    v_punishment_subj = pm.Normal('v_catPunLevel_subj',mu=v_punishment_mean,sigma=v_punishment_sd,dims='id')

    a_punishment_mean = pm.Normal('a_catPunLevel',mu=0,sigma=1)
    a_punishment_sd = pm.HalfCauchy('sigma_a_catPunLevel',beta=0.2)
    a_punishment_subj = pm.Normal('a_catPunLevel_subj',mu=a_punishment_mean,sigma=a_punishment_sd,dims='id')

    theta_punishment_mean = pm.Normal('theta_catPunLevel',mu=0,sigma=1)
    theta_punishment_sd = pm.HalfCauchy('sigma_theta_catPunLevel',beta=0.2)
    theta_punishment_subj = pm.Normal('theta_catPunLevel_subj',mu=theta_punishment_mean,
                                      sigma=theta_punishment_sd,dims='id')

    theta_mean = pm.Normal('mu_theta',mu=0,sigma=1)
    theta_sd = pm.HalfCauchy('sigma_theta',beta=0.2)
    theta_subj = pm.Normal('theta_subj',mu=theta_mean,sigma=theta_sd,dims='id')

    a_mean = pm.Normal('mu_a',mu=0,sigma=1)
    a_sd = pm.HalfCauchy('sigma_a',beta=0.2)
    a_subj = pm.Normal('a_subj',mu=a_mean,sigma=a_sd,dims='id')

    z_mean = pm.Normal("mu_z", mu=0,sigma=1)
    z_sd = pm.HalfCauchy("sigma_z",beta=0.2)
    z_subj = pm.Normal("z_subj", mu=z_mean,sigma=z_sd,dims='id')

    t_mean = pm.Normal("mu_t", mu=0,sigma=1)
    t_sd = pm.HalfCauchy("sigma_t",beta=0.2)
    t_subj = pm.Normal("t_subj", mu=t_mean,sigma=t_sd,dims='id')

    # Lapse (outlier) probability
    lapse_mean = pm.Normal("mu_p", mu=0,sigma=1)
    lapse_sd = pm.HalfCauchy("sigma_p",beta=0.2)
    lapse_subj = pm.Normal("p_subj", mu=lapse_mean,sigma=lapse_sd,dims='id')

    deadline = pm.ConstantData('deadline',DEADLINE)

    subject_answered = pm.ConstantData('idx1',answered_trials.subject,dims='observation1')
    subject_all = pm.ConstantData('idx2',trials.subject,dims='observation2')

    reward_answered = pm.ConstantData('rewardLevel1',answered_trials.catRewLevel,dims='observation1')
    reward_all = pm.ConstantData('rewardLevel2',trials.catRewLevel,dims='observation2')
    punishment_answered = pm.ConstantData('punishLevel1',answered_trials.catPunLevel,dims='observation1')
    punishment_all = pm.ConstantData('punishLevel2',trials.catPunLevel,dims='observation2')
    reward_x_punishment_answered = pm.ConstantData('rewpun1',answered_trials.reward_x_punishment,dims='observation1')
    reward_x_punishment_all = pm.ConstantData('rewpun2',trials.reward_x_punishment,dims='observation2')
    congruency_answered = pm.ConstantData('cong1',answered_trials.catCong,dims='observation1')
    congruency_all = pm.ConstantData('cong2',trials.catCong,dims='observation2')

    v_answered = pm.Deterministic('v_trial1',v_subj[subject_answered] +
                                  reward_answered * v_reward_subj[subject_answered] +
                                  punishment_answered * v_punishment_subj[subject_answered] +
                                  congruency_answered * v_congruency_subj[subject_answered] +
                                  reward_x_punishment_answered * v_reward_x_punishment_subj[subject_answered])
    v_all = pm.Deterministic('v_trial2',v_subj[subject_all] +
                             reward_all * v_reward_subj[subject_all] +
                             punishment_all * v_punishment_subj[subject_all] +
                             congruency_all * v_congruency_subj[subject_all] +
                             reward_x_punishment_all * v_reward_x_punishment_subj[subject_all])
    a_answered = pm.Deterministic('a_trial1',a_subj[subject_answered] +
                                  reward_answered * a_reward_subj[subject_answered] +
                                  punishment_answered * a_punishment_subj[subject_answered] +
                                  congruency_answered * a_congruency_subj[subject_answered] +
                                  reward_x_punishment_answered * a_reward_x_punishment_subj[subject_answered])
    a_all = pm.Deterministic('a_trial2',a_subj[subject_all] +
                             reward_all * a_reward_subj[subject_all] +
                             punishment_all * a_punishment_subj[subject_all] +
                             congruency_all * a_congruency_subj[subject_all] +
                             reward_x_punishment_all * a_reward_x_punishment_subj[subject_all])
    theta_answered = pm.Deterministic('theta_trial1',theta_subj[subject_answered] +
                                      reward_answered * theta_reward_subj[subject_answered] +
                                      punishment_answered * theta_punishment_subj[subject_answered] +
                                      congruency_answered * theta_congruency_subj[subject_answered] +
                                      reward_x_punishment_answered * theta_reward_x_punishment_subj[subject_answered])
    theta_all = pm.Deterministic('theta_trial2',theta_subj[subject_all] +
                                 reward_all * theta_reward_subj[subject_all] +
                                 punishment_all * theta_punishment_subj[subject_all] +
                                 congruency_all * theta_congruency_subj[subject_all] +
                                 reward_x_punishment_all * theta_reward_x_punishment_subj[subject_all])
    z_answered = pm.Deterministic('z_trial1',z_subj[subject_answered] +
                                  congruency_answered * z_congruency_subj[subject_answered])
    z_all = pm.Deterministic('z_trial2',z_subj[subject_all] + congruency_all * z_congruency_subj[subject_all])

    pm.CustomDist("choice_rt", v_answered,
                               a_answered,
                               z_answered,
                               t_subj[subject_answered],
                               theta_answered,
                               lapse_subj[subject_answered],
                  logp=lan_logp_op,observed=answered_trials[['rt','response']])
    pm.CustomDist("omission", v_all,
                              a_all,
                              z_all,
                              t_subj[subject_all],
                              theta_all,
                              deadline,
                  logp=omission_logp_op,observed=trials['oe'])
    idata = pymc_jax.sample_numpyro_nuts(
            chains=2, draws=1000, tune=1000,initvals={'mu_v':0,'mu_a':0,'mu_theta':0,"mu_z":0,"mu_t":0,
                                                     "v_catRewLevel":0,"a_catRewLevel":0,
                                                    'theta_catRewLevel':0,"v_catPunLevel":0,"a_catPunLevel":0,
                                                    'theta_catPunLevel':0,"v_cong":0,"a_cong":0,
                                                    'theta_cong':0,'z_cong':0,'v_catRewLevelcatPunLevel':0,'mu_p':0,
                                                    'a_catRewLevelcatPunLevel':0,'theta_catRewLevelcatPunLevel':0}
    )

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
idata.to_netcdf(args.out)
