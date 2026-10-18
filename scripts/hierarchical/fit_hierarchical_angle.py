"""Hierarchical angle-model fit to the task data.

Choices and RTs of responses are scored with the LAN (with a uniform lapse), the
omission of each trial with the omission network. Drift, boundary, starting point,
non-decision time, collapse and lapse vary by subject; v, a and theta also depend
on reward, punishment (and their interaction) and congruency, z on congruency.
Trials with RT above 1.25 s are treated as omissions. Parameters are sampled on an
unbounded scale and mapped into `bounds`.
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
from jax.config import config
from pymc.sampling import jax as pmj

config.update("jax_enable_x64", False)

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

bounds = {'v': (-3, 3), 'a': (0.2, 2.5), 'z': (0.1, 0.9), 't': (0.01, 0.5), 'theta': (0, 1.2)}
lan_logp_op = make_trialwise_logp_op(
    make_trialwise_lan_logp(load_network("angle", "lan"),
                            ['v', 'a', 'z', 't', 'theta', 'p_outlier'], {**bounds, 'p_outlier': (0, 0.05)},
                            lapse_density=1 / 2.5),
    params_is_reg=[True] * 6)
cpn_logp_op = make_trialwise_logp_op(
    make_trialwise_omission_logp(load_network("angle", "cpn"),
                                 ['v', 'a', 'z', 't', 'theta', 'deadline'], bounds),
    params_is_reg=[True] * 5 + [False])

df = pd.read_csv(args.data)

if args.n_subjects > 0:
    df = df[df['subj_idx'].isin(df['subj_idx'].unique()[:args.n_subjects])].reset_index(drop=True)

df['subject'] = df['subj_idx'].factorize()[0]

df['catRewLevel'] = df['catRewLevel'].factorize()[0]

df['catPunLevel'] = df['catPunLevel'].factorize()[0]

df['catCong'] = df['catCong'].factorize()[0]

df['oe'].mask(df['rt'] > 1.25, True, inplace=True)

df['response'] = df['response'] * 2 - 1

df['catRewLevel'] = df['catRewLevel'] * 2 - 1

df['catPunLevel'] = df['catPunLevel'] * 2 - 1

df['rewpunLevel'] = df['catRewLevel'] * df['catPunLevel']

df['catCong'] = df['catCong'] * 2 - 1

df_commission = df.loc[df['oe'] == False]

coords = {
    "id": df.subject.unique(),  # actual group names
    "observation1": np.arange(df_commission.shape[0]),  # or use this, `data.index.values
    "observation2": np.arange(df.shape[0])
}

with pm.Model(coords=coords) as hierarchical:
    # Hyperpriors
    mu_v = pm.Normal('mu_v',mu=0,sigma=1)
    sigma_v = pm.HalfCauchy('sigma_v',beta=0.2)
    v_subj = pm.Normal('v_subj',mu=mu_v,sigma=sigma_v,dims='id')

    mu_v_catRewLevel = pm.Normal('v_catRewLevel',mu=0,sigma=1)
    sigma_v_catRewLevel = pm.HalfCauchy('sigma_v_catRewLevel',beta=0.2)
    v_catRewLevel_subj = pm.Normal('v_catRewLevel_subj',mu=mu_v_catRewLevel,sigma=sigma_v_catRewLevel,dims='id')
    
    mu_a_catRewLevel = pm.Normal('a_catRewLevel',mu=0,sigma=1)
    sigma_a_catRewLevel = pm.HalfCauchy('sigma_a_catRewLevel',beta=0.2)
    a_catRewLevel_subj = pm.Normal('a_catRewLevel_subj',mu=mu_a_catRewLevel,sigma=sigma_a_catRewLevel,dims='id')
    
    mu_theta_catRewLevel = pm.Normal('theta_catRewLevel',mu=0,sigma=1)
    sigma_theta_catRewLevel = pm.HalfCauchy('sigma_theta_catRewLevel',beta=0.2)
    theta_catRewLevel_subj = pm.Normal('theta_catRewLevel_subj',mu=mu_theta_catRewLevel,sigma=sigma_theta_catRewLevel,dims='id')    

    mu_v_catRewLevelcatPunLevel = pm.Normal('v_catRewLevelcatPunLevel',mu=0,sigma=0.5)
    sigma_v_catRewLevelcatPunLevel = pm.HalfCauchy('sigma_v_catRewLevelPunLevel',beta=0.2)
    v_catRewLevelcatPunLevel_subj = pm.Normal('v_catRewLevelcatPunLevel_subj',mu=mu_v_catRewLevelcatPunLevel,
                                              sigma=sigma_v_catRewLevelcatPunLevel,dims='id')
    mu_a_catRewLevelcatPunLevel = pm.Normal('a_catRewLevelcatPunLevel',mu=0,sigma=0.5)
    sigma_a_catRewLevelcatPunLevel = pm.HalfCauchy('sigma_a_catRewLevelcatPunLevel',beta=0.2)
    a_catRewLevelcatPunLevel_subj = pm.Normal('a_catRewLevelcatPunLevel_subj',mu=mu_a_catRewLevelcatPunLevel,sigma=sigma_a_catRewLevelcatPunLevel,dims='id')
    
    mu_theta_catRewLevelcatPunLevel = pm.Normal('theta_catRewLevelcatPunLevel',mu=0,sigma=0.5)
    sigma_theta_catRewLevelcatPunLevel = pm.HalfCauchy('sigma_theta_catRewLevelcatPunLevel',beta=0.2)
    theta_catRewLevelcatPunLevel_subj = pm.Normal('theta_catRewLevelcatPunLevel_subj',
                                                  mu=mu_theta_catRewLevelcatPunLevel,sigma=sigma_theta_catRewLevelcatPunLevel,dims='id')    

    mu_v_cong = pm.Normal('v_cong',mu=0,sigma=1)
    sigma_v_cong = pm.HalfCauchy('sigma_v_cong',beta=0.2)
    v_cong_subj = pm.Normal('v_cong_subj',mu=mu_v_cong,sigma=sigma_v_cong,dims='id')
    
    mu_a_cong = pm.Normal('a_cong',mu=0,sigma=1)
    sigma_a_cong = pm.HalfCauchy('sigma_a_cong',beta=0.2)
    a_cong_subj = pm.Normal('a_cong_subj',mu=mu_a_cong,sigma=sigma_a_cong,dims='id')
    
    mu_theta_cong = pm.Normal('theta_cong',mu=0,sigma=1)
    sigma_theta_cong = pm.HalfCauchy('sigma_theta_cong',beta=0.2)
    theta_cong_subj = pm.Normal('theta_cong_subj',mu=mu_theta_cong,sigma=sigma_theta_cong,dims='id')   
    
    mu_z_cong = pm.Normal('z_cong',mu=0,sigma=1)
    sigma_z_cong = pm.HalfCauchy('sigma_z_cong',beta=0.2)
    z_cong_subj = pm.Normal('z_cong_subj',mu=mu_z_cong,sigma=sigma_z_cong,dims='id')
    
    mu_v_catPunLevel = pm.Normal('v_catPunLevel',mu=0,sigma=1)
    sigma_v_catPunLevel = pm.HalfCauchy('sigma_v_catPunLevel',beta=0.2)
    v_catPunLevel_subj = pm.Normal('v_catPunLevel_subj',mu=mu_v_catPunLevel,sigma=sigma_v_catPunLevel,dims='id')
    
    mu_a_catPunLevel = pm.Normal('a_catPunLevel',mu=0,sigma=1)
    sigma_a_catPunLevel = pm.HalfCauchy('sigma_a_catPunLevel',beta=0.2)
    a_catPunLevel_subj = pm.Normal('a_catPunLevel_subj',mu=mu_a_catPunLevel,sigma=sigma_a_catPunLevel,dims='id')
    
    mu_theta_catPunLevel = pm.Normal('theta_catPunLevel',mu=0,sigma=1)
    sigma_theta_catPunLevel = pm.HalfCauchy('sigma_theta_catPunLevel',beta=0.2)
    theta_catPunLevel_subj = pm.Normal('theta_catPunLevel_subj',mu=mu_theta_catPunLevel,sigma=sigma_theta_catPunLevel,dims='id')    
 
    mu_theta = pm.Normal('mu_theta',mu=0,sigma=1)
    sigma_theta = pm.HalfCauchy('sigma_theta',beta=0.2)
    theta_subj = pm.Normal('theta_subj',mu=mu_theta,sigma=sigma_theta,dims='id')
    
    mu_a = pm.Normal('mu_a',mu=0,sigma=1)
    sigma_a = pm.HalfCauchy('sigma_a',beta=0.2)
    a_subj = pm.Normal('a_subj',mu=mu_a,sigma=sigma_a,dims='id')
        
    mu_z = pm.Normal("mu_z", mu=0,sigma=1)
    sigma_z = pm.HalfCauchy("sigma_z",beta=0.2)
    z_subj = pm.Normal("z_subj", mu=mu_z,sigma=sigma_z,dims='id')
    
    mu_t = pm.Normal("mu_t", mu=0,sigma=1)
    sigma_t = pm.HalfCauchy("sigma_t",beta=0.2)
    t_subj = pm.Normal("t_subj", mu=mu_t,sigma=sigma_t,dims='id')
    

    mu_p = pm.Normal("mu_p", mu=0,sigma=1)
    sigma_p = pm.HalfCauchy("sigma_p",beta=0.2)
    p_subj = pm.Normal("p_subj", mu=mu_p,sigma=sigma_p,dims='id')
    
    deadline = pm.ConstantData('deadline',1.25)

    idx1 = pm.ConstantData('idx1',df_commission.subject,dims='observation1')
    idx2 = pm.ConstantData('idx2',df.subject,dims='observation2')
    
    rewardLevel1 = pm.ConstantData('rewardLevel1',df_commission.catRewLevel,dims='observation1')
    rewardLevel2 = pm.ConstantData('rewardLevel2',df.catRewLevel,dims='observation2')
    punishLevel1 = pm.ConstantData('punishLevel1',df_commission.catPunLevel,dims='observation1')
    punishLevel2 = pm.ConstantData('punishLevel2',df.catPunLevel,dims='observation2')
    rewpun1 = pm.ConstantData('rewpun1',df_commission.rewpunLevel,dims='observation1')
    rewpun2 = pm.ConstantData('rewpun2',df.rewpunLevel,dims='observation2')
    cong1 = pm.ConstantData('cong1',df_commission.catCong,dims='observation1')
    cong2 = pm.ConstantData('cong2',df.catCong,dims='observation2')
    
    v_trial1 = pm.Deterministic('v_trial1',v_subj[idx1] + rewardLevel1 * v_catRewLevel_subj[idx1] + 
                                punishLevel1 * v_catPunLevel_subj[idx1] +
                                cong1 * v_cong_subj[idx1] + rewpun1 * v_catRewLevelcatPunLevel_subj[idx1])
    v_trial2 = pm.Deterministic('v_trial2',v_subj[idx2] + rewardLevel2 * v_catRewLevel_subj[idx2] + 
                                punishLevel2 * v_catPunLevel_subj[idx2] + 
                                cong2 * v_cong_subj[idx2] + rewpun2 * v_catRewLevelcatPunLevel_subj[idx2])
    a_trial1 = pm.Deterministic('a_trial1',a_subj[idx1] + rewardLevel1 * a_catRewLevel_subj[idx1] + 
                                punishLevel1 * a_catPunLevel_subj[idx1] + 
                                cong1 * a_cong_subj[idx1] + rewpun1 * a_catRewLevelcatPunLevel_subj[idx1])
    a_trial2 = pm.Deterministic('a_trial2',a_subj[idx2] + rewardLevel2 * a_catRewLevel_subj[idx2] + 
                                punishLevel2 * a_catPunLevel_subj[idx2] + 
                                cong2 * a_cong_subj[idx2] + rewpun2 * a_catRewLevelcatPunLevel_subj[idx2])   
    theta_trial1 = pm.Deterministic('theta_trial1',theta_subj[idx1] + rewardLevel1 * theta_catRewLevel_subj[idx1] + 
                                    punishLevel1 * theta_catPunLevel_subj[idx1] + 
                                    cong1 * theta_cong_subj[idx1] + rewpun1 * theta_catRewLevelcatPunLevel_subj[idx1])
    theta_trial2 = pm.Deterministic('theta_trial2',theta_subj[idx2] + rewardLevel2 * theta_catRewLevel_subj[idx2] + 
                                    punishLevel2 * theta_catPunLevel_subj[idx2] + 
                                    cong2 * theta_cong_subj[idx2] + rewpun2 * theta_catRewLevelcatPunLevel_subj[idx2])
    z_trial1 = pm.Deterministic('z_trial1',z_subj[idx1] + cong1 * z_cong_subj[idx1])  
    z_trial2 = pm.Deterministic('z_trial2',z_subj[idx2] + cong2 * z_cong_subj[idx2])
    
    pm.CustomDist("choice_rt", v_trial1,
                                          a_trial1,
                                          z_trial1, 
                                          t_subj[idx1], 
                                          theta_trial1,
                                          p_subj[idx1],
                 logp=lan_logp_op,observed=df_commission[['rt','response']])
    pm.CustomDist("omission", v_trial2,
                                         a_trial2, 
                                         z_trial2, 
                                         t_subj[idx2], 
                                         theta_trial2,
                                         deadline,
                 logp=cpn_logp_op,observed=df['oe'])   
    ddm_blog_traces_numpyro_d = pmj.sample_numpyro_nuts(
            chains=2, draws=1000, tune=1000,initvals={'mu_v':0,'mu_a':0,'mu_theta':0,"mu_z":0,"mu_t":0,
                                                     "v_catRewLevel":0,"a_catRewLevel":0,
                                                    'theta_catRewLevel':0,"v_catPunLevel":0,"a_catPunLevel":0,
                                                    'theta_catPunLevel':0,"v_cong":0,"a_cong":0,
                                                    'theta_cong':0,'z_cong':0,'v_catRewLevelcatPunLevel':0,'mu_p':0,
                                                    'a_catRewLevelcatPunLevel':0,'theta_catRewLevelcatPunLevel':0}
    )
    
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
ddm_blog_traces_numpyro_d.to_netcdf(args.out)
