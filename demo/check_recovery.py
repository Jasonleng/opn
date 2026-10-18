"""Compare the demo fit with the true parameters the demo data were simulated from.

    python demo/check_recovery.py [fit folder]   (default: demo/output)
"""

import json
import os
import sys

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
fit_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")
truth = pd.read_csv(os.path.join(HERE, "demo_truth.csv"))
info = json.load(open(os.path.join(fit_dir, "fit_info.json")))
post = az.from_netcdf(os.path.join(fit_dir, "posterior.nc")).posterior
bounds = info["parameter_bounds"]
coding = info["predictor_coding"]["difficulty"]


def to_range(x, name):
    low, high = bounds[name]
    return low + (high - low) * expit(x)


# Posterior draws on the natural scale, per subject (in the order of the fit)
draws = {"a": post["a_subj_natural"], "t": post["t_subj_natural"], "p_lapse": post["p_lapse_subj_natural"]}
for level, x in coding.items():
    draws[f"v_{level}"] = to_range(post["v_subj"] + x * post["v_difficulty_subj"], "v")

rows = []
for i, subject in enumerate(info["subjects"]):
    true = truth.set_index("participant").loc[subject]
    for name, d in draws.items():
        samples = d.isel(subject=i).values.ravel()
        low, high = az.hdi(samples, hdi_prob=0.94)
        rows.append({"participant": subject, "parameter": name, "true": true[name],
                     "estimate": samples.mean(), "hdi_3%": low, "hdi_97%": high,
                     "covered": low <= true[name] <= high})
table = pd.DataFrame(rows)
pd.set_option("display.width", 120)
print(table.round(3).to_string(index=False))
print("\nPer parameter: correlation of estimate with truth across participants, and 94% HDI coverage")
for name, g in table.groupby("parameter", sort=False):
    r = np.corrcoef(g["true"], g["estimate"])[0, 1] if g["true"].std() > 0 else np.nan
    print(f"  {name:8s} r = {r:5.2f}   covered {g['covered'].sum()}/{len(g)}")
print(f"\nOverall {table['covered'].sum()}/{len(table)} true values inside the 94% HDI "
      "(expect most, not necessarily all).")
table.to_csv(os.path.join(fit_dir, "recovery.csv"), index=False)
