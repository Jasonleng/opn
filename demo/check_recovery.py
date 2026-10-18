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
fit_info = json.load(open(os.path.join(fit_dir, "fit_info.json")))
posterior = az.from_netcdf(os.path.join(fit_dir, "posterior.nc")).posterior
bounds = fit_info["parameter_bounds"]
difficulty_coding = fit_info["predictor_coding"]["difficulty"]


def to_range(unbounded_value, name):
    low, high = bounds[name]
    return low + (high - low) * expit(unbounded_value)


# Posterior draws on the natural scale, per subject (in the order of the fit)
natural_draws = {"a": posterior["a_subj_natural"], "t": posterior["t_subj_natural"],
                 "p_lapse": posterior["p_lapse_subj_natural"]}
for level, code in difficulty_coding.items():
    natural_draws[f"v_{level}"] = to_range(posterior["v_subj"] + code * posterior["v_difficulty_subj"], "v")

recovery_rows = []
for subject_index, subject in enumerate(fit_info["subjects"]):
    true_values = truth.set_index("participant").loc[subject]
    for name, parameter_draws in natural_draws.items():
        samples = parameter_draws.isel(subject=subject_index).values.ravel()
        hdi_low, hdi_high = az.hdi(samples, hdi_prob=0.94)
        recovery_rows.append({"participant": subject, "parameter": name, "true": true_values[name],
                              "estimate": samples.mean(), "hdi_3%": hdi_low, "hdi_97%": hdi_high,
                              "covered": hdi_low <= true_values[name] <= hdi_high})
recovery = pd.DataFrame(recovery_rows)
pd.set_option("display.width", 120)
print(recovery.round(3).to_string(index=False))
print("\nPer parameter: correlation of estimate with truth across participants, and 94% HDI coverage")
for name, parameter_rows in recovery.groupby("parameter", sort=False):
    correlation = (np.corrcoef(parameter_rows["true"], parameter_rows["estimate"])[0, 1]
                   if parameter_rows["true"].std() > 0 else np.nan)
    print(f"  {name:8s} r = {correlation:5.2f}   covered {parameter_rows['covered'].sum()}/{len(parameter_rows)}")
print(f"\nOverall {recovery['covered'].sum()}/{len(recovery)} true values inside the 94% HDI "
      "(expect most, not necessarily all).")
recovery.to_csv(os.path.join(fit_dir, "recovery.csv"), index=False)
