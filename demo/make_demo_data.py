"""Simulate the demo dataset: 6 participants, 200 trials each, from the hierarchical DDM.

Writes demo/demo_data.csv (the data, formatted like a typical experiment export) and
demo/demo_truth.csv (the true parameters of every participant). The values are drawn
from the same model that demo/fit_demo.py fits, so check_recovery.py can compare them.

    python demo/make_demo_data.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from opn.simulate import simulate

SEED = 2026
N_SUBJECTS, N_TRIALS = 6, 200
DEADLINE = 1.5
STARTING_POINT = 0.5
# Parameter ranges used by the fit (ssm-simulators' DDM ranges; lapse 0-5 %)
BOUNDS = {"v": (-3.0, 3.0), "a": (0.3, 2.5), "t": (0.0, 2.0), "p_lapse": (0.0, 0.05)}


def to_range(unbounded_value, name):
    low, high = BOUNDS[name]
    return low + (high - low) * expit(unbounded_value)


def from_range(natural_value, name):
    low, high = BOUNDS[name]
    return logit((natural_value - low) / (high - low))


rng = np.random.default_rng(SEED)
# Group means (natural scale) and between-subject spreads (unbounded scale)
group_params = {"v": (1.0, 0.3), "a": (1.0, 0.15), "t": (0.3, 0.15), "p_lapse": (0.02, 0.3)}
difficulty_effect_mean, difficulty_effect_sd = -0.35, 0.1  # on v, unbounded scale; easy = -1, hard = +1

trial_rows, true_params = [], []
for subject_index in range(N_SUBJECTS):
    subject_unbounded = {param: from_range(group_mean, param) + group_sd * rng.normal()
                         for param, (group_mean, group_sd) in group_params.items()}
    difficulty_effect = difficulty_effect_mean + difficulty_effect_sd * rng.normal()
    a, t, p_lapse = (to_range(subject_unbounded[param], param) for param in ("a", "t", "p_lapse"))
    subject_record = {"participant": f"P{subject_index + 1:02d}", "a": a, "t": t, "p_lapse": p_lapse}
    for level, code in (("easy", -1.0), ("hard", 1.0)):
        v = to_range(subject_unbounded["v"] + code * difficulty_effect, "v")
        subject_record[f"v_{level}"] = v
        simulated = simulate("ddm", [v, a, STARTING_POINT, t], N_TRIALS // 2, random_state=int(rng.integers(2**31)))
        is_lapse = rng.random(len(simulated)) < p_lapse                 # lapses: uniform RT before the deadline
        simulated[is_lapse, 0] = rng.uniform(0, DEADLINE, is_lapse.sum())
        simulated[is_lapse, 1] = rng.choice([-1.0, 1.0], is_lapse.sum())
        for rt, choice in simulated:
            answered = rt < DEADLINE
            trial_rows.append({"participant": subject_record["participant"], "difficulty": level,
                               "rt": round(float(rt), 4) if answered else np.nan,      # no response recorded
                               "choice": int(choice > 0) if answered else np.nan})     # 1 = upper, 0 = lower
    true_params.append(subject_record)

demo_data = pd.DataFrame(trial_rows)
# Interleave the conditions within each participant, like a real session
demo_data = (demo_data.groupby("participant", group_keys=False)
             .apply(lambda participant_trials: participant_trials.sample(frac=1, random_state=SEED))
             .reset_index(drop=True))
demo_data.insert(1, "trial", demo_data.groupby("participant").cumcount() + 1)
demo_data["choice"] = demo_data["choice"].astype("Int64")                   # 0/1, empty when omitted
demo_data.to_csv(os.path.join(HERE, "demo_data.csv"), index=False)
pd.DataFrame(true_params).round(4).to_csv(os.path.join(HERE, "demo_truth.csv"), index=False)
print(f"wrote {len(demo_data)} trials ({demo_data['rt'].isna().mean():.1%} omitted at {DEADLINE} s) "
      "to demo/demo_data.csv")
