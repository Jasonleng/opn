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
Z = 0.5
# Parameter ranges used by the fit (ssm-simulators' DDM ranges; lapse 0-5 %)
BOUNDS = {"v": (-3.0, 3.0), "a": (0.3, 2.5), "t": (0.0, 2.0), "p_lapse": (0.0, 0.05)}


def to_range(x, name):
    low, high = BOUNDS[name]
    return low + (high - low) * expit(x)


def from_range(value, name):
    low, high = BOUNDS[name]
    return logit((value - low) / (high - low))


rng = np.random.default_rng(SEED)
# Group means (natural scale) and between-subject spreads (unbounded scale)
group = {"v": (1.0, 0.3), "a": (1.0, 0.15), "t": (0.3, 0.15), "p_lapse": (0.02, 0.3)}
difficulty_effect = (-0.35, 0.1)  # on v, unbounded scale; easy = -1, hard = +1

rows, truth = [], []
for s in range(N_SUBJECTS):
    subj = {p: from_range(m, p) + sd * rng.normal() for p, (m, sd) in group.items()}
    effect = difficulty_effect[0] + difficulty_effect[1] * rng.normal()
    a, t, p_lapse = (to_range(subj[p], p) for p in ("a", "t", "p_lapse"))
    record = {"participant": f"P{s + 1:02d}", "a": a, "t": t, "p_lapse": p_lapse}
    for level, x in (("easy", -1.0), ("hard", 1.0)):
        v = to_range(subj["v"] + x * effect, "v")
        record[f"v_{level}"] = v
        sim = simulate("ddm", [v, a, Z, t], N_TRIALS // 2, random_state=int(rng.integers(2**31)))
        lapse = rng.random(len(sim)) < p_lapse                 # lapses: uniform RT before the deadline
        sim[lapse, 0] = rng.uniform(0, DEADLINE, lapse.sum())
        sim[lapse, 1] = rng.choice([-1.0, 1.0], lapse.sum())
        for rt, choice in sim:
            answered = rt < DEADLINE
            rows.append({"participant": record["participant"], "difficulty": level,
                         "rt": round(float(rt), 4) if answered else np.nan,      # no response recorded
                         "choice": int(choice > 0) if answered else np.nan})     # 1 = upper, 0 = lower
    truth.append(record)

data = pd.DataFrame(rows)
# Interleave the conditions within each participant, like a real session
data = (data.groupby("participant", group_keys=False)
        .apply(lambda d: d.sample(frac=1, random_state=SEED)).reset_index(drop=True))
data.insert(1, "trial", data.groupby("participant").cumcount() + 1)
data["choice"] = data["choice"].astype("Int64")                   # 0/1, empty when omitted
data.to_csv(os.path.join(HERE, "demo_data.csv"), index=False)
pd.DataFrame(truth).round(4).to_csv(os.path.join(HERE, "demo_truth.csv"), index=False)
print(f"wrote {len(data)} trials ({data['rt'].isna().mean():.1%} omitted at {DEADLINE} s) to demo/demo_data.csv")
