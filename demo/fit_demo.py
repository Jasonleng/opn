"""Hierarchical fit of the demo dataset, and the template for fitting your own data.

To fit your own data, copy this file (e.g. to scripts/hierarchical/fit_<study>.py) and
edit the definition section. Then run it from the repository root:

    python demo/fit_demo.py                                    # full fit
    python demo/fit_demo.py --n-subjects 3 --draws 200 --tune 200 --out-dir tmp/check   # quick check

The data are 6 simulated participants (see make_demo_data.py); check_recovery.py
compares the result with the true parameters.
"""

import argparse
import os
import sys

import pytensor

pytensor.config.floatX = "float32"  # the networks are float32; set before building the model
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from opn import hierarchical as hier

# ==================================== Definition =========================================

# Data: one row per trial. Columns: participant, RT in seconds (empty = no response),
# choice (0/1 or -1/+1), and any condition columns used below.
DATA_FILE = os.path.join(HERE, "demo_data.csv")
DEADLINE = 1.5                      # seconds; no response before the deadline = omission
COLUMNS = dict(subject="participant", rt="rt", response="choice")
# omitted="missed" would add an optional true/false column marking omissions

# Model: which SSM, parameters held fixed, and condition effects on parameters.
SPEC = hier.ModelSpec(
    model="ddm",                     # "ddm", "angle" or "weibull"
    fixed={"z": 0.5},                # unbiased starting point
    regressions={"v": ["difficulty"]},   # drift depends on difficulty (easy = -1, hard = +1)
    # bounds={"t": (0.05, 1.0)},     # narrower parameter ranges than the simulator's
    # random_slopes=False,           # one difficulty effect for everyone
    # lapse=False,                   # no lapse probability
)

SAMPLING = dict(chains=4, draws=1000, tune=1000, target_accept=0.95, seed=1)
OUT_DIR = os.path.join(HERE, "output")

# =========================================================================================

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n-subjects", type=int, default=0, help="fit only the first N participants (0 = all)")
parser.add_argument("--draws", type=int, help=f"override draws per chain ({SAMPLING['draws']})")
parser.add_argument("--tune", type=int, help=f"override tuning steps per chain ({SAMPLING['tune']})")
parser.add_argument("--out-dir", default=OUT_DIR, help="output folder (default: %(default)s)")
args = parser.parse_args()
SAMPLING.update({setting: getattr(args, setting) for setting in ("draws", "tune")
                 if getattr(args, setting) is not None})

trial_table = hier.first_subjects(pd.read_csv(DATA_FILE), COLUMNS["subject"], args.n_subjects)
data = hier.prepare_data(trial_table, DEADLINE, **COLUMNS, predictors=SPEC.predictors)
hier.fit(data, SPEC, args.out_dir, **SAMPLING)
