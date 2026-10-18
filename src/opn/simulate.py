"""Simulating datasets for the parameter-recovery analyses."""

import numpy as np
import ssms

from opn.likelihoods import OUTLIER_RT_MAX


def simulate(model, theta, n_samples, random_state=None):
    """Simulate (rt, choice) pairs from an ssm-simulators model as an (n_samples, 2) float32 array."""
    sim_out = ssms.basic_simulators.simulator.simulator(model=model, theta=list(theta), n_samples=n_samples,
                                                        random_state=random_state)
    return np.hstack([sim_out["rts"], sim_out["choices"]]).astype(np.float32)


def contaminate(data, p_outlier, rt_max=OUTLIER_RT_MAX, rng=None):
    """Replace a fraction p_outlier of trials with contaminants.

    A contaminant has a signed RT drawn uniformly from [-rt_max, rt_max]; its
    absolute value is the RT and its sign the choice (+1 / -1). Uses the global
    NumPy random state unless `rng` (a numpy Generator) is given.
    """
    rng = np.random if rng is None else rng
    n = len(data)
    rt_twoside = rng.uniform(-rt_max, rt_max, n)
    outliers = np.column_stack([np.abs(rt_twoside), 2 * (rt_twoside >= 0) - 1])
    order = np.arange(n)
    rng.shuffle(order)
    mask = order < p_outlier * n
    data = data.copy()
    data[mask, :] = outliers[mask, :]
    return data


def split_by_deadline(data, deadline):
    """Split data at a response deadline.

    Returns (commission, omitted): the (rt, choice) rows with rt < deadline, and a
    float32 indicator per trial that is 1 where rt >= deadline.
    """
    commission = data[data[:, 0] < deadline, :]
    omitted = (data[:, 0] >= deadline).astype(np.float32)
    return commission, omitted
