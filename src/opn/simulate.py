"""Simulating datasets for the parameter-recovery analyses."""

import numpy as np
import ssms

from opn.likelihoods import OUTLIER_RT_MAX


def simulate(model, param_values, n_samples, random_state=None):
    """Simulate (rt, choice) pairs from an ssm-simulators model as an (n_samples, 2) float32 array."""
    simulator_output = ssms.basic_simulators.simulator.simulator(model=model, theta=list(param_values),
                                                                 n_samples=n_samples, random_state=random_state)
    return np.hstack([simulator_output["rts"], simulator_output["choices"]]).astype(np.float32)


def contaminate(data, p_outlier, rt_max=OUTLIER_RT_MAX, rng=None):
    """Replace a fraction p_outlier of trials with contaminants.

    A contaminant has a signed RT drawn uniformly from [-rt_max, rt_max]; its
    absolute value is the RT and its sign the choice (+1 / -1). Uses the global
    NumPy random state unless `rng` (a numpy Generator) is given.
    """
    rng = np.random if rng is None else rng
    n_trials = len(data)
    signed_rts = rng.uniform(-rt_max, rt_max, n_trials)
    contaminant_trials = np.column_stack([np.abs(signed_rts), 2 * (signed_rts >= 0) - 1])
    shuffled_ranks = np.arange(n_trials)
    rng.shuffle(shuffled_ranks)
    is_contaminated = shuffled_ranks < p_outlier * n_trials
    data = data.copy()
    data[is_contaminated, :] = contaminant_trials[is_contaminated, :]
    return data


def split_by_deadline(data, deadline):
    """Split data at a response deadline.

    Returns (commission, omitted): the (rt, choice) rows with rt < deadline, and a
    float32 indicator per trial that is 1 where rt >= deadline.
    """
    commission = data[data[:, 0] < deadline, :]
    omitted = (data[:, 0] >= deadline).astype(np.float32)
    return commission, omitted
