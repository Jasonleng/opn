"""JAX log-likelihoods built from the trained networks.

Two families, matching how the analyses use them:

* Summed (`make_lan_logp`, `make_omission_logp`): parameters are shared by all
  trials and the function returns one summed log-likelihood. Used by the angle
  and DDM parameter-recovery simulations; wrap with `ops.make_summed_logp_op`.
* Trialwise (`make_trialwise_lan_logp`, `make_trialwise_omission_logp`): the
  function scores a single trial, parameters are on an unbounded scale and are
  mapped into `bounds` with a scaled sigmoid. Vectorised over trials (and over
  trial-varying parameters) by `ops.make_trialwise_logp_op`. Used by the Weibull
  simulation and the hierarchical fit.

Outliers are modelled as a mixture with a uniform contaminant whose signed RT is
uniform on [-OUTLIER_RT_MAX, OUTLIER_RT_MAX] (see `simulate.contaminate`).
"""

import jax.numpy as jnp
from jax.scipy.special import expit

EPSILON = 1e-64
OUTLIER_RT_MAX = 5.0
# Density of a contaminant (rt, choice) pair: signed RT uniform on [-5, 5]
OUTLIER_DENSITY = 1 / (2 * OUTLIER_RT_MAX)


def scaled_sigmoid(unbounded_value, low, high):
    """Map an unbounded value into (low, high)."""
    return expit(unbounded_value) * (high - low) + low


def outlier_probability(unbounded_weight):
    """Map an unbounded value to an outlier probability in (0, 0.1) (summed family)."""
    return 0.1 * jnp.exp(unbounded_weight) / (1 + jnp.exp(unbounded_weight))


def omission_probability(network_output, network="cpn"):
    """P(omission) from an omission-network output.

    network="cpn": the network outputs log P(response before deadline) (angle, DDM).
    network="opn": the network outputs log P(omission) (Weibull).
    """
    if network == "cpn":
        return 1 - jnp.exp(network_output) + EPSILON
    if network == "opn":
        return jnp.exp(network_output) + EPSILON
    raise ValueError(f"network must be 'cpn' or 'opn', got {network!r}")


def mix_outlier_omission(p_omission, p_outlier, deadline, rt_max=OUTLIER_RT_MAX):
    """Omission probability when a fraction p_outlier of trials are contaminants (|RT| uniform on [0, rt_max])."""
    return (1 - p_outlier) * p_omission + p_outlier * (1 - deadline / rt_max)


# --- Summed family -----------------------------------------------------------

def make_lan_logp(forward, outlier=False):
    """Summed log-likelihood of (rt, choice) data under a LAN.

    Returns ``logp(data, *params)`` with data an (n_trials, >=2) array of
    (rt, choice). With ``outlier=True`` the last parameter is an unbounded outlier
    weight, mapped by `outlier_probability`, mixed with a uniform contaminant.
    """
    def logp(data, *params):
        if outlier:
            params, p_outlier = params[:-1], outlier_probability(params[-1])
        params_matrix = jnp.repeat(jnp.stack(params).reshape(1, -1), axis=0, repeats=data.shape[0])
        input_matrix = jnp.hstack([params_matrix, data[:, :2]])
        likelihood = jnp.exp(forward(input_matrix))
        if outlier:
            likelihood = p_outlier * OUTLIER_DENSITY + (1 - p_outlier) * likelihood
        return jnp.sum(jnp.squeeze(jnp.log(likelihood)))

    return logp


def make_omission_logp(forward, network="cpn", outlier=False, rt_max=OUTLIER_RT_MAX):
    """Summed log-likelihood of the omitted trials under an omission network.

    Returns ``logp(data, *params, deadline[, outlier])`` with data a per-trial
    omission indicator (1 = omitted). With ``outlier=True`` the last parameter is
    an unbounded outlier weight (see `make_lan_logp`); contaminant RTs are uniform
    on [0, rt_max].
    """
    def logp(data, *params):
        n_omitted = jnp.sum(data > 0)
        network_params = params[:-1] if outlier else params
        network_output = jnp.squeeze(forward(jnp.stack(network_params).reshape(1, -1)))
        p_omission = omission_probability(network_output, network)
        if outlier:
            p_omission = mix_outlier_omission(p_omission, outlier_probability(params[-1]), params[-2], rt_max)
        return jnp.log(p_omission) * n_omitted

    return logp


# --- Trialwise family --------------------------------------------------------

def transform_params(params, param_names, bounds):
    """Apply `scaled_sigmoid` to every parameter named in `bounds`; leave the rest unchanged."""
    return [scaled_sigmoid(param_value, *bounds[name]) if name in bounds else param_value
            for param_value, name in zip(params, param_names)]


def make_trialwise_lan_logp(forward, param_names, bounds, lapse_density):
    """Log-likelihood of one (rt, choice) trial under a LAN with a uniform lapse.

    Returns ``logp(trial, *params)``; the last entry of `param_names` must be the
    lapse probability, mixed with a uniform density `lapse_density`.
    """
    def logp(trial, *params):
        natural_values = transform_params(params, param_names, bounds)
        network_input = jnp.concatenate((jnp.array(natural_values[:-1]), trial))
        p_lapse = natural_values[-1]
        likelihood = jnp.multiply(jnp.exp(forward(network_input)), 1 - p_lapse) + p_lapse * lapse_density
        return jnp.sum(jnp.squeeze(jnp.log(likelihood)))

    return logp


def make_trialwise_omission_logp(forward, param_names, bounds, network="cpn", outlier=False,
                                 rt_max=OUTLIER_RT_MAX):
    """Log-likelihood contribution of one trial's omission indicator.

    Returns ``logp(omitted, *params)``: log P(omission) if the trial was omitted,
    otherwise 0. `param_names` ends with "deadline", or with ("deadline",
    "p_outlier") when ``outlier=True``: a fraction p_outlier of trials are
    contaminants with RT uniform on [0, rt_max]. With rt_max equal to the deadline
    (a lapse that always responds in time) this is (1 - p_outlier) * P(omission).
    """
    def logp(omitted, *params):
        natural_values = transform_params(params, param_names, bounds)
        network_values = natural_values[:-1] if outlier else natural_values
        network_output = jnp.squeeze(forward(jnp.array(network_values)))
        p_omission = omission_probability(network_output, network)
        if outlier:
            p_omission = mix_outlier_omission(p_omission, natural_values[-1], natural_values[-2], rt_max)
        return jnp.log(p_omission) * jnp.sum(omitted > 0)

    return logp
