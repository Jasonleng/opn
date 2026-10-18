"""Loading the trained networks (lanfactory / flax) used by the analyses."""

import os

import lanfactory
import ssms

from opn.paths import NETWORK_DIR

# (model, kind) -> (network config, train state), relative to networks/jax/.
# kind "lan": log-likelihood of (rt, choice) given parameters; input = parameters + (rt, choice).
# kind "cpn": omission network; input = parameters + deadline. The angle and DDM networks
# output log P(response before deadline), the Weibull network outputs log P(omission).
NETWORKS = {
    ("angle", "lan"): ("angle/lan/96f2b24a933211ee99b9a0423f3e9a40_lan_angle__network_config.pickle",
                       "angle/lan/96f2b24a933211ee99b9a0423f3e9a40_lan_angle__train_state.jax"),
    ("angle", "cpn"): ("angle/cpn/338ff01ca91911ee91a3a0423f3e9b42_cpn_angle__network_config.pickle",
                       "angle/cpn/338ff01ca91911ee91a3a0423f3e9b42_cpn_angle__train_state.jax"),
    ("ddm", "lan"): ("ddm/lan/e91014e8b9a011eeafa5a0423f3e9be0_lan_ddm__network_config.pickle",
                     "ddm/lan/e91014e8b9a011eeafa5a0423f3e9be0_lan_ddm__train_state.jax"),
    ("ddm", "cpn"): ("ddm/cpn/0ea678a4b93911ee8857a0423f3e9a40_cpn_ddm__network_config.pickle",
                     "ddm/cpn/0ea678a4b93911ee8857a0423f3e9a40_cpn_ddm__train_state.jax"),
    ("weibull", "lan"): ("weibull/lan/2b8c70363dff11ef8680a0423f3e9b42_lan_weibull_network_config.pickle",
                         "weibull/lan/2b8c70363dff11ef8680a0423f3e9b42_lan_weibull__train_state.jax"),
    ("weibull", "cpn"): ("weibull/cpn/cd5cafb23ec111efa0bca0423f3e9b5e_opn_weibull_deadline_network_config.pickle",
                         "weibull/cpn/cd5cafb23ec111efa0bca0423f3e9b5e_cpn_weibull_deadline__train_state.jax"),
}


# What the omission ("cpn") network of each model outputs: log P(response before the
# deadline) ("cpn") or log P(omission) ("opn"); see likelihoods.omission_probability.
OMISSION_NETWORK_OUTPUT = {"angle": "cpn", "ddm": "cpn", "weibull": "opn"}


def input_dim(model, kind):
    """Network input size: the model parameters plus (rt, choice) for a LAN, plus the deadline for a CPN."""
    n_params = ssms.config.model_config[model]["n_params"]
    return n_params + (2 if kind == "lan" else 1)


def network_files(model, kind):
    """Absolute paths of the (config, state) files for a registered network."""
    config_file, state_file = NETWORKS[(model, kind)]
    return os.path.join(NETWORK_DIR, config_file), os.path.join(NETWORK_DIR, state_file)


def load_network_from_files(config_path, state_path, n_inputs, seed=42, retries=1):
    """Load a lanfactory network with n_inputs inputs and return its (un-jitted) forward pass.

    Loading is retried `retries` times, as the original analysis scripts did.
    """
    for attempt in range(retries + 1):
        try:
            network_factory = lanfactory.trainers.MLPJaxFactory(network_config=str(config_path), train=False)
            forward, _ = network_factory.make_forward_partial(
                seed=seed, input_dim=n_inputs, state=str(state_path), add_jitted=False)
            return forward
        except Exception:
            if attempt == retries:
                raise


def load_network(model, kind):
    """Load a registered network, e.g. ``load_network("angle", "lan")``."""
    config_path, state_path = network_files(model, kind)
    return load_network_from_files(config_path, state_path, input_dim(model, kind))
