"""The trained networks referenced by the library exist, load, and give finite outputs."""

import os

import numpy as np
import pytest

from opn.networks import NETWORKS, input_dim, load_network, network_files
from opn.paths import NETWORK_DIR

# Plausible parameters per model (v, a, z, t, ...), followed by (rt, choice) or the deadline
PARAMS = {"angle": [1.0, 1.5, 0.5, 0.3, 0.6], "ddm": [1.0, 1.2, 0.5, 0.3],
          "weibull": [1.0, 1.1, 0.5, 0.3, 1.0, 1.5]}


def test_network_folder_exists():
    assert os.path.isdir(NETWORK_DIR)


@pytest.mark.parametrize("model_and_kind", sorted(NETWORKS))
def test_network_files_exist(model_and_kind):
    for path in network_files(*model_and_kind):
        assert os.path.isfile(path), path


@pytest.mark.parametrize("model,kind", sorted(NETWORKS))
def test_network_forward_pass(model, kind):
    forward = load_network(model, kind)
    extra_inputs = [0.7, 1.0] if kind == "lan" else [1.25]
    network_input = np.array([PARAMS[model] + extra_inputs] * 3, dtype=np.float32)
    assert network_input.shape[1] == input_dim(model, kind)
    network_output = np.asarray(forward(network_input))
    assert network_output.shape == (3, 1)
    assert np.all(np.isfinite(network_output))
