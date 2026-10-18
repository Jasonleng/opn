import jax.numpy as jnp
import pytensor
import pytest

# As in the analysis scripts: must be set before any likelihood Op is built
pytensor.config.floatX = "float32"


def _fake_forward(x):
    """Stand-in network: log-output -0.5 * sum(x**2) is <= 0, so it works as a LAN or omission network."""
    return -0.5 * jnp.sum(jnp.asarray(x) ** 2, axis=-1, keepdims=True)


@pytest.fixture
def fake_forward():
    return _fake_forward
