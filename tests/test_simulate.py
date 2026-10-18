import numpy as np
import pytest

from opn.simulate import contaminate, simulate, split_by_deadline


def test_split_by_deadline():
    data = np.array([[0.5, 1], [1.3, -1], [1.25, 1], [0.9, -1]], dtype=np.float32)
    commission, omitted = split_by_deadline(data, 1.25)
    np.testing.assert_array_equal(commission, data[[0, 3]])
    np.testing.assert_array_equal(omitted, [0, 1, 1, 0])
    assert omitted.dtype == np.float32


def test_contaminate_replaces_the_requested_fraction():
    data = np.column_stack([np.full(1000, 0.123), np.ones(1000)]).astype(np.float32)
    out = contaminate(data, 0.05, rng=np.random.default_rng(0))
    changed = np.any(out != data, axis=1)
    assert changed.sum() == 50
    assert np.all(out[changed, 0] <= 5) and np.all(out[changed, 0] >= 0)
    assert set(np.unique(out[changed, 1])) <= {-1.0, 1.0}
    assert out.dtype == np.float32
    assert np.all(data[:, 0] == np.float32(0.123)), "input must not be modified"


def test_contaminate_with_zero_fraction_is_identity():
    data = np.random.default_rng(1).random((100, 2)).astype(np.float32)
    np.testing.assert_array_equal(contaminate(data, 0.0, rng=np.random.default_rng(0)), data)


@pytest.mark.parametrize("model,theta", [("ddm", [1.0, 1.2, 0.5, 0.3]),
                                         ("angle", [1.0, 1.5, 0.5, 0.3, 0.6]),
                                         ("weibull", [1.0, 1.1, 0.5, 0.3, 1.0, 1.5])])
def test_simulate(model, theta):
    data = simulate(model, theta, n_samples=200)
    assert data.shape == (200, 2) and data.dtype == np.float32
    assert set(np.unique(data[:, 1])) <= {-1.0, 1.0}
    assert np.all(data[:, 0] > 0)
