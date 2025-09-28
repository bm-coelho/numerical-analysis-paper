import numpy as np
import pytest

from regkit.naive import regression_naive, wrap_lambdified


def test_wrap_lambdified_accepts_various_inputs():
    def base(x0, x1):
        return np.vstack((x0 + x1, 2 * x0 - x1))

    wrapped = wrap_lambdified(base, n=2, m=2)

    matrix = wrapped(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(matrix, [[3.0, 0.0], [7.0, 2.0]])

    broadcast = wrapped(np.array([1.0, 3.0]), np.array([2.0, 4.0]))
    np.testing.assert_allclose(broadcast, [[3.0, 0.0], [7.0, 2.0]])

    with pytest.raises(ValueError):
        wrapped(np.array([[1.0, 2.0, 3.0]]))


@pytest.mark.filterwarnings("ignore:.*ill-conditioned.*")
def test_regression_naive_recovers_linear_map():
    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    Y = 2.0 * X + 1.0

    coeffs, approx, predict = regression_naive(X, Y, degree=1, quiet=True)

    preds = np.asarray(predict(X)).reshape(Y.shape)
    np.testing.assert_allclose(preds, Y, atol=1e-8)

    new_points = np.array([[-0.5], [0.5]])
    new_preds = np.asarray(predict(new_points)).reshape(new_points.shape)
    np.testing.assert_allclose(new_preds, 2.0 * new_points + 1.0, atol=1e-8)

    assert coeffs.shape[0] > 0
    assert approx.shape == (1, 1)
