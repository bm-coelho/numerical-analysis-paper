import numpy as np
import pytest

from regkit.optimized import (
    build_block_system,
    build_design,
    predict_from_blocks,
    regression_largeP,
    regression_largeP_iter,
    regression_optimized,
    solve_blocks,
)


def _make_dataset():
    X = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    Y = np.hstack((2.0 * X + 1.0, -0.5 * X + 0.3))
    return X, Y


def test_build_block_system_matches_design_matrix():
    X, Y = _make_dataset()
    G_block, B, poly = build_block_system(X, Y, degree=2)
    V, _ = build_design(X, degree=2)

    np.testing.assert_allclose(G_block, V.T @ V)
    np.testing.assert_allclose(B, V.T @ Y)
    assert poly.n_output_features_ == V.shape[1]


def test_build_block_system_with_weights():
    X, Y = _make_dataset()
    weights = np.linspace(1.0, 2.0, X.shape[0])

    G_block, B, poly = build_block_system(X, Y, degree=2, weights=weights)
    V = poly.transform(X)
    W = np.diag(weights)

    np.testing.assert_allclose(G_block, V.T @ W @ V)
    np.testing.assert_allclose(B, V.T @ W @ Y)


def test_solve_blocks_adds_ridge_term():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(4, 4))
    G = G.T @ G  # make symmetric positive definite
    B = rng.normal(size=(4, 2))

    A = solve_blocks(G.copy(), B, ridge=0.1)
    expected = np.linalg.solve(G + 0.1 * np.eye(4), B)
    np.testing.assert_allclose(A, expected)


def test_regression_optimized_predictions_match_least_squares():
    X, Y = _make_dataset()
    A, poly, predict = regression_optimized(X, Y, degree=2)

    V, _ = build_design(X, degree=2)
    A_expected, *_ = np.linalg.lstsq(V, Y, rcond=None)

    np.testing.assert_allclose(A, A_expected, atol=1e-12)
    np.testing.assert_allclose(predict(X), V @ A_expected, atol=1e-12)


@pytest.mark.parametrize("degree", [1, 2])
def test_regression_largeP_matches_batchless(degree):
    X, Y = _make_dataset()
    A_full, poly_full, predict_full = regression_optimized(X, Y, degree=degree)

    A_stream, poly_stream, predict_stream = regression_largeP(
        X,
        Y,
        degree=degree,
        batch_size=3,
    )

    np.testing.assert_allclose(A_stream, A_full, atol=1e-12)
    np.testing.assert_allclose(predict_stream(X), predict_full(X), atol=1e-12)


def test_regression_largeP_iter_matches_batch_version():
    X, Y = _make_dataset()

    batches = [
        (X[:4], Y[:4]),
        (X[4:7], Y[4:7]),
        (X[7:], Y[7:]),
    ]

    A_iter, poly_iter, predict_iter = regression_largeP_iter(iter(batches), degree=2)
    A_full, poly_full, predict_full = regression_optimized(X, Y, degree=2)

    np.testing.assert_allclose(A_iter, A_full, atol=1e-12)
    np.testing.assert_allclose(predict_iter(X), predict_full(X), atol=1e-12)
    np.testing.assert_allclose(
        predict_from_blocks(X, poly_iter, A_iter), predict_full(X), atol=1e-12
    )
