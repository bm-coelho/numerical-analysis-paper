"""Optimized polynomial regression routines.

These helpers implement the numerical counterpart of the symbolic approach in
``regkit.naive``. The design matrix is built via ``PolynomialFeatures`` and the
resulting normal equations are solved in blocks so that the work is shared
across output dimensions. Streaming variants are provided to handle datasets
that do not fit in memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "build_design",
    "build_block_system",
    "solve_blocks",
    "predict_from_blocks",
    "regression_optimized",
    "regression_largeP",
    "regression_largeP_iter",
]

Batch = Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]


def build_design(
    X: ArrayLike,
    degree: int,
    include_bias: bool = True,
    interaction_only: bool = False,
):
    """Return the design matrix and fitted :class:`PolynomialFeatures`.

    Parameters
    ----------
    X:
        Input samples with shape ``(P, n)``.
    degree:
        Maximum polynomial degree of the expansion.
    include_bias:
        When ``True`` the constant feature is kept in the design matrix.
    interaction_only:
        If ``True``, restricts the basis to interaction terms.

    Returns
    -------
    tuple[np.ndarray, PolynomialFeatures]
        The Vandermonde-like matrix ``V`` and the fitted
        :class:`~sklearn.preprocessing.PolynomialFeatures` transformer.
    """

    X = np.asarray(X, dtype=float)
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
        order="C",
    )
    V = poly.fit_transform(X)
    return V, poly


def build_block_system(
    X: ArrayLike,
    Y: ArrayLike,
    degree: int,
    *,
    weights: ArrayLike | None = None,
    include_bias: bool = True,
    interaction_only: bool = False,
):
    """Build the normal-equations block system.

    The function mirrors the symbolic construction but leverages the compact
    block structure that arises from vector-valued outputs.

    Parameters
    ----------
    X, Y:
        Input and output samples with matching leading dimension.
    degree, include_bias, interaction_only:
        Configuration forwarded to :func:`build_design`.
    weights:
        Optional per-sample weights used to form a weighted least-squares
        system.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, PolynomialFeatures]
        The Gram block ``G_block``, the right-hand side matrix ``B`` and the
        fitted feature transformer.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    V, poly = build_design(X, degree, include_bias, interaction_only)

    if weights is None:
        G_block = V.T @ V
        B = V.T @ Y
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        G_block = V.T @ (w * V)
        B = V.T @ (w * Y)

    return G_block, B, poly


def solve_blocks(G_block: np.ndarray, B: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Solve ``G_block A = B`` optionally adding a ridge term.

    Parameters
    ----------
    G_block:
        Symmetric Gram matrix produced by :func:`build_block_system`.
    B:
        Right-hand side matrix storing the projections against the targets.
    ridge:
        Non-negative regularisation parameter. When positive, ``ridge`` times
        the identity matrix is added to ``G_block`` before solving.
    """

    if ridge:
        G_block = G_block + ridge * np.eye(G_block.shape[0])
    return np.linalg.solve(G_block, B)


def predict_from_blocks(X_new: ArrayLike, poly: PolynomialFeatures, A: np.ndarray) -> np.ndarray:
    """Evaluate predictions at new points using the learned coefficients.

    Parameters
    ----------
    X_new:
        Evaluation points. The leading dimension corresponds to samples.
    poly:
        Polynomial transformer returned by :func:`build_block_system`.
    A:
        Coefficient matrix obtained from :func:`solve_blocks` or
        :func:`regression_optimized`.
    """

    V_new = poly.transform(np.asarray(X_new, dtype=float))
    return V_new @ A


def regression_optimized(X, Y, degree: int, quiet: bool = False):
    """Fit a multivariate polynomial regression using the block Gram system.

    Parameters
    ----------
    X, Y:
        Matrices representing the inputs and outputs of the regression task.
    degree:
        Polynomial degree of the approximation.
    quiet:
        Suppresses the numerical conditioning warning when ``True``.

    Returns
    -------
    tuple[np.ndarray, PolynomialFeatures, callable]
        The coefficient matrix ``A``, the fitted transformer, and a convenience
        prediction callable.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if not (X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]):
        msg = "Shape mismatch between X and Y"
        raise ValueError(msg)

    G_block, B, poly = build_block_system(
        X,
        Y,
        degree,
        weights=None,
        include_bias=True,
        interaction_only=False,
    )

    cond = np.linalg.cond(G_block)
    if not np.isfinite(cond) or cond > 1e12:
        if not quiet:
            print(
                "[regression] Warning: Gram block is ill-conditioned "
                f"(cond≈{cond:.2e}); falling back to lstsq on design matrix."
            )
        V, _ = build_design(X, degree, include_bias=True, interaction_only=False)
        A, *_ = np.linalg.lstsq(V, Y, rcond=None)
    else:
        A = solve_blocks(G_block, B)

    def predict(X_new):
        return predict_from_blocks(X_new, poly, A)

    return A, poly, predict


def regression_largeP(
    X: np.ndarray,
    Y: np.ndarray,
    degree: int,
    *,
    batch_size: int = 8192,
    weights: np.ndarray | None = None,
    include_bias: bool = True,
    interaction_only: bool = False,
    ridge: float = 0.0,
    quiet: bool = False,
):
    """Out-of-core regression for large ``P`` using batches from arrays.

    Parameters
    ----------
    X, Y:
        Arrays containing the full dataset with matching leading dimension.
    degree, include_bias, interaction_only:
        Configuration forwarded to :func:`_init_poly_from_n` and
        :func:`predict_from_blocks`.
    batch_size:
        Number of rows processed at a time when accumulating the Gram block.
    weights:
        Optional weights with length ``P`` to perform weighted least squares.
    ridge:
        Regularisation parameter applied to the Gram matrix.
    quiet:
        Disables conditioning warnings when ``True``.

    Returns
    -------
    tuple[np.ndarray, PolynomialFeatures, callable]
        The coefficient matrix, the fitted polynomial transformer, and a
        prediction helper compatible with new data.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if not (X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]):
        raise ValueError("Shape mismatch between X and Y")

    P, n = X.shape
    m = Y.shape[1]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != P:
            raise ValueError("Weights must have length equal to number of rows in X")

    poly = _init_poly_from_n(n, degree, include_bias, interaction_only)
    T = poly.n_output_features_

    G = np.zeros((T, T), dtype=float)
    B = np.zeros((T, m), dtype=float)

    for start in range(0, P, batch_size):
        stop = min(start + batch_size, P)
        Xb = X[start:stop]
        Yb = Y[start:stop]
        wb = None if weights is None else weights[start:stop]
        _accumulate_GB(G, B, Xb, Yb, poly, wb)

    if ridge:
        G = G + ridge * np.eye(T)

    cond = np.linalg.cond(G)
    if not np.isfinite(cond) or cond > 1e12:
        if not quiet:
            print(
                "[regression_largeP] Warning: Gram block ill-conditioned "
                f"(cond≈{cond:.2e}). Adding ridge {max(ridge,1e-8):.1e}."
            )
        G = G + max(ridge, 1e-8) * np.eye(T)

    A = np.linalg.solve(G, B)

    def predict(X_new):
        return predict_from_blocks(X_new, poly, A)

    return A, poly, predict


def regression_largeP_iter(
    batch_iter: Iterable[Batch],
    degree: int,
    *,
    n: int | None = None,
    include_bias: bool = True,
    interaction_only: bool = False,
    ridge: float = 0.0,
    quiet: bool = False,
):
    """Out-of-core regression where batches are provided by an iterator.

    Parameters
    ----------
    batch_iter:
        Iterator yielding ``(Xb, Yb)`` or ``(Xb, Yb, wb)`` batches.
    degree, include_bias, interaction_only:
        Configuration forwarded to the polynomial feature generator.
    n:
        Input dimensionality. When omitted, it is inferred from the first
        batch.
    ridge:
        Optional ridge regularisation applied to the accumulated Gram matrix.
    quiet:
        Suppresses the conditioning warnings emitted when solving the system.

    Returns
    -------
    tuple[np.ndarray, PolynomialFeatures, callable]
        The coefficient matrix, the fitted polynomial transformer, and a
        prediction helper for new data points.
    """

    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise ValueError("Empty iterator") from exc

    if len(first) == 2:
        Xb0, Yb0 = first
        wb0 = None
    elif len(first) == 3:
        Xb0, Yb0, wb0 = first
    else:
        raise ValueError("Batches must be (Xb, Yb) or (Xb, Yb, wb)")

    Xb0 = np.asarray(Xb0, dtype=float)
    Yb0 = np.asarray(Yb0, dtype=float)
    if n is None:
        n = Xb0.shape[1]
    m = Yb0.shape[1]

    poly = _init_poly_from_n(n, degree, include_bias, interaction_only)
    T = poly.n_output_features_
    G = np.zeros((T, T), dtype=float)
    B = np.zeros((T, m), dtype=float)

    _accumulate_GB(G, B, Xb0, Yb0, poly, wb0)
    for batch in it:
        if len(batch) == 2:
            Xb, Yb = batch
            wb = None
        else:
            Xb, Yb, wb = batch
        _accumulate_GB(G, B, Xb, Yb, poly, wb)

    if ridge:
        G = G + ridge * np.eye(T)

    cond = np.linalg.cond(G)
    if not np.isfinite(cond) or cond > 1e12:
        if not quiet:
            print(
                "[regression_largeP_iter] Warning: Gram block ill-conditioned "
                f"(cond≈{cond:.2e}). Adding ridge {max(ridge,1e-8):.1e}."
            )
        G = G + max(ridge, 1e-8) * np.eye(T)

    A = np.linalg.solve(G, B)

    def predict(X_new):
        return predict_from_blocks(X_new, poly, A)

    return A, poly, predict


def _init_poly_from_n(
    n: int,
    degree: int,
    include_bias: bool,
    interaction_only: bool,
) -> PolynomialFeatures:
    """Initialise ``PolynomialFeatures`` knowing only the input dimension ``n``."""

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
        order="C",
    )
    poly.fit(np.zeros((1, n), dtype=float))
    return poly


def _accumulate_GB(
    G: np.ndarray,
    B: np.ndarray,
    Xb: ArrayLike,
    Yb: ArrayLike,
    poly: PolynomialFeatures,
    weights: ArrayLike | None,
) -> None:
    """Accumulate Gram and right-hand side contributions from a batch."""

    Xb = np.asarray(Xb, dtype=float)
    Yb = np.asarray(Yb, dtype=float)
    Vb = poly.transform(Xb)

    if weights is None:
        G += Vb.T @ Vb
        B += Vb.T @ Yb
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        G += Vb.T @ (w * Vb)
        B += Vb.T @ (w * Yb)


