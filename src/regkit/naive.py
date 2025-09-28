"""Naive/symbolic polynomial regression helpers.

The routines in this module mirror the derivations from the exploratory
notebooks. They rely on SymPy to construct a symbolic polynomial basis and
provide NumPy-friendly wrappers so that the expressions can be evaluated
efficiently when running experiments.
"""

from __future__ import annotations

import itertools as it
from typing import Iterable, Sequence

import numpy as np
import sympy as sp
from sympy import itermonomials
from tqdm.auto import tqdm

__all__ = [
    "build_terms",
    "build_gram_num",
    "build_rhs_num",
    "wrap_lambdified",
    "regression_naive",
]


def build_terms(n: int, m: int, r: int) -> tuple[list[sp.Expr], list[sp.Lambda], Sequence[sp.Symbol]]:
    """Construct symbolic basis terms and their numerical counterparts.

    Parameters
    ----------
    n:
        Number of input dimensions.
    m:
        Number of output dimensions.
    r:
        Polynomial degree used to generate the monomial basis.

    Returns
    -------
    tuple[list[sp.Expr], list[sp.Lambda], Sequence[sp.Symbol]]
        The SymPy expressions, the NumPy-callable lambdas, and the symbolic
        variables ``(x0, …, x(n-1))`` used to build the model.
    """
    x = sp.symbols(f"x0:{n}")  # (x0, x1, ..., xn-1)
    e = sp.symbols(f"e0:{m}")  # (e0, e1, ..., e(m-1))

    terms = list(itermonomials(x, r))
    pairs = list(it.product(terms, e))
    exprs = [term * vec for term, vec in pairs]

    I = sp.eye(m)
    subs_map = {e[i]: I[:, i] for i in range(m)}
    exprs_sub = [ex.subs(subs_map) for ex in exprs]

    lambda_expr = [sp.lambdify(x, ex, "numpy") for ex in exprs_sub]
    return exprs_sub, lambda_expr, x


def build_gram_num(
    terms_num: Sequence[sp.Lambda],
    X: Iterable[Sequence[float]] | np.ndarray,
    *,
    quiet: bool = False,
) -> np.ndarray:
    """Numerically assemble the Gram matrix for the symbolic basis.

    Parameters
    ----------
    terms_num:
        Iterable of lambdified basis functions as produced by
        :func:`build_terms`.
    X:
        Collection of input samples. The function accepts iterables of vectors
        or NumPy arrays with shape ``(P, n)``.
    quiet:
        When ``True``, progress bars from :mod:`tqdm` are suppressed.

    Returns
    -------
    np.ndarray
        The symmetric Gram matrix ``G`` with shape ``(K, K)`` where ``K`` is the
        number of basis functions.
    """
    K = len(terms_num)
    G = np.zeros((K, K), dtype=float)

    for i in tqdm(range(K), desc="Gram rows", disable=quiet):
        fi = terms_num[i]
        for j in range(i + 1):
            fj = terms_num[j]
            v = sum(np.dot(fi(*pt).ravel(), fj(*pt).ravel()) for pt in X)
            G[i, j] = v
            G[j, i] = v

    return G


def build_rhs_num(
    terms_num: Sequence[sp.Lambda],
    X: Iterable[Sequence[float]] | np.ndarray,
    Y: Iterable[Sequence[float]] | np.ndarray,
    *,
    quiet: bool = False,
) -> np.ndarray:
    """Build the right-hand side vector for the least-squares system.

    Parameters
    ----------
    terms_num:
        Numerical basis functions obtained from :func:`build_terms`.
    X, Y:
        Matching collections of input and output samples.
    quiet:
        When ``True``, disables progress reporting during the accumulation.

    Returns
    -------
    np.ndarray
        The right-hand side vector ``B`` with one entry per basis function.
    """
    return np.array(
        [
            sum(np.dot(f(*pt).ravel(), y) for pt, y in zip(X, Y))
            for f in tqdm(terms_num, desc="Building RHS", disable=quiet)
        ],
        dtype=float,
    )


def wrap_lambdified(base_func, n: int, m: int):
    """Wrap a lambdified function to accept flexible NumPy inputs.

    The helper normalizes a variety of valid calling conventions—single array,
    per-dimension arrays, 1D and 2D inputs—into the layout expected by the
    lambdified SymPy expression. This mirrors the behaviour of the inline
    helpers used in the notebooks while keeping the callable ergonomic for
    downstream experiments.
    """

    def f(*args):  # type: ignore[override]
        if len(args) == 1:
            arr = np.array(args[0], ndmin=1)
            if arr.ndim == 1 and arr.size == n:
                cols = [arr]
            elif arr.ndim == 2 and arr.shape[1] == n:
                cols = [arr[:, i] for i in range(n)]
            else:
                raise ValueError(
                    f"Input must be length-{n} or shape (N,{n}); got {arr.shape}"
                )
        elif len(args) == n:
            cols = [np.array(a, ndmin=1) for a in args]
            length = max(col.size for col in cols)
            cols = [np.broadcast_to(col, (length,)) for col in cols]
        else:
            raise ValueError(f"Expected 1 or {n} args, got {len(args)}")

        res = base_func(*cols)
        out = np.array(res)
        if out.ndim == 1:
            return out.reshape(1, -1)
        if out.ndim == 2 and out.shape[0] == m and out.shape[1] == len(cols[0]):
            return out.T
        return out

    return f


def regression_naive(
    X,
    Y,
    degree: int,
    *,
    quiet: bool = False,
):
    """Fit the symbolic polynomial regression model described in the notebooks.

    Parameters
    ----------
    X, Y:
        Input and output samples matching the ``(n, m)`` regression task.
    degree:
        Polynomial degree of the approximation.
    quiet:
        Propagated to :func:`build_gram_num` and :func:`build_rhs_num` to
        silence progress bars.

    Returns
    -------
    tuple[sp.Matrix, sp.Expr, callable]
        The SymPy coefficient vector, the symbolic approximation, and a
        NumPy-callable predictor constructed from the symbolic expression.
    """

    if isinstance(X, list):
        X = np.array(X)
    if isinstance(Y, list):
        Y = np.array(Y)

    n = X.shape[1]
    m = Y.shape[1]

    terms_sym, terms_num, x_syms = build_terms(n, m, degree)
    G = build_gram_num(terms_num, X, quiet=quiet)
    B = build_rhs_num(terms_num, X, Y, quiet=quiet)

    a, *_ = np.linalg.lstsq(G, B, rcond=None)

    coeffs = sp.Matrix(a)
    zero = sp.zeros(*terms_sym[0].shape)
    approx = sum((coeffs[i] * terms_sym[i] for i in range(len(a))), zero)

    f_base = sp.lambdify(x_syms, approx, "numpy")
    f_approx = wrap_lambdified(f_base, n, m)

    return coeffs, approx, f_approx


