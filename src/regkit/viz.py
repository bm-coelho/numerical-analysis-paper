"""Visualization helpers for the regression datasets.

The plotting utilities select an appropriate dimensionality-reduction strategy
based on the shape of ``X`` and ``Y`` so that the examples from the notebooks
can be rendered without repeating boilerplate code.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

__all__ = ["plot_dataset"]


def plot_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    reduction: str = "pca",
    grid_shape: tuple[int, int] | None = None,
    reducer: PCA | TSNE | None = None,
    show: bool = True,
) -> PCA | TSNE | None:
    """Plot an ``n→m`` dataset ``(X, Y)`` in a visually meaningful way.

    Parameters
    ----------
    X, Y:
        Input and output arrays describing the dataset.
    reduction:
        Dimensionality reduction method to project high-dimensional data.
    grid_shape:
        Optional 2D grid shape when ``X`` describes a mesh grid.
    reducer:
        Previously fitted reducer to reuse. When provided, its ``transform``
        method is used so that multiple plots share the same projection.
    show:
        Whether to call :func:`matplotlib.pyplot.show`. Set to ``False`` when
        running in headless environments that only need to save figures.

    Returns
    -------
    PCA | TSNE | None
        The fitted reducer whenever one is applicable. ``None`` is returned
        for low-dimensional visualizations or t-SNE projections (which cannot
        be reused safely).
    """

    n, N = X.shape
    m, _ = Y.shape
    total_dim = n + m

    if total_dim == 2:
        Z = np.vstack([X, Y])
        plt.figure()
        plt.scatter(Z[0], Z[1], c="C0", marker="o")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"2D scatter of {n}+{m}=2 dims")
        plt.grid(True)
        if show:
            plt.show()
        return None

    if total_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        Z = np.vstack([X, Y])

        if (n, m) == (2, 1) and grid_shape is not None:
            nx, ny = grid_shape
            X1 = Z[0].reshape(nx, ny)
            X2 = Z[1].reshape(nx, ny)
            Y1 = Z[2].reshape(nx, ny)
            ax.plot_surface(X1, X2, Y1, cmap="viridis", edgecolor="none")
            ax.set_xlabel("X₁")
            ax.set_ylabel("X₂")
            ax.set_zlabel("Y")
        else:
            ax.scatter(Z[0], Z[1], Z[2], c="C1", marker="o")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")
            ax.set_title(f"3D scatter of {n}+{m}=3 dims")

        if show:
            plt.show()
        return None

    Z = np.vstack([X, Y]).T
    reducer_name = reduction.lower()
    if reducer is not None and hasattr(reducer, "transform"):
        Zr = reducer.transform(Z)
        method = reducer.__class__.__name__
    elif reducer_name == "tsne":
        reducer = TSNE(n_components=3)
        Zr = reducer.fit_transform(Z)
        method = "TSNE"
        reducer = None  # TSNE does not support transforming new data reliably.
    else:
        reducer = PCA(n_components=3)
        Zr = reducer.fit_transform(Z)
        method = "PCA"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Zr[:, 0], Zr[:, 1], Zr[:, 2], c="C2", marker="o")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(f"{method} projection of {n}+{m} dims → 3D")
    if show:
        plt.show()

    return reducer

