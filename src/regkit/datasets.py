"""Dataset generators used throughout the notebooks and examples."""

from __future__ import annotations

import numpy as np

__all__ = ["gen_exploration_dataset", "create_target_dataset"]


def gen_exploration_dataset(
    num_points: int,
    n: int,
    m: int,
    *,
    seed: int | None = None,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the exploratory manifold dataset used in the paper.

    Parameters
    ----------
    num_points:
        Number of samples to draw along the latent spiral manifold.
    n:
        Dimension of the input space.
    m:
        Dimension of the output space.
    seed:
        Optional seed forwarded to :func:`numpy.random.default_rng` for
        reproducibility.
    noise:
        Standard deviation of the isotropic Gaussian noise added to the
        samples. Set to zero for noiseless data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair ``(X, Y)`` where ``X`` has shape ``(n, num_points)`` and ``Y``
        has shape ``(m, num_points)``.
    """

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, num_points)

    D = n + m
    Z = np.zeros((D, num_points))

    for i in range(D // 2):
        Z[2 * i, :] = np.sin((i + 1) * t)
        Z[2 * i + 1, :] = np.cos((i + 1) * t)

    if D % 2:
        Z[-1, :] = t / t.max()

    if noise > 0:
        Z += rng.normal(scale=noise, size=Z.shape)

    X = Z[:n, :]
    Y = Z[n:, :]
    return X, Y


def create_target_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Recreate the structured target dataset described in the paper.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The pair ``(X, Y)`` matching the MATLAB prototype distributed with the
        project. Both arrays have shape ``(4, N)`` where ``N`` is the combined
        number of time and parameter samples.
    """

    t = np.arange(0.0, 2.6 + 0.2, 0.2)
    q = np.arange(1.0, 2.0 + 0.2, 0.2)

    Nt, Nq = t.size, q.size
    N = Nt + Nq - 1

    X = np.zeros((4, N))
    Y = np.zeros((4, N))

    for k in range(Nt):
        for ell in range(Nq):
            phi0 = np.array([1, 0, 0, 0])
            phi1 = np.array([0, 1, 0, 0])
            phi2 = np.array([0, 0, 1, 0])
            phi3 = np.array([0, 0, 0, 1])
            phi4 = np.array([t[k], 0, 0, 0])
            phi5 = np.array([0, t[k], 0, 0])
            phi6 = np.array([0, 0, q[ell], 0])
            phi7 = np.array([0, 0, 0, q[ell]])
            phi8 = np.array([t[k] ** 2, 0, 0, 0])
            phi9 = np.array([0, 0, 0, np.sin(t[k])])

            idx = k + ell
            X[:, idx] = phi0 + phi1 + phi2 + phi3 + phi4 + phi5 + phi6 + phi7
            Y[:, idx] = (
                2 * phi0
                + phi1
                - phi2
                + 4 * phi3
                + 5 * phi4
                + 2 * phi5
                - phi6
                + phi7
                + 0.1 * phi8
                + 0.2 * phi9
            )

    return X, Y


# Backwards compatibility for any existing notebook imports.
create_dataset = create_target_dataset

