User guide
==========

This guide provides a practical tour of RegKit's regression workflow—the same
one followed in the exploratory notebooks. The intent is to make it easier to
reproduce figures, tables, and sanity checks from the undergraduate study. The
examples assume familiarity with NumPy arrays and scikit-learn style
transformers.

Workflow overview
-----------------

1. Create or load a dataset using :mod:`regkit.datasets`.
2. Choose a solver—either the symbolic :mod:`regkit.naive` helpers for small
   problems or the numerical :mod:`regkit.optimized` implementations for larger
   experiments.
3. Evaluate the fitted model and visualise predictions with
   :func:`regkit.viz.plot_dataset` or custom plotting code.

Dataset generation
------------------

The dataset utilities produce matrix-shaped arrays matching the mathematical
notation used in the paper. ``X`` represents inputs of shape ``(n, N)`` and
``Y`` outputs of shape ``(m, N)``. For compatibility with scikit-learn
estimators, transpose them into ``(N, n)`` and ``(N, m)`` before calling the
optimised solvers.

.. code-block:: python

   from regkit.datasets import gen_exploration_dataset, create_target_dataset

   X_explore, Y_explore = gen_exploration_dataset(num_points=256, n=3, m=2, seed=1)
   X_target, Y_target = create_target_dataset()

   X_samples = X_explore.T  # shape (N, n)
   Y_samples = Y_explore.T  # shape (N, m)

The :func:`create_target_dataset` helper mirrors the deterministic dataset used
throughout the manuscript and the notebooks. Use it as a lightweight benchmark
when comparing symbolic and numerical runs.

Fitting models
--------------

The optimised block solver is the recommended default for exploratory work. It
leverages :class:`~sklearn.preprocessing.PolynomialFeatures` to build the design
matrix and solves for the coefficients jointly across outputs, just like in the
automation scripts.

.. code-block:: python

   from regkit.optimized import regression_optimized

   A, poly, predict = regression_optimized(X_samples, Y_samples, degree=3)
   Y_pred = predict(X_samples)

   residual = Y_samples - Y_pred
   rmse = (residual**2).mean(axis=0) ** 0.5
   print("Per-output RMSE:", rmse)

Switch to :func:`regkit.optimized.regression_largeP` when the dataset contains
hundreds of thousands of samples. The routine accumulates the normal equations
in batches, reducing the memory footprint while preserving the same API.
Iterative data sources are supported through
:func:`regkit.optimized.regression_largeP_iter`, which accepts generators or
streaming loaders yielding mini-batches. The experiments scripts use these
helpers purely to automate the sweeps captured in the notebooks.

Symbolic baseline
-----------------

The :mod:`regkit.naive` module mirrors the derivations discussed in the paper
and notebooks using SymPy to build the polynomial basis explicitly. Although
computationally heavier, the routine exposes the symbolic expressions and
provides insight into the structure of the approximation. This was especially
useful during the undergraduate study for debugging assumptions and validating
handwritten calculations.

.. code-block:: python

   from regkit.naive import regression_naive

   coeffs, approx_expr, approx_fun = regression_naive(X_explore.T, Y_explore.T, degree=2)
   print("Number of symbolic terms:", coeffs.size)
   print("Symbolic approximation:", approx_expr)

Visualising results
-------------------

Use :func:`regkit.viz.plot_dataset` to inspect datasets and predictions. The
helper automatically selects a sensible dimensionality-reduction strategy and
returns the fitted reducer when applicable so that subsequent plots reuse the
projection, the same workflow the notebooks follow when assembling figures.

.. code-block:: python

   from regkit.viz import plot_dataset

   reducer = plot_dataset(X_explore, Y_explore, reduction="pca")
   plot_dataset(X_explore, Y_pred.T, reducer=reducer)  # reuse PCA basis

For bespoke visualisations, call :func:`regkit.optimized.predict_from_blocks`
with the fitted transformer to obtain the design matrix, then use Matplotlib or
Plotly directly. Keep in mind that these utilities were written to support a
single undergraduate project, so they favour clarity over clever abstractions.
