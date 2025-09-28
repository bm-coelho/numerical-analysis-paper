Getting started
===============

This quick-start guide walks you through installing the lightweight RegKit
helpers, validating your Python environment, and running the smallest working
regression example.

Prerequisites
-------------

RegKit targets Python 3.12 or later. The reference environment uses NumPy,
scikit-learn, Matplotlib, and SymPy for the symbolic routines
demonstrated in the notebooks. Install the project in editable mode when
working from a clone of this repository:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate
   (.venv) $ pip install -r requirements.txt  # if available
   (.venv) $ pip install -e .[sympy]


Sanity check
------------

With the environment ready, run a small regression task to ensure the helper
package is importable and that compiled extensions from dependencies load
correctly. Execute the following snippet in an interactive shell or notebook
cell:

.. code-block:: python

   import numpy as np
   from regkit.datasets import gen_exploration_dataset
   from regkit.optimized import regression_optimized

   X, Y = gen_exploration_dataset(num_points=128, n=2, m=1, seed=7, noise=1e-3)
   # Transpose to match the ``(P, n)`` convention used by scikit-learn helpers.
   X_samples = X.T
   Y_samples = Y.T

   A, poly, predict = regression_optimized(X_samples, Y_samples, degree=3)
   Y_pred = predict(X_samples)
   err = np.linalg.norm(Y_samples - Y_pred) / np.linalg.norm(Y_samples)
   print(f"Relative training error: {err:.2%}")

The relative error should be well below ``1%`` for the default dataset and
parameters. A warning about an ill-conditioned Gram matrix may appear for
higher-degree polynomials; see :ref:`theory` for mitigation tips. In the study
this behaviour is discussed qualitatively rather than solved perfectly; the
project aims for understanding, not industrial robustness.

Next steps
----------

* Continue to :doc:`user-guide` for a walkthrough of the regression pipeline
  that feeds the exploratory notebooks.
* Browse the :doc:`api/modules` reference if you need details on the helper
  functions used throughout the undergraduate study.
