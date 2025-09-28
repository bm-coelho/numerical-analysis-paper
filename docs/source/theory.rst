.. _theory:

Theory and numerical considerations
===================================

RegKit implements two complementary strategies for multivariate polynomial
regression. Both approximate an unknown mapping ``f : R^n → R^m`` with a basis
of tensor-product monomials up to degree ``r``—the same framing explored in the
undergraduate notebooks. The difference lies in how the basis is constructed and
how the resulting linear system is solved.

Symbolic construction
---------------------

The :mod:`regkit.naive` module mirrors the derivations presented in the
exploratory notebooks. Symbolic monomials are generated using
:func:`sympy.polys.monomials.itermonomials`, paired with each output axis, and
assembled into a Gram matrix ``G`` and right-hand side vector ``B``. Solving the
system ``G a = B`` yields the coefficient vector that defines the approximation
``\hat{f}``.

This approach exposes rich intermediate artefacts: symbolic expressions for each
basis element, lambdified callables, and the exact coefficient matrix. These are
valuable when validating theoretical claims or deriving closed-form expressions,
which is why the notebooks lean on them heavily. However, the computational cost
grows rapidly with dimension and degree because the number of monomials scales
as ``O(\binom{n + r}{r})``.

Block-structured normal equations
---------------------------------

The optimised implementation leverages the observation that the regression task
can be expressed as a block system when outputs share the same polynomial basis.
Instead of instantiating symbolic monomials, :func:`regkit.optimized.build_design`
constructs the design matrix ``V`` via :class:`~sklearn.preprocessing.PolynomialFeatures`.
The normal equations take the form ``(V^T V) A = V^T Y`` where the coefficient
matrix ``A`` has one column per output dimension. Solving the dense block system
avoids redundant work and is compatible with NumPy and SciPy linear algebra
routines. This is the path chosen for the larger experiments purely to make the
undergraduate study tractable.

Ill-conditioning and regularisation
-----------------------------------

High-degree polynomial features are notoriously ill-conditioned, especially when
input variables vary across different scales. RegKit therefore exposes several
controls:

* All regression helpers accept a ``ridge`` parameter that adds ``\lambda I`` to
  the Gram matrix before solving.
* The large-sample solvers accumulate ``V^T V`` incrementally, which reduces
  catastrophic cancellation compared to forming the design matrix explicitly.
* The :func:`regkit.optimized.regression_optimized` routine falls back to
  :func:`numpy.linalg.lstsq` when the condition number exceeds ``1e12``.

For more demanding problems consider feature scaling, orthogonal polynomial
bases, or regularisation paths via scikit-learn estimators. These ideas are
briefly touched on in the notebooks but remain outside the scope of this
exploratory work.

.. Complexity summary
.. ------------------

.. .. list-table::
..    :header-rows: 1
..    :widths: 28 72

..    * - Routine
..      - Computational characteristics
..    * - ``regression_naive``
..      - ``O(P K^2 + K^3)`` where ``K = m\,\binom{n+r}{r}`` counts the tensor
..        product monomials per output. The ``K``-nested loops in
..        :func:`regkit.naive.build_gram_num` and :func:`~regkit.naive.build_rhs_num`
..        drive the ``P K^2`` term; the ``K^3`` contribution stems from the
..        fallback to :func:`numpy.linalg.lstsq` when solving ``G a = B``.
..    * - ``regression_optimized``
..      - ``O(P T^2 + P T m + T^3 + T^2 m)`` with ``T`` denoting the expanded
..        feature count (:math:`p` in ``paper/common/sections/sec_otimizacao_regressao.tex``). The
..        ``V^T V`` and ``V^T Y`` products contribute the ``P``-linear terms,
..        while :func:`numpy.linalg.solve` on the ``T \times T`` Gram block yields
..        ``T^3 + T^2 m``. When ``T = p`` the expression matches the paper's
..        ``O(N p^2 + p^3 + p^2 m)``.
..    * - ``regression_largeP``
..      - ``O(P T^2 + P T m + T^3 + T^2 m)`` from the streaming ``V_b^T V_b`` and
..        ``V_b^T Y_b`` accumulations in :func:`regkit.optimized._accumulate_GB`
..        and the dense solve on the aggregated Gram block.
..    * - ``regression_largeP_iter``
..      - Same ``O(P T^2 + P T m + T^3 + T^2 m)`` cost as
..        :func:`~regkit.optimized.regression_largeP` with batches supplied by an
..        iterator.

.. These estimates ignore constant factors from NumPy and scikit-learn routines but
.. serve as guidelines when sizing experiments or selecting hardware.
