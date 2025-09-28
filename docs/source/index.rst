RegKit documentation
====================

RegKit packages the supporting code for an undergraduate exploratory study on
multivariate, vector-valued polynomial regression. The heavy lifting, and the
research narrative, lives in the notebooks shipped with this repository. The
library simply keeps shared utilities in one place so the notebooks stay
manageable and repeatable.

The documentation is organised in a task-oriented fashion:

* :doc:`getting-started` highlights installation and validation steps so that
  you can reproduce the same small-scale experiments used in the notebooks.
* :doc:`user-guide` dives deeper into dataset generation, model fitting, and
  evaluation workflows that feed the figures and tables in the study.
* :doc:`theory` summarises the numerical background and design motivations
  behind the naïve and optimised regression solvers explored during the
  project.
* The :doc:`api/modules` reference is generated directly from the source code
  and lists every public entry point. Treat it as documentation for the helper
  scaffolding rather than a polished public API.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting-started
   user-guide
   theory

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
