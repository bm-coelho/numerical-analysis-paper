# Numerical Analysis Polynomial Regression Study

This repository documents an exploratory undergraduate research project on a
modest generalisation of multivariate, vector-valued polynomial regression.
Nothing here is meant to be revolutionary or production ready - the goal was to
learn numerical analysis techniques and record the journey. The actual
investigation, derivations, and discussion live in the notebooks; everything
else exists to keep those notebooks reproducible and tidy.

## Key components

- **`notebooks/`** – Jupyter notebooks containing the full research record:
  symbolic derivations, exploratory tests, and discussion of the results. Read
  these first; they are the project.
- **`experiments/`** – Small Python scripts that automate notebook routines
  (generating figures, re-running sweeps, saving tables). They exist purely for
  convenience and mirror the notebook logic.
- **`src/regkit/`** – A lightweight helper library that supports the notebooks
  by wrapping shared utilities such as dataset generators, the symbolic
  baseline, and the optimised solver. Treat it as scaffolding for the study, not
  a production package.
- **`paper/`** – Drafts prepared for conference submissions that summarise the
  notebook findings.
- **`spec/`** – The brief that kicked off the undergraduate project.

## Environment setup

The project targets Python 3.12 or newer. You can prepare the dependencies
manually with `pip` or let the `just` automation recipes handle the heavy
lifting. In either case, the editable install makes the `regkit` package
importable while you work on the source tree and enables the symbolic prototype
provided by `regkit.naive`.

### Manual (Python + pip)

1. Create and activate a virtual environment using your favourite tool (the
   commands below use the built-in `venv`).
2. Install the runtime requirements listed in `config/requirements.txt`.
3. Install the project in editable mode with the optional SymPy extra.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r config/requirements.txt
python -m pip install -e .[sympy]
```

### Automated with the Justfile

The repository ships with a [Just](https://just.systems/man/en/) command file
that can set up several workflow variants:

```bash
just install-venv        # Python's built-in venv + pip (uses config/requirements.txt)
just install-micromamba  # Micromamba/conda-forge environment defined in config/environment.yml
just install-conda       # Conda environment defined in config/environment.yml
just experiments         # Run all experiment scripts sequentially
```

Prerequisites:

- `just` must be installed on your system.
- `micromamba` or `conda` are required for the respective recipes.
- A TeX distribution that provides `latexmk` is needed for LaTeX compilation
  (see below).

Each recipe installs an IPython kernel named `num_analysis_env`, making the
environment available in Jupyter interfaces so you can open the notebooks with
the intended dependency set.

To build the paper once:

```bash
just latex-build
```

Run `just latex-watch` for a live-reloading LaTeX preview (requires `latexmk`
to be available on your `PATH`).

The experimental scripts in `experiments/` are exposed through dedicated
recipes so you can reproduce individual runs or execute the entire suite:

```bash
just experiment-naive        # Symbolic baseline experiment
just experiment-optimized    # Optimized solver evaluation
just experiment-usage        # Usage walkthrough example
just experiment-benchmark    # Performance benchmarks
just experiments             # Run all of the above sequentially
```

The LaTeX recipes automatically expose the files in `paper/CNMAC/` (the custom
class file, bibliography, and figures) by extending the TeX search paths. On
Debian/Ubuntu systems you can install the required toolchain with:

```bash
sudo apt-get install latexmk texlive-latex-extra texlive-lang-portuguese \
    texlive-fonts-extra texlive-science texlive-bibtex-extra biber
```

## Using the regression toolkit

While the main narrative stays in the notebooks, you can poke at the helper
library directly in Python if that is more comfortable:

```python
import numpy as np
from regkit import (
    create_target_dataset,
    regression_naive,
    regression_optimized,
)

X, Y = create_target_dataset()

# Symbolic prototype – returns SymPy objects and a NumPy-friendly callable
coeffs, expr, f_symbolic = regression_naive(X.T, Y.T, degree=2, quiet=True)

# Optimized numerical solver – returns coefficient matrix and a predictor
A, poly, predict = regression_optimized(X.T, Y.T, degree=2)
Y_pred = predict(X.T)
```

Additional helpers cover out-of-core regression (`regression_largeP`,
`regression_largeP_iter`), dataset generation for exploratory plots, and
principal-component / t-SNE visualization (`plot_dataset`). They keep the
 experiments manageable and are not tuned for real-world
deployments.

## Working with the notebooks

The notebooks in `notebooks/` are the primary deliverable. They document the
full computational workflow and contain the commentary that ties the ideas
together:

1. **01 – Naive solution**: symbolic derivations of the regression system.
2. **02 – Optimized solution**: refines the solver using block Gram systems.
3. **03 – RegKit usage**: demonstrates the packaged utilities on new data.
4. **04 – Benchmark regression**: compares performance on larger datasets.

To run them locally, install Jupyter alongside the project and start a notebook
server from the repository root:

```bash
python -m pip install jupyterlab  # or `notebook` if you prefer the classic UI
jupyter lab
```

## Development workflow

This is a learning project, so the workflow is intentionally light:

- Prefer editing the notebooks and re-running the automation scripts to keep
  figures in sync.
- Log new observations directly in the relevant notebook so the research trail
  stays contiguous.
