set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Project paths and environment names
env_file := "config/environment.yml"
requirements_file := "config/requirements.txt"
venv_dir := ".venv"
env_name := "num_analysis_env"
ipy_display := "Python (num_analysis_env)"
latex_build_dir := "build"
experiments_dir := "experiments"
experiments_scripts := "run_naive_solution.py run_optimized_solution.py run_regkit_usage.py run_benchmark_regression.py run_wing_surface_regression.py"
docs_source_dir := "docs/source"
docs_build_dir := "docs/_build/html"

# Default target lists the available recipes
default:
    @just --list

# Clean up build artifacts
clean:
    rm -rf {{latex_build_dir}}

# Install the project using the default (micromamba) workflow
install: install-micromamba

# Install the project using micromamba (default)
install-micromamba:
    if micromamba env list | awk '{print $1}' | grep -qx '{{env_name}}'; then \
        micromamba env update -y -n {{env_name}} -f {{env_file}}; \
    else \
        micromamba env create -y -f {{env_file}}; \
    fi
    micromamba run -n {{env_name}} python -m pip install --upgrade pip
    micromamba run -n {{env_name}} python -m pip install -e '.[sympy]'
    micromamba run -n {{env_name}} python -m ipykernel install --user --name {{env_name}} --display-name "{{ipy_display}}"

# Install the project using conda
install-conda:
    if conda env list | awk '{print $1}' | grep -qx '{{env_name}}'; then \
        conda env update --yes -n {{env_name}} -f {{env_file}} --prune; \
    else \
        conda env create --yes -f {{env_file}}; \
    fi
    conda run -n {{env_name}} python -m pip install --upgrade pip
    conda run -n {{env_name}} python -m pip install -e '.[sympy]'
    conda run -n {{env_name}} python -m ipykernel install --user --name {{env_name}} --display-name "{{ipy_display}}"

# Install the project using Python's built-in venv
install-venv:
    python3 -m venv {{venv_dir}}
    source {{venv_dir}}/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r {{requirements_file}}
    python -m pip install -e '.[sympy]'
    python -m ipykernel install --user --name {{env_name}} --display-name "{{ipy_display}}"

paper: experiments paper-cnmac paper-wmma
    cp {{latex_build_dir}}/cnmac/v1_23_Julho_2025.pdf ./cnmac_paper.pdf
    cp {{latex_build_dir}}/wmma/v1_23_Julho_2025.pdf ./wmma_paper.pdf

watch:
    if command -v watchexec >/dev/null 2>&1; then \
        watchexec --watch paper --watch img --exts tex,bib,cls,sty,bst,png,jpg --restart --clear -- just paper; \
    else \
        echo "watchexec is required for 'just watch'. Install it from https://github.com/watchexec/watchexec."; \
        exit 1; \
    fi

# Build the CNMAC LaTeX document once
latex-build:
    mkdir -p {{latex_build_dir}}
    mkdir -p {{latex_build_dir}}/texmf-var/cnmac
    TEXINPUTS="$(pwd)/paper/CNMAC:$(pwd)/paper/common:" \
    BIBINPUTS="$(pwd)/paper/common:" \
    BSTINPUTS="$(pwd)/paper/CNMAC:" \
    TEXMFVAR="$(pwd)/{{latex_build_dir}}/texmf-var/cnmac" \
    latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir="$(pwd)/{{latex_build_dir}}/cnmac" "paper/CNMAC/v1_23_Julho_2025.tex"

# Build the CNMAC LaTeX document in preview mode (interactive)
latex-watch:
    mkdir -p {{latex_build_dir}}
    mkdir -p {{latex_build_dir}}/texmf-var/cnmac
    TEXINPUTS="$(pwd)/paper/CNMAC:$(pwd)/paper/common:" \
    BIBINPUTS="$(pwd)/paper/common:" \
    BSTINPUTS="$(pwd)/paper/CNMAC:" \
    TEXMFVAR="$(pwd)/{{latex_build_dir}}/texmf-var/cnmac" \
    latexmk -cd -pdf -pvc -interaction=nonstopmode -halt-on-error -outdir="$(pwd)/{{latex_build_dir}}/cnmac" "paper/CNMAC/v1_23_Julho_2025.tex"

paper-cnmac: latex-build

paper-wmma:
    mkdir -p {{latex_build_dir}}
    mkdir -p {{latex_build_dir}}/texmf-var/wmma
    TEXINPUTS="$(pwd)/paper/WMMA\ 2025:$(pwd)/paper/common:" \
    BIBINPUTS="$(pwd)/paper/common:" \
    BSTINPUTS="$(pwd)/paper/WMMA\ 2025:" \
    TEXMFVAR="$(pwd)/{{latex_build_dir}}/texmf-var/wmma" \
    latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir="$(pwd)/{{latex_build_dir}}/wmma" "paper/WMMA 2025/v1_23_Julho_2025.tex"

latex-watch-wmma:
    mkdir -p {{latex_build_dir}}
    mkdir -p {{latex_build_dir}}/texmf-var/wmma
    TEXINPUTS="$(pwd)/paper/WMMA\ 2025:$(pwd)/paper/common:" \
    BIBINPUTS="$(pwd)/paper/common:" \
    BSTINPUTS="$(pwd)/paper/WMMA\ 2025:" \
    TEXMFVAR="$(pwd)/{{latex_build_dir}}/texmf-var/wmma" \
    latexmk -cd -pdf -pvc -interaction=nonstopmode -halt-on-error -outdir="$(pwd)/{{latex_build_dir}}/wmma" "paper/WMMA 2025/v1_23_Julho_2025.tex"

# Internal helper to run Python commands inside the configured environment
_run-python +args:
    if command -v micromamba >/dev/null 2>&1; then \
        if ! micromamba env list | awk '{print $1}' | grep -qx '{{env_name}}'; then \
            just install-micromamba; \
        fi; \
        micromamba run -n {{env_name}} python {{args}}; \
    else \
        python {{args}}; \
    fi

# Run the experimental scripts individually or in sequence
experiment-naive:
    just _run-python experiments/run_naive_solution.py

experiment-optimized:
    just _run-python experiments/run_optimized_solution.py

experiment-usage:
    just _run-python experiments/run_regkit_usage.py

experiment-benchmark:
    just _run-python experiments/run_benchmark_regression.py

experiment-wing:
    just _run-python experiments/run_wing_surface_regression.py

experiments:
    just _run-python experiments/ensure_experiment_artifacts.py

# Run the project test suite
test:
    if command -v micromamba >/dev/null 2>&1; then \
        if ! micromamba env list | awk '{print $1}' | grep -qx '{{env_name}}'; then \
            just install-micromamba; \
        fi; \
        micromamba run -n {{env_name}} python -m pip install --upgrade pip; \
        micromamba run -n {{env_name}} python -m pip install pytest; \
        micromamba run -n {{env_name}} pytest; \
    else \
        if ! python -c "import sympy" >/dev/null 2>&1; then \
            python -m pip install --upgrade pip; \
            python -m pip install -e '.[sympy]'; \
        fi; \
        python -m pytest; \
    fi

# Run all experiments sequentially and store their generated artifacts
run-experiments:
    if command -v micromamba >/dev/null 2>&1; then \
        if ! micromamba env list | awk '{print $1}' | grep -qx '{{env_name}}'; then \
            just install-micromamba; \
        fi; \
        micromamba run -n {{env_name}} python {{experiments_dir}}/ensure_experiment_artifacts.py --force; \
    else \
        if ! python -c "import importlib.util, sys; required = ('numpy', 'matplotlib', 'scikit-learn', 'sympy', 'tqdm'); missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]; sys.exit(0 if not missing else 1)"; then \
            python -m pip install --upgrade pip; \
            python -m pip install -r {{requirements_file}}; \
            python -m pip install -e '.[sympy]'; \
        fi; \
        python {{experiments_dir}}/ensure_experiment_artifacts.py --force; \
    fi

# Build the Sphinx documentation
docs-build:
    mkdir -p {{docs_build_dir}}
    if command -v micromamba >/dev/null 2>&1; then \
        if ! micromamba env list | awk '{print $1}' | grep -qx '{{env_name}}'; then \
            just install-micromamba; \
        fi; \
        micromamba run -n {{env_name}} sphinx-build -b html {{docs_source_dir}} {{docs_build_dir}}; \
    else \
        if ! python -c "import importlib.util, sys; required = ('sphinx', 'sphinx_autodoc_typehints', 'furo'); missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]; sys.exit(0 if not missing else 1)"; then \
            python -m pip install --upgrade pip; \
            python -m pip install -r {{requirements_file}}; \
            python -m pip install -e '.[sympy]'; \
        fi; \
        sphinx-build -b html {{docs_source_dir}} {{docs_build_dir}}; \
    fi


docs: docs-build
    # open {{docs_build_dir}}/index.html in default web browser
    if command -v xdg-open >/dev/null 2>&1; then \
        xdg-open {{docs_build_dir}}/index.html; \
    elif command -v open >/dev/null 2>&1; then \
        open {{docs_build_dir}}/index.html; \
    else \
        echo "Documentation built at {{docs_build_dir}}/index.html"; \
    fi
