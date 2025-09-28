#!/usr/bin/env bash
# 1) Create env with all deps at once
micromamba create -n num_analysis_env -c conda-forge \
    python=3.12 \
    numpy \
    matplotlib \
    scikit-learn \
    tqdm \
    ipykernel \
    sympy \
    pandas \
    pytest \
    sphinx \
    sphinx-autodoc-typehints \
    furo \
    jupyterlab \
    -y

# 2) Hook micromamba into this shell
eval "$(micromamba shell hook -s bash)"

# 3) Activate it
micromamba activate num_analysis_env

# 4) Register the kernel
python -m ipykernel install \
  --user \
  --name num_analysis_env \
  --display-name "Python (num_analysis_env)"

echo "✅ Done! Use 'micromamba activate num_analysis_env' and select the kernel in Jupyter."
