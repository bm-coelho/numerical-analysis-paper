# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys


# -- Path setup --------------------------------------------------------------

ROOT = os.path.abspath(os.path.join('..', '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RegKit'
copyright = '2025, RegKit Authors'
author = 'RegKit Authors'

version = '0.1.0'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
typehints_defaults = 'comma'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    'sidebar_hide_name': False,
    'light_css_variables': {
        'color-brand-primary': '#2a7ae2',
        'color-brand-content': '#1f4b99',
    },
    'dark_css_variables': {
        'color-brand-primary': '#8ab4f8',
        'color-brand-content': '#aecbfa',
    },
}
