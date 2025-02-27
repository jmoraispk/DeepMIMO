"""Configuration file for the Sphinx documentation builder."""

import os
import sys
import tomllib
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'DeepMIMO'
copyright = '2025, Wireless Intelligence Lab'
author = 'João Morais, Umut Demirhan, Ahmed Alkhateeb'

# The full version, including alpha/beta/rc tags
try:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    release = pyproject["project"]["version"]
except FileNotFoundError:
    release = "4.0.0aX" # default version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Theme configuration
html_theme = 'furo'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Furo theme options
# html_theme_options = {
#     "sidebar_hide_name": False,
#     "navigation_with_keys": True,
#     "announcement": "This is the latest version of DeepMIMO documentation.",
#     "light_css_variables": {
#         "color-brand-primary": "#2962ff",
#         "color-brand-content": "#2962ff",
#     },
#     "dark_css_variables": {
#         "color-brand-primary": "#5c85ff",
#         "color-brand-content": "#5c85ff",
#     },
# }

# Other settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'