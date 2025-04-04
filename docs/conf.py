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

# -- Theme Selection -----------------------------------------------------
# Set this to 'furo' or 'sphinx_rtd_theme' to switch themes
THEME = 'sphinx_rtd_theme'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Add autosummary settings
autosummary_generate = True

# Add nbsphinx settings
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

# Theme configuration
html_theme = THEME
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# Theme-specific configurations
if THEME == 'furo':
    # Furo theme configuration
    html_sidebars = {
        "**": [
            "sidebar/search.html",
            "sidebar/navigation.html",
        ]
    }
    
    # Furo theme options
    html_theme_options = {
        # Sidebar behavior
        "sidebar_hide_name": False,
        "navigation_with_keys": True,
        
        # Light mode styling
        "light_css_variables": {
            # Brand colors
            "color-brand-primary": "#0EA5E9",
            "color-brand-content": "#0284C7",
            
            # Sidebar styling
            "color-sidebar-background": "#F8F9FC",
            "color-sidebar-item-background--hover": "#F2F4F8",
            "color-sidebar-link": "#374151",
            "color-sidebar-link--hover": "#0EA5E9",
            
            # TOC styling
            "toc-title-font-size": "1rem",
            "toc-spacing-vertical": "1.5rem",
            "toc-font-size": "0.9rem",
            
            # Content styling
            "font-size--normal": "1rem",
            "font-size--small": "0.9rem",
            "content-padding": "3rem",
            "color-foreground-primary": "#111827",
            "color-foreground-secondary": "#374151",
            "color-foreground-muted": "#4B5563",
            
            # Admonition styling
            "color-admonition-background": "#F5F7FA",
        },
        
        # Dark mode styling
        "dark_css_variables": {
            "color-brand-primary": "#4A7DFF",
            "color-brand-content": "#185dbe",
            "color-sidebar-background": "#0F0F0F",
            "color-sidebar-item-background--hover": "#1A1A1A",
            "color-sidebar-link": "#E0E0E0",
            "color-sidebar-link--hover": "#6eafeb",
            "color-foreground-primary": "#FFFFFF",
            "color-foreground-secondary": "#E0E0E0",
            "color-foreground-muted": "#C0C0C0",
            "color-admonition-background": "#141414",
        },
        
        # Other settings
        # "announcement": "This is the latest version of DeepMIMO documentation.",
        "show_toc_level": 4
    }
elif THEME == 'sphinx_rtd_theme':
    # Read the Docs theme options
    html_theme_options = {
        'navigation_depth': 4,
        'titles_only': False,
        'logo_only': False,
        'style_external_links': True,
        'style_nav_header_background': '#2980B9',
        'collapse_navigation': True,
        'sticky_navigation': True,
        'includehidden': True,
        'prev_next_buttons_location': 'both',
    }

# Show/hide settings
html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False

# Other settings
autodoc_member_order = 'bysource'
add_module_names = False

# Add custom CSS to change fonts
html_css_files = [
    'css/custom.css',
]
# autodoc_typehints = 'description'
