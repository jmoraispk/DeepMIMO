import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'DeepMIMO'
copyright = '2025, Wireless Intelligence Lab'
author = 'João Morais, Umut Demirhan, Ahmed Alkhateeb'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Theme configuration
html_theme = 'furo'
html_static_path = ['_static']

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": "This is the latest version of DeepMIMO documentation.",
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5c85ff",
        "color-brand-content": "#5c85ff",
    },
}

# Other settings
autodoc_member_order = 'bysource'
add_module_names = False 