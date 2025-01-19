import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'DeepMIMO'
copyright = '2025, Wireless Intelligence Lab'
author = 'Umut Demirhan, Ahmed Alkhateeb, João Morais'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
add_module_names = False 