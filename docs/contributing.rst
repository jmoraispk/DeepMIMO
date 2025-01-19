Contributing
====================

We welcome contributions to DeepMIMO! Here's how you can help.

Setting up for Development
---------------------------

1. Fork the repository on GitHub
2. Clone your fork locally::

    git clone git@github.com:your_name_here/DeepMIMO.git

3. Install development dependencies::

    pip install -e ".[dev]"

Building Documentation
------------------------

To build the documentation locally:

1. Install Sphinx and the theme::

    pip install sphinx sphinx-rtd-theme

2. Build the HTML documentation::

    cd docs
    sphinx-build -b html . _build/html

The built documentation will be available in ``docs/_build/html``.

Serving Documentation
------------------------

There are two ways to serve the documentation:

1. Locally using Python's built-in server::

    cd docs/_build/html
    python -m http.server 8000

   You'll see a message like ``Serving HTTP on :: port 8000 (http://[::]:8000/)``.
   You can then access the documentation by opening any of these URLs in your browser:

   * http://localhost:8000
   * http://127.0.0.1:8000

2. Through GitHub Pages:

   a. Push the ``docs/_build/html`` contents to the ``gh-pages`` branch::

        git subtree push --prefix docs/_build/html origin gh-pages

   b. Enable GitHub Pages in your repository settings to serve from the ``gh-pages`` branch
   c. The documentation will be available at ``https://<username>.github.io/DeepMIMO``

Pull Request Guidelines
------------------------

1. Create a branch for your changes
2. Make your changes
3. Add tests if applicable
4. Update documentation if needed
5. Submit a pull request

Code Style
-------------

* Follow PEP 8 guidelines
* Add docstrings to new functions
* Write clear commit messages 