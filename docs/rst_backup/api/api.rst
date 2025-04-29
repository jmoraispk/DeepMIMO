Database API
============

This module provides functionality for interfacing with the DeepMIMO database and managing scenarios.

Remote Repository Interface
---------------------------------

.. automodule:: deepmimo.api
   :members:
   :undoc-members:
   :show-inheritance:

   Functions for interacting with the DeepMIMO remote repository:

   * Upload scenarios to the DeepMIMO database
   * Download scenarios from the database
   * Version control and caching
   * Authentication and authorization

Basic Usage
~~~~~~~~~~~~

.. code-block:: python

    import deepmimo as dm

    # Download a scenario from the database (implicitely called in ....)
    dm.download('asu_campus_3p5')

    # Upload your own scenario (requires API key)
    dm.upload('my_scenario', 'your-api-key',
             details=['Custom scenario at 3.5 GHz'])


Authentication
~~~~~~~~~~~~~~~

To upload scenarios, you need an API key. You can obtain one by:

1. Go to "Contribute"
2. Create an account on the DeepMIMO website
3. Generate an API key from the dashboard: https://dev.deepmimo.net/dashboard?tab=uploadKey
