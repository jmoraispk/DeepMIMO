Installation
============

You can install DeepMIMO using pip:

.. code-block:: bash

   pip install deepmimo

Requirements
------------

DeepMIMO requires Python 3.10 or later. The main dependencies are:

* matplotlib (>= 3.8.2)
* numpy (>= 1.19.5)
* scipy (>= 1.6.2)
* tqdm (>= 4.59.0)

For development installation, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/jmoraispk/DeepMIMO.git
   cd DeepMIMO
   pip install -e .

Documentation Dependencies
-------------------------

To build the documentation locally, you'll need:

.. code-block:: bash

   pip install sphinx furo 