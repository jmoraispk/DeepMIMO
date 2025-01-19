Quickstart Guide
===============

This guide will help you get started with DeepMIMO quickly.

Basic Usage
-----------

Here's a simple example of using DeepMIMO:

.. code-block:: python

    import deepmimo as dm

    # Create a new scenario from the Ray Tracing source
    scenario_name = dm.create_scenario('./ray_tracing/O1/')

    # Load a scenario
    scenario = dm.load_scenario(scenario_name)

    # Generate channel data
    channel_data = dm.generate_channel()

Advanced Features
----------------

For more advanced usage and features, please refer to the API documentation. 