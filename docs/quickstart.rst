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

    # Generate channel data with default parameters
    channel_data = dm.generate_channel()

    # Access channel data components
    H = channel_data.channel  # Complex channel matrices
    tx_positions = channel_data.tx_positions  # Transmitter positions
    rx_positions = channel_data.rx_positions  # Receiver positions

Advanced Features
----------------

DeepMIMO offers several advanced features for customizing your channel generation:

1. Custom Antenna Patterns
-------------------------

.. code-block:: python

    # Define custom antenna parameters
    antenna_params = {
        'array_type': 'UPA',  # Uniform Planar Array
        'n_rows': 8,          # Number of antenna rows
        'n_cols': 8,          # Number of antenna columns
        'spacing': 0.5,       # Element spacing in wavelengths
    }
    
    # Generate channels with custom antenna
    channel_data = dm.generate_channel(antenna_params=antenna_params)

2. Ray Selection
---------------

.. code-block:: python

    # Generate channels with specific ray selection criteria
    channel_data = dm.generate_channel(
        max_paths=10,         # Maximum number of paths to consider
        min_power_db=-100     # Minimum path power threshold in dB
    )

3. Batch Processing
------------------

.. code-block:: python

    # Process data in batches for memory efficiency
    for batch in dm.generate_channel_batches(batch_size=1000):
        # Process each batch
        process_channels(batch.channel)
        process_positions(batch.tx_positions, batch.rx_positions)

For more advanced usage and features, please refer to the API documentation. 