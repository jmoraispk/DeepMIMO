Visualization
=============

This module provides visualization utilities for the DeepMIMO dataset.

.. automodule:: deepmimo.generator.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Core Visualization Functions
------------------------------

Coverage Maps
~~~~~~~~~~~~~~

The module provides powerful tools for visualizing coverage maps and signal characteristics:

.. code-block:: python

    import deepmimo as dm
    import numpy as np

    # Create coverage map
    fig, ax, cbar = dm.plot_coverage(
        rxs=user_positions,          # Shape: (n_users, 3)
        cov_map=received_power,      # Values to visualize
        bs_pos=base_station_pos,     # Optional: BS location
        bs_ori=base_station_ori,     # Optional: BS orientation
        proj_3D=True,               # 3D visualization
        cmap='viridis'              # Color scheme
    )

Ray Path Visualization
~~~~~~~~~~~~~~~~~~~~~~

Visualize ray paths between transmitters and receivers:

.. code-block:: python

    # Plot ray paths for a specific user
    fig, ax = dm.plot_rays(
        rx_loc=user_position,        # Receiver location
        tx_loc=transmitter_pos,      # Transmitter location
        inter_pos=interaction_pos,   # Interaction points
        inter=interaction_types,     # Interaction types
        color_by_type=True,         # Color code by interaction type
        proj_3D=True                # 3D visualization
    )

Data Export
~~~~~~~~~~~~

Export visualization data to external formats:

.. code-block:: python

    # Export to CSV for external tools
    dm.export_xyz_csv(
        data=dataset,               # DeepMIMO dataset
        z_var=values_to_export,     # Values to export
        outfile='coverage.csv',     # Output filename
        google_earth=True           # Convert to geo coordinates
    )

Customization Options
---------------------

The visualization module supports extensive customization:

* Multiple color schemes via matplotlib colormaps
* 2D and 3D projections
* Adjustable figure sizes and DPI
* Custom axis limits and scaling
* Categorical and continuous color bars
* Geographic coordinate transformation
* Export capabilities for external tools

Helper Functions
-----------------

Internal helper functions for specialized visualization tasks:

* Coordinate transformation between Cartesian and geographic systems
* Custom colorbar creation for categorical and continuous data
* Automatic axis scaling and limits 