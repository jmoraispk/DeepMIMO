Generator Module
================

The generator module is the core component of DeepMIMO that handles the generation of MIMO channel datasets from ray-tracing data.

Core Components
-------------------------

Dataset Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.dataset
   :members:
   :undoc-members:
   :show-inheritance:

   The Dataset class is the primary container for DeepMIMO data, providing:
   
   * Channel matrices and path information storage
   * Automatic computation of derived quantities
   * Field of view and antenna pattern application
   * Grid-based sampling capabilities

Core Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.core
   :members:
   :undoc-members:
   :show-inheritance:

   Core functionality for:
   
   * Dataset generation and scenario management
   * Ray-tracing data loading and processing
   * Channel computation and parameter validation
   * Multi-user MIMO channel generation

Channel Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.channel
   :members:
   :undoc-members:
   :show-inheritance:

   Channel generation components including:
   
   * Channel parameter management (ChannelGenParameters)
   * OFDM processing and path verification
   * MIMO channel matrix computation

Geometry and Array Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.geometry
   :members:
   :undoc-members:
   :show-inheritance:

   Geometric computations for:
   
   * Antenna array response calculation
   * Angle rotation and transformation
   * Field of view filtering
   * Array indexing utilities

Antenna Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.ant_patterns
   :members:
   :undoc-members:
   :show-inheritance:

   Antenna pattern functionality:
   
   * Standard antenna pattern implementations
   * Pattern application to path gains
   * Custom pattern support

Utilities and Tools
--------------------------------

Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.visualization
   :members:
   :undoc-members:
   :show-inheritance:

   The visualization module provides comprehensive tools for visualizing DeepMIMO datasets:

   Coverage Map Visualization
   ***************************

   Create detailed coverage maps with customizable parameters:

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
   ***********************

   Visualize ray paths and interaction points:

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
   ************

   Export visualization data to external tools:

   .. code-block:: python

       # Export to CSV for external tools
       dm.export_xyz_csv(
           data=dataset,               # DeepMIMO dataset
           z_var=values_to_export,     # Values to export
           outfile='coverage.csv',     # Output filename
           google_earth=True           # Convert to geo coordinates
       )

   Customization Features
   **********************

   The visualization module supports:

   * Multiple color schemes via matplotlib colormaps
   * 2D and 3D projections
   * Adjustable figure sizes and DPI
   * Custom axis limits and scaling
   * Categorical and continuous color bars
   * Geographic coordinate transformation
   * Export capabilities for external tools

Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.utils
   :members:
   :undoc-members:
   :show-inheritance:

   General utility functions for:
   
   * Unit conversion
   * Array manipulation
   * Parameter validation
   * Helper functions

Integrations
---------------------

Sionna Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.sionna_adapter
   :members:
   :undoc-members:
   :show-inheritance:

   Integration with Sionna:
   
   * Channel format conversion
   * Parameter mapping
   * Dataset compatibility
