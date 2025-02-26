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

   Visualization tools for:
   
   * Channel statistics plotting
   * Path visualization
   * Array geometry display
   * Coverage analysis

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

Data Management
------------------

Scenario Download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.generator.downloader
   :members:
   :undoc-members:
   :show-inheritance:

   Scenario management:
   
   * Scenario downloading
   * Version control
   * Cache management 