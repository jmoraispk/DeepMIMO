Converter Module
================

This module provides functionality to convert ray-tracing simulation results into the DeepMIMO dataset format.

Main Interface
-------------------------

.. automodule:: deepmimo.converter
   :members:
   :undoc-members:
   :show-inheritance:

   The main converter interface provides a single entry point ``convert()`` function that automatically detects and handles different ray-tracing formats:

   .. code-block:: python

       import deepmimo as dm
       
       # Convert ray-tracing data to DeepMIMO format
       scen_name = dm.convert(
           './ray_tracing/my_scenario',     # Ray-tracing folder
           scenario_name='my_scenario',      # Custom name
           vis_scene=True                    # Visualize after conversion
       )

Supported Ray Tracers
-------------------------

The converter supports multiple ray-tracing tools through specialized modules:

.. mermaid::

   graph TD
       A[dm.convert] --> B[Detect Format]
       B --> C[Wireless InSite]
       B --> D[Sionna RT]
       B --> E[AODT]
       
       C --> F[insite_converter]
       D --> G[sionna_converter]
       E --> H[aodt_converter]
       
       F --> I[Standardized Format]
       G --> I
       H --> I

Wireless InSite Converter
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.converter.wireless_insite.insite_converter
   :members:
   :undoc-members:
   :show-inheritance:

   Specialized converter for Wireless InSite ray-tracing data. Handles:
   
   * Setup files (.setup)
   * TX/RX configurations (.txrx)
   * Path data (.p2m)
   * Material properties (.city, .ter, .veg)
   * Scene geometry

Sionna RT Converter
~~~~~~~~~~~~~~~~~~~~

.. automodule:: deepmimo.converter.sionna_rt.sionna_converter
   :members:
   :undoc-members:
   :show-inheritance:

   Specialized converter for Sionna ray-tracing data. Handles:
   
   * Path information (angles, delays, powers)
   * TX/RX locations and parameters
   * Scene geometry and materials
   * Interaction types

AODT Converter
~~~~~~~~~~~~~~~

.. automodule:: deepmimo.converter.aodt.aodt_converter
   :members:
   :undoc-members:
   :show-inheritance:

   Specialized converter for AODT ray-tracing data (coming soon).

Implementation Details
-------------------------

Each converter module follows a similar structure:

1. Parameter Parsing
   - Reading configuration files
   - Validating parameters
   - Converting to standardized format

2. TX/RX Configuration
   - Processing antenna configurations
   - Converting coordinate systems
   - Handling array geometries

3. Path Processing
   - Converting path data
   - Processing interactions
   - Computing channel parameters

4. Scene Conversion
   - Converting geometry
   - Processing materials
   - Handling terrain

5. Output Generation
   - Creating standardized files
   - Generating metadata
   - Optional visualization

