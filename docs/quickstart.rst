Quickstart Guide
================

This guide will help you get started with DeepMIMO quickly.

Basic Usage
-----------

Here's a simple end-to-end example:

.. code-block:: python

    import deepmimo as dm
    
    # Load a pre-built scenario
    dataset = dm.load_scenario('asu_campus_3p5')
    
    # Generate channels with default parameters
    dataset.compute_channels()

.. image:: _static/basic_scene.png
   :alt: Basic scene visualization
   :align: center

Loading Data
-----------

DeepMIMO offers flexible ways to load specific parts of a scenario:

.. code-block:: python

    # Option 1: Load specific TX-RX points using dictionaries
    tx_sets = {1: [0]}  # First point from TX set 1
    rx_sets = {2: [0,1,3]}  # Points (users) 0, 1, and 3 from RX set 2
    dataset1 = dm.load_scenario('asu_campus_3p5', 
                              tx_sets=tx_sets, 
                              rx_sets=rx_sets)

    # Option 2: Load entire sets using lists
    dataset2 = dm.load_scenario('asu_campus_3p5', 
                              tx_sets=[1], 
                              rx_sets=[2])

    # Option 3: Load all points from all tx and rx sets (default)
    dataset3 = dm.load_scenario('asu_campus_3p5', 
                              tx_sets='all', 
                              rx_sets='all')

    # Load specific matrices and limit paths
    dataset4 = dm.load_scenario(
        'asu_campus_3p5',
        matrices=['aoa_az', 'aoa_el', 'inter_pos', 'delay'],
        max_paths=10
    )

Channel Generation
----------------

Customize channel generation with detailed parameters:

.. code-block:: python

    # Create channel parameters
    ch_params = dm.ChannelGenParameters()

    # Base station antenna configuration
    ch_params.bs_antenna.rotation = [30, 40, 30]  # [az, el, pol] in degrees
    ch_params.bs_antenna.fov = [360, 180]        # [az, el] in degrees
    ch_params.bs_antenna.shape = [8, 8]          # [horizontal, vertical] elements
    ch_params.bs_antenna.spacing = 0.5           # Element spacing in wavelengths

    # User equipment antenna configuration
    ch_params.ue_antenna.rotation = [0, 0, 0]    # [az, el, pol] in degrees
    ch_params.ue_antenna.fov = [120, 180]        # [az, el] in degrees
    ch_params.ue_antenna.shape = [4, 4]          # [horizontal, vertical] elements
    ch_params.ue_antenna.spacing = 0.5           # Element spacing in wavelengths

    # Frequency domain parameters
    ch_params.freq_domain = True                 # Enable frequency domain channels
    ch_params.bandwidth = 0.1                    # Bandwidth in GHz
    ch_params.num_subcarriers = 64               # Number of subcarriers

    # Generate channels
    dataset.compute_channels(ch_params)

Scene Analysis
-------------

Explore the physical environment and materials:

.. code-block:: python

    # Access scene objects
    scene = dataset.scene
    buildings = scene.get_objects('buildings')
    terrain = scene.get_objects('terrain')
    vegetation = scene.get_objects('vegetation')

    # Analyze materials
    materials = dataset.materials
    building_materials = buildings.get_materials()

    # Filter objects
    material_idx = building_materials[0]
    buildings_with_material = scene.get_objects(
        label='buildings', 
        material=material_idx
    )

    # Get object properties
    building = buildings[0]
    print(f"Building height: {building.height:.2f}m")
    print(f"Building volume: {building.volume:.2f}m³")
    print(f"Building footprint area: {building.footprint_area:.2f}m²")

.. image:: _static/scene_analysis.png
   :alt: Scene analysis visualization
   :align: center

Visualization
------------

DeepMIMO provides rich visualization tools:

.. code-block:: python

    # Plot the scene
    scene.plot()  # Basic view
    scene.plot(mode='tri_faces')  # With triangular faces

    # Plot coverage maps
    dm.plot_coverage(dataset.rx_pos, dataset.aoa_az[:,0], 
                    bs_pos=dataset.tx_pos.T)

    # Plot ray paths for a specific user
    dm.plot_rays(dataset.rx_pos[10], dataset.tx_pos[0],
                dataset.inter_pos[10], dataset.inter[10],
                proj_3D=True, color_by_type=True)

    # Plot various channel properties
    properties = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 
                 'delay', 'power', 'phase', 'los', 
                 'distances', 'num_paths']
    
    for prop in properties:
        values = dataset[prop][:,0] if dataset[prop].ndim == 2 else dataset[prop]
        dm.plot_coverage(dataset.rx_pos, values, 
                        bs_pos=dataset.tx_pos.T, title=prop)

.. image:: _static/coverage_map.png
   :alt: Coverage map visualization
   :align: center

.. image:: _static/ray_paths.png
   :alt: Ray paths visualization
   :align: center

For more advanced usage and features, please refer to the API documentation. 