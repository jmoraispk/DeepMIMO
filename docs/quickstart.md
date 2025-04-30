# Quickstart

This guide will help you get started with DeepMIMO quickly.

Basic Usage

Load a scenario and generate channels with default settings:

``python
    import deepmimo as dm
    
    # Load a pre-built scenario
    dataset = dm.load('asu_campus_3p5')
    
    # Generate channels with default parameters
    dataset.compute_channels()
``

    # Print channel dimensions: [n_users, n_bs_ant, n_ue_ant, n_freqs]
    print(dataset.channels.shape)  # e.g., (1000, 64, 16, 1) for 1000 users, 
                                   # 8x8 BS array, 4x4 UE array, 1 subcarrier
    # Plot the scene
    dataset.scene.plot()

.. image:: _static/basic_scene.png
   :alt: Basic scene visualization
   :align: center

Customize Loading

DeepMIMO offers flexible ways to load specific parts of a scenario:

``python
    # Load first base station and all users
    dataset1 = dm.load('asu_campus_3p5',
                              tx_sets={1: [0]},      # First BS from set 1
                              rx_sets={2: 'all'})    # All users from set 2
``

    # Load specific users and channel matrices
    dataset2 = dm.load(
        'asu_campus_3p5',
        tx_sets={1: [0]},                           # First BS
        rx_sets={2: [0,1,3]},                       # Users 0, 1, and 3
        matrices=['aoa_az', 'aoa_el', 'power'],     # Specific matrices
        max_paths=10                                # Limit paths per user
    )

Flexible Channel Generation

Customize channel generation with detailed parameters:

``python
    # Create channel parameters
    ch_params = dm.ChannelGenParameters()
``

    # Base station antenna parameters
    ch_params.bs_antenna.shape = [8, 8]          # 8x8 array
    ch_params.bs_antenna.spacing = 0.5           # Half-wavelength spacing
    ch_params.bs_antenna.rotation = [30,40,30]   # [az,el,pol] rotation
    ch_params.bs_antenna.fov = [360, 180]        # Full azimuth coverage

    # User equipment antenna parameters
    ch_params.ue_antenna.shape = [4, 4]          # 4x4 array
    ch_params.ue_antenna.fov = [120, 180]        # 120° azimuth coverage

    # Frequency domain parameters
    ch_params.freq_domain = True
    ch_params.bandwidth = 0.1                    # 100 MHz
    ch_params.num_subcarriers = 64

    # Generate channels
    dataset.compute_channels(ch_params)

Download, Convert and Upload

Work with your own ray-tracing data:

``python
    # Convert ray-tracing data to DeepMIMO format
    scen_name = dm.convert(
        './ray_tracing/my_scenario',     # Ray-tracing folder
        scenario_name='my_scenario',     # Custom name
        vis_scene=True                   # Visualize after conversion
    )
``

    # Upload to DeepMIMO server (requires free API key)
    dm.upload(scen_name, 'your-api-key',
             details=['Custom scenario at 3.5 GHz'])
    
    # The scenario becomes available in the DeepMIMO library is accessible by other users
    dm.download(scen_name)

Visualization Examples

Plot coverage maps and ray paths:

``python
    # Plot power coverage map
    dm.plot_coverage(dataset.rx_pos, dataset.power[:,0], 
                    bs_pos=dataset.tx_pos.T,
                    title="Power Coverage Map (dB)")
``

.. image:: _static/coverage_map.png
   :alt: Coverage map visualization
   :align: center

``python
    # Plot ray paths for user with most paths
    user_idx = np.argmax(dataset.num_paths)
    dm.plot_rays(dataset.rx_pos[user_idx], 
                dataset.tx_pos[0],
                dataset.inter_pos[user_idx], 
                dataset.inter[user_idx],
                proj_3D=True, 
                color_by_type=True)
``

.. image:: _static/ray_paths.png
   :alt: Ray paths visualization
   :align: center

For more advanced usage and features, please refer to the API documentation. 