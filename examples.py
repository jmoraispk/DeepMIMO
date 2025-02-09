import time
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt
from pprint import pprint

#%% Basic End-to-End Example
print("\nBasic End-to-End Example")
print("-" * 50)

# Convert a Wireless Insite scenario to DeepMIMO format
scen_name = dm.convert(r'.\P2Ms\asu_campus\study_area_asu5',
                      overwrite=True,  # Whether to overwrite existing scenario
                      old=False,       # Whether to use v3 converter
                      scenario_name=None,  # Custom name for the scenario
                      vis_scene=True)  # Visualize the scene after conversion

# Load the scenario with all available options
dataset = dm.load_scenario(
    scen_name,
    tx_sets='all',  # Can be 'all', list of sets, or dict of {set: [indices]}
    rx_sets='all',  # Same as tx_sets
    load_paths=True,  # Whether to load ray paths
    load_materials=True,  # Whether to load material information
)

# Create channel generation parameters with all options
ch_params = dm.ChannelGenParameters()
ch_params.bs_antenna.rotation = np.array([30, 40, 30])  # [az, el, pol] in degrees
ch_params.bs_antenna.fov = np.array([360, 180])  # [az, el] in degrees
ch_params.ue_antenna.fov = np.array([120, 180])  # [az, el] in degrees
ch_params.freq_domain = True  # Whether to compute frequency domain channels

# Generate channels
dataset._compute_channels(ch_params)

#%% Detailed Conversion Example
print("\nDetailed Conversion Example")
print("-" * 50)

# Example with ASU campus scenario
p2m_folder = r'.\P2Ms\asu_campus\study_area_asu5'
old_params_dict = {
    'num_bs': 1,  # Number of base stations
    'user_grid': [1, 411, 321],  # [z_points, x_points, y_points]
    'freq': 3.5e9  # Carrier frequency in Hz
}

# Convert using v3 converter
scen_name_v3 = dm.convert(
    p2m_folder,
    overwrite=True,
    old=True,
    old_params=old_params_dict,
    scenario_name='asu_campus_v3',
    vis_scene=True
)

# Convert using v4 converter
scen_name_v4 = dm.convert(
    p2m_folder,
    overwrite=True,
    old=False,
    scenario_name='asu_campus_v4',
    vis_scene=True
)

#%% Detailed Loading Example
print("\nDetailed Loading Example")
print("-" * 50)

# Example 1: Load specific TX/RX sets using dictionaries
tx_sets_dict = {1: [0, 1], 2: [0, 1, 2]}  # Load first 2 points from set 1 and first 3 from set 2
rx_sets_dict = {1: [0], 2: range(10)}  # Load first point from set 1 and first 10 from set 2

dataset1 = dm.load_scenario(
    scen_name_v4,
    tx_sets=tx_sets_dict,
    rx_sets=rx_sets_dict,
    load_paths=True,
    load_materials=True
)

# Example 2: Load specific TX/RX sets using lists
tx_sets_list = [1, 2]  # Load all points from sets 1 and 2
rx_sets_list = [1]     # Load all points from set 1

dataset2 = dm.load_scenario(
    scen_name_v4,
    tx_sets=tx_sets_list,
    rx_sets=rx_sets_list
)

# Example 3: Load all TX/RX sets
dataset3 = dm.load_scenario(
    scen_name_v4,
    tx_sets='all',
    rx_sets='all'
)

#%% Channel Generation Example
print("\nChannel Generation Example")
print("-" * 50)

# Create channel parameters with all options
ch_params = dm.ChannelGenParameters()

# Base station antenna parameters
ch_params.bs_antenna.rotation = np.array([30, 40, 30])  # [az, el, pol] in degrees
ch_params.bs_antenna.fov = np.array([360, 180])        # [az, el] in degrees
ch_params.bs_antenna.array_size = np.array([8, 8])     # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = np.array([0.5, 0.5])    # Element spacing in wavelengths

# User equipment antenna parameters
ch_params.ue_antenna.rotation = np.array([0, 0, 0])    # [az, el, pol] in degrees
ch_params.ue_antenna.fov = np.array([120, 180])        # [az, el] in degrees
ch_params.ue_antenna.array_size = np.array([4, 4])     # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = np.array([0.5, 0.5])    # Element spacing in wavelengths

# Channel computation parameters
ch_params.freq_domain = True     # Whether to compute frequency domain channels
ch_params.bandwidth = 100e6      # Bandwidth in Hz
ch_params.num_subcarriers = 64   # Number of subcarriers

# Generate channels
dataset._compute_channels(ch_params)

#%% Scene and Materials Example
print("\nScene and Materials Example")
print("-" * 50)

# Load a scenario
dataset = dm.load_scenario('simple_street_canyon_test')
scene = dataset.scene

# 1. Basic scene information
print("\n1. Scene Overview:")
print(f"Total objects: {len(scene.objects)}")

# Get objects by category
buildings = scene.get_objects('buildings')
terrain = scene.get_objects('terrain')
vegetation = scene.get_objects('vegetation')

print(f"Buildings: {len(buildings)}")
print(f"Terrain: {len(terrain)}")
print(f"Vegetation: {len(vegetation)}")

# 2. Materials and Filtering
print("\n2. Materials and Filtering:")
materials = dataset.materials

# Get materials used by buildings
building_materials = buildings.get_materials()
print(f"Materials used in buildings: {building_materials}")

# Different ways to filter objects
print("\nFiltering examples:")

# Filter by label only
buildings = scene.get_objects(label='buildings')
print(f"Buildings: {len(buildings)}")

# Filter by material only
material_idx = building_materials[0]
objects_with_material = scene.get_objects(material=material_idx)
print(f"Objects with material {material_idx}: {len(objects_with_material)}")

# Filter by both label and material
buildings_with_material = scene.get_objects(label='buildings', material=material_idx)
print(f"Buildings with material {material_idx}: {len(buildings_with_material)}")

# Print material properties
material = materials[material_idx]
print(f"\nMaterial {material_idx} properties:")
print(f"- Name: {material.name}")
print(f"- Permittivity: {material.permittivity}")
print(f"- Conductivity: {material.conductivity}")

# 3. Object Properties
print("\n3. Object Properties:")
building = buildings[0]
print(f"Building faces: {len(building.faces)}")
print(f"Building height: {building.height:.2f}m")
print(f"Building volume: {building.volume:.2f}m³")
print(f"Building footprint area: {building.footprint_area:.2f}m²")

# 4. Bounding Boxes
print("\n4. Bounding Boxes:")
bb = buildings.bounding_box
print(f"Buildings bounding box:")
print(f"- Width (X): {bb.width:.2f}m")
print(f"- Length (Y): {bb.length:.2f}m")
print(f"- Height (Z): {bb.height:.2f}m")

#%% Visualization Examples
print("\nVisualization Examples")
print("-" * 50)

# Plot the full scene
scene.plot()

# Plot coverage map
dm.visualization.plot_coverage(
    dataset['rx_pos'],
    dataset['aoa_az'][:, 0],
    bs_pos=dataset['tx_pos'].T
)

# Add specific point highlight
plt.scatter(dataset['rx_pos'][100,0], dataset['rx_pos'][100,1], c='k', s=20) 