#%% Imports

import time
import numpy as np
import deepmimo as dm

from pprint import pprint

import matplotlib.pyplot as plt

#%% V3 & V4 Conversion

# path_to_p2m_outputs = r'.\P2Ms\asu_campus\study_area_asu5'
path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon_test\study_rays=0.25_res=2m_3ghz'

if 'asu_campus' in path_to_p2m_outputs:
    old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu
else:
    old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61],   'freq': 3.5e9} # simple canyon

old = False
scen_name = path_to_p2m_outputs.split('\\')[2] + ('_old' if old else '')
scen_name = dm.convert(path_to_p2m_outputs,
                       overwrite=True, 
                       old=old,
                       old_params=old_params_dict,
                       scenario_name=scen_name,
                       vis_scene=True)

#%% V4 Generation

# Start timing
start_time = time.time()

scen_name = 'simple_street_canyon_test'
# scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
# rx_sets = {1: [0]}
rx_sets = {2: [0,1,2,3,4,5,6,7,8,9,10]}

# Option 2 - lists with tx/rx set (assumes all points inside the set)
# tx_sets = [1]
# rx_sets = [2]

# Option 3 - string 'all' (generates all points of all tx/rx sets) (default)
# tx_sets = rx_sets = 'all'

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 25}
dataset = dm.load_scenario(scen_name, **load_params)
# pprint(dataset)

# dataset.info() # print available tx-rx information

# V4 from Dataset

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
ch_params.bs_antenna.rotation = np.array([30,40,30])
ch_params.bs_antenna.fov = np.array([360, 180])
ch_params.ue_antenna.fov = np.array([120, 180])
ch_params.freq_domain = True

# Basic computations
p = dataset.power_linear  # Will be computed from dataset.power

dataset.power_linear *= 1000  # JUST TO BE COMPATIBLE WITH V3

# Other computations
_ = dataset._compute_channels(ch_params)

# End timing
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

#%%

# dataset.num_paths  # Triggers _compute_num_paths

# # Rotated angles computation
# # These will be computed when first accessed:
dataset.aoa_az_rot  # Triggers _compute_rotated_angles
# dataset.aoa_el_rot
# dataset.aod_az_rot
# dataset.aod_el_rot

# # FoV filtered angles computation
# # These will be computed when first accessed:
dataset.aoa_az_rot_fov  # Triggers _compute_fov
# dataset.aoa_el_rot_fov
# dataset.aod_az_rot_fov
# dataset.aod_el_rot_fov
# dataset.fov_mask

# # Compute received power with antenna pattern
dataset.power_linear_ant_gain  # Triggers _compute_received_power

dataset.channel  # Triggers _compute_channels with ch_params
dataset.pathloss  # Triggers _compute_pathloss
dataset.distances  # Triggers _compute_distances

# The aliases are already defined in the Dataset class
# So these will work automatically:
dataset.pwr      # Alias for power
dataset.pwr_lin  # Alias for power_linear
dataset.ch       # Alias for channel
dataset.pl       # Alias for pathloss
dataset.rx_loc   # Alias for rx_pos
dataset.tx_loc   # Alias for tx_pos
dataset.dist     # Alias for distances

#%% V3 Generation

# Start timing
start_time = time.time()

# scen_name = 'simple_street_canyon_test_old'
scen_name = 'asu_campus_old'
params = dm.Parameters_old(scen_name)
params['bs_antenna']['rotation'] = np.array([30,40,30])
params['bs_antenna']['fov'] = np.array([360, 180])
params['ue_antenna']['fov'] = np.array([120, 180])
params['freq_domain'] = True

params['user_rows'] = np.arange(1)
dataset2 = dm.generate_old(params)

# End timing
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

# Verification
i = 10
a = dataset['ch'][i]
b = dataset2[0]['user']['channel'][i]
c = a-b
pprint(a.flatten()[-10:])
pprint(b.flatten()[-10:])
pprint(np.max(np.abs(c)))

#%% Demo

# TODO: include this as an end2end example and test, with available asu source files
import deepmimo as dm
scen_name = dm.convert(r'.\P2Ms\asu_campus\study_area_asu5')
dataset = dm.generate(scen_name)

#%% Visualization check

dm.visualization.plot_coverage(dataset['rx_pos'], dataset['aoa_az'][:, 0],
                               bs_pos=dataset['tx_pos'].T)

plt.scatter(dataset['rx_pos'][100,0], dataset['rx_pos'][100,1], c='k', s=20)

#%% Scene and Materials Example

print("\nScene and Materials Example")
print("-" * 50)

# Load a scenario
scen_name = 'simple_street_canyon_test'
dataset = dm.load_scenario(scen_name)

# Get the scene
scene = dataset.scene

# TODO: put this into some info object

# 1. Basic scene information
print("\n1. Scene Overview:")
print(f"Total objects: {len(scene.objects)}")

# Get objects by category
buildings = scene.get_objects('buildings') # TIP: use help(buildings) to see methods
terrain = scene.get_objects('terrain')
vegetation = scene.get_objects('vegetation')

print(f"Buildings: {len(buildings)}")
print(f"Terrain: {len(terrain)}")
print(f"Vegetation: {len(vegetation)}")

# 2. Materials
print("\n2. Materials Usage:")
materials = dataset.materials

# Get materials used by buildings
building_materials = buildings.get_materials()
print(f"Materials used in buildings: {building_materials}")

# Get objects using a specific material
material_idx = list(building_materials)[0]  # Get first material index
objects_with_material = scene.get_objects_by_material(material_idx)
print(f"Objects using material {material_idx}: {len(objects_with_material)}")

# TODO: add a __repr__ for the material or something such that we don't have to refer 
# to it by the index, especially in prints like here

# Print material properties
material = materials[material_idx]
print(f"\nMaterial {material_idx} properties:")

print(f"- Name: {material.name}")
print(f"- Permittivity: {material.permittivity}")
print(f"- Conductivity: {material.conductivity}")

# 3. Object Properties
print("\n3. Object Properties:")
building = buildings[0]  # Get first building
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

# 5. Visualization
print("\n5. Visualization:")
print("Plotting scene with different categories highlighted...")

# Plot the scene
scene.plot()

# Plot just the buildings
buildings_scene = dm.Scene()
for obj in buildings:
    buildings_scene.add_object(obj)
    
buildings_scene.plot()

# TODO: make hull better (to contain the whole building, not just each face)
