#%%
import numpy as np
import deepmimo as dm

#%% Basic End-to-End Example

scen_name = dm.convert(r'.\P2Ms\asu_campus\study_area_asu5', vis_scene=True)
dm.generate(scen_name)


#%% Detailed Conversion Example
print("\nDetailed Conversion Example")
print("-" * 50)

# Example with ASU campus scenario
# rt_folder = './P2Ms/asu_campus/study_area_asu5'
# rt_folder = './P2Ms/simple_street_canyon_test/study_rays=0.25_res=2m_3ghz'
rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder/'

# Convert a Wireless Insite scenario to DeepMIMO format
scen_name = dm.convert(rt_folder,
                       overwrite=True,  # Whether to overwrite existing scenario
                       scenario_name=None,  # Custom name for the scenario
                       vis_scene=True)  # Visualize the scene after conversion

#%%

dataset = dm.load_scenario('asu_campus')

#%%

dm.summary('asu_campus')

dm.info()


#%% Loading Example
print("\nLoading Example")
print("-" * 50)

# Example 1: Load specific TX/RX sets using dictionaries
tx_sets_dict = {1: [0]}  # Load first 2 points from set 1 and first 3 from set 2
rx_sets_dict = {2: np.arange(10)}  # Load first point from set 1 and first 10 from set 2

dataset1 = dm.load_scenario(
    scen_name,
    tx_sets=tx_sets_dict,
    rx_sets=rx_sets_dict,
    matrices=['aoa_az', 'aoa_el', 'inter_pos', 'inter'], 
    max_paths=10
)

# Example 2: Load specific TX/RX sets using lists
dataset = dm.load_scenario(scen_name, tx_sets=[1], rx_sets=[2])

# Example 3: Load all TX/RX sets
dataset3 = dm.load_scenario(scen_name, tx_sets='all', rx_sets='all')

#%% Channel Generation Example
print("\nChannel Generation Example")
print("-" * 50)

# Create channel parameters with all options
ch_params = dm.ChannelGenParameters()

# Base station antenna parameters
ch_params.bs_antenna.rotation = np.array([30, 40, 30])  # [az, el, pol] in degrees
ch_params.bs_antenna.fov = np.array([360, 180])         # [az, el] in degrees
ch_params.bs_antenna.shape = np.array([8, 8])           # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5                      # Element spacing in wavelengths

# User equipment antenna parameters
ch_params.ue_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.ue_antenna.fov = np.array([120, 180])      # [az, el] in degrees
ch_params.ue_antenna.shape = np.array([4, 4])        # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = 0.5                   # Element spacing in wavelengths

# # Channel computation parameters
ch_params.freq_domain = True     # Whether to compute frequency domain channels
ch_params.bandwidth = 0.1      # Bandwidth in GHz
ch_params.num_subcarriers = 64   # Number of subcarriers

# Generate channels
dataset.compute_channels(ch_params)
dataset.channel.shape

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

dataset = dm.load_scenario('asu_campus', tx_sets=[1], rx_sets=[2])

# Plot the full scene
scene.plot()

# Plot the scene with triangular faces
scene.plot(mode='tri_faces')

# Plot coverage map
dm.plot_coverage(dataset.rx_pos, dataset.aoa_az[:,0], bs_pos=dataset.tx_pos.T)

#%% Plot all coverage maps
main_keys = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'delay', 'power', 'phase', 
             'los', 'distances', 'num_paths']
for key in main_keys:
    plt_var = dataset[key][:,0] if dataset[key].ndim == 2 else dataset[key]
    dm.plot_coverage(dataset.rx_pos, plt_var, bs_pos=dataset.tx_pos.T, title=key)



#%%
idxs = dataset.get_uniform_idxs([4,4])
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], bs_pos=dataset.tx_pos.T)

#%%

dm.plot_rays(dataset.rx_pos[10], dataset.tx_pos[0],
             dataset.inter_pos[10], dataset.inter[10],
             proj_3D=True, color_by_type=True)
