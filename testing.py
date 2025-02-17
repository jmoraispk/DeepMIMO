#%% Imports

import time
import numpy as np
import deepmimo as dm

from pprint import pprint

import matplotlib.pyplot as plt

#%% V3 & V4 Conversion

def convert_scenario(rt_folder: str, use_v3: bool = False) -> str:
    """Convert a Wireless Insite scenario to DeepMIMO format.
    
    Args:
        rt_folder (str): Path to the ray tracing folder
        use_v3 (bool): Whether to use v3 converter. Defaults to False.
        
    Returns:
        str: Name of the converted scenario
    """
    # Set parameters based on scenario
    if 'asu_campus' in rt_folder:
        old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu
    else:
        old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61], 'freq': 3.5e9} # simple canyon

    # Convert to unix path
    rt_folder = rt_folder.replace('\\', '/')
    # Make sure it ends with a /
    if 'P2Ms' in rt_folder:
        i = -2
    else:
        rt_folder += '/' if not rt_folder.endswith('/') else ''
        i = -1
    # -2 for Wireless Insite, -1 for other raytracers
    # because for Wireless Insite, we give the P2M INSIDE the rt_folder.. 

    # Get scenario name
    scen_name = rt_folder.split('/')[i] + ('_old' if use_v3 else '')

    # Convert using appropriate converter
    if use_v3:
        return dm.insite_rt_converter_v3(rt_folder, None, None, old_params_dict, scen_name)
    else:
        return dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

# Example usage
# rt_folder = './P2Ms/asu_campus/study_area_asu5'
# rt_folder = './P2Ms/simple_street_canyon_test/study_rays=0.25_res=2m_3ghz'
rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'

# Convert using v4 converter
scen_name = convert_scenario(rt_folder, use_v3=False)

#%% V4 Generation

# Start timing
start_time = time.time()

# scen_name = 'DeepMIMO_folder'
# scen_name = 'simple_street_canyon_test'
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

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets}
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

#%% V3 Generation

# Start timing
start_time = time.time()
params = dm.Parameters_old(scen_name + '_old')
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

import deepmimo as dm
scen_name = dm.convert(r'.\P2Ms\asu_campus\study_area_asu5')
dataset = dm.generate(scen_name)

#%% Visualization check

dm.plot_coverage(dataset['rx_pos'], dataset['aoa_az'][:, 0],
                               bs_pos=dataset['tx_pos'].T)

plt.scatter(dataset['rx_pos'][10,0], dataset['rx_pos'][10,1], c='k', s=20)

#%%

dm.plot_rays(dataset['rx_pos'][10], dataset['tx_pos'][0],
             dataset['inter_pos'][10], dataset['inter'][10],
             proj_3D=True, color_by_type=True)

# Next: determine which buildings interact with each ray. 
#       make a set of those buildings for all the rays in the user.
#       plot the buildings that matter to that user along with the rays.
#       (based on the building bounding boxes)
#       Use the PhysicalObjects class to plot a group of buildings.
#### NEXT: Make a plot of just SOME of the buildings
# buildings_scene = dm.Scene()
# for obj in buildings[:3]:
#     buildings_scene.add_object(obj)
    
# buildings_scene.plot()
#####

#%% An exmaple of mapping params.mat values to web data


# TODO: move this summary to dm.summary(scenario_name)


# DM scenario summary

# Read params.mat and provide TXRX summary, total number of tx & rx, scene size, 
# and other relevant parameters, computed/extracted from the all dicts, not just rt_params

scen_folder = dm.consts.SCENARIOS_FOLDER + '/' + scen_name

mat_file = f'C:/Users/jmora/Documents/GitHub/DeepMIMO/{scen_folder}/params.mat'

params_dict = dm.generator.python.core.load_mat_file_as_dict(mat_file)['params']

# pprint(params_dict)

# Print scenario summary
print("\n" + "="*50)
print(f"DeepMIMO {scen_name} Scenario Summary")
print("="*50)

print("\n[Ray-Tracing Configuration]")
print(f"- Ray-tracer: {params_dict['raytracer']} v{params_dict['raytracer_version']}")
print(f"- Frequency: {params_dict['rt_params']['frequency']/1e9:.1f} GHz")

print("\n[Ray-tracing parameters]")
print(f"- Max reflections: {params_dict['rt_params']['max_reflections']}")
print(f"- Max path depth: {params_dict['rt_params']['path_depth']}")
print(f"- Diffuse scattering: {params_dict['rt_params']['diffuse_scattering']} ({params_dict['rt_params']['diffuse_reflections']} reflections, {params_dict['rt_params']['diffuse_diffractions']} diffractions)")
print(f"- Polarization: {params_dict['rt_params']['polarization']}")
print(f"- Antenna type: {params_dict['rt_params']['antenna_type']}")

print("\n[Scene Composition]")
# Get scene object counts using Scene class method
scene = dm.Scene.from_data(params_dict['scene'], scen_folder)
label_counts = scene.count_objects_by_label()
objects_summary = ', '.join(f'{count} {label}' for label, count in label_counts.items())

# Count faces from scene metadata
normal_faces = sum(len(obj['faces']) for obj in params_dict['scene']['objects'])

print(f"- Total objects: {params_dict['scene']['n_objects']} ({objects_summary})")
print(f"- Vertices: {params_dict['scene']['n_vertices']}")
print(f"- Faces: {normal_faces:,} (decomposed into {params_dict['scene']['n_triangular_faces']:,} triangular faces)")

# Get scene boundaries from scene bounding box
bbox = scene.bounding_box
print("\n[Scene boundaries]")
print(f"- X: {bbox.x_min:.2f}m to {bbox.x_max:.2f}m (width: {bbox.width:.2f}m)")
print(f"- Y: {bbox.y_min:.2f}m to {bbox.y_max:.2f}m (length: {bbox.length:.2f}m)")
print(f"- Z: {bbox.z_min:.2f}m to {bbox.z_max:.2f}m (height: {bbox.height:.2f}m)")
print(f"- Area: {bbox.width * bbox.length:,.2f}m²")

print("\n[Materials]")
print(f"Total materials: {len(params_dict['materials'])}")
for mat_name, mat_props in params_dict['materials'].items():
    print(f"\n{mat_props['name']}:")
    print(f"- Permittivity: {mat_props['permittivity']:.2f}")
    print(f"- Conductivity: {mat_props['conductivity']:.2f} S/m")
    print(f"- Scattering model: {mat_props['scattering_model']}")
    print(f"- Scattering coefficient: {mat_props['scattering_coefficient']:.2f}")
    print(f"- Cross-polarization coefficient: {mat_props['cross_polarization_coefficient']:.2f}")

print("\n[Transmitter/Receiver Configuration]")
for set_name, set_info in params_dict['txrx'].items():
    print(f"\n{set_name} ({set_info['name']}):")
    role = []
    if set_info['is_tx']: role.append("TX")
    if set_info['is_rx']: role.append("RX")
    print(f"- Role: {' & '.join(role)}")
    print(f"- Total points: {set_info['num_points']:,}")
    print(f"- Active points: {set_info['num_active_points']:,}")
    print(f"- Antennas per point: {set_info['num_ant']}")

print(f"\n[Version Information]")
print(f"- DeepMIMO Version: {params_dict['version']}")
print(f"- Dynamic scenes: {params_dict['dynamic_scenario_scenes']}")


# Sum total number of receivers and transmitters
n_rx = sum(set_info['num_active_points'] for set_info in params_dict['txrx'].values()
           if set_info['is_rx'] and not set_info['is_tx'])
n_tx = sum(set_info['num_active_points'] for set_info in params_dict['txrx'].values()
           if set_info['is_tx'] and not set_info['is_rx'])
print(f"Total number of receivers: {n_rx}")
print(f"Total number of transmitters: {n_tx}")



#%%
