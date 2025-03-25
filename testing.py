#%% Imports

import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

from my_api_key import API_KEY as MY_API_KEY

#%% V4 Conversion

# Example usage
rt_folder = './P2Ms/asu_campus'
# rt_folder = './P2Ms/simple_street_canyon_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_export_test2'
# rt_folder = r'C:\Users\jmora\Downloads\DeepMIMOv4-hao-test\all_runs\run_03-08-2025_15H38M57S\NewYork\sionna_export_full'
# rt_folder = r'C:\Users\jmora\Documents\GitHub\AutoRayTracing\all_runs\run_03-09-2025_18H18M51S\NewYork\sionna_export_RX'

scen_name = os.path.basename(rt_folder)
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

#%%

import pickle
def load_pickle(filename: str):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
from pprint import pprint

p  = load_pickle(os.path.join(rt_folder, 'sionna_paths.pkl'))
m  = load_pickle(os.path.join(rt_folder, 'sionna_materials.pkl'))
mi = load_pickle(os.path.join(rt_folder, 'sionna_material_indices.pkl'))
rt = load_pickle(os.path.join(rt_folder, 'sionna_rt_params.pkl'))
v  = load_pickle(os.path.join(rt_folder, 'sionna_vertices.pkl'))
o  = load_pickle(os.path.join(rt_folder, 'sionna_objects.pkl'))

#%%

scen_name = 'asu_campus_3p5'
dataset = dm.load(scen_name)[0]
# dataset = dm.load(model, matrices=relevant_mats)[0]

ch_params = dm.ChannelGenParameters()
ch_params.bs_antenna.shape = np.array([10, 1])

# Reduce dataset size with uniform sampling 
uni_idxs = dataset.get_uniform_idxs([3,3])
dataset_u = dataset.subset(uni_idxs)

# Consider only active users for redundancy reduction
dataset_t = dataset_u.subset(dataset_u.get_active_idxs())  # ...

print(f'dataset.n_ue = {dataset.n_ue}')
print(f'dataset_u.n_ue = {dataset_u.n_ue}')
print(f'dataset_t.n_ue = {dataset_t.n_ue}')


#%%

dataset = dm.load(scen_name)[0]

# Create channel parameters with all options
ch_params = dm.ChannelGenParameters()

# Antenna parameters

# Base station antenna parameters
ch_params.bs_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.bs_antenna.shape = np.array([8, 1])        # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5                   # Element spacing in wavelengths

# User equipment antenna parameters
ch_params.ue_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.ue_antenna.shape = np.array([1, 1])        # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = 0.5                   # Element spacing in wavelengths

# Channel parameters
ch_params.freq_domain = True  # Whether to compute frequency domain channels
ch_params.num_paths = 10      # Number of paths

# OFDM parameters
ch_params.ofdm.bandwidth = 10e6                      # Bandwidth in Hz
ch_params.ofdm.subcarriers = 512                     # Number of subcarriers
ch_params.ofdm.selected_subcarriers = np.arange(1)   # Which subcarriers to generate
ch_params.ofdm.rx_filter = 0                         # Receive Low Pass / ADC Filter

ch_params.ofdm.aaa = 0
dataset.set_channel_params(ch_params)

# Generate channels
# dataset.compute_channels(ch_params)
dataset.channel.shape


#%% [!] Load matrices

import os
import numpy as np

rt_scens = ['asu_campus_3p5', 'city_0_newyork_3p5', ]

ch_params = dm.ChannelGenParameters()
ch_params.ofdm.selected_subcarriers = np.arange(10 * 12)
ch_params.bs_antenna.shape = np.array([10, 1])
ch_params.bs_antenna.rotation = np.array([0, 0, -135])

data_matrices = {}

data_folder = 'stochastic_data'
models = rt_scens
for model in models:
    dataset = dm.load(model)[0]
    data_matrices[model] = dataset.compute_channels(ch_params)


#%%

dataset = dm.load(scen_name)

dm.plot_coverage(dataset.rx_pos, dataset.los, bs_pos=dataset.tx_pos[0].T)


#%% V4 Generation

# scen_name = 'DeepMIMO_folder'
# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
rx_sets = {0: 'all'}

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 25}
dataset = dm.load(scen_name, **load_params)

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
ch_params.num_paths = 5
ch_params.ue_antenna.shape = np.array([1,1])

dataset.compute_channels(ch_params)

print(f'channel parameters = {ch_params}')
print(f'channel.shape = {dataset.channel.shape}')


#%% Test available_txrx_pairs

# Get all available TX-RX pairs
txrx_sets = dm.get_txrx_sets(scen_name)
pairs = dm.get_txrx_pairs(txrx_sets)

#%% PLOT RAYS

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

#%% LOOP for zip (no upload)

# Get all available scenarios using function
scenarios = dm.get_available_scenarios()

metadata_dict = {
    'bbCoords': {
        "minLat": 40.68503298,
        "minLon": -73.84682129, 
        "maxLat": 40.68597435,
        "maxLon": -73.84336302
    },
    'digitalTwin': True,
    'environment': 'indoor',
    "city": "New York"
}
metadata_dict = {}

# Zip the filtered scenarios
for scenario in scenarios:
    scen_path = dm.get_scenario_folder(scenario)
    # dm.zip(scen_path)
    if not ('boston' in scenario):
        continue
    print(f"\nProcessing: {scenario}")
    # continue
    dm.upload(scenario, MY_API_KEY, skip_zip=False, extra_metadata=metadata_dict)

#%% LOOP all scenarios (summary, load, plot)

import deepmimo as dm
import matplotlib.pyplot as plt
base_path = 'F:/deepmimo_loop_ready'
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Load all scenarios
for subfolder in subfolders[:-5]:
    scen_name = os.path.basename(subfolder)
    
    # dm.summary(scen_name)

    # dm.load(scen_name, 'matrices': None)

    try:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={2: []}, matrices=None)
    except Exception as e:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={4: []}, matrices=None)
    _, ax = d.scene.plot(show=False)
    ax.set_title(scen_name + ': ' + ax.get_title())
    plt.show()

