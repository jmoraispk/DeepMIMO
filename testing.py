#%% Imports

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
scen_name = dm.create_scenario(path_to_p2m_outputs,
                               overwrite=True, 
                               old=old,
                               old_params=old_params_dict,
                               scenario_name=scen_name,
                               convert_buildings=True, vis_buildings=True)

#%% V4 Generation

# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
rx_sets = {2: [0,1,2,3,4,5,6,7,8,9,10]}

# Option 2 - lists with tx/rx set (assumes all points inside the set)
# tx_sets = [1]
# rx_sets = [2]

# Option 3 - string 'all' (generates all points of all tx/rx sets) (default)
# tx_sets = rx_sets = 'all'

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 5,
               'matrices': 'all'}
dataset = dm.load_scenario(scen_name, **load_params)

# dataset.info() # print available tx-rx information
# from pprint import pprint
# pprint(dataset)

#%% V4 from Dataset
scen_name = 'asu_campus'
dataset = dm.load_scenario(scen_name, **load_params)

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
# ch_params['bs_antenna']['rotation'] = np.array([30,40,30])
ch_params.bs_antenna.FoV = np.array([360, 180])
ch_params.ue_antenna.FoV = np.array([120, 180])
ch_params.OFDM_channels = True

# Basic computations
p = dataset.power_linear  # Will be computed from dataset.power

dataset.power_linear *= 1000  # JUST TO BE COMPATIBLE WITH V3

# TODO: NECESSARY FOR ANGLE ROTATION -> RAISE WARNING IN ANGLE ROTATION COMPUTATION
dataset.ch_params = ch_params

dataset.aoa_az_rot  # Triggers _compute_rotated_angles
# dataset.aoa_az_rot_fov  # Triggers _compute_fov
dataset.power_linear_ant_gain  # Triggers _compute_received_power

# Other computations
_ = dataset._compute_channels(ch_params)

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

# scen_name = 'simple_street_canyon_test_old'
scen_name = 'asu_campus_old'
params = dm.Parameters_old(scen_name)
# params['bs_antenna']['rotation'] = np.array([30,40,30])
params['bs_antenna']['FoV'] = np.array([360, 180])
params['ue_antenna']['FoV'] = np.array([120, 180])
params['OFDM_channels'] = True

params['user_rows'] = np.arange(1)
dataset2 = dm.generate_old(params)

chs2 = dataset2[0]['user']['channel']

# Verification
i = 10
a = dataset['ch'][i]
b = chs2[i]

pprint(a.flatten()[-10:])
pprint(b.flatten()[-10:])

#%% Demo

scen_name = dm.create_scenario(r'.\P2Ms\asu_campus\study_area_asu5')
dataset = dm.generate(scen_name)

#%% Demo part 2
# load_params = {'tx_sets': [1], 'rx_sets': [2], 'max_paths': 1}
load_params = {'tx_sets': [1], 'rx_sets': {2: 'active'}}
# load_params = {'tx_sets': [1], 'rx_sets': {2: [1,2,3]}}
dataset = dm.load_scenario(scen_name, **load_params)
# dataset = dm.load_scenario('city_10_austin')

#%% Visualization check

dm.visualization.plot_coverage(dataset['rx_pos'], dataset['aoa_az'][:, 0],
                               bs_pos=dataset['tx_pos'].T)

plt.scatter(dataset['rx_pos'][100,0], dataset['rx_pos'][100,1], c='k', s=20)
