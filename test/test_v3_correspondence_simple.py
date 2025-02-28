
#%%
import numpy as np
import deepmimo as dm
import os
from pprint import pprint

# Import V3 functions directly from their new locations
from deepmimo_old_v3.generator.python.generator import generate_data as generate_old
from deepmimo_old_v3.generator.python.params import Parameters as Parameters_old
from deepmimo_old_v3.converter.wireless_insite.insite_converter_v3 import insite_rt_converter_v3

#%% V3 & V4 Conversion

def convert_scenario(rt_folder: str, use_v3: bool = False) -> str:
    """Convert a Wireless Insite scenario to DeepMIMO format.
    
    Args:
        rt_folder (str): Path to the ray tracing folder
        use_v3 (bool): Whether to use v3 converter. Defaults to False.
        
    Returns:
        str: Name of the converted scenario
    """
    if use_v3:
        # Set parameters based on scenario
        if 'asu_campus' in rt_folder:
            old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu
        else:
            old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61], 'freq': 3.5e9} # simple canyon

    # Convert to unix path
    rt_folder = rt_folder.replace('\\', '/')  # Is this needed?

    # Get scenario name
    scen_name = os.path.basename(rt_folder)

    # Convert using appropriate converter
    if use_v3:
        return insite_rt_converter_v3(rt_folder, None, None, old_params_dict, scen_name)
    else:
        return dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

# Example usage
rt_folder = './P2Ms/asu_campus'

# Convert using v4 converter
scen_name = convert_scenario(rt_folder, use_v3=False)

#%% V4 Generation

scen_name = 'asu_campus'

tx_sets = {1: [0]}
rx_sets = {2: [0,1,2,3,4,5,6,7,8,9,10]}

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 25}
dataset = dm.load(scen_name, **load_params)

# V4 from Dataset

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
# ch_params.bs_antenna.rotation = np.array([30,40,30])
# ch_params.bs_antenna.fov = np.array([360, 180])
# ch_params.ue_antenna.fov = np.array([120, 180])
# ch_params.freq_domain = True
ch_params.num_paths = 5
ch_params.ofdm.subcarriers = 64
# ch_params.ofdm.selected_subcarriers = np.arange(11)
ch_params.ue_antenna.shape = np.array([1,1])

# Other computations
dataset.compute_channels(ch_params)

#%% V3 Generation


# Generate dataset using V3
params = Parameters_old('asu_campus1')
# params = Parameters_old('O1_3p5')
# params['bs_antenna']['rotation'] = np.array([30,40,30])
# params['bs_antenna']['fov'] = np.array([360, 180])
# params['ue_antenna']['fov'] = np.array([120, 180])
# params['freq_domain'] = True
params['user_rows'] = np.arange(1)
params['num_paths'] = 5
params['ofdm']['selected_subcarriers'] = np.arange(11)

dataset2 = generate_old(params)

#%%

# Verification
i = 10
a = dataset['ch'][i]
b = dataset2[0]['user']['channel'][i]
pprint(a.flatten()[-10:])
pprint(b.flatten()[-10:])
pprint(np.max(np.abs(a-b)))