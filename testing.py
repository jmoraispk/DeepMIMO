#%%
# The directives below are for ipykernel to auto reload updated modules (e.g. in vscode)
# %reload_ext autoreload
# %autoreload 2

#%%

import deepmimo as dm
# path_to_p2m_folder = r'.\P2Ms\ASU_campus_just_p2m\study_area_asu5'
# path_to_p2m_folder = r'.\P2Ms\simple_street_canyon\study_rays=0.25_res=2m_3ghz'
path_to_p2m_folder = r'.\P2Ms\simple_street_canyon'

scen_name = dm.create_scenario(path_to_p2m_folder,
                               copy_source=False, tx_ids=[1], rx_ids=[2],
                               overwrite=True)

#%%
import deepmimo as dm
scen_name = 'simple_street_canyon'
params = dm.Parameters(scen_name)#asu_campus')
params.get_params_dict()['user_rows'] = [1,2]
dataset = dm.generate(params)

#%% READ Setup

# Conversion: 
# 1) Put all of this inside insite_converter [DONE]
# 2) Switch to reading the new way [DONE]
# 3) Save params.mat [DONE]
# 4) Make params.mat more flexible (don't hardcode stuff and read from dicts)
# 5) Add n_antennas to txrx_set (n_tx_ant and n_rx_ant)

# Generation:
# 6) Generate a <info> field with all sorts of information
# 7) Generate scenario automatically for ASU and street canyon
# 8) Save channels for validation
# 9) Time conversion and Generation speeds to compare with new formats

# 10) Redo Insite Converter to use the new format (and don't store empty users?)

from scipy.io import loadmat

mat = loadmat('./deepmimo_scenarios2/simple_street_canyon/params.mat')
mat.keys()



