#%%

# The directives below are for ipykernel to auto reload updated modules
# %reload_ext autoreload
# %autoreload 2

import deepmimo as dm

#%%

# path_to_p2m_folder = r'.\P2Ms\ASU_campus_just_p2m\study_area_asu5'
# path_to_p2m_folder = r'.\P2Ms\simple_street_canyon\study_rays=0.25_res=2m_3ghz'
path_to_p2m_folder = r'.\P2Ms\simple_street_canyon'

scen_name = dm.create_scenario(path_to_p2m_folder,
                               copy_source=False, tx_ids=[1], rx_ids=[2])

# TODO: map p2m dict to universal dict to write to params (almost the same)

#%%
scen_name = 'simple_street_canyon'
params = dm.Parameters(scen_name)#asu_campus')

dataset = dm.generate(params)

#%% READ Setup

# TODO: 
# 1) Put all of this inside insite_converter
# 2) Save params.mat
# 3) Generate a <info> field with all sorts of information

# 4) Generate scenario and channels for validation
# 5) Redo Insite Converter to use the new format (and don't store empty users?)





