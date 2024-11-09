#%%

# The directives below are for ipykernel to auto reload updated modules (avoid importlib)
%reload_ext autoreload
%autoreload 2

import deepmimo as dm

#%%

path_to_p2m_folder = r'C:\Users\jmora\Documents\GitHub\DeepMIMO\P2Ms\ASU_campus_just_p2m\study_area_asu5'

dm.create_scenario(path_to_p2m_folder)

#%%
params = dm.Parameters('city_2_chicago')#asu_campus')

dataset = dm.generate(params)

    