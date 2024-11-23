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

#%%

from deepmimo.converter.wireless_insite.parser import tokenize_file, parse_document

# tks = tokens("P2Ms/simple_street_canyon/simple_street_canyon_test.txrx")
# tks = tokens("P2Ms/simple_street_canyon/simple_street_canyon_test.setup")
tks = tokenize_file("P2Ms/simple_street_canyon/simple_street_canyon_floor.ter")
#tks = tokenize_file("P2Ms/simple_street_canyon/simple_street_canyon_buildings.city")

document = parse_document(tks)


# JTODO: Generation:
#   - add option to generate only users with channels
#   - add read parameters for auto generation
