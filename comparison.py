# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:16:57 2025

@author: joao

Comparing DeepMIMOv3 with DeepMIMO(v4)
"""

import DeepMIMOv3 as dm_v3
import deepmimo as dm

# Objectives of v4:
#   1- add converter(s)
#   2- add new features to enhance user experience
#   3- improve DeepMIMO format (simpler, more efficient, more user-friendly)
#      (avoids dictionary bottleneck -> "Dicts of Lists >> Lists of Dicts")
#   4- Prepare for open-source: 
#     - improve code cleaniness, organization, readibility and documentation

#%% Creating a scenario (v3)

# Generators not available in dm_v3
# Need to go into personal repository and run the generator function
# We have loaded those function into dm

import time
import deepmimo as dm

path_to_p2m_outputs = r'.\P2Ms\asu_campus\study_area_asu5'
# path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon_test\study_rays=0.25_res=2m_3ghz'

# old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61],   'freq': 3.5e9}   # simple canyon
old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu


t1 = time.time()

# V3
# scen_name = dm.create_scenario(path_to_p2m_outputs, overwrite=True, 
#                                old=True, old_params=old_params_dict)
                               # old has STATIC PARAMS (manually defined)

# V4
scen_name = dm.create_scenario(path_to_p2m_outputs, overwrite=True)

t2 = time.time()
print(f'{t2 - t1:.2f}s')

# Diff
# - speed: 32.75s vs 8.51s
# - space: vs 3.80 MB(*1) [vs 4.31 MB(*2) vs 12.2 MB(*3)]
#   *1 = (same data)
#   *2 = with interaction types but not locations
#   *3 = all
# - space (with interaction types and locations):  vs 
# - less necessary files for conversion: 
#   - .cir, .doa, .dod, .pl, .paths[.t001_{tx_id}.r{rx_id}.p2m], 
#   vs
#    - .pl and .paths (and only .paths if we didn't want positions without paths)
# - no intermediate files during generation
# - ready to upload zipped to DeepMIMO database
# - other:
#   - outputs to deepmimo_scenarios
#   - saves interactions and interactions types (R, D, S, T, F/X)
#   - not yet supported: 
#       - dynamic scenes (easy) 
#       - dual polarization (= ray tracing multi antenna) (done poorly before)
#       - doppler (done in the wrong place before)

#%% Generating a scenario (v4)
# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'

t1 = time.time()

# V3
# params = dm.Parameters_old(scen_name)
# dataset_old = dm.generate_old(params)

# V4
dataset = dm.generate(scen_name) # wrapper to elementary operations load() and compute_chs()

t2 = time.time()
print(f'{t2 - t1:.2f}s')

# Diff
# - speed:
#   - fair speed (with channel gen): ... vs ...
#   - speed: ... vs ... (vs -> generating only active users)
# - simpler and more consistent API
#   - dataset['aoa']
#   vs
#   - dataset[0]['user']['paths'][0]['DoA_phi']
# - gen features
#   - generates 'all' TX-RX sets by default
#   - can generate only selected users (not rows)
#   - downloads scenarios automatically by name (no need for manual download)
#   - can generate only active users (that have paths)
#   - decouples num_paths, pathloss, distances, and channel generation, because
#     they are non essential parts of RT datasets
#     (via smart object that says when something needs computation)
#       - compute_num_paths() [done]
#       - compute_num_interactions() [done]
#       - compute_pl()
#       - compute_pg()
#       - compute_dist()
#       - compute_channels() [done]
#       - get_min_usr_spacing()
#       - ...
# - info features
#   - scenario information
#   - help about any function or parameter (including website ref)
# - plot features
#   - buildings

