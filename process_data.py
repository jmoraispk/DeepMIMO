# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""

from ChannelDataLoader import WIChannelConverter
from ChannelDataFormatter import DeepMIMODataFormatter
from scenario_utils import ScenarioParameters
import os

#%% Parameters to be given for the scenario
p2m_folder = r'C:\Users\Umt\Desktop\Boston5G_3p5_small\Boston5G_3p5'
intermediate_folder = os.path.join(os.path.dirname(p2m_folder), 'intermediate_files')
output_folder = os.path.join(os.path.dirname(p2m_folder), 'mat_files')

#%% Convert P2M files to mat format
channel_data = WIChannelConverter(p2m_folder, intermediate_folder)

#%%
DeepMIMODataFormatter(intermediate_folder, output_folder, max_channels=100000)


 
# #%% Scenario parameters
# sp = ScenarioParameters()
# sp.add_BS(TX_ID=13, TX_subID=1, FOV_az=0, FOV_el=0)
# sp.add_BS(TX_ID=14, TX_subID=1, FOV_az=0, FOV_el=0)

# sp.set_transmission(carrier_freq=60e9, tx_power=0)
# sp.set_user_grids(user_grids=[[1, 551, 121], 
#                               [552, 1159, 86]]) # Start row - end row - num users
# sp.save(output_folder)

