# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""

from ChannelDataLoader import WIChannelLoader
from scenario_utils import ScenarioParameters
import os

#%% Parameters to be given for the scenario

p2m_folder = r'C:\Users\demir\OneDrive\Desktop\Boston5G_3p5_small\Boston5G_3p5'
output_folder = os.path.join(os.path.dirname(p2m_folder), 'mat_files')

#%% Load p2m files
channel_data = WIChannelLoader(p2m_folder)
channel_data.save_channels(output_folder, scene_idx=None, interacts=False) # Static scenarios

#%% Scenario parameters
sp = ScenarioParameters()
sp.add_BS(TX_ID=13, TX_subID=1, FOV_az=0, FOV_el=0)
sp.add_BS(TX_ID=14, TX_subID=1, FOV_az=0, FOV_el=0)

sp.set_transmission(carrier_freq=60e9, tx_power=0)
sp.set_user_grids(user_grids=[[1, 551, 121], 
                              [552, 1159, 86]]) # Start row - end row - num users
sp.save(output_folder)

