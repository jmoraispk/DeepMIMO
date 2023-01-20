# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""


from ChannelDataLoader import WIChannelLoader


p2m_folder = r'C:\Users\demir\OneDrive\Desktop\p2m\I3'
mat_folder = p2m_folder + '_mat'

channel_data = WIChannelLoader(p2m_folder)

channel_data.save_channels(mat_folder, scene_idx=1, interacts=False)