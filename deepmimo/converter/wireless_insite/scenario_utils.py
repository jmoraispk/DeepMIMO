# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:44:37 2023

@author: Umt
"""

import scipy.io as scio
import numpy as np
import os

class ScenarioParameters:
    def __init__(self):
        
        # Basestation Parameters
        self.TX_ID = [] # ID of the TX
        self.TX_subID = [] # Sub-ID of the TX
        self.FOV_az = [] # Azimuth direction center angle in degrees for FoV
        self.FOV_el = [] # Elevation direction center angle in degrees for FoV
        self.TX_order = []
        
        # General Parameters
        self.carrier_freq = None
        self.transmit_power = None
        
        # User Parameters
        self.user_grids = None
        
        # Automatic Extraction
        self.num_BS = None
        
    def add_BS(self, TX_ID, TX_subID, FOV_az, FOV_el, TX_order=None):

        if not TX_order:
            if not self.TX_order:
                TX_order = 1 # ID of the first added basestation is one
            else:
                TX_order = max(self.TX_order)+1 # Increment it by one
        self.TX_order.append(TX_order)
            
        self.TX_ID.append(TX_ID)
        self.TX_subID.append(TX_subID)
        self.FOV_az.append(FOV_az)
        self.FOV_el.append(FOV_el)
        
    def set_transmission(self, carrier_freq, tx_power):
        self.carrier_freq = carrier_freq
        self.transmit_power = tx_power
        
    def set_user_grids(self, user_grids):
        self.user_grids = np.array(user_grids)
        
    def save(self, mat_folder, version=2):
        self.num_BS = len(self.TX_order)
        
        data_dict = {'TX_ID_map': 
                      np.array([self.TX_order, 
                               self.TX_ID, 
                               self.TX_subID, 
                               self.FOV_az, 
                               self.FOV_el]).T.astype(float),
                      'version': version,
                      'carrier_freq': float(self.carrier_freq),
                      'transmit_power': float(self.transmit_power),
                      'user_grids': self.user_grids.astype(float),
                      'num_BS': float(self.num_BS),
                      'BS_grids': np.array([[i+1, i+1, 1] for i in range(self.num_BS)]).astype(float)
                     }
            
        scio.savemat(os.path.join(mat_folder, 'params.mat'), data_dict)