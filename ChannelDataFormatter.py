# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:37:10 2023

@author: Umt
"""

import os
import re
import pandas as pd
import scipy.io
import numpy as np
from tqdm import tqdm

class DeepMIMODataFormatter:
    def __init__(self, intermediate_folder, save_folder, TX_order=None, RX_order=None):
        self.intermediate_folder = intermediate_folder
        
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
            
        self.import_folder()
        
        self.TX_order = TX_order
        self.RX_order = RX_order
        self.sort_TX_RX()
        self.save_data()
        
    def sort_TX_RX(self):
        if self.TX_order and self.RX_order:
            assert len(self.TX_order)==len(self.TX_list), 'Number of provided TX IDs must match the available files from the scenario generation!'
            assert len(self.RX_order)==len(self.RX_list), 'Number of provided RX IDs must match the available files from the scenario generation!'
            assert set(self.TX_order) == set(self.TX_list), 'Provided TX IDs must match the available TXs from the scenario generation!'
            assert set(self.RX_order) == set(self.RX_list), 'Provided RX IDs must match the available RXs from the scenario generation!'
        else:
            self.TX_order = self.TX_list
            self.RX_order = self.RX_list
        
    def save_data(self):
        for tx_cnt, t in enumerate(tqdm(self.TX_order)):
            # BS-BS channel generation
            tx_files = self.df[self.df['TX'] == t]
            bs_bs_channels = []
            bs_bs_info = []
            for r in self.TX_order:
                file = tx_files[tx_files['RX'] == r]
                assert len(file) == 1, 'All TXs must be a transceiver and must have a receive file'
                file_path = os.path.join(self.intermediate_folder, file.iloc[0, 0])
                data = scipy.io.loadmat(file_path)
                bs_bs_channels.append(data['channels'][0])
                bs_bs_info.append(data['rx_locs'])
            bs_bs_channels = np.concatenate(bs_bs_channels)
            bs_bs_info = np.concatenate(bs_bs_info)
            
            
            bs_ue_channels = []
            bs_ue_info = []
            for r in self.RX_order:
                file = tx_files[tx_files['RX'] == r]
                assert len(file) == 1, 'All RXs must must have a single receive file'
                file_path = os.path.join(self.intermediate_folder, file.iloc[0, 0])
                data = scipy.io.loadmat(file_path)
                bs_ue_channels.append(data['channels'][0])
                bs_ue_info.append(data['rx_locs'])
            bs_ue_channels = np.concatenate(bs_ue_channels)
            bs_ue_info = np.concatenate(bs_ue_info)
            
            scipy.io.savemat(os.path.join(self.save_folder, 'BS%i_BS.mat'%(tx_cnt+1)), {'channels': bs_bs_channels, 'rx_locs': bs_bs_info})
            scipy.io.savemat(os.path.join(self.save_folder, 'BS%i_UE.mat'%(tx_cnt+1)), {'channels': bs_ue_channels, 'rx_locs': bs_ue_info})

                
                
    def import_folder(self):
        file_pattern = r'TX(\d+)-(\d+)_RX(\d+)\.mat'
        
        # Get a list of files in the folder
        file_list = os.listdir(self.intermediate_folder)
        
        # Filter files using regular expression pattern
        filtered_files = [filename for filename in file_list if re.match(file_pattern, filename)]
        
        # Extract numbers from the file names and create a list of dictionaries
        data = []
        for filename in filtered_files:
            match = re.match(file_pattern, filename)
            numbers = {
                'TX': int(match.group(1)),
                'TX_sub': int(match.group(2)),
                'RX': int(match.group(3)),
            }
            data.append(numbers)
        
        # Create a Pandas DataFrame
        df = pd.DataFrame(data)
        
        # Add 'File Name' column to the DataFrame
        df['Filename'] = filtered_files
        
        # Reorder columns
        self.df = df[['Filename', 'TX', 'TX_sub', 'RX']]

        assert len(df['TX_sub'].unique())==1, 'Multiple TX antennas is not supported!'

        self.TX_list = sorted(df['TX'].unique())
        self.RX_list = df['RX'].unique()
        self.RX_list = sorted(list(set(self.RX_list)-set(self.TX_list)))
        
    