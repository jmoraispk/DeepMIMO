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
    def __init__(self, intermediate_folder, save_folder, max_channels=100000, TX_order=None, RX_order=None, TX_polar=None, RX_polar=None):
        self.intermediate_folder = intermediate_folder
        self.max_channels = max_channels
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
            
        self.import_folder()
        self.tx_locs = scipy.io.loadmat(os.path.join(self.intermediate_folder, 'TX_locs.mat'))
        
        self.TX_order = TX_order
        self.RX_order = RX_order
        self.TX_polar = TX_polar
        self.RX_polar = RX_polar
        self.sort_TX_RX()
        if self.TX_polar:
            self.save_data_polar()
        else:
            self.save_data()
        
    def sort_TX_RX(self):
        if self.TX_order and self.RX_order and not self.TX_polar and not self.RX_polar:
            assert len(self.TX_order) == len(self.TX_list), 'Number of provided TX IDs must match the available files from the scenario generation!'
            assert len(self.RX_order) == len(self.RX_list), 'Number of provided RX IDs must match the available files from the scenario generation!'
            assert set(self.TX_order) == set(self.TX_list), 'Provided TX IDs must match the available TXs from the scenario generation!'
            assert set(self.RX_order) == set(self.RX_list), 'Provided RX IDs must match the available RXs from the scenario generation!'
        elif self.TX_order and self.RX_order and self.TX_polar and self.RX_polar:
            assert len(self.TX_order) + len(self.TX_polar) == len(self.TX_list), 'Number of provided TX IDs must match the available files from the scenario generation!'
            assert len(self.RX_order) + len(self.RX_polar) == len(self.RX_list), 'Number of provided RX IDs must match the available files from the scenario generation!'
            assert set(self.TX_order + self.TX_polar) == set(self.TX_list), 'Provided TX IDs must match the available TXs from the scenario generation!'
            assert set(self.RX_order + self.RX_polar) == set(self.RX_list), 'Provided RX IDs must match the available RXs from the scenario generation!'
        else:
            self.TX_order = self.TX_list
            self.RX_order = self.RX_list
        
    def save_data(self):
        for tx_cnt, t in enumerate(tqdm(self.TX_order)):
            
            # Load tx locations and append it to each file
            tx_loc = self.tx_locs['%s-1'%t].squeeze()
            tx_files = self.df[self.df['TX'] == t]

            bs_idx_files = [item for item in self.TX_order if item in tx_files["RX"].values]
            if bs_idx_files:
                bs_bs_channels, bs_bs_info = self.collect_data(tx_files, bs_idx_files, self.intermediate_folder)
                scipy.io.savemat(os.path.join(self.save_folder, 'BS%i_BS.mat'%(tx_cnt+1)), {'channels': bs_bs_channels, 'rx_locs': bs_bs_info, 'tx_loc': tx_loc})
            
            bs_ue_channels, bs_ue_info = self.collect_data(tx_files, self.RX_order, self.intermediate_folder)
            cur_count = 0
            num_ues = len(bs_ue_channels)
            while cur_count<num_ues:
                next_count = min(cur_count + self.max_channels, num_ues)
                scipy.io.savemat(os.path.join(self.save_folder, 'BS%i_UE_%i-%i.mat'%(tx_cnt+1, cur_count, next_count)), {'channels': bs_ue_channels[cur_count:next_count], 'rx_locs': bs_ue_info[cur_count:next_count], 'tx_loc': tx_loc})
                cur_count = next_count
             
    def collect_data(self, df_tx_files, rx_index, intermediate_folder):
        bs_ue_channels = []
        bs_ue_info = []
        for r in rx_index:
            file = df_tx_files[df_tx_files['RX'] == r]
            assert len(file) == 1, 'All RXs must must have a single receive file'
            file_path = os.path.join(intermediate_folder, file.iloc[0, 0])
            data = scipy.io.loadmat(file_path)
            bs_ue_channels.append(data['channels'][0])
            bs_ue_info.append(data['rx_locs'])
        if bs_ue_channels:
            bs_ue_channels = np.concatenate(bs_ue_channels)
            bs_ue_info = np.concatenate(bs_ue_info)
        else:
            bs_ue_channels = np.array(bs_ue_channels)
            bs_ue_info = np.array(bs_ue_info)
        return bs_ue_channels, bs_ue_info
             
    def save_data_polar(self):
        for tx_cnt, t in enumerate(tqdm(self.TX_order)):
            
            tx_loc = self.tx_locs['%s-1'%t].squeeze()

            # BS-BS channel generation
            tx_files = self.df[self.df['TX'] == t]
            tx_files_polar = self.df[self.df['TX'] == self.TX_polar[tx_cnt]]

            bs_idx_files = [item for item in zip(self.TX_order, self.TX_polar) if item in tx_files["RX"].values]
            if bs_idx_files:

                bs_bs_channels_VV, bs_bs_info = self.collect_data(tx_files, bs_idx_files[0], self.intermediate_folder)
                bs_bs_channels_VH, _ = self.collect_data(tx_files, bs_idx_files[1], self.intermediate_folder)

                bs_bs_channels_HV, _ = self.collect_data(tx_files_polar, bs_idx_files[0], self.intermediate_folder)
                bs_bs_channels_HH, _ = self.collect_data(tx_files_polar, bs_idx_files[1], self.intermediate_folder)
            
                scipy.io.savemat(os.path.join(self.save_folder, 'BS%i_BS.mat'%(tx_cnt+1)), {'channels_VV': bs_bs_channels_VV, 
                                                                                            'channels_VH': bs_bs_channels_VH,
                                                                                            'channels_HV': bs_bs_channels_HV,
                                                                                            'channels_HH': bs_bs_channels_HH,
                                                                                            'rx_locs': bs_bs_info,
                                                                                            'tx_loc': tx_loc
                                                                                            }
                                )
                
            bs_ue_channels_VV, bs_ue_info = self.collect_data(tx_files, self.RX_order, self.intermediate_folder)
            bs_ue_channels_VH, _ = self.collect_data(tx_files, self.RX_polar, self.intermediate_folder)
            
            bs_ue_channels_HV, _ = self.collect_data(tx_files_polar, self.RX_order, self.intermediate_folder)
            bs_ue_channels_HH, _ = self.collect_data(tx_files_polar, self.RX_polar, self.intermediate_folder)
            
            cur_count = 0
            num_ues = len(bs_ue_channels_VV)
            while cur_count<num_ues:
                next_count = min(cur_count + self.max_channels, num_ues)
                scipy.io.savemat(os.path.join(self.save_folder, 'BS%i_UE_%i-%i.mat'%(tx_cnt+1, cur_count, next_count)), 
                                     {'channels_VV': bs_ue_channels_VV[cur_count:next_count], 
                                      'channels_VH': bs_ue_channels_VH[cur_count:next_count],
                                      'channels_HV': bs_ue_channels_HV[cur_count:next_count],
                                      'channels_HH': bs_ue_channels_HH[cur_count:next_count], 
                                      'rx_locs': bs_ue_info[cur_count:next_count],
                                      'tx_loc': tx_loc
                                      }
                                 )
                cur_count = next_count
                
                
    
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
        
    