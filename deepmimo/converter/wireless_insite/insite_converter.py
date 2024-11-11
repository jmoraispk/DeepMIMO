# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""
import os
import shutil
import numpy as np
import scipy.io

from .. import converter_utils as cu
from ...general_utilities import PrintIfVerbose
from ... import consts as c

from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter

from typing import List, Dict

class InsiteMaterial():
    def __init__(self):
        """
        Diffuse model implemented from [1] + extended with cross-polarization scattering terms
        
        Diffuse scattering models explained in [2], slides 29-31. 
        
        Sources:
            [1] A Diffuse Scattering Model for Urban Propagation Prediction - Vittorio Degli-Esposti 2001
                https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=933491
            [2] https://x.webdo.cc/userfiles/Qiwell/files/Remcom_Wireless%20InSite_5G_final.pdf

        """
        self.name = ''
        self.type = 0
        self.diffuse_scat_model = '' # 'labertian', 'directive', 'directive_w_backscatter'
        self.scat_fact = 0           # 0-1, fraction of incident fields that are scattered
        self.cross_pol_frac = 0      # 0-1, fraction of the scattered field that is cross pol
        self.directive_alpha = 0     # 1-10, defines how broad forward beam is
        self.directive_beta = 0      # 1-10, defines how broad backscatter beam is
        self.directive_lambda = 0    # 0-1, fraction of the scattered power in forward direction (vs back)
        
        self.conductivity = 0        # ..
        self.permittivity = 0        # ..
        self.roughness = 0           # .. affects ...
        self.thickness = 0           # [m]


def insite_rt_converter(rt_folder: str, copy_source: bool = False,
                        tx_ids: List[int] = None, rx_ids: List[int] = None,
                        verbose: bool = True, p2m_folder: str = None):

    # Setup output folder
    if p2m_folder:
        insite_sim_folder = os.path.dirname(p2m_folder)
    else:
        insite_sim_folder = rt_folder
        # if p2m folder is not provided, choose the last folder available
        p2m_folder = [name for name in os.listdir(insite_sim_folder)
                      if os.path.isdir(os.path.join(insite_sim_folder, name))][-1]

    output_folder = os.path.join(insite_sim_folder, 'mat_files') # SCEN_NAME!
    os.makedirs(output_folder, exist_ok=True)

    # DELETE??
    intermediate_folder = os.path.join(insite_sim_folder, 'intermediate_files')
    os.makedirs(intermediate_folder, exist_ok=True)

    # Check if necessary files exist
    verify_sim_folder(insite_sim_folder, verbose)
    
    # Copy ray tracing source files
    if copy_source: copy_rt_source_files(insite_sim_folder, verbose)
    
    files_in_sim_folder = [os.path.join(insite_sim_folder, file) 
                           for file in os.listdir(insite_sim_folder)]

    # Read setup (.setup)
    setup_file = cu.ext_in_list('.setup', files_in_sim_folder)[0]
    setup_dict = read_setup(setup_file, verbose, p2m_folder)
    return 
    # Read TXRX (.txrx)
    txrx_file = cu.ext_in_list('.txrx', files_in_sim_folder)[0]
    avail_tx_idxs, avail_rx_idxs, txrx_dict = read_txrx(txrx_file, verbose)
    
    tx_ids = tx_ids if tx_ids else avail_tx_idxs
    rx_ids = rx_ids if rx_ids else avail_rx_idxs

    # Read Terrain and Buildings Materials (.ter, .city):
    city_files = cu.ext_in_list('.city', files_in_sim_folder)
    ter_files = cu.ext_in_list('.ter', files_in_sim_folder)
    city_materials = read_city(city_files, verbose)
    ter_materials = read_ter(ter_files, verbose)
    # For more info, inspect the raytracing source available in {website}. 
    # (or offer option to dload RT source)

    export_params_dict(output_folder, setup_dict, txrx_dict, 
                       city_materials, ter_materials)

    # P2Ms (.cir, .doa, .dod, .paths[.t{tx_id}_{??}.r{rx_id}.p2m] e.g. .t001_01.r001.p2m)

    # Convert P2M files to mat format
    WIChannelConverter(p2m_folder, intermediate_folder)

    dm = DeepMIMODataFormatter(intermediate_folder, output_folder, 
                               TX_order=tx_ids, RX_order=rx_ids)
    #                          # TODO: read this automatically from P2M
    scen_name = export_scenario(insite_sim_folder, output_folder, overwrite=False)
    return scen_name

    # JTODO 2: write parameters to DeepMIMO metadata (accessible via print and website)

    # JTODO: Generation:
    #   - add option to generate only users with channels
    #   - add read parameters for auto generation

    # TODO: REFACTOR (2 days!)
    #   Eliminate the intermediate files -> ****** no need ******
    #   REUSE only parsers for CIR, PATHS and DoD/A
    #   Save in matrices (faster)
    #   DoA DoD -> AoA AoD
    #   Generate per user index, not row


def verify_sim_folder(sim_folder: str, verbose: bool):
    
    files_in_sim_folder = os.listdir(sim_folder)
    for ext in ['.setup', '.txrx']:
        files_found_with_ext = cu.ext_in_list(ext, files_in_sim_folder)
        if verbose:
            print(f'Found {files_found_with_ext}')
        if len(files_found_with_ext) == 0:
            raise Exception(f'{ext} not found in {sim_folder}')
        elif len(files_found_with_ext) > 1:
            raise Exception(f'Several {ext} found in {sim_folder}')

def copy_rt_source_files(sim_folder: str, verbose: bool = True):
    
    vprint = PrintIfVerbose(verbose) # prints if verbose 
    rt_source_folder = 'raytracing_source'
    files_in_sim_folder = os.listdir(sim_folder)
    vprint('Copying raytracing source files to "rt_source_folder"')
    zip_temp_folder = os.path.join(sim_folder, rt_source_folder)
    os.makedirs(zip_temp_folder)
    for ext in ['.setup', '.txrx', '.ter', '.city']:
        # copy all files with extensions to temp folder
        for file in cu.ext_in_list(ext, files_in_sim_folder):
            curr_file_path = os.path.join(sim_folder, file)
            new_file_path  = os.path.join(zip_temp_folder, file)
            
            vprint(f'Copying {file}')
            shutil.copy(curr_file_path, new_file_path)
    
    vprint('Zipping')
    cu.zip_folder(zip_temp_folder)
    
    vprint(f'Deleting temp folder {os.path.basename(zip_temp_folder)}')
    shutil.rmtree(zip_temp_folder)
    
    vprint('Done')

def read_setup(file: str, verbose: bool, p2m_folder):
    if verbose:
        print(f'Reading setup file: {os.path.basename(file)}')
    
    # Read the study area that matches the p2m folder name
    study_area_name = os.path.basename(p2m_folder)
    
    setup_dict = { # make sure the keys exist in .setup study areas
        'diffuse_scattering': 0, # ony one not in .setup -> has specific mapping
        'diffuse_reflections': 0,
        'diffuse_diffractions': 0,
        'diffuse_transmissions': 0,
        'final_interaction_only': 0,

        'max_reflections': 0,
        'max_transmissions': 0,
        'max_wedge_diffractions': 1,
        'foliage_attenuation_vert': 1, # 0/1 = OFF/ON attenuation
        'foliage_attenuation_hor': 1,  # 0/1 = OFF/ON attenuation
    } # Tip: make them follow the same order as .setup for max performance

    inside_study_area = False
    line = ''
    with open(file, 'r') as fp:
        while True:
            last_line = line # needed because "enabled" refers to the previous line
            line = fp.readline()
            if line == '': # end of file
                raise Exception(f'Reached end of file - ensure {p2m_folder} matches'
                                f' study area name in {file}')
            
            if line != f'begin_<studyarea> {study_area_name}\n' and not inside_study_area:
                continue
            
            inside_study_area = True
            if line == 'end_<studyarea>\n': break

            line_split = line.split(' ')
            key = line_split[0]
            val = line_split[-1][:-1]
            if key == 'enabled':
                if last_line == 'begin_<diffuse_scattering> \n':
                    if val == 'yes':
                        setup_dict['diffuse_scattering'] = 1

            if key in setup_dict.keys():
                if not setup_dict['diffuse_scattering']:
                    if key in ['diffuse_reflections', 'diffuse_reflections',
                               'diffuse_diffractions', 'diffuse_transmissions',
                               'final_interaction_only']:
                        continue # ignore certain keys if DS is off
                if len(val) > 1:
                    val = 1 if val == 'yes' else 0
                else:
                    val = int(val)
                
                setup_dict[key] = val
    
    if verbose:
        print(f'Read the following dict: {setup_dict}')

    return setup_dict

def read_txrx(file: str, verbose: bool):
    print(f'Reading txrx file: {os.path.basename(file)}')
    
    txrx_dict = {}

    # txrx <grid>
    # side1 410.00000
    # side2 320.00000
    # spacing 1.00000
    
    # power 0.00000

    return [], [], txrx_dict

def read_city(files: List[str], verbose: bool):
    return read_components_file(files, verbose)

def read_ter(files: List[str], verbose: bool):
    return read_components_file(files, verbose)

def read_components_file(files: List[str], verbose: bool):
    print(f'Reading materials files: {[os.path.basename(f) for f in files]}')
    material_list = [InsiteMaterial() for i in range(1)]
    return material_list

def export_params_dict(output_folder: str, setup_dict: Dict, txrx_dict: Dict, 
                       city_mat_dict: Dict, ter_mat_dict: Dict):
    data_dict = {
                'version': 2,
                'carrier_freq': 28e9, ############# REAAAAD
                'transmit_power': 0.0, #dB from the scenario ############# REAAAAD
                # Start row - end row - num users - Num users must be larger than the maximum number of dynamic receivers
                'user_grids': np.array([[1, 411, 321]], dtype=float), ############# REAAAAD
                'num_BS': 1, #len(dm.TX_order), ############# REAAAAD
                'dual_polar_available': 0, ############# REAAAAD
                'doppler_available': 0
                #'BS_grids': np.array([[i+1, i+1, 1] for i in range(self.num_BS)]).astype(float)

                # n_paths, n_reflections, type of difusion, ...

                }
        
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)

def export_scenario(sim_folder, output_folder, overwrite=False):
    name = os.path.basename(sim_folder)
    scen_path = c.SCENARIOS_FOLDER + f'/{name}'
    if os.path.exists(scen_path) and not overwrite:
        
        print('Scenario with name "{name}" already exists in {c.SCENARIOS_FOLDER}. Delete? (Y/n)')
        ans = input()
        if ('n' in ans.lower()):
            return
        else:
            shutil.rmtree(scen_path)
    
    shutil.move(output_folder, scen_path)

    return name