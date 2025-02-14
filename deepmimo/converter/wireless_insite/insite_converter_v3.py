"""Legacy v3 Wireless Insite to DeepMIMO Converter.

This module provides the v3 converter functionality for Wireless Insite raytracing 
simulation outputs into DeepMIMO-compatible scenario files.
"""

import os
import shutil
from typing import List, Dict
import numpy as np
import scipy.io

from ... import consts_v3 as c
from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter

def insite_rt_converter_v3(p2m_folder: str, tx_ids: List[int] | None = None, rx_ids: List[int] | None = None, 
                          params_dict: Dict | None = None, scenario_name: str = '') -> str:
    """Convert Wireless Insite files to DeepMIMO format using legacy v3 converter.

    Args:
        p2m_folder (str): Path to folder containing .p2m files.
        tx_ids (List[int]): List of transmitter IDs to process.
        rx_ids (List[int]): List of receiver IDs to process.
        params_dict (Dict): Dictionary containing simulation parameters.
        scenario_name (str): Custom name for output folder. Uses p2m parent folder name if empty.

    Returns:
        str: Path to output folder containing converted files.
    """
    # Loads P2Ms (.cir, .doa, .dod, .paths[.t001_{tx_id}.r{rx_id}.p2m] eg: .t001_01.r001.p2m)
    
    insite_sim_folder = os.path.dirname(p2m_folder)

    intermediate_folder = os.path.join(insite_sim_folder, 'intermediate_files')
    output_folder = os.path.join(insite_sim_folder, 'mat_files') # SCEN_NAME!
    
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Convert P2M files to mat format
    WIChannelConverter(p2m_folder, intermediate_folder)

    DeepMIMODataFormatter(intermediate_folder, output_folder, TX_order=tx_ids, RX_order=rx_ids)
    
    data_dict = {
                c.LOAD_FILE_SP_VERSION: c.VERSION,
                c.LOAD_FILE_SP_CF: params_dict['freq'], 
                c.LOAD_FILE_SP_USER_GRIDS: np.array([params_dict['user_grid']], dtype=float),
                c.LOAD_FILE_SP_NUM_BS: params_dict['num_bs'],
                c.LOAD_FILE_SP_TX_POW: 0,
                c.LOAD_FILE_SP_NUM_RX_ANT: 1,
                c.LOAD_FILE_SP_NUM_TX_ANT: 1,
                c.LOAD_FILE_SP_POLAR: 0,
                c.LOAD_FILE_SP_DOPPLER: 0
                }
    
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)
    
    # export
    scen_name = scenario_name if scenario_name else os.path.basename(os.path.dirname(output_folder))
    scen_path = c.SCENARIOS_FOLDER + f'/{scen_name}'
    if os.path.exists(scen_path):
        shutil.rmtree(scen_path)
    shutil.copytree(output_folder, './' + scen_path)
    
    return output_folder 