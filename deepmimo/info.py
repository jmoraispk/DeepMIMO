"""
Module for providing help and information about DeepMIMO dataset parameters and materials.

This module contains utilities to display helpful information about various DeepMIMO
parameters, material properties, and dataset variables through a simple info() interface.
"""

from . import consts as c

# Dictionary of help messages for DeepMIMO dataset parameters
DATASET_HELP_MESSAGES = {
    c.AOA_AZ_PARAM_NAME: 
        'Angle of arrival in azimuth for each path in radians\n'
        '\t[num_rx, num_paths]',
    c.AOA_EL_PARAM_NAME: 
        'Angle of arrival in elevation for each path in radians\n'
        '\t[num_rx, num_paths]',
    c.AOD_AZ_PARAM_NAME:
        'Angle of departure in azimuth for each path in radians\n'
        '\t[num_rx, num_paths]',
    c.AOD_EL_PARAM_NAME:
        'Angle of departure in elevation for each path in radians\n'
        '\t[num_rx, num_paths]',
    c.DELAY_PARAM_NAME:
        'Propagation delay for each path in seconds\n'
        '\t[num_rx, num_paths]',
    c.POWER_PARAM_NAME:
        'Received power in dBm for each path\n'
        '\t[num_rx, num_paths]',
    c.PHASE_PARAM_NAME:
        'Phase of received signal for each path in radians\n'
        '\t[num_rx, num_paths]',
    c.RX_POS_PARAM_NAME:
        'Receiver positions in 3D coordinates in meters\n'
        '\t[num_rx, 3]',
    c.TX_POS_PARAM_NAME:
        'Transmitter positions in 3D coordinates in meters\n'
        '\t[num_tx, 3]',
    c.INTERACTIONS_PARAM_NAME:
        'Type of interaction for each path segment\n'
        '\tCodes: 0: LOS, 1: Reflection, 2: Diffraction, 3: Scattering, 4: Transmission\n'
        '\tCode meaning: 121 -> Tx-R-D-R-Rx\n'
        '\t[num_rx, num_paths, max_interactions]',
    c.INTERACTIONS_POS_PARAM_NAME:
        '3D coordinates in meters of each interaction point along paths\n'
        '\t[num_rx, num_paths, max_interactions, 3]',
    c.NUM_PATHS_PARAM_NAME:
        'Number of paths for each user\n'
        '\t[num_rx]',
    c.NUM_PATHS_FOV_PARAM_NAME:
        'Number of paths within FoV for each user\n'
        '\t[num_rx]',
    c.PWR_LINEAR_PARAM_NAME:
        'Linear power for each path (W)\n'
        '\t[num_rx, num_paths]',
    c.PATHLOSS_PARAM_NAME:
        'Pathloss for each path (dB)\n'
        '\t[num_rx, num_paths]',
    c.DIST_PARAM_NAME:
        'Distance between TX and RX for each path (m)\n'
        '\t[num_rx, num_paths]',
    c.CHANNEL_PARAM_NAME:
        'Channel matrix between TX and RX antennas\n'
        '\t[num_rx, num_rx_ant, num_tx_ant]',
    c.MATERIALS_PARAM_NAME:
        'List of available materials and their electromagnetic properties',
    c.RT_PARAMS_PARAM_NAME:
        'Ray-tracing parameters',
    c.SCENE_PARAM_NAME:
        'Scene parameters',
    c.TXRX_PARAM_NAME:
        'Transmitter/receiver parameters',
}

CHANNEL_HELP_MESSAGES = {
    c.PARAMSET_OFDM_SC_NUM:
        'Number of subcarriers in OFDM',
    c.PARAMSET_OFDM_SC_SAMP:
        'Subcarriers to generate in OFDM',
    c.PARAMSET_OFDM_BANDWIDTH:
        'Bandwidth of OFDM',
    c.PARAMSET_OFDM_LPF:
        'Whether to apply a low-pass filter to the OFDM signal',
}



def info(param_name: str | object | None = None) -> None:
    """Display help information about DeepMIMO dataset parameters and materials.
    
    Args:
        param_name: Name of the parameter to get info about, or object to get help for.
                   If a string, must be one of the valid parameter names or 'materials'.
                   If an object, displays Python's built-in help for that object.
                   If None or 'all', displays information for all parameters.
    
    Returns:
        None
    """
    if not isinstance(param_name, (str, type(None))):
        help(param_name)
    elif param_name is None or param_name == 'all':
        print("DeepMIMO Dataset Parameters:")
        print("=" * 30)
        for param, msg in sorted(DATASET_HELP_MESSAGES.items()):
            print(f"{param}: {msg}")
    elif param_name in ['ch_params', 'channel_params']:
        print("Channel Generation Parameters:")
        print("=" * 30)
        for param, msg in sorted(CHANNEL_HELP_MESSAGES.items()):
            print(f"{param}: {msg}")
    else:
        print(f"Unknown parameter: {param_name}")
        print("Use info() or info('all') to see all available parameters")
    
    return
            