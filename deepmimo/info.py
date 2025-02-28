"""
Module for providing help and information about DeepMIMO dataset parameters and materials.

This module contains utilities to display helpful information about various DeepMIMO
parameters, material properties, and dataset variables through a simple info() interface.
"""

from . import consts as c

# Dictionary of help messages for fundamental matrices
FUNDAMENTAL_MATRICES_HELP = {
    # Power and phase
    c.POWER_PARAM_NAME:
        'Tap power. Received power in dBW for each path, assuming 0 dBW transmitted power. \n'
        '10*log10(|a|²), where a is the complex channel amplitude\n'
        '\t[num_rx, num_paths]',
    c.PHASE_PARAM_NAME:
        'Tap phase. Phase of received signal for each path in degrees. \n'
        '∠a (angle of a), where a is the complex channel amplitude\n'
        '\t[num_rx, num_paths]',
    # Delay
    c.DELAY_PARAM_NAME:
        'Tap delay. Propagation delay for each path in seconds\n'
        '\t[num_rx, num_paths]',
    # Angles
    c.AOA_AZ_PARAM_NAME: 
        'Angle of arrival (azimuth) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    c.AOA_EL_PARAM_NAME: 
        'Angle of arrival (elevation) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    c.AOD_AZ_PARAM_NAME:
        'Angle of departure (azimuth) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    c.AOD_EL_PARAM_NAME:
        'Angle of departure (elevation) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    # Interactions
    c.INTERACTIONS_PARAM_NAME:
        'Type of interaction for each path segment\n'
        '\tCodes: 0: LOS, 1: Reflection, 2: Diffraction, 3: Scattering, 4: Transmission\n'
        '\tCode meaning: 121 -> Tx-R-D-R-Rx\n'
        '\t[num_rx, num_paths, max_interactions]',
    c.INTERACTIONS_POS_PARAM_NAME:
        '3D coordinates in meters of each interaction point along paths\n'
        '\t[num_rx, num_paths, max_interactions, 3]',
    # Positions
    c.RX_POS_PARAM_NAME:
        'Receiver positions in 3D coordinates in meters\n'
        '\t[num_rx, 3]',
    c.TX_POS_PARAM_NAME:
        'Transmitter positions in 3D coordinates in meters\n'
        '\t[num_tx, 3]',
}

# Dictionary of help messages for computed/derived matrices
COMPUTED_MATRICES_HELP = {
    c.CHANNEL_PARAM_NAME:
        'Channel matrix between TX and RX antennas\n'
        '\t[num_rx, num_rx_ant, num_tx_ant]',
    c.PWR_LINEAR_PARAM_NAME:
        'Linear power for each path (W)\n'
        '\t[num_rx, num_paths]',
    c.PATHLOSS_PARAM_NAME:
        'Pathloss for each path (dB)\n'
        '\t[num_rx, num_paths]',
    c.DIST_PARAM_NAME:
        'Distance between TX and RX for each path (m)\n'
        '\t[num_rx, num_paths]',
    c.NUM_PATHS_PARAM_NAME:
        'Number of paths for each user\n'
        '\t[num_rx]',
    c.NUM_PATHS_FOV_PARAM_NAME:
        'Number of paths within FoV for each user\n'
        '\t[num_rx]',
}

# Dictionary of help messages for configuration/other parameters
CONFIG_HELP = {
    c.SCENE_PARAM_NAME:
        'Scene parameters',
    c.MATERIALS_PARAM_NAME:
        'List of available materials and their electromagnetic properties',
    c.TXRX_PARAM_NAME:
        'Transmitter/receiver parameters',
    c.RT_PARAMS_PARAM_NAME:
        'Ray-tracing parameters',
}

CHANNEL_HELP_MESSAGES = {
    c.PARAMSET_OFDM_BANDWIDTH:
        'Bandwidth of OFDM',
    c.PARAMSET_OFDM_SC_NUM:
        'Number of subcarriers in OFDM',
    c.PARAMSET_OFDM_SC_SAMP:
        'Subcarriers to generate in OFDM',
    c.PARAMSET_OFDM_LPF:
        'Whether to apply a low-pass filter to the OFDM signal',
}

# Combined dictionary for parameter lookups
ALL_PARAMS = {
    **FUNDAMENTAL_MATRICES_HELP,
    **COMPUTED_MATRICES_HELP,
    **CONFIG_HELP,
    **CHANNEL_HELP_MESSAGES
}

def _print_section(title: str, params: dict) -> None:
    """Helper function to print a section of parameter descriptions.
    
    Args:
        title: Section title to display
        params: Dictionary of parameter names and their descriptions
    """
    print(f"\n{title}:")
    print("=" * 30)
    for param, msg in params.items():
        print(f"{param}: {msg}")

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
        return

    if param_name is None or param_name == 'all':
        _print_section("Fundamental Matrices", FUNDAMENTAL_MATRICES_HELP)
        _print_section("Computed/Derived Matrices", COMPUTED_MATRICES_HELP) 
        _print_section("Configuration Parameters", CONFIG_HELP)
    
    elif param_name in ['ch_params', 'channel_params']:
        _print_section("Channel Generation Parameters", CHANNEL_HELP_MESSAGES)
    
    else:
        if param_name in ALL_PARAMS:
            print(f"{param_name}: {ALL_PARAMS[param_name]}")
        else:
            print(f"Unknown parameter: {param_name}")
            print("Use info() or info('all') to see all available parameters")
    
    return
            