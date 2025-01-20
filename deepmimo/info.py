"""
Module for providing help and information about DeepMIMO dataset parameters and materials.

This module contains utilities to display helpful information about various DeepMIMO
parameters, material properties, and dataset variables through a simple info() interface.
"""

from typing import Union

from . import consts as c


MSG_DATASET_VAR_INTERACTIONS_LOC = \
    """Location of interactions, Tx -> interaction_1 -> interction_2 -> .. -> Rx"""


def info(param_name: Union[str, object]) -> None:
    """Display help information about DeepMIMO parameters and materials.
    
    Args:
        param_name: Name of the parameter to get info about, or object to get help for.
                   If a string, must be one of the valid parameter names or 'materials'.
                   If an object, displays Python's built-in help for that object.
    
    Returns:
        None
    """
    if not isinstance(param_name, str):
        help(param_name)
        return

    if param_name == 'materials':
        website = 'deepmimo.net'  # put more accurate website
        print('Materials are unique. If 2 names appear, they must have different, '
              'properties. ')
        print('For more info, inspect the raytracing source available '
              f'in {website}. ')  # TODO: offer option to dload RT source from link
        return

    help_messages = {
        c.CHS_PARAM_NAME: '...',
        c.AOA_AZ_PARAM_NAME: '...',
        c.AOA_EL_PARAM_NAME: '...',
        c.AOD_AZ_PARAM_NAME: '...',
        c.AOD_EL_PARAM_NAME: '...',
        c.TOA_PARAM_NAME: '...',
        c.PWR_PARAM_NAME: '...',
        c.PHASE_PARAM_NAME: '...',
        c.RX_POS_PARAM_NAME: '...',
        c.TX_POS_PARAM_NAME: '...',
        c.INTERACTIONS_PARAM_NAME: '...', 
        c.INTERACTIONS_POS_PARAM_NAME: MSG_DATASET_VAR_INTERACTIONS_LOC,
        'materials': '...'
    }
    
    print(help_messages[param_name])
    
    return
            