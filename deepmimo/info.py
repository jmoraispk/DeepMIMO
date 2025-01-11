
from . import consts as c


MSG_DATASET_VAR_INTERACTIONS_LOC = \
    """Location of interactions, Tx -> interaction_1 -> interction_2 -> .. -> Rx"""



def info(s):
    if type(s) != str:
        help(s)
    elif s == 'materials':
        website = 'deepmimo.net' # put more accurate website
        print('Materials are unique. If 2 names appear, they must have different, '
              'properties. ')
        print('For more info, inspect the raytracing source available '
              f'in {website}. ') # JTODO: offer option to dload RT source from link
        pass
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
    
    print(help_messages[s])
    
    return 
            