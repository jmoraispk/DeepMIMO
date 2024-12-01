
from . import consts as c
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
        c.RX_LOC_PARAM_NAME: '...',
        c.TX_LOC_PARAM_NAME: '...',
        c.INTERACTIONS_PARAM_NAME: '...', 
        c.INTERACTIONS_LOC_PARAM_NAME: '...',
        'materials': '...'
        }
            