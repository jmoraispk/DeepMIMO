

import numpy as np

def ext_in_list(s, l):
    return np.any([el.endswith(s) for el in l])