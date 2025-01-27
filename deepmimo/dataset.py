
"""
Plan: possibly 3 objects

1- Macro dataset (containing the information from several transmitters)
2- Dataset (containing 1tx worth of information)
3- Matrix (like a numpy matrix, but supports a few more operations

Other notes:

* make dataset.aoa_az and dataset['aoa_az'] both possible
* @property can be used to define functions to be called when doing a.smth
* @descriptors can be used to define meta properties
* __help__, __getitem__, __setitem__, __getattr__, can be used to make our dataset object
"""


# 1. 19 - If anyone asks for aoa or aod, ask if they would like to concatenate them. 
# 2.      np.concatenate(dataset['aoa_az'], dataset['aoa_el'] )
# 3. dataset['aoa'] # N x PATHS x 2 (azimuth / elevation)
# 4. dataset['aoa'] = # concat...aoa_el / aoa_az
