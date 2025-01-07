#%%
# The directives below are for ipykernel to auto reload updated modules (e.g. in vscode)
# %reload_ext autoreload
# %autoreload 2

#%%

import deepmimo as dm
# path_to_p2m_folder = r'.\P2Ms\ASU_campus_just_p2m\study_area_asu5'
path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon\study_rays=0.25_res=2m_3ghz'

scen_name = dm.create_scenario(path_to_p2m_outputs)

#%%
scen_name = dm.create_scenario(path_to_p2m_outputs,
                               copy_source=True, 
                               tx_ids=[1], rx_ids=[2],
                               overwrite=True, 
                               old=False)

#%%
import deepmimo as dm
scen_name = 'simple_street_canyon'
params = dm.Parameters(scen_name)#asu_campus')
# params.get_params_dict()['user_rows'] = np.arange(91)
dataset = dm.generate(params)

#%% Dream

import deepmimo as dm
scen_name = dm.create_scenario(r'.\P2Ms\simple_street_canyon')
dataset = dm.generate(scen_name)


#%%

path = './P2Ms/simple_street_canyon/study_rays=0.25_res=2m_3ghz/simple_street_canyon_test.pl.t001_01.r002.p2m'

import re
import numpy as np
import time

class TxRxSet():
    """
    Requirements: 
        - each BS must be a transceiver (tx and rx) so we can get their positions
    """
    def __init__(self, name: str = '', index: int = 0, 
                 is_tx: bool = False, is_rx: bool = False):
        self.name = name
        self.index = index
        self.is_tx = is_tx
        self.is_rx = is_rx
        # self.xyz_pos = np.array([]).astype(np.float32)
        # self.rx_enabled = 0 # 0 = not enabled | 1 = enabled
        # self.n_active = sum(self.rx_enabled)
    
    
def read_pl_p2m_file(filename: str):
    """
    Returns xyz, distance, pl from p2m file.
    """
    assert filename.endswith('.p2m') # should be a .p2m file
    assert '.pl.' in filename        # should be the pathloss p2m

    # Initialize empty lists for matrices
    xyz_list = []
    dist_list = []
    path_loss_list = []

    # Define patterns to match header data lines
    data_pattern = re.compile(r"^(\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)$")
    
    # If we want to preallocate matrices, count lines
    # num_lines = sum(1 for _ in open(filename, 'rb'))
    
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    
    for line in lines:
        data_match = data_pattern.match(line)
        if data_match:
            xyz_list.append([float(data_match.group(2)),         # X (m)
                             float(data_match.group(3)),         # Y (m)
                             float(data_match.group(4))])        # Z (m)
            dist_list.append([float(data_match.group(5))])       # distance (m)
            path_loss_list.append([float(data_match.group(6))])  # path loss (dB)

    # Convert lists to numpy arrays
    xyz_matrix = np.array(xyz_list, dtype=np.float32)
    dist_matrix = np.array(dist_list, dtype=np.float32)
    path_loss_matrix = np.array(path_loss_list, dtype=np.float32)

    return xyz_matrix, dist_matrix, path_loss_matrix


def gen_txrx_p2m_file(n_lines=100):

    file_content = \
        """# <Transmitter Set: Tx: 1 BS - Point 1> 
        # <Receiver Set: Rx: 2 ue_grid> 
        # <X(m)> <Y(m)> <Z(m)> <Distance(m)> <PathLoss(dB)>"""
    file_content += ''.join([f"\n{i} -90 -60 1.53079 96.0221 108.661"
                             for i in range(1, int(n_lines) + 1)])
    
    test_file = f'file.pl.n={n_lines}.p2m'
    with open(test_file, 'w') as fp:
        fp.writelines(file_content)
    return test_file
    
test_file = gen_txrx_p2m_file(10000)

# Line count execution
t1 = time.time()
line_count = sum(1 for line in open(test_file, 'rb'))
t2 = time.time()
print('Line count time:', t2 - t1, "seconds")

# Time the execution
t1 = time.time()
xyz_matrix, dist_matrix, path_loss_matrix = read_pl_p2m_file(test_file)
t2 = time.time()

# Print the results
print("Execution Time:", t2 - t1, "seconds")
print("XYZ Matrix Shape:", xyz_matrix.shape)
print("Distance Matrix Shape:", dist_matrix.shape)
print("Path Loss Matrix Shape:", path_loss_matrix.shape)




#%% READ Setup

# Generation:
# 6) Generate a <info> field with all sorts of information
# dataset[tx]['info']
# maybe also dm.info('chs') | dm.info('aoa_az') | ..
dm.info('params num_paths')
dm.info('params')['num_paths']

def info(s):
    if ' ' in s:
        s1, s2 = s.split()
        info(s2)
    
    a = 'num_paths'
    b = 'Influences the sizes of the matrices aoa, aod, etc... and the generated channels'

# 7) Generate scenario automatically for ASU and street canyon
# 8) Save channels for validation
# 9) Time conversion and Generation speeds to compare with new formats

# 10) Redo Insite Converter to use the new format (and don't store empty users?)

