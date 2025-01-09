import os
import numpy as np
from tqdm import tqdm
from ... import consts as c
from .utils import dbm2pow
import scipy.io

def read_raytracing():
    pass

# MAKE THIS be the dataset!
def load_variables(path_params, params):
    num_max_paths = params[c.PARAMSET_NUM_PATHS]
    user_data = dict()
    user_data[c.OUT_PATH_PHASE] = path_params[0, :num_max_paths]
    user_data[c.OUT_PATH_TOA] = path_params[1, :num_max_paths]
    user_data[c.OUT_PATH_RX_POW] = dbm2pow(path_params[2, :num_max_paths] + 30)
    user_data[c.OUT_PATH_DOA_PHI] = path_params[3, :num_max_paths]
    user_data[c.OUT_PATH_DOA_THETA] = path_params[4, :num_max_paths]
    user_data[c.OUT_PATH_DOD_PHI] = path_params[5, :num_max_paths]
    user_data[c.OUT_PATH_DOD_THETA] = path_params[6, :num_max_paths]
    

def load_scenario_params(scenario_params_path):
    data = scipy.io.loadmat(scenario_params_path)
    scenario_params = {
                        c.PARAMSET_SCENARIO_PARAMS_CF: data[c.LOAD_FILE_SP_CF].astype(float).item(),
                        c.PARAMSET_SCENARIO_PARAMS_NUM_BS: data[c.LOAD_FILE_SP_NUM_BS].astype(int).item(),
                        c.PARAMSET_SCENARIO_PARAMS_USER_GRIDS: data[c.LOAD_FILE_SP_USER_GRIDS].astype(int),
                        c.PARAMSET_SCENARIO_PARAMS_DOPPLER_EN: data[c.LOAD_FILE_SP_DOPPLER].astype(int).item(),
                        c.PARAMSET_SCENARIO_PARAMS_POLAR_EN: data[c.LOAD_FILE_SP_POLAR].astype(int).item()
                      }
    return scenario_params

