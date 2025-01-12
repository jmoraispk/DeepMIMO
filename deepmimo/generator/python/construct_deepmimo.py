import numpy as np
from tqdm import tqdm
from .ant_patterns import AntennaPattern
from ... import consts as c

from typing import Dict


def generate_MIMO_channel(dataset,
                          ofdm_params: Dict, 
                          tx_ant_params: Dict, 
                          rx_ant_params: Dict, 
                          freq_domain: int | bool = True,
                          carrier_freq: int | float = 3e9):
    """
    Output is a numpy matrix [n_rx, n_rx_ant, n_tx_ant, n_subcarriers or n_paths]
    """
    bandwidth = ofdm_params[c.PARAMSET_OFDM_BW] * c.PARAMSET_OFDM_BW_MULT
    
    kd_tx = 2 * np.pi * tx_ant_params[c.PARAMSET_ANT_SPACING]
    kd_rx = 2 * np.pi * rx_ant_params[c.PARAMSET_ANT_SPACING]
    Ts = 1 / bandwidth
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    path_gen = OFDM_PathGenerator(ofdm_params, subcarriers)
    antennapattern = AntennaPattern(tx_pattern=tx_ant_params[c.PARAMSET_ANT_RAD_PAT],
                                    rx_pattern=rx_ant_params[c.PARAMSET_ANT_RAD_PAT])

    M_tx = np.prod(tx_ant_params[c.PARAMSET_ANT_SHAPE])
    M_rx = np.prod(rx_ant_params[c.PARAMSET_ANT_SHAPE])
    
    ant_tx_ind = ant_indices(tx_ant_params[c.PARAMSET_ANT_SHAPE])
    ant_rx_ind = ant_indices(rx_ant_params[c.PARAMSET_ANT_SHAPE])
    
    n_ues = dataset[c.RX_POS_PARAM_NAME].shape[0]
    max_paths = dataset[c.LOAD_PARAMS_PARAM_NAME][c.LOAD_PARAM_MAX_PATH]
    last_ch_dim = len(subcarriers) if freq_domain else max_paths
    channel = np.zeros((n_ues, M_rx, M_tx, last_ch_dim), dtype=np.csingle)
    
    for i in tqdm(range(n_ues), desc='Generating channels'):
        if dataset[c.NUM_PATHS_PARAM_NAME][i] == 0:
            continue
        dod_theta, dod_phi = rotate_angles(rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
                                           theta=dataset[c.AOD_EL_PARAM_NAME][i],
                                           phi=dataset[c.AOD_AZ_PARAM_NAME][i])
        
        doa_theta, doa_phi = rotate_angles(rotation=rx_ant_params[c.PARAMSET_ANT_ROTATION][i],
                                           theta=dataset[c.AOA_EL_PARAM_NAME][i],
                                           phi=dataset[c.AOA_AZ_PARAM_NAME][i])
        
        # Compute and apply FoV (field of view) - selects allowed angles
        FoV_tx = apply_FoV(tx_ant_params[c.PARAMSET_ANT_FOV], dod_theta, dod_phi)
        FoV_rx = apply_FoV(rx_ant_params[c.PARAMSET_ANT_FOV], doa_theta, doa_phi)
        FoV = np.logical_and(FoV_tx, FoV_rx)
        dod_theta = dod_theta[FoV]
        dod_phi = dod_phi[FoV]
        doa_theta = doa_theta[FoV]
        doa_phi = doa_phi[FoV]
        
        # TODO: FoV will trim angles and paths.. check this..
        # (for now, assume FoV = off)
        # for key in dataset[i].keys(): 
        #     if key == 'num_paths':
        #         dataset[i][key] = FoV.sum()
        #     else:
        #         dataset[i][key] = dataset[i][key][FoV]
        
        array_response_TX = array_response(ant_ind=ant_tx_ind, 
                                           theta=dod_theta, 
                                           phi=dod_phi, 
                                           kd=kd_tx)
        
        array_response_RX = array_response(ant_ind=ant_rx_ind, 
                                           theta=doa_theta, 
                                           phi=doa_phi,
                                           kd=kd_rx)
        
        power = antennapattern.apply(power=dataset[c.PWR_LINEAR_PARAM_NAME][i], 
                                     doa_theta=doa_theta, 
                                     doa_phi=doa_phi, 
                                     dod_theta=dod_theta, 
                                     dod_phi=dod_phi)
        
        # TODO: MODIFIED POWER (not the original!)
        # dataset[c.PWR_PARAM_NAME][i] = power
        
        if freq_domain: # OFDM
            path_const = path_gen.generate(pwr=power, #dataset[c.PWR_PARAM_NAME][i],
                                           toa=dataset[c.TOA_PARAM_NAME][i],
                                           phs=dataset[c.PHASE_PARAM_NAME][i],
                                           Ts=Ts)
            
            channel[i] = np.sum(array_response_RX[:, None, None, :] * 
                                array_response_TX[None, :, None, :] * 
                                path_const.T[None, None, :, :], axis=3)
                
            if ofdm_params[c.PARAMSET_OFDM_LPF]: # apply LPF
                channel[i] = channel[i] @ path_gen.delay_to_OFDM
        
        else: # TD channel
            channel[i, :, :, :dataset[c.NUM_PATHS_PARAM_NAME][i]] = \
                (array_response_RX[:, None, :] * array_response_TX[None, :, :] *
                 (np.sqrt(power) * np.exp(1j*np.deg2rad(dataset[c.PHASE_PARAM_NAME][i])))[None, None, :])

    return channel


def array_response(ant_ind, theta, phi, kd):        
    gamma = array_response_phase(theta, phi, kd)
    return np.exp(ant_ind@gamma.T)
    
def array_response_phase(theta, phi, kd):
    gamma_x = 1j * kd * np.sin(theta) * np.cos(phi)
    gamma_y = 1j * kd * np.sin(theta) * np.sin(phi)
    gamma_z = 1j * kd * np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T
 
def ant_indices(panel_size):
    gamma_x = np.tile(np.arange(1), panel_size[0]*panel_size[1])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[0]), 1), panel_size[1])
    gamma_z = np.repeat(np.arange(panel_size[1]), panel_size[0])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T

def apply_FoV(FoV, theta, phi):
    theta = np.mod(theta, 2*np.pi)
    phi = np.mod(phi, 2*np.pi)
    FoV = np.deg2rad(FoV)
    path_inclusion_phi = np.logical_or(phi <= 0+FoV[0]/2, phi >= 2*np.pi-FoV[0]/2)
    path_inclusion_theta = np.logical_and(theta <= np.pi/2+FoV[1]/2, theta >= np.pi/2-FoV[1]/2)
    path_inclusion = np.logical_and(path_inclusion_phi, path_inclusion_theta)
    return path_inclusion

def rotate_angles(rotation, theta, phi): # Input all degrees - output radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    if rotation is not None:
        rotation = np.deg2rad(rotation)
    
        sin_alpha = np.sin(phi - rotation[2])
        sin_beta = np.sin(rotation[1])
        sin_gamma = np.sin(rotation[0])
        cos_alpha = np.cos(phi - rotation[2])
        cos_beta = np.cos(rotation[1])
        cos_gamma = np.cos(rotation[0])
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        theta = np.arccos(cos_beta*cos_gamma*cos_theta +
                          sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha))
        phi = np.angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta +
                       1j*(cos_beta*sin_gamma*cos_theta + 
                           sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha)))
    return theta, phi


class OFDM_PathGenerator:
    def __init__(self, params, subcarriers):
        self.OFDM_params = params
        
        self.subcarriers = subcarriers
        self.total_subcarriers = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params['subcarriers'])
        self.delay_to_OFDM = np.exp(-1j * 2 * np.pi / self.total_subcarriers * 
                                    np.outer(self.delay_d, self.subcarriers))
    
    def generate(self, pwr, toa, phs, Ts):
        
        power = pwr.reshape(-1, 1)
        delay_n = toa.reshape(-1, 1) / Ts
        phase = phs.reshape(-1, 1)
    
        # Ignore paths over CP
        paths_over_FFT = (delay_n >= self.OFDM_params['subcarriers'])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params['subcarriers']
        
        path_const = np.sqrt(power / self.total_subcarriers) * np.exp(1j * np.deg2rad(phase))
        if self.OFDM_params[c.PARAMSET_OFDM_LPF]: # Low-pass filter (LPF) convolution
            path_const *= np.sinc(self.delay_d - delay_n)
        else: # Path construction without LPF
            path_const *= np.exp(-1j * (2 * np.pi / self.total_subcarriers) * 
                                 np.outer(delay_n, self.subcarriers))

        return path_const
    