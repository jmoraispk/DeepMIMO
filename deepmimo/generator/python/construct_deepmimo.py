import numpy as np
from tqdm import tqdm
from .ant_patterns import AntennaPattern
from ... import consts as c

# Generates common parameters first. The output is a numpy matrix.
def generate_MIMO_channel(raydata, params, tx_ant_params, rx_ant_params):
    
    bandwidth = params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_BW] * c.PARAMSET_OFDM_BW_MULT
    
    kd_tx = 2 * np.pi * tx_ant_params[c.PARAMSET_ANT_SPACING]
    kd_rx = 2 * np.pi * rx_ant_params[c.PARAMSET_ANT_SPACING]
    Ts = 1 / bandwidth
    subcarriers = params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_SAMP]
    path_gen = OFDM_PathGenerator(params, subcarriers)
    antennapattern = AntennaPattern(tx_pattern=tx_ant_params[c.PARAMSET_ANT_RAD_PAT],
                                    rx_pattern=rx_ant_params[c.PARAMSET_ANT_RAD_PAT])

    M_tx = np.prod(tx_ant_params[c.PARAMSET_ANT_SHAPE])
    M_rx = np.prod(rx_ant_params[c.PARAMSET_ANT_SHAPE])
    
    ant_tx_ind = ant_indices(tx_ant_params[c.PARAMSET_ANT_SHAPE])
    ant_rx_ind = ant_indices(rx_ant_params[c.PARAMSET_ANT_SHAPE])
    
    n_ues = len(ray_data)
    last_ch_dim = len(subcarriers) if params[c.PARAMSET_FDTD] else params[c.PARAMSET_NUM_PATHS]
    channel = np.zeros((n_ues, M_rx, M_tx, last_ch_dim), dtype=np.csingle)
    
    for i in tqdm(range(n_ues), desc='Generating channels'):
        
        dod_theta, dod_phi = rotate_angles(rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
                                           theta=raydata[i][c.OUT_PATH_DOD_THETA],
                                           phi=raydata[i][c.OUT_PATH_DOD_PHI])
        
        doa_theta, doa_phi = rotate_angles(rotation=rx_ant_params[c.PARAMSET_ANT_ROTATION][i],
                                           theta=raydata[i][c.OUT_PATH_DOA_THETA],
                                           phi=raydata[i][c.OUT_PATH_DOA_PHI])
        
        # Compute and apply FoV (field of view) - selects allowed angles
        FoV_tx = apply_FoV(tx_ant_params[c.PARAMSET_ANT_FOV], dod_theta, dod_phi)
        FoV_rx = apply_FoV(rx_ant_params[c.PARAMSET_ANT_FOV], doa_theta, doa_phi)
        FoV = np.logical_and(FoV_tx, FoV_rx)
        dod_theta = dod_theta[FoV]
        dod_phi = dod_phi[FoV]
        doa_theta = doa_theta[FoV]
        doa_phi = doa_phi[FoV]
        
        for key in raydata[i].keys():
            if key == 'num_paths':
                raydata[i][key] = FoV.sum()
            else:
                raydata[i][key] = raydata[i][key][FoV]
        
        array_response_TX = array_response(ant_ind=ant_tx_ind, 
                                           theta=dod_theta, 
                                           phi=dod_phi, 
                                           kd=kd_tx)
        
        array_response_RX = array_response(ant_ind=ant_rx_ind, 
                                           theta=doa_theta, 
                                           phi=doa_phi,
                                           kd=kd_rx)
        
        power = antennapattern.apply(power=raydata[i][c.OUT_PATH_RX_POW], 
                                     doa_theta=doa_theta, 
                                     doa_phi=doa_phi, 
                                     dod_theta=dod_theta, 
                                     dod_phi=dod_phi)
        
        raydata[i][c.OUT_PATH_RX_POW] = power
        
        if  params[c.PARAMSET_FDTD]: # OFDM
            path_const = path_gen.generate(raydata[i], Ts)
            
            # The next step is to be defined
            channel[i] = np.sum(array_response_RX[:, None, None, :] * 
                                array_response_TX[None, :, None, :] * 
                                path_const.T[None, None, :, :], axis=3)
            
            if params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_LPF] == 0:
                channel[i] = channel[i] @ path_gen.delay_to_OFDM
        
        else: # TD channel
            channel[i, :, :, :raydata[i][c.OUT_PATH_NUM]] = \
                (array_response_RX[:, None, :] * array_response_TX[None, :, :] *
                 (np.sqrt(power) * np.exp(1j*np.deg2rad(raydata[i][c.OUT_PATH_PHASE])))[None, None, :])

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
        self.params = params
        self.OFDM_params = params[c.PARAMSET_OFDM]
        
        self.subcarriers = subcarriers
        self.total_subcarriers = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params['subcarriers'])
        self.delay_to_OFDM = np.exp(-1j*2*np.pi/self.total_subcarriers*np.outer(self.delay_d, self.subcarriers))
    
    def generate(self, raydata, Ts):
        use_LPF = self.OFDM_params[c.PARAMSET_OFDM_LPF]
        power = raydata[c.OUT_PATH_RX_POW].reshape(-1, 1)
        delay_n = (raydata[c.OUT_PATH_TOA] / Ts).reshape(-1, 1)
        phase = raydata[c.OUT_PATH_PHASE].reshape(-1, 1)
    
        # Ignore paths over CP
        paths_over_FFT = (delay_n >= self.OFDM_params['subcarriers'])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params['subcarriers']
        
        path_const = np.sqrt(power / self.total_subcarriers) * np.exp(1j * np.deg2rad(phase))
        if use_LPF: # LPF convolution
            path_const *= np.sinc(self.delay_d - delay_n)
        else: # Path construction without LPF
            path_const *= np.exp(-1j * (2 * np.pi / self.total_subcarriers) * np.outer(delay_n, self.subcarriers))
    
        # Apply Doppler effect if enabled
        doppler_available = self.params[c.PARAMSET_SCENARIO_PARAMS][c.PARAMSET_SCENARIO_PARAMS_DOPPLER_EN]
        doppler_selected_for_gen = self.params[c.PARAMSET_DOPPLER_EN]
        if doppler_selected_for_gen and doppler_available:
            doppler_vel = raydata[c.OUT_PATH_DOP_VEL].reshape(-1, 1)
            doppler_acc = raydata[c.OUT_PATH_DOP_ACC].reshape(-1, 1)
            carr_freq = self.params[c.PARAMSET_SCENARIO_PARAMS][c.PARAMSET_SCENARIO_PARAMS_CF]
    
            delay = Ts * self.delay_d.T if use_LPF else raydata[c.OUT_PATH_TOA].reshape(-1, 1)
            
            doppler_phase = np.exp(-1j * 2 * np.pi * carr_freq * 
                                   (doppler_vel * delay / c.LIGHTSPEED +
                                    doppler_acc * (delay ** 2) / (2 * c.LIGHTSPEED)))
            path_const *= doppler_phase

        return path_const
    