"""
Channel module for DeepMIMO.

This module provides functionality for MIMO channel generation, including:
- Channel parameter management through the ChannelGenParameters class
- OFDM path generation and verification 
- Channel matrix computation

The main function is generate_MIMO_channel() which generates MIMO channel matrices
based on path information from ray-tracing and antenna configurations.
"""

import numpy as np
from tqdm import tqdm
from typing import Dict
from ... import consts as c
from ...general_utilities import DotDict

class ChannelGenParameters(DotDict):
    """Class for managing channel generation parameters.
    
    This class provides an interface for setting and accessing various parameters
    needed for MIMO channel generation, including:
    - BS/UE antenna array configurations
    - OFDM parameters
    - Channel domain settings (time/frequency)
    
    The parameters can be accessed directly using dot notation (e.g. params.bs_antenna.shape)
    or using dictionary notation (e.g. params['bs_antenna']['shape']).
    """
    def __init__(self):
        """Initialize channel generation parameters with default values."""
        super().__init__({
            # BS Antenna Parameters
            c.PARAMSET_ANT_BS: {
                c.PARAMSET_ANT_SHAPE: np.array([8, 4]), # Antenna dimensions in X - Y - Z
                c.PARAMSET_ANT_SPACING: 0.5,
                c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]), # Rotation around X - Y - Z axes
                c.PARAMSET_ANT_FOV: np.array([360, 180]), # Horizontal-Vertical FoV
                c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0] # 'omni-directional'
            },
            
            # UE Antenna Parameters
            c.PARAMSET_ANT_UE: {
                c.PARAMSET_ANT_SHAPE: np.array([4, 2]), # Antenna dimensions in X - Y - Z
                c.PARAMSET_ANT_SPACING: 0.5,
                c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]), # Rotation around X - Y - Z axes
                c.PARAMSET_ANT_FOV: np.array([360, 180]), # Horizontal-Vertical FoV
                c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0] # 'omni-directional'
            },
            
            c.PARAMSET_DOPPLER_EN: 0,
            c.PARAMSET_POLAR_EN: 0,
            
            c.PARAMSET_FD_CH: 1, # OFDM channel if 1, Time domain if 0
            
            # OFDM Parameters
            c.PARAMSET_OFDM: {
                c.PARAMSET_OFDM_SC_NUM: 512, # Number of total subcarriers
                c.PARAMSET_OFDM_SC_SAMP: np.arange(1), # Select subcarriers to generate
                c.PARAMSET_OFDM_BW: 0.05, # GHz
                c.PARAMSET_OFDM_LPF: 0 # Receive Low Pass / ADC Filter
            }
        })

class PathVerifier:
    """Class for verifying and validating paths based on configuration parameters.
    
    This class checks path validity against OFDM parameters and provides warnings
    when paths exceed the OFDM symbol duration.
    
    Attributes:
        params (dict): Channel generation parameters
        FFT_duration (float): OFDM symbol duration
        max_ToA (float): Maximum time of arrival seen
        path_ratio_FFT (list): Ratios of clipped path powers
    """
    
    def __init__(self, params: Dict):
        """Initialize path verifier.
        
        Args:
            params (dict): Channel generation parameters
        """
        self.params = params
        if self.params[c.PARAMSET_FD_CH]: # IF OFDM
            Ts = 1 / (params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_BW]*c.PARAMSET_OFDM_BW_MULT)
            self.FFT_duration = params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_NUM] * Ts
            self.max_ToA = 0
            self.path_ratio_FFT = []
    
    def verify_path(self, ToA: float, power: float) -> None:
        """Verify a path's time of arrival against OFDM parameters.
        
        Args:
            ToA (float): Time of arrival
            power (float): Path power
        """
        if self.params[c.PARAMSET_FD_CH]: # OFDM CH
            m_toa = np.max(ToA)
            self.max_ToA = max(self.max_ToA, m_toa)
            
            if m_toa > self.FFT_duration:
                violating_paths = ToA > self.FFT_duration
                self.path_ratio_FFT.append(sum(power[violating_paths])/sum(power))
                        
    def notify(self) -> None:
        """Print notification about paths exceeding OFDM duration if needed."""
        if self.params[c.PARAMSET_FD_CH]:
            avg_ratio_FFT = 0
            if len(self.path_ratio_FFT) != 0:
                avg_ratio_FFT = np.mean(self.path_ratio_FFT)*100
                
            if self.max_ToA > self.FFT_duration and avg_ratio_FFT >= 1.:
                print(f'ToA of some paths of {len(self.path_ratio_FFT)} channels '
                      f'with an average total power of {avg_ratio_FFT:.2f}% exceed '
                      'the useful OFDM symbol duration and are clipped.')

class OFDM_PathGenerator:
    """Class for generating OFDM paths with specified parameters.
    
    This class handles the generation of OFDM paths including optional
    low-pass filtering.
    
    Attributes:
        OFDM_params (dict): OFDM parameters
        subcarriers (array): Selected subcarrier indices
        total_subcarriers (int): Total number of subcarriers
        delay_d (array): Delay domain array
        delay_to_OFDM (array): Delay to OFDM transform matrix
    """
    
    def __init__(self, params: Dict, subcarriers: np.ndarray):
        """Initialize OFDM path generator.
        
        Args:
            params (dict): OFDM parameters
            subcarriers (array): Selected subcarrier indices
        """
        self.OFDM_params = params
        self.subcarriers = subcarriers  # selected
        self.total_subcarriers = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        self.delay_to_OFDM = np.exp(-1j * 2 * np.pi / self.total_subcarriers * 
                                   np.outer(self.delay_d, self.subcarriers))
    
    def generate(self, pwr: np.ndarray, toa: np.ndarray, phs: np.ndarray, Ts: float) -> np.ndarray:
        """Generate OFDM paths.
        
        Args:
            pwr (array): Path powers
            toa (array): Times of arrival
            phs (array): Path phases
            Ts (float): Sampling period
            
        Returns:
            array: Generated OFDM paths
        """
        power = pwr.reshape(-1, 1)
        delay_n = toa.reshape(-1, 1) / Ts
        phase = phs.reshape(-1, 1)
    
        # Ignore paths over CP
        paths_over_FFT = (delay_n >= self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        path_const = np.sqrt(power / self.total_subcarriers) * np.exp(1j * np.deg2rad(phase))
        if self.OFDM_params[c.PARAMSET_OFDM_LPF]: # Low-pass filter (LPF) convolution
            path_const = path_const * np.sinc(self.delay_d - delay_n) @ self.delay_to_OFDM
        else: # Path construction without LPF
            path_const *= np.exp(-1j * (2 * np.pi / self.total_subcarriers) * 
                               np.outer(delay_n, self.subcarriers))
        return path_const

def generate_MIMO_channel(array_response_product: np.ndarray,
                         powers: np.ndarray,
                         delays: np.ndarray,
                         phases: np.ndarray,
                         ofdm_params: Dict,
                         freq_domain: bool = True) -> np.ndarray:
    """Generate MIMO channel matrices.
    
    This function generates MIMO channel matrices based on path information and
    pre-computed array responses. It supports both time and frequency domain
    channel generation.
    
    Args:
        array_response_product: Product of TX and RX array responses [n_users, M_rx, M_tx, n_paths]
        powers: Linear path powers [W] with antenna gains applied [n_users, n_paths]
        toas: Times of arrival [n_users, n_paths]
        phases: Path phases [n_users, n_paths]
        ofdm_params: OFDM parameters
        freq_domain: Whether to generate frequency domain channel. Defaults to True.
        
    Returns:
        numpy.ndarray: MIMO channel matrices with shape (n_users, n_rx_ant, n_tx_ant, n_paths/subcarriers)
    """
    bandwidth = ofdm_params[c.PARAMSET_OFDM_BW] * c.PARAMSET_OFDM_BW_MULT
    Ts = 1 / bandwidth
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    path_gen = OFDM_PathGenerator(ofdm_params, subcarriers)

    n_ues = powers.shape[0]
    max_paths = powers.shape[1]
    M_rx, M_tx = array_response_product.shape[1:3]
    
    last_ch_dim = len(subcarriers) if freq_domain else max_paths
    channel = np.zeros((n_ues, M_rx, M_tx, last_ch_dim), dtype=np.csingle)
    
    # Pre-compute NaN masks for all users using powers
    nan_masks = ~np.isnan(powers)  # [n_users, n_paths]
    valid_path_counts = np.sum(nan_masks, axis=1)  # [n_users]

    # Generate channels for each user
    for i in tqdm(range(n_ues), desc='Generating channels'):
        # Get valid paths for this user
        non_nan_mask = nan_masks[i]
        n_paths = valid_path_counts[i]
        
        # Skip users with no valid paths
        if n_paths == 0:
            continue
            
        # Get pre-computed array product for this user (with NaN handling)
        array_product = array_response_product[i][..., non_nan_mask]  # [M_rx, M_tx, n_valid_paths]
        
        # Get pre-computed values for this user
        power = powers[i, non_nan_mask]
        delays_user = delays[i, non_nan_mask]
        phases_user = phases[i, non_nan_mask]
        
        if freq_domain: # OFDM
            path_gains = path_gen.generate(pwr=power, toa=delays_user, phs=phases_user, Ts=Ts).T
            channel[i] = np.nansum(array_product[..., None, :] * 
                                   path_gains[None, None, :, :], axis=-1)
        else: # TD channel
            path_gains = np.sqrt(power) * np.exp(1j*np.deg2rad(phases_user))
            channel[i, ..., :n_paths] = array_product * path_gains[None, None, :]

    return channel 