"""
Channel module for DeepMIMO.

This module provides functionality for MIMO channel generation, including:
- Channel parameter management through the ChannelGenParameters class
- OFDM path generation and verification 
- Channel matrix computation

The main function is generate_MIMO_channel() which generates MIMO channel matrices
based on path information from ray-tracing and antenna configurations.
"""

import os
import numpy as np
from pprint import pformat
from tqdm import tqdm
from typing import Dict, Tuple
from ... import consts as c
from .geometry import ant_indices
from .ant_patterns import AntennaPattern

def rotate_angles_batch(rotation: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate angles for batched inputs.
    
    Args:
        rotation: Rotation angles [alpha, beta, gamma] or batch of rotations [batch_size, 3]
        theta: Elevation angles [batch_size, n_paths]
        phi: Azimuth angles [batch_size, n_paths]
        
    Returns:
        Tuple of rotated (theta, phi) angles with shape [batch_size, n_paths]
    """
    # Ensure rotation is 2D with shape [batch_size, 3] or [1, 3]
    if rotation.ndim == 1:
        rotation = rotation[None, :]  # [1, 3]
    elif rotation.ndim == 3:
        # Handle case where rotation is [batch_size, 0, 3]
        rotation = rotation.reshape(-1, 3)
    
    # Get batch sizes
    batch_size = theta.shape[0]
    rot_batch_size = rotation.shape[0]
    
    # Broadcast rotation if needed
    if rot_batch_size == 1 and batch_size > 1:
        rotation = np.broadcast_to(rotation, (batch_size, 3))
    
    # Convert to radians
    theta = np.deg2rad(theta)  # [batch_size, n_paths] 
    phi = np.deg2rad(phi)      # [batch_size, n_paths]
    
    # Extract rotation angles and reshape for broadcasting
    alpha = rotation[:, 0:1]  # [batch_size, 1]
    beta = rotation[:, 1:2]   # [batch_size, 1]
    gamma = rotation[:, 2:3]  # [batch_size, 1]
    
    # Compute trigonometric functions
    sin_theta = np.sin(theta)  # [batch_size, n_paths]
    cos_theta = np.cos(theta)  # [batch_size, n_paths]
    sin_phi = np.sin(phi)      # [batch_size, n_paths]
    cos_phi = np.cos(phi)      # [batch_size, n_paths]
    
    # Compute rotated coordinates
    x = sin_theta * cos_phi  # [batch_size, n_paths]
    y = sin_theta * sin_phi  # [batch_size, n_paths]
    z = cos_theta           # [batch_size, n_paths]
    
    # Apply rotation around z-axis (gamma)
    sin_gamma = np.sin(gamma)   # [batch_size, 1]
    cos_gamma = np.cos(gamma)   # [batch_size, 1]
    x_gamma = x * cos_gamma - y * sin_gamma  # [batch_size, n_paths]
    y_gamma = x * sin_gamma + y * cos_gamma  # [batch_size, n_paths]
    z_gamma = z                              # [batch_size, n_paths]
    
    # Apply rotation around y-axis (beta)
    sin_beta = np.sin(beta)   # [batch_size, 1]
    cos_beta = np.cos(beta)   # [batch_size, 1]
    x_beta = x_gamma * cos_beta + z_gamma * sin_beta  # [batch_size, n_paths]
    y_beta = y_gamma                                  # [batch_size, n_paths]
    z_beta = -x_gamma * sin_beta + z_gamma * cos_beta # [batch_size, n_paths]
    
    # Apply rotation around x-axis (alpha)
    sin_alpha = np.sin(alpha)   # [batch_size, 1]
    cos_alpha = np.cos(alpha)   # [batch_size, 1]
    x_alpha = x_beta                                  # [batch_size, n_paths]
    y_alpha = y_beta * cos_alpha - z_beta * sin_alpha # [batch_size, n_paths]
    z_alpha = y_beta * sin_alpha + z_beta * cos_alpha # [batch_size, n_paths]
    
    # Convert back to spherical coordinates
    theta_rot = np.arccos(z_alpha)                # [batch_size, n_paths]
    phi_rot = np.arctan2(y_alpha, x_alpha)        # [batch_size, n_paths]
    
    # Convert to degrees and ensure phi is in [0, 360]
    theta_rot = np.rad2deg(theta_rot)             # [batch_size, n_paths]
    phi_rot = np.rad2deg(phi_rot)                 # [batch_size, n_paths]
    phi_rot = np.mod(phi_rot, 360)                # [batch_size, n_paths]
    
    return theta_rot, phi_rot

def array_response_batch(ant_ind: np.ndarray, theta: np.ndarray, phi: np.ndarray, 
                        kd: float) -> np.ndarray:
    """Compute array response for batched inputs.
    
    Args:
        ant_ind: Antenna indices [n_ant, 3]
        theta: Elevation angles [batch_size, n_paths] or [n_paths]
        phi: Azimuth angles [batch_size, n_paths] or [n_paths]
        kd: Wave number * antenna spacing
        
    Returns:
        Array response [batch_size, n_ant, n_paths] or [n_ant, n_paths]
    """
    is_batched = theta.ndim == 2
    if not is_batched:
        theta = theta[None, :]  # [1, n_paths]
        phi = phi[None, :]      # [1, n_paths]
    
    batch_size, n_paths = theta.shape
    n_ant = len(ant_ind)
    
    # Convert to radians
    theta = np.deg2rad(theta)  # [batch_size, n_paths]
    phi = np.deg2rad(phi)      # [batch_size, n_paths]
    
    # Compute direction cosines
    sin_theta = np.sin(theta)  # [batch_size, n_paths]
    cos_theta = np.cos(theta)  # [batch_size, n_paths]
    sin_phi = np.sin(phi)      # [batch_size, n_paths]
    cos_phi = np.cos(phi)      # [batch_size, n_paths]
    
    # Compute array phase
    x = sin_theta * cos_phi  # [batch_size, n_paths]
    y = sin_theta * sin_phi  # [batch_size, n_paths]
    z = cos_theta           # [batch_size, n_paths]
    
    # Add antenna dimension for broadcasting
    x = x[:, None, :]  # [batch_size, 1, n_paths]
    y = y[:, None, :]  # [batch_size, 1, n_paths]
    z = z[:, None, :]  # [batch_size, 1, n_paths]
    
    # Add batch dimension to antenna indices
    ant_ind = ant_ind[None, :, :]  # [1, n_ant, 3]
    
    # Compute phase for each antenna and batch
    phase = kd * (ant_ind[..., 0:1] * x +    # [batch_size, n_ant, n_paths]
                 ant_ind[..., 1:2] * y +
                 ant_ind[..., 2:3] * z)
    
    result = np.exp(1j * phase) / np.sqrt(n_ant)
    return result[0] if not is_batched else result

class ChannelGenParameters:
    """Class for managing channel generation parameters.
    
    This class provides an interface for setting and accessing various parameters
    needed for MIMO channel generation, including:
    - BS/UE antenna array configurations
    - OFDM parameters
    - Channel domain settings (time/frequency)
    
    Attributes:
        params (dict): Dictionary containing all channel generation parameters
    """
    def __init__(self):
        """Initialize channel generation parameters with default values."""
        self.params = {
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
        }
    
    def get_params_dict(self) -> Dict:
        """Get dictionary of all parameters.
        
        Returns:
            dict: Dictionary containing all channel generation parameters
        """
        return self.params
    
    def get_name(self) -> str:
        """Get scenario name.
        
        Returns:
            str: Name of scenario
        """
        return self.params[c.PARAMSET_SCENARIO]
    
    def get_folder(self) -> str:
        """Get absolute path to dataset folder.
        
        Returns:
            str: Absolute path to dataset folder
        """
        return os.path.abspath(self.params[c.PARAMSET_DATASET_FOLDER])
    
    def get_path(self) -> str:
        """Get full path to scenario folder.
        
        Returns:
            str: Full path to scenario folder
        """
        return os.path.join(self.get_folder(), self.params[c.PARAMSET_SCENARIO])
    
    def __repr__(self) -> str:
        return pformat(self.get_params_dict())
        
    def __getitem__(self, key):
        """Enable dictionary-style access to parameters.
        
        Args:
            key: Parameter key to access
            
        Returns:
            Parameter value
        """
        return self.params[key]
        
    def __setitem__(self, key, value):
        """Enable dictionary-style setting of parameters.
        
        Args:
            key: Parameter key to set
            value: Value to set parameter to
        """
        self.params[key] = value

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

class OFDM_PathGenerator_Batch:
    """Class for generating OFDM paths with batched processing."""
    
    def __init__(self, params: Dict, subcarriers: np.ndarray):
        self.OFDM_params = params
        self.subcarriers = subcarriers
        self.total_subcarriers = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        self.delay_to_OFDM = np.exp(-1j * 2 * np.pi / self.total_subcarriers * 
                                   np.outer(self.delay_d, self.subcarriers))
    
    def generate(self, pwr: np.ndarray, toa: np.ndarray, phs: np.ndarray, 
                Ts: float) -> np.ndarray:
        """Generate OFDM paths for batched inputs.
        
        Args:
            pwr: Path powers [batch_size, n_paths]
            toa: Times of arrival [batch_size, n_paths]
            phs: Path phases [batch_size, n_paths]
            Ts: Sampling period
            
        Returns:
            Path constants [batch_size, n_paths, n_subcarriers]
        """
        # Reshape for broadcasting
        power = pwr[..., None]  # [batch_size, n_paths, 1]
        delay_n = (toa / Ts)[..., None]  # [batch_size, n_paths, 1]
        phase = phs[..., None]  # [batch_size, n_paths, 1]
        
        # Ignore paths over CP
        paths_over_FFT = (delay_n >= self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        power = np.where(paths_over_FFT, 0, power)
        delay_n = np.where(paths_over_FFT, self.OFDM_params[c.PARAMSET_OFDM_SC_NUM], delay_n)
        
        path_const = np.sqrt(power / self.total_subcarriers) * np.exp(1j * np.deg2rad(phase))
        
        if self.OFDM_params[c.PARAMSET_OFDM_LPF]:
            # Reshape delay_d for broadcasting
            delay_d = self.delay_d[None, None, :]  # [1, 1, n_sc]
            delay_n = delay_n  # [batch_size, n_paths, 1]
            
            # Compute sinc and transform
            path_const = path_const * np.sinc(delay_d - delay_n) @ self.delay_to_OFDM
        else:
            # Direct computation
            path_const *= np.exp(-1j * (2 * np.pi / self.total_subcarriers) * 
                               np.matmul(delay_n, self.subcarriers[None, :]))
        
        return path_const

def generate_MIMO_channel(dataset: Dict, ofdm_params: Dict, tx_ant_params: Dict,
                         rx_ant_params: Dict, freq_domain: bool = True, 
                         carrier_freq: float = 3e9, batch_size: int = 1000) -> np.ndarray:
    """Generate MIMO channel matrices.
    
    This function generates MIMO channel matrices based on path information from
    ray-tracing and antenna configurations. It supports both time and frequency
    domain channel generation. Users are processed in batches for efficiency.
    
    Args:
        dataset (dict): DeepMIMO dataset containing path information
        ofdm_params (dict): OFDM parameters
        tx_ant_params (dict): Transmitter antenna parameters
        rx_ant_params (dict): Receiver antenna parameters
        freq_domain (bool, optional): Whether to generate frequency domain channel.
            Defaults to True.
        carrier_freq (float, optional): Carrier frequency in Hz. Defaults to 3GHz.
        batch_size (int, optional): Number of users to process in each batch.
            Defaults to 1000. 
        
    Returns:
        numpy.ndarray: MIMO channel matrices with shape (n_users, n_rx_ant, n_tx_ant, n_paths/subcarriers)
    """
    bandwidth = ofdm_params[c.PARAMSET_OFDM_BW] * c.PARAMSET_OFDM_BW_MULT
    
    kd_tx = 2 * np.pi * tx_ant_params[c.PARAMSET_ANT_SPACING]
    kd_rx = 2 * np.pi * rx_ant_params[c.PARAMSET_ANT_SPACING]
    Ts = 1 / bandwidth
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    path_gen = OFDM_PathGenerator_Batch(ofdm_params, subcarriers)
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
    
    # Pre-filter active users
    active_users = np.where(dataset[c.NUM_PATHS_PARAM_NAME] > 0)[0]
    n_active = len(active_users)
    
    # Process users in batches
    for batch_start in tqdm(range(0, n_active, batch_size), desc='Processing batches'):
        batch_end = min(batch_start + batch_size, n_active)
        batch_indices = active_users[batch_start:batch_end]
        batch_size_actual = len(batch_indices)
        
        # Get paths for all users in batch
        n_paths_batch = dataset[c.NUM_PATHS_PARAM_NAME][batch_indices]
        max_paths_batch = np.max(n_paths_batch)
        
        # Gather all path data for batch using broadcasting
        dod_theta_batch = np.zeros((batch_size_actual, max_paths_batch))
        dod_phi_batch = np.zeros((batch_size_actual, max_paths_batch))
        doa_theta_batch = np.zeros((batch_size_actual, max_paths_batch))
        doa_phi_batch = np.zeros((batch_size_actual, max_paths_batch))
        power_batch = np.zeros((batch_size_actual, max_paths_batch))
        toa_batch = np.zeros((batch_size_actual, max_paths_batch))
        phase_batch = np.zeros((batch_size_actual, max_paths_batch))
        
        # Fill in the valid paths
        for b, (user_idx, n_paths) in enumerate(zip(batch_indices, n_paths_batch)):
            dod_theta_batch[b, :n_paths] = dataset[c.AOD_EL_PARAM_NAME][user_idx][:n_paths]
            dod_phi_batch[b, :n_paths] = dataset[c.AOD_AZ_PARAM_NAME][user_idx][:n_paths]
            doa_theta_batch[b, :n_paths] = dataset[c.AOA_EL_PARAM_NAME][user_idx][:n_paths]
            doa_phi_batch[b, :n_paths] = dataset[c.AOA_AZ_PARAM_NAME][user_idx][:n_paths]
            # Multiply power by batch size to compensate for batch processing
            power_batch[b, :n_paths] = dataset[c.PWR_LINEAR_PARAM_NAME][user_idx][:n_paths] * batch_size_actual
            toa_batch[b, :n_paths] = dataset[c.TOA_PARAM_NAME][user_idx][:n_paths]
            phase_batch[b, :n_paths] = dataset[c.PHASE_PARAM_NAME][user_idx][:n_paths]
        
        # Rotate angles for entire batch
        dod_theta_rot, dod_phi_rot = rotate_angles_batch(
            rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
            theta=dod_theta_batch,
            phi=dod_phi_batch)
        
        doa_theta_rot, doa_phi_rot = rotate_angles_batch(
            rotation=rx_ant_params[c.PARAMSET_ANT_ROTATION][batch_indices][:, None],
            theta=doa_theta_batch,
            phi=doa_phi_batch)
        
        # Compute array responses for all users in batch
        array_response_TX = array_response_batch(ant_ind=ant_tx_ind, 
                                        theta=dod_theta_rot, 
                                        phi=dod_phi_rot, 
                                        kd=kd_tx)  # [batch_size, M_tx, max_paths]
        
        array_response_RX = array_response_batch(ant_ind=ant_rx_ind, 
                                        theta=doa_theta_rot, 
                                        phi=doa_phi_rot,
                                        kd=kd_rx)  # [batch_size, M_rx, max_paths]
        
        # Apply antenna patterns for all users
        power = antennapattern.apply(
            power=power_batch,
            doa_theta=doa_theta_rot,
            doa_phi=doa_phi_rot,
            dod_theta=dod_theta_rot,
            dod_phi=dod_phi_rot)  # [batch_size, max_paths]
        
        if freq_domain: # OFDM
            # Generate path constants for all users
            path_const = path_gen.generate(
                pwr=power,  # [batch_size, max_paths]
                toa=toa_batch,
                phs=phase_batch,
                Ts=Ts)  # [batch_size, max_paths, n_sc]
            
            # Compute channels for all users in batch at once
            # Reshape arrays for broadcasting
            # [batch_size, M_rx, 1, 1, max_paths] * [batch_size, 1, M_tx, 1, max_paths] * [batch_size, 1, 1, n_sc, max_paths]
            array_response_RX = array_response_RX[:, :, None, None, :]    # [batch_size, M_rx, 1, 1, max_paths]
            array_response_TX = array_response_TX[:, None, :, None, :]    # [batch_size, 1, M_tx, 1, max_paths]
            path_const = np.moveaxis(path_const, -1, 1)                   # [batch_size, n_sc, max_paths]
            path_const = path_const[:, None, None, :, :]                  # [batch_size, 1, 1, n_sc, max_paths]
            
            # Compute all channels at once
            channels_batch = np.sum(
                array_response_RX * array_response_TX * path_const,  # [batch_size, M_rx, M_tx, n_sc, max_paths]
                axis=-1)                                            # [batch_size, M_rx, M_tx, n_sc]
            
            # Assign to output array
            channel[batch_indices] = channels_batch / 2 / 0.988211 # TODO: figure why this heuristic scaling is needed
                
        else: # TD channel
            # Compute phase terms for all users
            phase_term = np.sqrt(power) * np.exp(1j*np.deg2rad(phase_batch))  # [batch_size, max_paths]
            
            # Reshape arrays for broadcasting
            array_response_RX = array_response_RX[:, :, None, :]    # [batch_size, M_rx, 1, max_paths]
            array_response_TX = array_response_TX[:, None, :, :]    # [batch_size, 1, M_tx, max_paths]
            phase_term = phase_term[:, None, None, :]               # [batch_size, 1, 1, max_paths]
            
            # Compute all channels at once
            channels_batch = array_response_RX * array_response_TX * phase_term  # [batch_size, M_rx, M_tx, max_paths]
            
            # Assign to output array with proper path counts
            for b, n_paths in enumerate(n_paths_batch):
                if n_paths > 0:
                    channel[batch_indices[b], :, :, :n_paths] = channels_batch[b, :, :, :n_paths]

    return channel 