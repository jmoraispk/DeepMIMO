"""
Geometry module for DeepMIMO channel generation.

This module provides geometric calculations and transformations needed for MIMO systems:
- Array response calculations
- Antenna array geometry functions
- Field of view constraints
- Angle rotations and transformations
- Steering vector computation

The functions handle both single values and numpy arrays for vectorized operations.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


def array_response(ant_ind: NDArray, theta: float, phi: float, kd: float) -> NDArray:
    """Calculate the array response vector for given antenna indices and angles.
    
    This function computes the complex array response based on antenna positions and 
    arrival angles using the standard array response formula.
    
    Args:
        ant_ind (NDArray): Array of antenna indices with shape (N,3) containing [x,y,z] positions
        theta (float): Elevation angle in radians
        phi (float): Azimuth angle in radians  
        kd (float): Product of wavenumber k and antenna spacing d
        
    Returns:
        NDArray: Complex array response vector with shape matching ant_ind
    """
    gamma = array_response_phase(theta, phi, kd)
    return np.exp(ant_ind@gamma.T)


def array_response_phase(theta: float, phi: float, kd: float) -> NDArray:
    """Calculate the phase components of the array response.
    
    This function computes the phase terms for each spatial dimension x,y,z
    used in array response calculations.
    
    Args:
        theta (float): Elevation angle in radians
        phi (float): Azimuth angle in radians
        kd (float): Product of wavenumber k and antenna spacing d
        
    Returns:
        NDArray: Array of phase components with shape (N,3) for [x,y,z] dimensions
    """
    gamma_x = 1j * kd * np.sin(theta) * np.cos(phi)
    gamma_y = 1j * kd * np.sin(theta) * np.sin(phi)
    gamma_z = 1j * kd * np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T


def ant_indices(panel_size: Tuple[int, int, int]) -> NDArray:
    """Generate antenna element indices for a rectangular panel.
    
    This function creates an array of indices representing antenna positions 
    in 3D space for a rectangular antenna panel.
    
    Args:
        panel_size (Tuple[int, int, int]): Panel dimensions as tuple (Mx, My, Mz)
        
    Returns:
        NDArray: Array of antenna indices with shape (N,3) where N is total number of elements
    """
    gamma_x = np.tile(np.arange(1), panel_size[0]*panel_size[1])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[0]), 1), panel_size[1])
    gamma_z = np.repeat(np.arange(panel_size[1]), panel_size[0])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T


def apply_FoV(FoV: Tuple[float, float, float, float], theta: float, phi: float) -> NDArray:
    """Apply field of view constraints to angles.
    
    This function filters angles based on specified field of view limits
    in both elevation and azimuth directions.
    
    Args:
        FoV (Tuple[float, float, float, float]): Field of view limits (theta_min, theta_max, phi_min, phi_max) in degrees
        theta (float): Elevation angle in degrees
        phi (float): Azimuth angle in degrees
        
    Returns:
        NDArray: Boolean mask indicating which angles are within the field of view
    """
    theta = np.mod(theta, 2*np.pi)
    phi = np.mod(phi, 2*np.pi)
    FoV = np.deg2rad(FoV)
    path_inclusion_phi = np.logical_or(phi <= 0+FoV[0]/2, phi >= 2*np.pi-FoV[0]/2)
    path_inclusion_theta = np.logical_and(theta <= np.pi/2+FoV[1]/2, theta >= np.pi/2-FoV[1]/2)
    return np.logical_and(path_inclusion_phi, path_inclusion_theta)


def rotate_angles(rotation: Optional[Tuple[float, float, float]], theta: float, 
                 phi: float) -> Tuple[float, float]:
    """Rotate angles according to specified rotation angles.
    
    This function applies 3D rotation to direction angles using rotation matrix
    decomposition with Euler angles.
    
    Args:
        rotation (Optional[Tuple[float, float, float]]): Rotation angles around x, y, z axes in degrees
        theta (float): Elevation angle in degrees
        phi (float): Azimuth angle in degrees
        
    Returns:
        Tuple[float, float]: Tuple of rotated angles (theta, phi) in radians
    """
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


def steering_vec(array: NDArray, phi: float = 0, theta: float = 0, spacing: float = 0.5) -> NDArray:
    """Calculate the steering vector for an antenna array.
    
    This function computes the normalized array response vector for a given array
    geometry and steering direction.
    
    Args:
        array (NDArray): Array of antenna positions
        phi (float): Azimuth angle in degrees. Defaults to 0.
        theta (float): Elevation angle in degrees. Defaults to 0.
        spacing (float): Antenna spacing in wavelengths. Defaults to 0.5.
        
    Returns:
        NDArray: Complex normalized steering (array response) vector
    """
    idxs = ant_indices(array)
    resp = array_response(idxs, phi*np.pi/180, theta*np.pi/180 + np.pi/2, 2*np.pi*spacing)
    return resp / np.linalg.norm(resp)