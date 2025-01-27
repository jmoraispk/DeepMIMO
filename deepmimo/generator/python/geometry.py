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
    decomposition with Euler angles. The rotation sequence is:
    1. First rotation (gamma) around z-axis
    2. Second rotation (beta) around y-axis
    3. Third rotation (alpha) around x-axis
    
    The rotation is applied as: R = R_x(alpha) * R_y(beta) * R_z(gamma)
    
    Args:
        rotation (Optional[Tuple[float, float, float]]): Rotation angles [alpha, beta, gamma] 
            around x, y, z axes in degrees. If None, no rotation is applied.
        theta (float): Elevation angle in degrees
        phi (float): Azimuth angle in degrees
        
    Returns:
        Tuple[float, float]: Tuple of rotated angles (theta, phi) in radians
        
    Note:
        The function uses a specific formulation for rotation that directly computes
        the final angles without intermediate Cartesian coordinate transformations.
        The formulation is:
        theta_rot = arccos(cos_beta*cos_gamma*cos_theta + 
                         sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha))
        phi_rot = angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta +
                       1j*(cos_beta*sin_gamma*cos_theta + 
                           sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha)))
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


def rotate_angles_batch(rotation: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate angles for batched inputs.
    
    This is a vectorized version of rotate_angles() that can process multiple users
    and paths simultaneously. It uses the same rotation sequence and mathematical
    formulation as rotate_angles():
    1. First rotation (gamma) around z-axis
    2. Second rotation (beta) around y-axis
    3. Third rotation (alpha) around x-axis
    
    The rotation is applied as: R = R_x(alpha) * R_y(beta) * R_z(gamma)
    
    Args:
        rotation: Rotation angles [alpha, beta, gamma] with shape [3] for single rotation
            or [batch_size, 3] for per-user rotations. Angles in degrees.
        theta: Elevation angles with shape [n_paths] for single user or 
            [batch_size, n_paths] for multiple users. Angles in degrees.
        phi: Azimuth angles with shape [n_paths] for single user or
            [batch_size, n_paths] for multiple users. Angles in degrees.
        
    Returns:
        Tuple of rotated angles (theta, phi) with same shape as input in radians:
        - If input is [n_paths]: output is [n_paths]
        - If input is [batch_size, n_paths]: output is [batch_size, n_paths]
        
    Note:
        The function uses the same direct angle computation as rotate_angles():
        theta_rot = arccos(cos_beta*cos_gamma*cos_theta + 
                         sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha))
        phi_rot = angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta +
                       1j*(cos_beta*sin_gamma*cos_theta + 
                           sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha)))
                           
        Broadcasting is used to handle both single rotations applied to all users
        and per-user rotations efficiently.
    """
    is_batched = theta.ndim == 2
    if not is_batched:
        theta = theta[None, :]  # [1, n_paths]
        phi = phi[None, :]      # [1, n_paths]
    
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
    rotation = np.deg2rad(rotation)  # [batch_size, 3]
    
    # Extract rotation angles
    alpha = rotation[:, 0:1]  # [batch_size, 1]
    beta = rotation[:, 1:2]   # [batch_size, 1]
    gamma = rotation[:, 2:3]  # [batch_size, 1]
    
    # Compute trigonometric functions - exactly matching original function
    sin_alpha = np.sin(phi - gamma)    # phi - gamma, matches original
    sin_beta = np.sin(beta)            # beta, matches original
    sin_gamma = np.sin(alpha)          # alpha, matches original
    cos_alpha = np.cos(phi - gamma)    # phi - gamma, matches original
    cos_beta = np.cos(beta)            # beta, matches original
    cos_gamma = np.cos(alpha)          # alpha, matches original
    
    sin_theta = np.sin(theta)  # [batch_size, n_paths]
    cos_theta = np.cos(theta)  # [batch_size, n_paths]
    
    # Compute rotated angles using the same formulation as original function
    theta_rot = np.arccos(cos_beta*cos_gamma*cos_theta +
                         sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha))
    
    phi_rot = np.angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta +
                      1j*(cos_beta*sin_gamma*cos_theta + 
                          sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha)))
    
    # Convert back to degrees (not needed)
    # theta_rot = np.rad2deg(theta_rot)  # [batch_size, n_paths]
    # phi_rot = np.rad2deg(phi_rot)      # [batch_size, n_paths]
    # phi_rot = np.mod(phi_rot, 360)     # [batch_size, n_paths]

    # Return angles in radians to match original function
    return (theta_rot[0] if not is_batched else theta_rot,
            phi_rot[0] if not is_batched else phi_rot)


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