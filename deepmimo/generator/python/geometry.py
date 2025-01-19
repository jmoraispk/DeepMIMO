"""
Geometry module for DeepMIMO.
Contains functions for spatial calculations, array responses, rotations, and coordinate transforms.
Core geometric operations used in MIMO channel generation.
"""

import numpy as np

def array_response(ant_ind, theta, phi, kd):
    """Calculate the array response for given antenna indices and angles.

    Args:
        ant_ind (numpy.ndarray): Array of antenna indices [x, y, z]
        theta (float): Elevation angle in radians
        phi (float): Azimuth angle in radians
        kd (float): Product of wavenumber k and antenna spacing d

    Returns:
        numpy.ndarray: Complex array response vector
    """
    gamma = array_response_phase(theta, phi, kd)
    return np.exp(ant_ind@gamma.T)
    
def array_response_phase(theta, phi, kd):
    """Calculate the phase component of the array response.

    Args:
        theta (float): Elevation angle in radians
        phi (float): Azimuth angle in radians
        kd (float): Product of wavenumber k and antenna spacing d

    Returns:
        tuple: Phase components (x, y, z) for each dimension
    """
    gamma_x = 1j * kd * np.sin(theta) * np.cos(phi)
    gamma_y = 1j * kd * np.sin(theta) * np.sin(phi)
    gamma_z = 1j * kd * np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T
 
def ant_indices(panel_size):
    """Generate antenna element indices for a rectangular panel.

    Args:
        panel_size (tuple): Panel dimensions (Mx, My, Mz)

    Returns:
        numpy.ndarray: Array of antenna indices with shape (N, 3) where N is total number of elements
    """
    gamma_x = np.tile(np.arange(1), panel_size[0]*panel_size[1])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[0]), 1), panel_size[1])
    gamma_z = np.repeat(np.arange(panel_size[1]), panel_size[0])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T

def apply_FoV(FoV, theta, phi):
    """Apply field of view constraints to angles.

    Args:
        FoV (tuple): Field of view limits (theta_min, theta_max, phi_min, phi_max) in degrees
        theta (float): Elevation angle in degrees
        phi (float): Azimuth angle in degrees

    Returns:
        tuple: Boolean mask indicating which angles are within FoV, and the filtered angles
    """
    theta = np.mod(theta, 2*np.pi)
    phi = np.mod(phi, 2*np.pi)
    FoV = np.deg2rad(FoV)
    path_inclusion_phi = np.logical_or(phi <= 0+FoV[0]/2, phi >= 2*np.pi-FoV[0]/2)
    path_inclusion_theta = np.logical_and(theta <= np.pi/2+FoV[1]/2, theta >= np.pi/2-FoV[1]/2)
    path_inclusion = np.logical_and(path_inclusion_phi, path_inclusion_theta)
    return path_inclusion

def rotate_angles(rotation, theta, phi):
    """Rotate angles according to specified rotation angles.

    Args:
        rotation (tuple): Rotation angles (alpha, beta, gamma) in degrees
        theta (float): Elevation angle in degrees
        phi (float): Azimuth angle in degrees

    Returns:
        tuple: Rotated angles (theta, phi) in radians
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

def steering_vec(array, phi=0, theta=0, spacing=0.5):
    """Calculate the steering vector for an antenna array.

    Args:
        array (numpy.ndarray): Array of antenna positions
        phi (float, optional): Azimuth angle in degrees. Defaults to 0.
        theta (float, optional): Elevation angle in degrees. Defaults to 0.
        spacing (float, optional): Antenna spacing in wavelengths. Defaults to 0.5.

    Returns:
        numpy.ndarray: Complex (normalized) steering (array response) vector
    """
    idxs = ant_indices(array)
    resp = array_response(idxs, phi*np.pi/180, theta*np.pi/180 + np.pi/2, 2*np.pi*spacing)
    return resp / np.linalg.norm(resp) 