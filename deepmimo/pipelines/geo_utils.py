"""
Geographic utilities for coordinate conversion.

This module provides functions for converting between geographic coordinates (latitude/longitude)
and Cartesian coordinates, as well as bounding box transformations.
"""

import numpy as np
import utm
from typing import Tuple


def xy_from_latlong(lat: float | np.ndarray, long: float | np.ndarray) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Convert latitude and longitude to UTM coordinates.
    
    Assumes lat and long are along row. Returns same row vec/matrix on
    cartesian coordinates.
    
    Args:
        lat (Union[float, np.ndarray]): Latitude in degrees
        long (Union[float, np.ndarray]): Longitude in degrees
        
    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]: UTM coordinates (easting, northing)
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat, long)
    return x, y


def convert_GpsBBox2CartesianBBox(minlat: float, minlon: float, 
                                  maxlat: float, maxlon: float, 
                                  origin_lat: float, origin_lon: float, 
                                  pad: float = 0) -> Tuple[float, float, float, float]:
    """Convert a GPS bounding box to a Cartesian bounding box.
    
    Args:
        minlat (float): Minimum latitude in degrees
        minlon (float): Minimum longitude in degrees
        maxlat (float): Maximum latitude in degrees
        maxlon (float): Maximum longitude in degrees
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        pad (float, optional): Padding to add to the bounding box. Defaults to 0.
        
    Returns:
        Tuple[float, float, float, float]: Cartesian bounding box (xmin, ymin, xmax, ymax)
    """
    xmin, ymin = xy_from_latlong(minlat, minlon)
    xmax, ymax = xy_from_latlong(maxlat, maxlon)
    x_origin, y_origin = xy_from_latlong(origin_lat, origin_lon)

    xmin = xmin - x_origin
    xmax = xmax - x_origin
    ymin = ymin - y_origin
    ymax = ymax - y_origin
    
    return xmin-pad, ymin-pad, xmax+pad, ymax+pad


def convert_Gps2RelativeCartesian(lat: float | np.ndarray, 
                                  lon: float | np.ndarray,
                                  origin_lat: float, 
                                  origin_lon: float) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Convert GPS coordinates to relative Cartesian coordinates.
    
    Args:
        lat (Union[float, np.ndarray]): Latitude in degrees
        lon (Union[float, np.ndarray]): Longitude in degrees
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        
    Returns:
        Tuple[float | np.ndarray, float | np.ndarray]: Relative Cartesian coordinates (x, y)
    """
    x_origin, y_origin = xy_from_latlong(origin_lat, origin_lon)
    x, y = xy_from_latlong(lat, lon)
    
    return x - x_origin, y - y_origin