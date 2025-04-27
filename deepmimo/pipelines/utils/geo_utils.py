"""
Geographic utilities for coordinate conversion.

This module provides functions for converting between geographic coordinates (latitude/longitude)
and Cartesian coordinates, as well as bounding box transformations.
"""

import os
import requests
import numpy as np
import utm
from typing import Tuple, Optional


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


#############################################
# Google Maps API Utilities
#############################################

def get_city_name(lat: float, lon: float, api_key: str) -> str:
    """Fetch the city name from coordinates using Google Maps Geocoding API.
    
    Args:
        lat (float): Latitude coordinate in degrees
        lon (float): Longitude coordinate in degrees 
        api_key (str): Google Maps API key for authentication
        
    Returns:
        str: City name if found, "unknown" otherwise
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            # Look for the city in the address components
            for result in data["results"]:
                for component in result["address_components"]:
                    if "locality" in component["types"]:  # 'locality' typically means city
                        return component["long_name"]
            return "unknown"  # Fallback if no city is found
        else:
            print(f"Geocoding error: {data['status']}")
            return "unknown"
    else:
        print(f"Geocoding request failed: {response.status_code}")
        return "unknown"

def fetch_satellite_view(minlat: float, minlon: float, maxlat: float, maxlon: float, 
                         api_key: str, save_dir: str) -> Optional[str]:
    """Fetch a satellite view image of a bounding box.
    
    Args:
        minlat (float): Minimum latitude in degrees
        minlon (float): Minimum longitude in degrees
        maxlat (float): Maximum latitude in degrees
        maxlon (float): Maximum longitude in degrees
        api_key (str): Google Maps API key for authentication
        save_dir (str): Directory to save the satellite view image
        
    Returns:
        str: Path to the saved satellite view image, or None if the request fails
    """

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate the center of the bounding box
    center_lat = (minlat + maxlat) / 2
    center_lon = (minlon + maxlon) / 2

    # Parameters for the Static Maps API
    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": 18,  # Adjust zoom level (higher = more detailed)
        "size": "640x640",  # Image size in pixels (max 640x640 for free tier)
        "maptype": "satellite",  # Options: roadmap, satellite, hybrid, terrain
        "key": api_key
    }

    # API endpoint
    STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

    # Make the request
    response = requests.get(STATIC_MAP_URL, params=params)

    # Save the image in the specified directory with city name
    if response.status_code == 200:
        image_path = os.path.join(save_dir, "satellite_view.png")
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Satellite view saved as '{image_path}'")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        image_path = None
    
    return image_path