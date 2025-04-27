"""
Pipeline for generating bounding boxes from city coordinates.

This script:
1. Reads city coordinates from worldcities.csv
2. For each coordinate, generates a bounding box
3. Checks if the bounding box contains any buildings
4. Validates that the BS coordinate is not on a building footprint
5. If BS is on a building, finds a nearby coordinate outside any building
6. Only saves scenarios that have at least one building and valid BS location
7. Outputs results to a CSV file with updated BS coordinates

TODO:
    - Configure cell splitting feature:
        - Use cell splitting to split bounding box into smaller cells
        - Configure cell size to be ~80m longer in x and y compared to NY (~200x400m)
        - Save individual cells in bbox_cells csv file
    - Add number of basestations as a parameter
"""

#%%
import os
from typing import List, Dict, Tuple
import pandas as pd
import random
import numpy as np
import time
from dataclasses import dataclass
from shapely.geometry import Point
from deepmimo.pipelines.utils.osm_utils import (
    get_buildings, is_point_clear_of_buildings, find_nearest_clear_location,
    SEARCH_RADIUS, VALIDATION_RADIUS
)

# Constants
MAX_VALIDATION_ATTEMPTS: int = 3  # max validation retries
DELTA_LAT = 0.003  # delta degress in latitude (for bounding box size)
DELTA_LON = 0.003  # delta degress in longitude (for bounding box size)

@dataclass
class BoundingBox:
    """Class to store bounding box information."""
    minlat: float
    minlon: float
    maxlat: float
    maxlon: float
    bs_lat: float
    bs_lon: float

    def to_dict(self) -> Dict[str, str]:
        """Convert bounding box to dictionary format."""
        return {
            'minlat': f"{self.minlat:.6f}",
            'minlon': f"{self.minlon:.6f}",
            'maxlat': f"{self.maxlat:.6f}",
            'maxlon': f"{self.maxlon:.6f}",
            'bs': f"{self.bs_lat:.6f}, {self.bs_lon:.6f}"
        }

def validate_and_adjust_point(lat: float, lon: float) -> Tuple[float, float, bool]:
    """Validate point and adjust until a suitable location away from buildings is found.
    
    Makes multiple attempts to find a location that maintains minimum distance from buildings.
    Gives up after MAX_VALIDATION_ATTEMPTS.
    
    Args:
        lat (float): Initial latitude to check
        lon (float): Initial longitude to check
        
    Returns:
        Tuple[float, float, bool]: Tuple containing:
            - latitude of valid location (or last attempt)
            - longitude of valid location (or last attempt)
            - True if location is valid, False if no valid location found
    """
    for attempt in range(MAX_VALIDATION_ATTEMPTS):
        buildings = get_buildings(lat, lon, VALIDATION_RADIUS)
        point = Point(lon, lat)
        
        if is_point_clear_of_buildings(point, buildings):
            return lat, lon, True
        
        # Find new location clear of buildings
        new_lat, new_lon = find_nearest_clear_location(lat, lon, buildings)
        
        # Verify the new location with fresh data
        verify_buildings = get_buildings(new_lat, new_lon, VALIDATION_RADIUS)
        if is_point_clear_of_buildings(Point(new_lon, new_lat), verify_buildings):
            return new_lat, new_lon, True
        
        lat, lon = new_lat, new_lon
    
    return lat, lon, False

def generate_bounding_boxes(coords_array: np.ndarray) -> List[Dict[str, str]]:
    """Generate bounding boxes with guaranteed safe BS locations.
    
    Args:
        coords_array (np.ndarray): Array of city coordinates
        
    Returns:
        List[Dict[str, str]]: List of bounding boxes with valid BS locations
    """
    valid_boxes: List[Dict[str, str]] = []
    skipped = 0
    
    for box_idx, coords in enumerate(coords_array):
        city_lat, city_lon = coords
        
        # First check if there are any significant buildings in the area
        buildings = get_buildings(city_lat, city_lon, SEARCH_RADIUS)
        if not buildings:
            skipped += 1
            continue
        
        # Find and validate safe BS location
        bs_lat, bs_lon, is_valid = validate_and_adjust_point(city_lat, city_lon)
        
        if not is_valid:
            print(f"Warning: Could not find safe location for box {box_idx}, skipping")
            skipped += 1
            continue
        
        # Define bounding box
        bbox = BoundingBox(
            minlat=city_lat - DELTA_LAT/2,
            maxlat=city_lat + DELTA_LAT/2,
            minlon=city_lon - DELTA_LON/2,
            maxlon=city_lon + DELTA_LON/2,
            bs_lat=bs_lat,
            bs_lon=bs_lon
        )
        
        valid_boxes.append(bbox.to_dict())
        
        # Be kind to the OSM server
        time.sleep(1)
    
    print(f"Generated {len(valid_boxes)} valid boxes, skipped {skipped} boxes with no buildings")
    return valid_boxes

#%%
if __name__ == "__main__":
    """Main function to generate and save bounding boxes."""
    random.seed(42)
    
    # Load city coordinates
    cities = pd.read_csv("./dev/worldcities.csv") 
    urban_cities = cities[cities['population'] > 5000000]
    city_coords = urban_cities[['lat', 'lng']].values

    # Generate bounding boxes
    bounding_boxes = generate_bounding_boxes(city_coords[:1])

    # Save to CSV
    if False:
        df = pd.DataFrame(bounding_boxes)
        df.to_csv("./dev/bounding_boxes.csv", index=False)
        print(f"Saved {len(bounding_boxes)} valid bounding boxes to bounding_boxes.csv")
