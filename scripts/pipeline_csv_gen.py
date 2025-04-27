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
    - Add number of basestations as a parameter (placing them equaly spaced in the bbox)
    - Add parameter to set BS positioning when lat&lon intersects with building
        - Currently, BS is placed on the nearest point outside the building
        - Alternative: Place BS on top of the building
    - Add parameter to set BS height

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
DEFAULT_BS_HEIGHT = 10.0  # default BS height in meters

@dataclass
class ScenarioBboxInfo:
    """Class to store scenario information including bounding box and multiple BS parameters."""
    name: str
    minlat: float
    minlon: float
    maxlat: float
    maxlon: float
    bs_lats: List[float]  # List of BS latitudes
    bs_lons: List[float]  # List of BS longitudes
    bs_heights: List[float]  # List of BS heights

    def to_dict(self) -> Dict[str, str]:
        """Convert scenario info to dictionary format."""
        # Convert lists of coordinates to comma-separated strings
        bs_lats_str = ",".join([f"{lat:.8f}" for lat in self.bs_lats])
        bs_lons_str = ",".join([f"{lon:.8f}" for lon in self.bs_lons])
        bs_heights_str = ",".join([f"{h:.1f}" for h in self.bs_heights])
        
        return {
            'name': self.name,
            'min_lat': f"{self.minlat:.8f}",
            'min_lon': f"{self.minlon:.8f}",
            'max_lat': f"{self.maxlat:.8f}",
            'max_lon': f"{self.maxlon:.8f}",
            'bs_lat': bs_lats_str,
            'bs_lon': bs_lons_str,
            'bs_height': bs_heights_str
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

def generate_bounding_boxes(city_data: pd.DataFrame) -> List[Dict[str, str]]:
    """Generate bounding boxes with guaranteed safe BS locations.
    
    Args:
        city_data (pd.DataFrame): DataFrame containing city information
        
    Returns:
        List[Dict[str, str]]: List of bounding boxes with valid BS locations
    """
    valid_boxes: List[Dict[str, str]] = []
    skipped = 0
    NUM_BS = 3  # Number of base stations per scenario
    
    for box_idx, (_, city) in enumerate(city_data.iterrows()):
        city_lat, city_lon = city['lat'], city['lng']
        city_name = city['city'].lower().replace(' ', '')
        
        # First check if there are any significant buildings in the area
        buildings = get_buildings(city_lat, city_lon, SEARCH_RADIUS)
        if not buildings:
            print(f"Could not fetch buildings for {city['city']}, skipping")
            skipped += 1
            continue
        
        # Find multiple BS locations
        bs_lats, bs_lons = [], []
        bs_heights = []
        
        for _ in range(NUM_BS):
            # Add some random offset to spread out BS locations
            offset_lat = random.uniform(-DELTA_LAT/4, DELTA_LAT/4)
            offset_lon = random.uniform(-DELTA_LON/4, DELTA_LON/4)
            test_lat = city_lat + offset_lat
            test_lon = city_lon + offset_lon
            
            # Find and validate BS location
            bs_lat, bs_lon, is_valid = validate_and_adjust_point(test_lat, test_lon)
            
            if not is_valid:
                print(f"Warning: Could not find suitable BS location for {city['city']}, BS {len(bs_lats)+1}")
                continue
                
            bs_lats.append(bs_lat)
            bs_lons.append(bs_lon)
            bs_heights.append(DEFAULT_BS_HEIGHT)
        
        if not bs_lats:  # Skip if no valid BS locations found
            print(f"Warning: No valid BS locations found for {city['city']}, skipping")
            skipped += 1
            continue
        
        # Define bounding box
        bbox = ScenarioBboxInfo(
            name=f"city_{box_idx}_{city_name}_3p5",
            minlat=city_lat - DELTA_LAT/2,
            maxlat=city_lat + DELTA_LAT/2,
            minlon=city_lon - DELTA_LON/2,
            maxlon=city_lon + DELTA_LON/2,
            bs_lats=bs_lats,
            bs_lons=bs_lons,
            bs_heights=bs_heights
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

    # Generate bounding boxes
    bounding_boxes = generate_bounding_boxes(urban_cities[:1])

    # Save to CSV
    df = pd.DataFrame(bounding_boxes)
    print("DataFrame columns:", df.columns)  # Debug print
    df.to_csv("./dev/bounding_boxes.csv", index=False)
    print(f"Saved {len(bounding_boxes)} valid bounding boxes to bounding_boxes.csv")
