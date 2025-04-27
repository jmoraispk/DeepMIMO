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
import matplotlib.pyplot as plt
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

def generate_uniform_positions(city_lat: float, city_lon: float, num_bs: int) -> List[Tuple[float, float]]:
    """Generate uniformly spaced positions in the bounding box.
    
    For different numbers of BS:
    - 1 BS: Center
    - 2 BS: Diagonal corners
    - 3 BS: Triangle formation
    - 4 BS: Square formation
    - 5+ BS: Grid formation with remaining BS in center
    
    Args:
        city_lat (float): Center latitude
        city_lon (float): Center longitude
        num_bs (int): Number of base stations
        
    Returns:
        List[Tuple[float, float]]: List of (lat, lon) positions
    """
    # Calculate box boundaries (80% of full box size to keep BS away from edges)
    margin = 0.1  # 10% margin from edges
    lat_range = DELTA_LAT * (1 - 2*margin)
    lon_range = DELTA_LON * (1 - 2*margin)
    min_lat = city_lat - lat_range/2
    min_lon = city_lon - lon_range/2
    
    positions = []
    
    if num_bs == 1:
        # Center position
        positions.append((city_lat, city_lon))
    
    elif num_bs == 2:
        # Diagonal corners
        positions.extend([
            (min_lat, min_lon),
            (min_lat + lat_range, min_lon + lon_range)
        ])
    
    elif num_bs == 3:
        # Triangle formation
        positions.extend([
            (min_lat, min_lon),  # Bottom left
            (min_lat, min_lon + lon_range),  # Bottom right
            (min_lat + lat_range, min_lon + lon_range/2)  # Top center
        ])
    
    elif num_bs == 4:
        # Square formation
        positions.extend([
            (min_lat, min_lon),  # Bottom left
            (min_lat, min_lon + lon_range),  # Bottom right
            (min_lat + lat_range, min_lon),  # Top left
            (min_lat + lat_range, min_lon + lon_range)  # Top right
        ])
    
    else:
        raise NotImplementedError(f"Number of BSs {num_bs} not supported. Maximum number of BSs is 4.")
    
    return positions

def generate_bs_positions(city_lat: float, city_lon: float, num_bs: int, buildings: List, algorithm: str = 'uniform') -> Tuple[List[float], List[float], List[float]]:
    """Generate and validate base station positions.
    
    Args:
        city_lat (float): City center latitude
        city_lon (float): City center longitude
        num_bs (int): Number of base stations to generate
        buildings (List): List of building polygons in the area
        algorithm (str, optional): BS positioning algorithm ('uniform' or 'random'). Defaults to 'uniform'.
        
    Returns:
        Tuple[List[float], List[float], List[float]]: Lists of BS latitudes, longitudes, and heights
    """
    bs_lats, bs_lons, bs_heights = [], [], []
    
    if algorithm == 'random':
        # Random positioning
        for _ in range(num_bs):
            offset_lat = random.uniform(-DELTA_LAT/4, DELTA_LAT/4)
            offset_lon = random.uniform(-DELTA_LON/4, DELTA_LON/4)
            test_lat = city_lat + offset_lat
            test_lon = city_lon + offset_lon
            
            bs_lat, bs_lon, is_valid = validate_and_adjust_point(test_lat, test_lon)
            if is_valid:
                bs_lats.append(bs_lat)
                bs_lons.append(bs_lon)
                bs_heights.append(DEFAULT_BS_HEIGHT)
    
    else:  # uniform positioning
        # Generate uniform positions
        positions = generate_uniform_positions(city_lat, city_lon, num_bs)
        
        # Validate and adjust each position
        for test_lat, test_lon in positions:
            bs_lat, bs_lon, is_valid = validate_and_adjust_point(test_lat, test_lon)
            if is_valid:
                bs_lats.append(bs_lat)
                bs_lons.append(bs_lon)
                bs_heights.append(DEFAULT_BS_HEIGHT)
    
    return bs_lats, bs_lons, bs_heights

def plot_scenario(bbox_info: Dict[str, str], save_dir: str = "./plots"):
    """Plot the bounding box and BS positions for a scenario.
    
    Args:
        bbox_info (Dict[str, str]): Dictionary containing bounding box information
        save_dir (str, optional): Directory to save plots. Defaults to "./plots".
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract coordinates
    minlat, maxlat = float(bbox_info['min_lat']), float(bbox_info['max_lat'])
    minlon, maxlon = float(bbox_info['min_lon']), float(bbox_info['max_lon'])
    bs_lats = [float(x) for x in bbox_info['bs_lat'].split(',')]
    bs_lons = [float(x) for x in bbox_info['bs_lon'].split(',')]
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot bounding box
    plt.plot([minlon, maxlon, maxlon, minlon, minlon], 
             [minlat, minlat, maxlat, maxlat, minlat], 
             'k-', label='Bounding Box')
    
    # Plot BS positions
    plt.scatter(bs_lons, bs_lats, c='red', marker='^', s=100, label='Base Stations')
    
    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f"Scenario: {bbox_info['name']}\n{len(bs_lats)} Base Stations")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.show()

def generate_bounding_boxes(city_data: pd.DataFrame, num_bs: int = 3, bs_algorithm: str = 'uniform', plot: bool = False) -> List[Dict[str, str]]:
    """Generate bounding boxes with guaranteed safe BS locations.
    
    Args:
        city_data (pd.DataFrame): DataFrame containing city information
        num_bs (int, optional): Number of base stations per scenario. Defaults to 3.
        bs_algorithm (str, optional): Algorithm for BS positioning ('uniform' or 'random'). Defaults to 'uniform'.
        plot (bool, optional): Whether to plot the scenarios. Defaults to False.
        
    Returns:
        List[Dict[str, str]]: List of bounding boxes with valid BS locations
    """
    valid_boxes: List[Dict[str, str]] = []
    skipped = 0
    
    for box_idx, (_, city) in enumerate(city_data.iterrows()):
        city_lat, city_lon = city['lat'], city['lng']
        city_name = city['city'].lower().replace(' ', '')
        
        # First check if there are any significant buildings in the area
        buildings = get_buildings(city_lat, city_lon, SEARCH_RADIUS)
        if not buildings:
            print(f"Could not fetch buildings for {city['city']}, skipping")
            skipped += 1
            continue
        
        # Generate BS positions
        bs_lats, bs_lons, bs_heights = generate_bs_positions(city_lat, city_lon, num_bs, buildings, bs_algorithm)
        
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
    
    # Set parameters
    plot_enabled = True  # Whether to generate plots
    
    # Load city coordinates
    cities = pd.read_csv("./dev/worldcities.csv") 
    urban_cities = cities[cities['population'] > 5000000]

    # Generate bounding boxes with 3 uniformly spaced base stations per scenario
    bounding_boxes = generate_bounding_boxes(urban_cities[:1], num_bs=2, bs_algorithm='uniform', plot=plot_enabled)

    # Save to CSV
    df = pd.DataFrame(bounding_boxes)
    print("DataFrame columns:", df.columns)  # Debug print
    df.to_csv("./dev/bounding_boxes.csv", index=False)
    print(f"Saved {len(bounding_boxes)} valid bounding boxes to bounding_boxes.csv")
    
    # Plot scenarios
    if False:
        print("Generating plots...")
        for bbox in bounding_boxes:
            plot_scenario(bbox)

            # CHCEK THIIISSS
