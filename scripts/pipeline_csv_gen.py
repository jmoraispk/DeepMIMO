"""


TODO:
    - Configure cell size to be ~80m longer in x and y compared to NY. Maybe 200 x 400m. (2x4)

"""
#%%
import os
import pandas as pd
import random
import requests
import numpy as np
import time
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from math import radians, sin, cos, sqrt, atan2

random.seed(42)

# Constants
EARTH_RADIUS = 6371000  # meters
DEGREE_TO_METER = 111320  # approx. meters per degree at equator
MIN_DISTANCE_FROM_BUILDING = 2  # meters
SEARCH_RADIUS = 150  # meters for building checks
SPIRAL_STEP = 5  # meters between test points
MAX_SPIRAL_RADIUS = 100  # meters maximum search radius
VALIDATION_RADIUS = 50  # meters for final validation
MAX_VALIDATION_ATTEMPTS = 3  # max validation retries
MIN_BUILDING_AREA = 25  # sq meters (ignore small buildings)

#%%  

# Provided manually-selected city coordinates
coordinates = [(35.714261535059016, 139.79669047744733)]

# Create a DataFrame
df_cities = pd.DataFrame(coordinates, columns=['lat', 'lng'])

# Add a placeholder population column (optional, to mimic worldcities.csv)
df_cities['population'] = 10000000

# Save to CSV
df_cities.to_csv('custom_cities.csv', index=False)
print(f"Generated 'custom_cities.csv' with {len(coordinates)} coordinates.")

#%%
"""Extract n bounding boxes from the worldcities.csv file and save them in an InSite pipeline-compatible format.

The script:
1. Reads city coordinates from worldcities.csv
2. For each coordinate, generates a bounding box
3. Checks if the bounding box contains any buildings
4. Validates that the BS coordinate is not on a building footprint
5. If BS is on a building, finds a nearby coordinate outside any building
6. Only saves scenarios that have at least one building and valid BS location
7. Outputs results to a CSV file with updated BS coordinates
"""

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * EARTH_RADIUS * atan2(sqrt(a), sqrt(1-a))

def meter_to_degree(meters, latitude):
    """Convert meters to approximate degrees at given latitude."""
    return meters / (DEGREE_TO_METER * cos(radians(latitude)))

def get_buildings(lat, lon, radius=SEARCH_RADIUS):
    """Get all significant buildings in the area as Shapely polygons."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      way["building"](around:{radius},{lat},{lon});
      relation["building"](around:{radius},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': query}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"OSM query failed: {e}")
        return []

    buildings = []
    nodes_cache = {}
    
    # Cache all nodes first
    for element in data['elements']:
        if element['type'] == 'node':
            nodes_cache[element['id']] = (element['lon'], element['lat'])
    
    # Process ways (buildings)
    for element in data['elements']:
        if element['type'] == 'way' and 'tags' in element and 'building' in element['tags']:
            nodes = []
            for node_id in element.get('nodes', []):
                if node_id in nodes_cache:
                    nodes.append(nodes_cache[node_id])
            
            if len(nodes) >= 3:  # Need at least 3 points for a polygon
                try:
                    polygon = Polygon(nodes)
                    if polygon.is_valid and polygon.area > MIN_BUILDING_AREA/(DEGREE_TO_METER**2):
                        # Add buffer to account for OSM inaccuracies
                        buildings.append(polygon.buffer(0.00002))  # ~2.2m buffer
                except:
                    continue
    
    return buildings

def is_point_safe(point, buildings):
    """Check if point is safely outside all buildings with margin."""
    if not buildings:
        return True
    
    buffer_degrees = meter_to_degree(MIN_DISTANCE_FROM_BUILDING, point.y)
    
    for building in buildings:
        if building.distance(point) < buffer_degrees:
            return False
    return True

def find_safe_point(original_lat, original_lon, buildings):
    """Find nearest safe point outside buildings using multiple strategies."""
    original_point = Point(original_lon, original_lat)
    
    # Strategy 1: Check if original point is already safe
    if is_point_safe(original_point, buildings):
        return original_lat, original_lon
    
    # Strategy 2: Move directly away from nearest building edge
    if buildings:
        nearest_building = min(buildings, key=lambda b: b.distance(original_point))
        nearest_pt = nearest_points(original_point, nearest_building)[1]
        
        # Calculate direction away from building
        dx = original_point.x - nearest_pt.x
        dy = original_point.y - nearest_pt.y
        dist = sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Move MIN_DISTANCE + 3m away for safety
            scale = (MIN_DISTANCE_FROM_BUILDING + 3) / (dist * DEGREE_TO_METER)
            new_lon = original_point.x + dx * scale
            new_lat = original_point.y + dy * scale
            new_point = Point(new_lon, new_lat)
            
            if is_point_safe(new_point, buildings):
                return new_lat, new_lon
    
    # Strategy 3: Spiral search pattern
    for distance in np.arange(SPIRAL_STEP, MAX_SPIRAL_RADIUS, SPIRAL_STEP):
        points_to_test = max(8, min(36, int(2*np.pi*distance/SPIRAL_STEP)))
        for angle in np.linspace(0, 2*np.pi, points_to_test, endpoint=False):
            offset_lat = meter_to_degree(distance * sin(angle), original_lat)
            offset_lon = meter_to_degree(distance * cos(angle), original_lat)
            test_lat = original_lat + offset_lat
            test_lon = original_lon + offset_lon
            test_point = Point(test_lon, test_lat)
            
            if is_point_safe(test_point, buildings):
                return test_lat, test_lon
    
    # Final strategy: Random walk with increasing distance
    for attempt in range(1, 6):
        distance = SPIRAL_STEP * attempt
        angle = random.uniform(0, 2*np.pi)
        test_lat = original_lat + meter_to_degree(distance * sin(angle), original_lat)
        test_lon = original_lon + meter_to_degree(distance * cos(angle), original_lat)
        test_point = Point(test_lon, test_lat)
        
        if is_point_safe(test_point, buildings):
            return test_lat, test_lon
    
    # Ultimate fallback: move 25m north
    return original_lat + meter_to_degree(25, original_lat), original_lon

def validate_and_adjust_point(lat, lon):
    """Validate point and adjust until safe or give up after MAX_ATTEMPTS."""
    for attempt in range(MAX_VALIDATION_ATTEMPTS):
        buildings = get_buildings(lat, lon, VALIDATION_RADIUS)
        point = Point(lon, lat)
        
        if is_point_safe(point, buildings):
            return lat, lon, True
        
        # Find new safe point using these buildings
        new_lat, new_lon = find_safe_point(lat, lon, buildings)
        
        # Verify the new point with fresh data
        verify_buildings = get_buildings(new_lat, new_lon, VALIDATION_RADIUS)
        if is_point_safe(Point(new_lon, new_lat), verify_buildings):
            return new_lat, new_lon, True
        
        lat, lon = new_lat, new_lon
    
    return lat, lon, False

def generate_bounding_boxes(n, city_coords):
    """Generate bounding boxes with guaranteed safe BS locations."""
    valid_boxes = []
    skipped = 0
    
    for k in range(n):
        city_lat, city_lon = random.choice(city_coords)
        
        # First check if there are any significant buildings in the area
        buildings = get_buildings(city_lat, city_lon, SEARCH_RADIUS)
        if not buildings:
            skipped += 1
            continue
        
        # Define bounding box
        lat_diff = 0.003
        lon_diff = 0.003
        minlat = city_lat - lat_diff/2
        maxlat = city_lat + lat_diff/2
        minlon = city_lon - lon_diff/2
        maxlon = city_lon + lon_diff/2
        
        # Find and validate safe BS location
        bs_lat, bs_lon, is_valid = validate_and_adjust_point(city_lat, city_lon)
        
        if not is_valid:
            print(f"Warning: Could not find safe location for box {k}, skipping")
            skipped += 1
            continue
        
        valid_boxes.append({
            'minlat': f"{minlat:.6f}",
            'minlon': f"{minlon:.6f}",
            'maxlat': f"{maxlat:.6f}",
            'maxlon': f"{maxlon:.6f}",
            'bs': f"{bs_lat:.6f}, {bs_lon:.6f}"
        })
        
        # Be kind to the OSM server
        time.sleep(1.5)
    
    print(f"Generated {len(valid_boxes)} valid boxes, skipped {skipped} boxes with no buildings")
    return valid_boxes

# Load city coordinates
cities = pd.read_csv("worldcities.csv")
urban_cities = cities[cities['population'] > 5000000]
city_coords = urban_cities[['lat', 'lng']].values

# Generate bounding boxes
n = 1500
bounding_boxes = generate_bounding_boxes(n, city_coords)

# Save to CSV
if bounding_boxes:
    df = pd.DataFrame(bounding_boxes)
    df.to_csv("bounding_boxes.csv", index=False)
    print(f"Saved {len(bounding_boxes)} valid bounding boxes to bounding_boxes.csv")
else:
    print("No valid bounding boxes were generated")
