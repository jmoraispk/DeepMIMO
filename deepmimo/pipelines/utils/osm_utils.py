"""OpenStreetMap utilities for querying and validating building data.

This module provides functions for interacting with OpenStreetMap data,
specifically for querying building footprints and validating point locations
with respect to buildings. It includes functionality for:

- Querying building footprints from OpenStreetMap
- Checking if points are clear of building footprints
- Finding locations away from buildings
- Handling building polygon geometries

Constants:
    MIN_DISTANCE_FROM_BUILDING (float): Minimum distance required from buildings in meters
    SEARCH_RADIUS (float): Default radius for building searches in meters
    VALIDATION_RADIUS (float): Radius for validating point safety in meters
    SPIRAL_STEP (float): Step size for spiral search pattern in meters
    MAX_SPIRAL_RADIUS (float): Maximum radius for spiral search in meters
    MIN_BUILDING_AREA (float): Minimum building area to consider in square meters
    DEGREE_TO_METER (float): Conversion factor from degrees to meters at equator
"""

import requests
import numpy as np
from typing import List, Tuple, Dict
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from math import sin, cos, pi
from .geo_utils import meter_to_degree

# Constants
MIN_DISTANCE_FROM_BUILDING = 2  # meters
SEARCH_RADIUS = 150  # meters for building checks
VALIDATION_RADIUS = 50  # meters for final validation
SPIRAL_STEP = 5  # meters between test points
MAX_SPIRAL_RADIUS = 100  # meters maximum search radius
MIN_BUILDING_AREA = 25  # sq meters (ignore small buildings)
DEGREE_TO_METER = 111320  # approx. meters per degree at equator

def get_buildings(lat: float, lon: float, radius: float = SEARCH_RADIUS) -> List[Polygon]:
    """Get all significant buildings in the area as Shapely polygons.
    
    Args:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        radius (float): Search radius in meters, defaults to SEARCH_RADIUS
        
    Returns:
        List[Polygon]: List of building polygons with a 2.2m buffer
    """
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

    buildings: List[Polygon] = []
    nodes_cache: Dict[int, Tuple[float, float]] = {}
    
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

def is_point_clear_of_buildings(point: Point, buildings: List[Polygon]) -> bool:
    """Check if point maintains minimum distance from all building footprints.
    
    Args:
        point (Point): Point to check
        buildings (List[Polygon]): List of building polygons
        
    Returns:
        bool: True if point maintains minimum distance from all buildings
        
    Note:
        The minimum distance is defined by MIN_DISTANCE_FROM_BUILDING constant
    """
    if not buildings:
        return True
    
    buffer_degrees = meter_to_degree(MIN_DISTANCE_FROM_BUILDING, point.y)
    
    for building in buildings:
        if building.distance(point) < buffer_degrees:
            return False
    return True

def find_nearest_clear_location(original_lat: float, original_lon: float, buildings: List[Polygon]) -> Tuple[float, float]:
    """Find nearest location that maintains minimum distance from all buildings.
    
    Uses multiple strategies to find a suitable location:
    1. Checks if original point is already clear of buildings
    2. Moves away from nearest building edge
    3. Uses spiral search pattern
    4. Uses random walk with increasing distance
    5. Falls back to moving north if all else fails
    
    Args:
        original_lat (float): Original latitude
        original_lon (float): Original longitude
        buildings (List[Polygon]): List of building polygons
        
    Returns:
        Tuple[float, float]: Tuple of (latitude, longitude) for location clear of buildings
    """
    original_point = Point(original_lon, original_lat)
    
    # Strategy 1: Check if original point is already clear
    if is_point_clear_of_buildings(original_point, buildings):
        return original_lat, original_lon
    
    # Strategy 2: Move directly away from nearest building edge
    if buildings:
        nearest_building = min(buildings, key=lambda b: b.distance(original_point))
        nearest_pt = nearest_points(original_point, nearest_building)[1]
        
        # Calculate direction away from building
        dx = original_point.x - nearest_pt.x
        dy = original_point.y - nearest_pt.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Move MIN_DISTANCE + 3m away for safety
            scale = (MIN_DISTANCE_FROM_BUILDING + 3) / (dist * DEGREE_TO_METER)
            new_lon = original_point.x + dx * scale
            new_lat = original_point.y + dy * scale
            new_point = Point(new_lon, new_lat)
            
            if is_point_clear_of_buildings(new_point, buildings):
                return new_lat, new_lon
    
    # Strategy 3: Spiral search pattern
    for distance in np.arange(SPIRAL_STEP, MAX_SPIRAL_RADIUS, SPIRAL_STEP):
        points_to_test = max(8, min(36, int(2*pi*distance/SPIRAL_STEP)))
        for angle in np.linspace(0, 2*pi, points_to_test, endpoint=False):
            offset_lat = meter_to_degree(distance * sin(angle), original_lat)
            offset_lon = meter_to_degree(distance * cos(angle), original_lat)
            test_lat = original_lat + offset_lat
            test_lon = original_lon + offset_lon
            test_point = Point(test_lon, test_lat)
            
            if is_point_clear_of_buildings(test_point, buildings):
                return test_lat, test_lon
    
    # Final strategy: Random walk with increasing distance
    for attempt in range(1, 6):
        distance = SPIRAL_STEP * attempt
        angle = np.random.uniform(0, 2*pi)
        test_lat = original_lat + meter_to_degree(distance * sin(angle), original_lat)
        test_lon = original_lon + meter_to_degree(distance * cos(angle), original_lat)
        test_point = Point(test_lon, test_lat)
        
        if is_point_clear_of_buildings(test_point, buildings):
            return test_lat, test_lon
    
    # Ultimate fallback: move 25m north
    return original_lat + meter_to_degree(25, original_lat), original_lon 