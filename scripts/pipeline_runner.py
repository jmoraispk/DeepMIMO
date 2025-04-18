"""
Steps to run a pipeline:

1. pip install deepmimo

2. Setup Blender
   That's how we currently fetch OSM data and convert it to a format
	that can be used by a ray tracer.

3. Install dependencies
   - a ray tracer: 
	- Wireless InSite (3.3.x or 4.0.x)
	  + pip install lxml plyfile
	- Sionna (0.19.1)
	  + pip install ...
   - pip install utm

4. Change parameters in params.csv and in this file

5. Run the pipeline
   - python pipeline_runner.py

   
448d2ed8855745f9a215b59d32bd90bc60650173b6408609
"""


#%% Imports
import pandas as pd
import os
import numpy as np
from deepmimo.pipelines.utils.pipeline_utils import call_blender, get_origin_coords, load_params_from_row

# import sys
# sys.path.append("C:/Users/jmora/Documents/GitHub/DeepMIMO")
import deepmimo as dm  # type: ignore

from deepmimo.pipelines.TxRxPlacement import gen_rx_grid, gen_tx_pos

from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite
# from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

# Paths
# Windows versions
OSM_ROOT = "C:/Users/jmora/Downloads/osm_root"
BLENDER_PATH = "C:/Program Files/Blender Foundation/Blender 3.6/blender-launcher.exe"

# Linux versions
# OSM_ROOT = "/mnt/c/Users/jmora/Downloads/osm_root"
# BLENDER_PATH = "/home/joao/blender-3.6.0-linux-x64/blender"

# Wireless InSite
WI_ROOT = "C:/Program Files/Remcom/Wireless InSite 4.0.0"
WI_EXE = os.path.join(WI_ROOT, "bin/calc/wibatch.exe")
WI_MAT = os.path.join(WI_ROOT, "materials")
WI_LIC = "C:/Users/jmora/Documents/GitHub/DeepMIMO/executables/wireless insite"
WI_VERSION = "4.0.1"

# Material paths
BUILDING_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Concrete 3.5 GHz.mtl")
ROAD_MATERIAL_PATH = os.path.join(WI_MAT, "Asphalt_1GHz.mtl")
TERRAIN_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Wet earth 3.5 GHz.mtl")

COUNTER = 112
#%% Step 1: (Optional) Generate CSV with GPS coordinates for map and basestation placement

print('not implemented yet')
# TODO:
# - Configure cell size to be ~80m longer in x and y compared to NY. Maybe 200 x 400m. (2x4)

#%% Step 2: Iterate over rows of CSV file to extract the map, create TX/RX positions, and run RT

df = pd.read_csv('./pipeline_dev/params.csv')

# GPU definition (e.g. for Sionna)
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

# Parameters
p = {
	# Scenario parameters (to be loaded from CSV)
	'name': None,
	'city': None,
	'min_lat': None,
	'min_lon': None,
	'max_lat': None,
	'max_lon': None,
	'bs_lats': None, 
	'bs_lons': None,
	'bs_heights': None,

	# User placement parameters
	'ue_height': 1.5,
	'grid_spacing': 1,
	'pos_prec': 4, # Decimal places for coordinates

	# Paths required by Wireless InSite
	'wi_exe': WI_EXE,
	'wi_lic': WI_LIC,
	'wi_version': WI_VERSION,
	'building_material': BUILDING_MATERIAL_PATH,
	'road_material': ROAD_MATERIAL_PATH,
	'terrain_material': TERRAIN_MATERIAL_PATH,

	# Sionna specific parameters
	'batch_size': 15,  # Number of users to compute at a time
	                   # Heuristic: 1.5 per GB of GPU VRAM, if using scattering, 
					   # else 5-10 users per GB
	'mat': 0,

	# Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
	'carrier_freq': 3.5e9,  # Hz
	'bandwidth': 10e6,  # Hz
	'max_reflections': 5,
	'max_paths': 10,
	'ray_spacing': 0.25,  # m
	'max_transmissions': 0,
	'max_diffractions': 0,
	'ds_enable': False,
	'ds_max_reflections': 2,
	'ds_max_transmissions': 0,
	'ds_max_diffractions': 1,
	'ds_final_interaction_only': True,
	'conform_to_terrain': False,  # Whether to conform the terrain to the ray tracing grid
	                              # (if True, positions have added the terrain height)

	# Blender specific parameters
	'auto_install_addons': True, # Whether to automatically install add-ons
}

for index, row in df.iterrows():
	print(f"\n{'='*50}\nSTARTING SCENARIO {index+1}/{len(df)}: {row['name']}\n{'='*50}")

	# RT Phase 1: Load GPS coordinates from CSV
	load_params_from_row(row, p)

	# RT Phase 2: Extract OSM data
	COUNTER += 1
	osm_folder = os.path.join(OSM_ROOT, row['name']) + f'_{COUNTER}'
	call_blender(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
			     osm_folder, # Output folder to the Blender script
				 BLENDER_PATH, 
				 outputs=['insite']) # List of outputs to generate
	p['origin_lat'], p['origin_lon'] = get_origin_coords(osm_folder)

	# RT Phase 3: Generate RX and TX positions
	rx_pos = gen_rx_grid(p)  # N x 3 (N ~ 100k)
	tx_pos = gen_tx_pos(p)   # M x 3 (M ~ 3)
	
	# Optional: Round positions (visually *way* better)
	rx_pos = np.round(rx_pos, p['pos_prec'])
	tx_pos = np.round(tx_pos, p['pos_prec'])

	# break
	# RT Phase 4: Run Wireless InSite ray tracing
	rt_path = raytrace_insite(osm_folder, tx_pos, rx_pos, **p)
	
	# break
	# rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)

	# RT Phase 5: Convert to DeepMIMO format
	dm.config('wireless_insite_version', WI_VERSION)
	dm.config('sionna_version', '0.19.1')
	scen_name = dm.convert(rt_path, overwrite=True)

	# RT Phase 6: Test Conversion
	dataset = dm.load(scen_name)[0]
	dataset.plot_coverage(dataset.los)
	dataset.plot_coverage(dataset.pwr[:, 0])
	break

#%% ROAD PROCESSING
# Test code for road processing
import numpy as np
from typing import Tuple, List, NamedTuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class RoadGeometry(NamedTuple):
    """Container for road geometry processing results."""
    face_vertices: List[Tuple[float, float, float]]  # 3D vertices of the face
    centroid_3d: np.ndarray  # 3D centroid
    centroid_2d: np.ndarray  # 2D centroid in the plane
    points_2d: np.ndarray    # 2D coordinates in the plane
    normal: np.ndarray       # Normal vector of the plane
    properties: dict         # Road-specific properties
    left_endpoints: np.ndarray   # Left endpoint coordinates (2D)
    right_endpoints: np.ndarray  # Right endpoint coordinates (2D)

def filter_close_points(points: np.ndarray, min_dist_threshold: float = 0.5) -> np.ndarray:
    """Filter out points that are too close to each other.
    
    Args:
        points: Array of points (N x 3 for 3D points)
        min_dist_threshold: Minimum distance between points in meters
        
    Returns:
        np.ndarray: Filtered points array
    """
    n_points = len(points)
    
    # Calculate all pairwise distances
    distances = np.zeros((n_points, n_points))
    for i in range(n_points):
        distances[i] = np.sqrt(np.sum((points - points[i])**2, axis=1))
    
    # Keep points that don't have any neighbors closer than threshold
    # (except themselves, hence the distances[i] > 0 condition)
    to_keep = []
    for i in range(n_points):
        min_dist = np.min(distances[i][distances[i] > 0])
        if min_dist >= min_dist_threshold:
            to_keep.append(i)
    
    return points[to_keep]

def detect_endpoints(points_2d: np.ndarray, n_endpoints: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Detect the endpoints of a road by finding extreme points in X direction.
    
    Args:
        points_2d: Array of 2D points (N x 2)
        n_endpoints: Number of points to consider as endpoints on each side
        
    Returns:
        tuple: (left_end_indices, right_end_indices) - Arrays of indices for left and right endpoints
    """
    # Find the extreme points in X direction
    x_sorted_indices = np.argsort(points_2d[:, 0])
    left_end_indices = x_sorted_indices[:n_endpoints]  # leftmost points
    right_end_indices = x_sorted_indices[-n_endpoints:]  # rightmost points
    
    # Print endpoint information
    print("\nEndpoint Analysis:")
    print("Left endpoints:", left_end_indices)
    print("Left endpoint coordinates:")
    print(points_2d[left_end_indices])
    print("\nRight endpoints:", right_end_indices)
    print("Right endpoint coordinates:")
    print(points_2d[right_end_indices])
    
    return left_end_indices, right_end_indices

def process_road_vertices(vertices: np.ndarray) -> RoadGeometry:
    """Process road vertices to find the best-fit plane and ordered vertices."""
    if len(vertices) < 3:
        return RoadGeometry([], None, None, None, None, {}, 
                          np.array([]), np.array([]))
        
    # Find the best-fit plane using SVD
    centroid_3d = np.mean(vertices, axis=0)
    centered_pts = vertices - centroid_3d
    _, _, vh = np.linalg.svd(centered_pts)
    
    # Normal vector is the last right singular vector
    normal = vh[2]
    
    # Get two orthogonal vectors in the plane
    basis1 = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    if np.allclose(basis1, 0):
        basis1 = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    
    # Project points onto their best-fit plane
    projected_vertices = vertices - np.outer(np.dot(centered_pts, normal), normal)
    
    # Get 2D coordinates in the plane
    centered_projected = projected_vertices - centroid_3d
    x_coords = np.dot(centered_projected, basis1)
    y_coords = np.dot(centered_projected, basis2)
    points_2d = np.column_stack([x_coords, y_coords])
    centroid_2d = np.mean(points_2d, axis=0)
    
    # Print point coordinates and distances
    print("\nPoint coordinates and distances:")
    print("Point ID  |     X     |     Y     | Min Distance")
    print("-" * 55)
    for i, point in enumerate(points_2d):
        # Calculate distances to all other points
        distances = np.sqrt(np.sum((points_2d - point)**2, axis=1))
        # Get minimum non-zero distance (exclude distance to self)
        min_dist = np.min(distances[distances > 0])
        print(f"{i:8d} | {point[0]:8.2f} | {point[1]:8.2f} | {min_dist:8.2f}")
    
    # Detect endpoints
    left_indices, right_indices = detect_endpoints(points_2d)
    left_endpoints = points_2d[left_indices]
    right_endpoints = points_2d[right_indices]
    
    def segment_intersects(p1, p2, q1, q2):
        """Check if line segments p1p2 and q1q2 intersect."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)
    
    def path_will_cross(new_point, path):
        """Check if adding new_point will create any crossing edges."""
        if len(path) < 2:
            return False
        # Check against all segments except the last one
        new_segment = (path[-1], new_point)
        for i in range(len(path)-2):
            if segment_intersects(new_segment[0], new_segment[1],
                                path[i], path[i+1]):
                return True
        return False
    
    # Find the two most extreme points in X direction for endpoints
    x_sorted_indices = np.argsort(points_2d[:, 0])
    left_end_indices = x_sorted_indices[:2]  # 2 leftmost points
    right_end_indices = x_sorted_indices[-2:]  # 2 rightmost points
    
    # Start with the leftmost point
    start_idx = left_end_indices[0]
    current_point = points_2d[start_idx]
    
    # Initialize path with the starting point
    n_points = len(points_2d)
    path = [current_point]
    used_points = np.zeros(n_points, dtype=bool)
    used_points[start_idx] = True
    
    # Force second point to be the other left endpoint
    second_idx = left_end_indices[1]
    path.append(points_2d[second_idx])
    used_points[second_idx] = True
    
    # Rest of the path finding remains similar, but with endpoint awareness
    while len(path) < n_points:
        current_point = path[-1]
        
        # Special case: if we have 2 points remaining and they're the right endpoints
        remaining_points = np.where(~used_points)[0]
        if len(remaining_points) == 2 and set(remaining_points) == set(right_end_indices):
            # Add the right endpoints in order of x-coordinate
            for idx in sorted(remaining_points, key=lambda i: points_2d[i][0]):
                path.append(points_2d[idx])
                used_points[idx] = True
            break
            
        # Normal path finding for middle points
        unused_mask = ~used_points
        candidates = points_2d[unused_mask]
        if len(candidates) == 0:
            break
            
        # Calculate vectors and distances
        vectors = candidates - current_point
        distances = np.sqrt(np.sum(vectors**2, axis=1))
        directions = vectors / distances[:, np.newaxis]
        
        # Get current direction
        if len(path) > 1:
            current_direction = path[-1] - path[-2]
            current_direction = current_direction / np.linalg.norm(current_direction)
        else:
            current_direction = directions[np.argmin(distances)]
        
        # Calculate angles
        dot_products = np.dot(directions, current_direction)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        
        # Score candidates
        scores = distances * (1.0 + angles)  # Simplified scoring
        
        # Select best candidate that doesn't create crossings
        candidate_indices = np.argsort(scores)
        next_point = None
        
        for idx in candidate_indices:
            candidate = candidates[idx]
            if not path_will_cross(candidate, path):
                next_point = candidate
                break
        
        if next_point is None:
            break
            
        # Add point to path
        path.append(next_point)
        point_idx = np.where(np.all(points_2d == next_point, axis=1))[0][0]
        used_points[point_idx] = True
    
    # Convert path back to indices
    ordered_indices = []
    for point in path:
        idx = np.where(np.all(points_2d == point, axis=1))[0][0]
        ordered_indices.append(idx)
    
    # Apply the ordering to the filtered points
    ordered_vertices = projected_vertices[ordered_indices]
    ordered_2d = points_2d[ordered_indices]
    
    # Convert to list of tuples format
    face_vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in ordered_vertices]
    
    # Calculate area and perimeter
    area = 0.0
    perimeter = 0.0
    for i in range(len(ordered_2d)):
        j = (i + 1) % len(ordered_2d)
        area += ordered_2d[i][0] * ordered_2d[j][1]
        area -= ordered_2d[j][0] * ordered_2d[i][1]
        perimeter += np.sqrt(np.sum((ordered_2d[j] - ordered_2d[i])**2))
    area = abs(area) / 2.0
    
    # Analyze road properties
    properties = validate_road_properties(vertices, ordered_2d)
    properties['area'] = area
    properties['perimeter'] = perimeter
    
    return RoadGeometry(face_vertices, centroid_3d, centroid_2d, ordered_2d, normal, properties, left_endpoints, right_endpoints)

def plot_3d_road(ax: plt.Axes, geometry: RoadGeometry, vertices: np.ndarray):
    """Plot 3D visualization of road processing.
    
    Args:
        ax: Matplotlib 3D axes to plot on
        geometry: Processed road geometry
        vertices: Original vertices
    """
    # Plot original vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c='blue', marker='o', label='Original Vertices')
    
    # Plot projected vertices and face
    face_array = np.array(geometry.face_vertices)
    ax.scatter(face_array[:, 0], face_array[:, 1], face_array[:, 2], 
              c='green', marker='s', label='Projected Vertices')
    
    # Plot centroid
    ax.scatter(geometry.centroid_3d[0], geometry.centroid_3d[1], geometry.centroid_3d[2], 
              c='red', marker='*', s=200, label='Centroid')
    
    # Plot face as a polygon
    face_poly = Poly3DCollection([face_array], alpha=0.2)
    face_poly.set_facecolor('blue')
    ax.add_collection3d(face_poly)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def plot_2d_road(ax: plt.Axes, geometry: RoadGeometry):
    """Plot 2D visualization of road processing with road properties."""
    points = geometry.points_2d
    
    # Plot regular points in blue
    ax.scatter(points[:, 0], points[:, 1], 
              c='blue', marker='o', label='Regular Points')
    
    # Add vertex numbers
    for i, (x, y) in enumerate(points):
        ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot endpoints in red
    ax.scatter(geometry.left_endpoints[:, 0], geometry.left_endpoints[:, 1],
              c='red', marker='o', label='Left Endpoints')
    ax.scatter(geometry.right_endpoints[:, 0], geometry.right_endpoints[:, 1],
              c='red', marker='o')
    
    # Plot 2D centroid
    ax.scatter(geometry.centroid_2d[0], geometry.centroid_2d[1], 
              c='green', marker='*', s=200, label='Centroid')
    
    # Plot edges with arrows to show direction
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        ax.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.9, (p2[1]-p1[1])*0.9,
                head_width=0.1, head_length=0.2, fc='g', ec='g', alpha=0.5)
    
    # Fill the enclosed area with semi-transparent color
    ax.fill(points[:, 0], points[:, 1], alpha=0.2, color='lightblue', label='Enclosed Area')
    
    ax.set_xlabel('X in plane')
    ax.set_ylabel('Y in plane')
    ax.axis('equal')
    
    ax.legend()

def validate_road_properties(vertices: np.ndarray, points_2d: np.ndarray) -> dict:
    """Analyze road-specific properties with enhanced curve detection."""
    properties = {}
    
    # Basic validation
    n_points = len(vertices)
    properties['valid_point_count'] = n_points >= 4
    properties['n_points'] = n_points
    
    if n_points < 4:
        return properties
    
    # Segment Analysis
    segments = np.diff(points_2d, axis=0)
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    
    # Calculate angles between consecutive segments
    segment_vectors = segments / segment_lengths[:, np.newaxis]
    dot_products = np.sum(segment_vectors[:-1] * segment_vectors[1:], axis=1)
    angle_changes = np.arccos(np.clip(dot_products, -1.0, 1.0))
    
    # Curvature Analysis (excluding endpoints)
    middle_angles = angle_changes[1:-1]  # Exclude first and last angle changes
    properties['max_angle_change'] = np.max(np.abs(middle_angles)) if len(middle_angles) > 0 else 0
    properties['total_curvature'] = np.sum(np.abs(middle_angles))
    properties['avg_curvature_per_segment'] = properties['total_curvature'] / max(1, len(middle_angles))
    
    # Classify road type
    properties['is_straight'] = properties['max_angle_change'] < np.pi/6  # < 30 degrees
    properties['is_curved'] = properties['total_curvature'] > np.pi/2  # > 90 degrees total
    properties['has_sharp_turns'] = properties['max_angle_change'] > np.pi/3  # > 60 degrees
    
    return properties

def check_path_intersections(points_2d: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Check if any non-adjacent segments in the path intersect.
    
    Args:
        points_2d: Array of 2D points ordered in a path
        
    Returns:
        list: List of tuples (i,j,k,l) where segment i->j intersects with k->l
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def segments_intersect(p1, p2, q1, q2):
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    
    intersections = []
    n_points = len(points_2d)
    
    # Check each pair of non-adjacent segments
    for i in range(n_points - 1):
        for j in range(i + 2, n_points - 1):  # Start from i+2 to skip adjacent segments
            if segments_intersect(points_2d[i], points_2d[i+1], 
                                points_2d[j], points_2d[j+1]):
                intersections.append((i, i+1, j, j+1))
    
    return intersections

def analyze_path_angles(points_2d: np.ndarray) -> np.ndarray:
    """Calculate angles at each point relative to the previous line segment.
    
    Args:
        points_2d: Array of 2D points ordered in a path
        
    Returns:
        np.ndarray: Array of angles in degrees at each point.
        Angle is measured as the deviation from a straight line:
        0° or 180° = straight line (no turn)
        90° = right angle turn
        Angles are in [0, 180].
    """
    n_points = len(points_2d)
    if n_points < 3:
        return np.zeros(n_points)
        
    # Calculate vectors between consecutive points, including from last to first
    vectors = np.zeros((n_points, 2))
    vectors[:-1] = np.diff(points_2d, axis=0)  # Regular vectors
    vectors[-1] = points_2d[0] - points_2d[-1]  # Vector from last to first point
    
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    # Calculate angles between consecutive vectors
    angles = np.zeros(n_points)
    for i in range(n_points):
        # For any point i, we want the vector arriving at i and the vector leaving i
        v1 = vectors[i-1]  # Vector arriving at point
        v2 = vectors[i]    # Vector leaving point
        
        # Calculate the angle between vectors using dot product
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)  # Clip to avoid numerical errors
        angle = np.degrees(np.arccos(dot_product))
        
        # The angle we want is either angle or 180-angle, whichever is smaller
        angles[i] = min(angle, 180 - angle)
    
    return angles

# Load road vertices
road_vertices = np.load('road_vertices_roads_3.npy')
print(f"\nLoaded road vertices: {len(road_vertices)} points")

# Filter close points
filtered_vertices = filter_close_points(road_vertices)
print(f"After filtering: {len(filtered_vertices)} points")

# Process the road geometry
geometry = process_road_vertices(filtered_vertices)

# Check for path intersections
intersections = check_path_intersections(geometry.points_2d)
if intersections:
    print("\nFound path intersections between segments:")
    for i, j, k, l in intersections:
        print(f"Segment {i}->{j} intersects with {k}->{l}")
else:
    print("\nNo path intersections found")

# Analyze angles
angles = analyze_path_angles(geometry.points_2d)
print("\nAngles at each vertex (deviation from straight line):")
print("Point |  Angle  | Note")
print("-" * 40)
for i, angle in enumerate(angles):
    note = ""
    if angle > 1:  # Only print angles > 1 degree
        if angle > 150:
            note = "hairpin turn!"
        elif angle > 120:
            note = "very sharp turn!"
        elif angle > 90:
            note = "sharp turn!"
        elif angle > 60:
            note = "moderate turn"
        elif angle > 30:
            note = "gentle turn"
        print(f"{i:5d} | {angle:7.2f}° | {note}")

# 2D visualization
f, ax = plt.subplots(dpi=100)
plot_2d_road(ax, geometry)
plt.tight_layout()
plt.show()

print("\nSuggestions for improvement:")
print("1. The path should be reordered to minimize deviations from straight lines")
print("2. Consider using a path optimization that:")
print("   - Starts from endpoints")
print("   - At each step, chooses the next point that creates the smallest angle deviation")
print("   - Avoids creating intersections")
print("3. Could use dynamic programming to find the optimal path that:")
print("   - Minimizes the angles at each vertex (prefers straight lines)")
print("   - Penalizes intersections heavily")
print("   - Enforces endpoint constraints")

# %% NEW CODE HERE! (extracting functions)

# Re-import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Function to calculate angle deviation
def calculate_angle_deviation(p1, p2, p3):
    """Calculate the deviation from a straight line at point p2.
    Returns angle in degrees, where:
    - 0° means the path p1->p2->p3 forms a straight line
    - 180° means the path doubles back on itself
    """
    if np.allclose(p1, p2) or np.allclose(p2, p3):
        return 180.0
    v1 = p2 - p1  # Vector from p1 to p2
    v2 = p3 - p2  # Vector from p2 to p3
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    return np.degrees(np.arccos(dot_product))

# Plot helper
def plot_points(points, path=None, title=""):
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    for i, (x, y) in enumerate(points):
        plt.text(x + 1, y + 1, str(i), fontsize=9)
    if path:
        for i in range(len(path) - 1):
            p1, p2 = points[path[i]], points[path[i+1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Intersection check for line segments
def segments_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

# Held-Karp TSP with angle penalty + intersection check
def tsp_held_karp_no_intersections(points):
    n = len(points)
    C = {}
    
    for k in range(1, n):
        dist = np.linalg.norm(points[0] - points[k])
        C[(1 << k, k)] = (dist, [0, k])

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    prev_cost, prev_path = C.get((prev_bits, m), (float('inf'), []))
                    if not prev_path:
                        continue
                    # Check for intersections
                    new_seg = (points[m], points[k])
                    intersects = False
                    for i in range(len(prev_path) - 2):
                        a, b = prev_path[i], prev_path[i + 1]
                        if segments_intersect(points[a], points[b], new_seg[0], new_seg[1]):
                            intersects = True
                            break
                    if intersects:
                        continue
                    angle_cost = calculate_angle_deviation(points[prev_path[-2]], points[m], points[k]) if len(prev_path) > 1 else 0
                    cost = prev_cost + np.linalg.norm(points[m] - points[k]) + angle_cost
                    res.append((cost, prev_path + [k]))
                if res:
                    C[(bits, k)] = min(res)

    bits = (1 << n) - 2
    res = []
    for k in range(1, n):
        if (bits, k) not in C:
            continue
        cost, path = C[(bits, k)]
        new_seg = (points[k], points[0])
        intersects = False
        for i in range(len(path) - 2):
            a, b = path[i], path[i + 1]
            if segments_intersect(points[a], points[b], new_seg[0], new_seg[1]):
                intersects = True
                break
        if intersects:
            continue
        angle_cost = calculate_angle_deviation(points[path[-2]], points[k], points[0])
        final_cost = cost + np.linalg.norm(points[k] - points[0]) + angle_cost
        res.append((final_cost, path + [0]))

    return min(res) if res else (float('inf'), [])


def trim_points(points, max_points=14):
    """ Deletes the point that is closest to the average of all points. """
    while len(points) > max_points:
        dists = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        np.fill_diagonal(dists, np.inf)
        _, j = np.unravel_index(np.argmin(dists), dists.shape)
        points = np.delete(points, j, axis=0)
    return points

#%% FILTERING WITH PROTECTED POINTS

def filter_with_protected(points, protected_indices, target_count):
    assert target_count >= len(protected_indices), "Target count must be >= number of protected points"
    
    protected_indices = set(protected_indices)
    total_indices = list(range(len(points)))
    removable_indices = [i for i in total_indices if i not in protected_indices]

    # Build full distance matrix
    dists = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    np.fill_diagonal(dists, np.inf)  # avoid self-pairs

    # While we have too many points, remove the closest removable point
    while len(protected_indices) + len(removable_indices) > target_count:
        # Consider only distances between removable points
        min_dist = float('inf')
        to_remove = None
        for i in removable_indices:
            for j in removable_indices:
                if i >= j:
                    continue
                if dists[i][j] < min_dist:
                    min_dist = dists[i][j]
                    to_remove = i if np.mean(dists[i]) < np.mean(dists[j]) else j
        removable_indices.remove(to_remove)

    # Final set of indices to keep
    final_indices = sorted(list(protected_indices) + removable_indices)
    return points[final_indices], final_indices


def trim_points_protected(points, protected_indices, max_points=14):
    """ Deletes points while preserving protected indices until max_points is reached. """
    protected_indices = set(protected_indices)
    assert len(points) >= max_points >= len(protected_indices), "max_points must be >= number of protected points"
    
    while len(points) > max_points:
        # Calculate pairwise distances
        dists = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Mask out protected indices
        for idx in protected_indices:
            dists[idx, :] = np.inf
            dists[:, idx] = np.inf
            
        # Find closest pair of non-protected points
        _, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Update protected indices after deletion
        new_protected = set()
        for idx in protected_indices:
            if idx < j:
                new_protected.add(idx)
            else:
                new_protected.add(idx - 1)
        protected_indices = new_protected
        
        points = np.delete(points, j, axis=0)
        
    return points


#%% tsp_held_karp_custom

# Provided list of vertices
points_raw = np.array([
    [26.87, -37.58],
    [41.20, -77.20],
    [27.76, -41.83],
    [0.90, 12.30],
    [-4.58, 7.95],
    [-76.82, 107.70],
    [-67.64, 83.88],
    [30.11, -26.84],
    [-85.58, 107.70],
    [35.19, -77.20],
    [24.14, -18.42],
    [18.49, -22.55],
    [-62.15, 88.22],
    [34.61, -40.39],
    [23.92, -30.21],
    [33.59, -35.54]
])


# points_raw = np.array([
#     [-85.58, 107.70], 
#     [-76.82, 107.70],
#     [26.87, -37.58],
#     [41.20, -77.20],
#     [27.76, -41.83],
#     [0.90, 12.30],
#     [-4.58, 7.95],
#     [-67.64, 83.88],
#     [30.11, -26.84],
#     [35.19, -77.20],
#     [24.14, -18.42],
#     [18.49, -22.55],
#     [-62.15, 88.22],
#     [34.61, -40.39],
#     [23.92, -30.21],
#     [33.59, -35.54]
# ])

# Trim points if more than 15 (keep only farthest-spread points)
def trim_points(points, max_points=14):
    while len(points) > max_points:
        dists = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        np.fill_diagonal(dists, np.inf)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        points = np.delete(points, j, axis=0)
    return points

# Plot before trimming
plot_points(points_raw, title="Original Points")

# Trim if needed
if len(points_raw) > 15:
    points = trim_points(points_raw)
else:
    points = points_raw

# Plot after trimming
plot_points(points, title="Trimmed Points")

# Held-Karp TSP with angle penalty as cost
def tsp_held_karp_custom(points):
    n = len(points)
    C = {}

    # Initial state with custom cost (start at 0)
    for k in range(1, n):
        cost = np.linalg.norm(points[0] - points[k])
        C[(1 << k, k)] = (cost, [0, k])

    # DP computation
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    prev_cost, prev_path = C[(prev_bits, m)]
                    angle_cost = calculate_angle_deviation(points[prev_path[-2]], points[m], points[k]) if len(prev_path) > 1 else 0
                    cost = prev_cost + np.linalg.norm(points[m] - points[k]) + angle_cost
                    res.append((cost, prev_path + [k]))
                C[(bits, k)] = min(res)

    # Return to start
    bits = (1 << n) - 2
    res = []
    for k in range(1, n):
        cost, path = C[(bits, k)]
        final_cost = cost + np.linalg.norm(points[k] - points[0]) + calculate_angle_deviation(points[path[-2]], points[k], points[0])
        res.append((final_cost, path + [0]))

    return min(res)

# Run custom TSP
best_cost, best_path = tsp_held_karp_custom(points)

# Plot final path
plot_points(points, best_path, title="Optimal Path with Angle-Aware Cost")

# TODO: ADD 2 endpoints to the start of the path. (2 at the end are okay too. )
# TODO: Possibly use distance as cost IN ADDITION to angle.

#%% ANGLE-BASED TSP (WORKING VERY WELL!)

points_raw = np.array([
    [26.87,  -37.58],
    [41.20,  -77.20],
    [27.76,  -41.83],
    [0.90,    12.30],
    [-4.58,    7.95],
    [-76.82, 107.70],
    [-67.64,  83.88],
    [30.11,  -26.84], # less great if this is removed
    [-85.58, 107.70],
    [35.19,  -77.20]
])

# Run and plot
points = points_raw
# points = filtered_points
# plot_points(points, title="Input Points (10 total)")
best_cost, best_path = tsp_held_karp_no_intersections(points)
plot_points(points, best_path, title="Non-Intersecting Angle-Aware Optimal Path")

#%% FILTERING WITH PROTECTED POINTS (GOOD!!)

# Test with your full list of 16 points and 4 protected ones
points_raw = np.array([
    [26.87, -37.58],
    [41.20, -77.20],
    [27.76, -41.83],
    [0.90, 12.30],     # Protected
    [-4.58, 7.95],
    [-76.82, 107.70],
    [-67.64, 83.88],
    [30.11, -26.84],   # Protected
    [-85.58, 107.70],
    [35.19, -77.20],
    [24.14, -18.42],   # Protected
    [18.49, -22.55],
    [-62.15, 88.22],
    [34.61, -40.39],
    [23.92, -30.21],   # Protected
    [33.59, -35.54]
])

# protected_indices = [3, 7, 10, 14]
protected_indices = [8, 5, 2, 9]
filtered_points, kept_indices = filter_with_protected(points_raw, protected_indices, target_count=10)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Before filtering
axs[0].scatter(points_raw[:, 0], points_raw[:, 1], color='blue')
axs[0].set_title(f"Input Points ({points_raw.shape[0]} total)")

# After filtering
axs[1].scatter(filtered_points[:, 0], filtered_points[:, 1], color='blue')
axs[1].set_title(f"Filtered Points ({filtered_points.shape[0]} total)")

plt.tight_layout()
plt.show()

#%% ENFORCED EDGES (Try without)

# Reuse points and enforced edges
points_raw = np.array([
    [26.87, -37.58],    # 0
    [41.20, -77.20],    # 1
    [27.76, -41.83],    # 2
    [0.90, 12.30],      # 3
    [-4.58, 7.95],      # 4
    [-76.82, 107.70],   # 5 - enforced
    [-67.64, 83.88],    # 6
    [-85.58, 107.70],   # 7 - enforced
    [35.19, -77.20]     # 8 - enforced
])

# enforced_edges = [(1, 8)]  # Fixed connections to always include
enforced_edges = [(0, 2)]  # Fixed connections to always include

def check_all_intersections(points, path):
    edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    for i, (a1, a2) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            b1, b2 = edges[j]
            if len(set([a1, a2, b1, b2])) < 4:
                continue  # sharing a point, skip
            if segments_intersect(points[a1], points[a2], points[b1], points[b2]):
                return True
    return False

def analyze_path(points, path):  # JUST PRINTING!
    """Analyze a path's angles and distances."""
    print("\nPath Analysis:")
    print(f"Path sequence: {path}")
    total_cost = 0
    for i in range(len(path) - 1):
        dist = np.linalg.norm(points[path[i]] - points[path[i+1]])
        angle = calculate_angle_deviation(points[path[i-1]], points[path[i]], points[path[i+1]]) if i > 0 else 0
        print(f"Edge {path[i]}->{path[i+1]}: distance={dist:.2f}, angle={angle:.2f}°")
        total_cost += dist + (angle if i > 0 else 0)
    print(f"Total cost: {total_cost:.2f}")
    return total_cost

# FINAL CORRECT VERSION: Fully tries all enforced edge orderings and valid insertions of remaining nodes
def fully_correct_tsp(points, enforced_edges):
    n = len(points)
    all_nodes = set(range(n))
    enforced_paths = [[a, b] for a, b in enforced_edges]
    forced_nodes = set(i for a, b in enforced_edges for i in (a, b))
    remaining_nodes = list(all_nodes - forced_nodes)

    print("\nStarting TSP with:")
    print(f"Enforced edges: {enforced_edges}")
    print(f"Forced nodes: {forced_nodes}")
    print(f"Remaining nodes: {remaining_nodes}")

    best_cost = float('inf')
    best_path = []
    paths_tried = 0
    valid_paths = 0

    # Try all orderings of enforced segments
    for enforced_order in itertools.permutations(enforced_paths):
        # Flatten into an initial path
        base_path = []
        used_nodes = set()
        for seg in enforced_order:
            if base_path and base_path[-1] == seg[0]:
                base_path.append(seg[1])
            elif base_path and base_path[-1] == seg[1]:
                base_path.append(seg[0])
            elif not base_path:
                base_path.extend(seg)
            else:
                break  # Can't chain segments directly
            used_nodes.update(seg)
        else:  # Only continue if we successfully chained all enforced segments
            print(f"\nTrying base path: {base_path}")
            
            # Try all permutations of remaining nodes and all insertion positions
            for perm in itertools.permutations(remaining_nodes):
                for insertion_points in itertools.product(range(1, len(base_path)), repeat=len(perm)):
                    paths_tried += 1
                    temp_path = base_path[:]
                    offset = 0
                    for node, idx in zip(perm, insertion_points):
                        insert_idx = idx + offset
                        temp_path.insert(insert_idx, node)
                        offset += 1
                    temp_path.append(temp_path[0])  # Close the loop

                    if len(set(temp_path)) != n:
                        continue
                    if check_all_intersections(points, temp_path):
                        continue

                    valid_paths += 1
                    # Compute cost and analyze path
                    cost = 0
                    print(f"\nAnalyzing path {valid_paths}: {temp_path}")
                    for i in range(len(temp_path) - 1):
                        dist = np.linalg.norm(points[temp_path[i]] - points[temp_path[i+1]])
                        angle = 0
                        if i >= 1:
                            angle = calculate_angle_deviation(
                                points[temp_path[i-1]], 
                                points[temp_path[i]], 
                                points[temp_path[i+1]]
                            )
                        cost += dist + angle
                        if i >= 1:
                            print(f"  Vertex {temp_path[i]}: angle={angle:.2f}°, dist={dist:.2f}")

                    print(f"  Total cost: {cost:.2f} (current best: {best_cost:.2f})")
                    if cost < best_cost:
                        print(f"  New best path found!")
                        best_cost = cost
                        best_path = temp_path

    print(f"\nSearch complete!")
    print(f"Paths tried: {paths_tried}")
    print(f"Valid paths: {valid_paths}")
    print(f"Best path found: {best_path}")
    print(f"Best cost: {best_cost:.2f}")
    
    # Analyze the best path in detail
    if best_path:
        analyze_path(points, best_path)

    return best_cost, best_path

# Run the fully correct implementation
final_cost, final_path = fully_correct_tsp(points_raw, enforced_edges)

plot_points(points_raw, final_path, title="Final Non-Intersecting Path with Enforced Edges")
#%% FULL EXAMPLE

for i in range(10):

    road_vertices = np.load(f'road_vertices_roads_{i}.npy')[:, :2]
    print(f"\nLoaded road vertices: {len(road_vertices)} points")

    points_raw = trim_points(road_vertices, max_points=10)

    # plot_points(road_vertices, title="raw points")
    # plot_points(points_raw, title="filtered points")

    best_cost, best_path = tsp_held_karp_no_intersections(points_raw)
    # plot_points(points_raw, best_path, title="filtered")