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

#%% Step 1: (Optional) Generate CSV with GPS coordinates for map and basestation placement

print('not implemented yet')
# TODO:
# - Configure cell size to be ~80m longer in x and y compared to NY. Maybe 200 x 400m. (2x4)

COUNTER = 109
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

#%%
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


def process_road_vertices(vertices: np.ndarray) -> RoadGeometry:
    """Process road vertices to find the best-fit plane and ordered vertices."""
    if len(vertices) < 3:
        return RoadGeometry([], None, None, None, None, {})
        
    # Find the best-fit plane using SVD
    centroid_3d = np.mean(vertices, axis=0)
    centered_pts = vertices - centroid_3d
    _, s, vh = np.linalg.svd(centered_pts)
    
    # Normal vector is the last right singular vector
    normal = vh[2]
    
    # Get two orthogonal vectors in the plane
    basis1 = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    if np.allclose(basis1, 0):
        basis1 = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    
    # Project points onto their best-fit plane
    projected_vertices = vertices - np.outer(
        np.dot(centered_pts, normal),
        normal
    )
    
    # Get 2D coordinates in the plane
    centered_projected = projected_vertices - centroid_3d
    x_coords = np.dot(centered_projected, basis1)
    y_coords = np.dot(centered_projected, basis2)
    points_2d = np.column_stack([x_coords, y_coords])
    
    # Print point coordinates and distances
    print("\nPoint coordinates and nearest neighbor distances:")
    print("Point ID  |     X     |     Y     | Nearest Dist")
    print("-" * 45)
    for i, point in enumerate(points_2d):
        dists = np.sqrt(np.sum((points_2d - point)**2, axis=1))
        dists[i] = np.inf
        print(f"{i:8d} | {point[0]:8.2f} | {point[1]:8.2f} | {min(dists):11.2f}")
    
    # Filter points closer than threshold
    min_dist_threshold = .5
    print(f"Filtering points closer than {min_dist_threshold:.2f} units")
    
    filtered_indices = []
    used_mask = np.zeros(len(points_2d), dtype=bool)
    for i in range(len(points_2d)):
        if not used_mask[i]:
            filtered_indices.append(i)
            used_mask |= np.sqrt(np.sum((points_2d[i] - points_2d)**2, axis=1)) < min_dist_threshold
    
    print(f"Filtered from {len(points_2d)} to {len(filtered_indices)} points")
    print("Kept points:", filtered_indices)
    
    points_2d = points_2d[filtered_indices]
    projected_vertices = projected_vertices[filtered_indices]
    centroid_2d = np.mean(points_2d, axis=0)
    
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
    
    # Print endpoint information
    print("\nEndpoint Analysis:")
    print("Left endpoints:", left_end_indices)
    print("Left endpoint coordinates:")
    print(points_2d[left_end_indices])
    print("\nRight endpoints:", right_end_indices)
    print("Right endpoint coordinates:")
    print(points_2d[right_end_indices])
    
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
    properties = {}
    properties['area'] = area
    properties['perimeter'] = perimeter
    
    return RoadGeometry(face_vertices, centroid_3d, centroid_2d, ordered_2d, normal, properties)


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
    
    # Plot normal vector
    normal_scale = np.max(np.ptp(vertices, axis=0)) * 0.2  # 20% of max range
    ax.quiver(geometry.centroid_3d[0], geometry.centroid_3d[1], geometry.centroid_3d[2],
             geometry.normal[0], geometry.normal[1], geometry.normal[2],
             length=normal_scale, color='orange', label='Normal')
    
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
    n_points = len(points)
    
    # Identify endpoints (first/last 4 points)
    endpoint_mask = np.zeros(n_points, dtype=bool)
    endpoint_mask[:4] = True  # First 4 points
    endpoint_mask[-4:] = True  # Last 4 points
    
    # Plot regular points in blue, endpoints in red
    regular_points = points[~endpoint_mask]
    end_points = points[endpoint_mask]
    ax.scatter(regular_points[:, 0], regular_points[:, 1], 
              c='blue', marker='o', label='Regular Points')
    ax.scatter(end_points[:, 0], end_points[:, 1],
              c='red', marker='o', label='End Points')
    
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
    

def visualize_road_processing(vertices: np.ndarray, name: str = "road"):
    """Create complete visualization of road processing.
    
    Args:
        vertices: Array of vertex coordinates (shape: N x 3)
        name: Name of the road for plot titles
    """
    # Process vertices
    geometry = process_road_vertices(vertices)
    if not geometry.face_vertices:
        print("Not enough vertices to process")
        return
        
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_road(ax1, geometry, vertices)
    ax1.set_title(f'{name} - 3D View')
    
    # 2D plot
    ax2 = fig.add_subplot(122)
    plot_2d_road(ax2, geometry)
    ax2.set_title(f'{name} - 2D Projection')
    
    plt.tight_layout()
    plt.savefig(f'{name}_analysis.png')
    plt.show()
    
    return geometry

# Load and process road vertices
road_vertices = np.load('road_vertices_roads_3.npy')
print("\nProcessing road vertices:")
print(f"Number of vertices: {len(road_vertices)}")

# Process and visualize
geometry = visualize_road_processing(road_vertices, "road_3")

# Print detailed road analysis
if geometry and geometry.properties:
    print("\nRoad Properties Analysis:")
    props = geometry.properties
    print(f"Valid point count: {props.get('valid_point_count', False)}")
    print(f"Number of points: {props.get('n_points', 0)}")
    print(f"Has parallel edges: {props.get('has_parallel_edges', False)}")
    print(f"Parallel edge count: {props.get('parallel_edge_count', 0)}")
    
    if props.get('has_parallel_edges', False):
        print(f"Aspect ratio: {props.get('aspect_ratio', 'N/A'):.2f}")
        print(f"Valid aspect ratio: {props.get('valid_aspect_ratio', False)}")
        print(f"Min parallel angle: {np.rad2deg(props.get('min_parallel_angle', np.pi)):.1f} degrees")
    
    print(f"Is curved: {props.get('is_curved', False)}")
    if props.get('is_curved', False):
        print(f"Total curvature: {np.rad2deg(props.get('total_curvature', 0)):.1f} degrees")
    print(f"Max angle change: {np.rad2deg(props.get('max_angle_change', 0)):.1f} degrees")
    print(f"Has sharp turns: {props.get('has_sharp_turns', False)}")
    
    if props.get('consistent_width', False):
        print(f"Average width: {props.get('avg_width', 0):.2f}")
        print(f"Width variation: {props.get('width_consistency', 0)*100:.1f}%")

# %%
