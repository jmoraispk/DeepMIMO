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

# Absolute Paths
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

COUNTER = 118
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
	# break

#%% ROAD PROCESSING ESSENTIAL FUNCTIONS

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

def compress_path(points, path, angle_threshold=1.0):
    """Compress a path by removing points that are nearly collinear with their neighbors.
    
    Args:
        points: Array of point coordinates (N x 2)
        path: List of indices forming the path
        angle_threshold: Minimum angle deviation (in degrees) to keep a point
        
    Returns:
        List of indices forming the compressed path
    """
    if len(path) <= 3:  # Can't compress paths with 3 or fewer points
        return path
        
    # We'll build the compressed path starting with the first point
    compressed = [path[0]]
    
    # Iterate through interior points (skip first and last)
    for i in range(1, len(path)-1):
        # Get the previous, current, and next points
        prev_idx = compressed[-1]  # Last point in compressed path
        curr_idx = path[i]        # Current point we're considering
        next_idx = path[i+1]      # Next point in original path
        
        # Calculate angle at current point
        angle = calculate_angle_deviation(
            points[prev_idx],
            points[curr_idx],
            points[next_idx]
        )
        
        # If angle is significant (> threshold), keep the point
        if angle > angle_threshold:
            compressed.append(curr_idx)
    
    # Always add the last point to close the loop
    compressed.append(path[-1])
    
    return compressed

#%% FULL EXAMPLE

for i in [3]:#range(10):

    road_vertices = np.load(f'road_vertices_roads_{i}.npy')[:, :2]
    print(f"\nLoaded road vertices: {len(road_vertices)} points")

    points_raw = trim_points(road_vertices, max_points=10)

    plot_points(road_vertices, title="raw points")
    plot_points(points_raw, title="filtered points")

    best_cost, best_path = tsp_held_karp_no_intersections(points_raw)
    print(f"Best path: {best_path}")
    plot_points(points_raw, best_path, title="filtered")

    # Compress the path by removing the points that are in line with the previous point
    compressed_path = compress_path(points_raw, best_path, angle_threshold=1.0)
    print(f"Compressed path: {compressed_path}")
    length_compressed, length_raw = len(compressed_path) - 1, len(best_path) - 1
    print(f"Length compressed: {length_compressed}, length raw: {length_raw}")
    points_compressed = points_raw[compressed_path[:-1]]
    plot_points(points_compressed, [i for i in range(length_compressed)]+[0], title="compressed")


# %%
