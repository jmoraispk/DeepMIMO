import bpy
import math
from mathutils import Vector

def get_obj_by_name(name):
    """
    Get an object by its name.
    
    Args:
        name (str): Name of the object
    Returns:
        bpy.types.Object or None: The object if found, else None
    """
    return bpy.data.objects.get(name)

def clear_blender():
    """Remove all datablocks from Blender to start with a clean slate."""
    block_lists = [
        bpy.data.collections, bpy.data.objects, bpy.data.meshes, bpy.data.materials,
        bpy.data.textures, bpy.data.images, bpy.data.curves, bpy.data.cameras
    ]
    for block_list in block_lists:
        for block in list(block_list):
            block_list.remove(block, do_unlink=True)

def get_bounding_box(obj):
    """
    Calculate the world-space bounding box of an object.
    
    Args:
        obj (bpy.types.Object): Mesh object
    Returns:
        tuple: (min_x, max_x, min_y, max_y) in world coordinates
    """
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    if obj.type != 'MESH':
        return None
    bbox = obj.bound_box
    matrix_world = obj.matrix_world
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for corner in bbox:
        world_coord = matrix_world @ Vector((corner[0], corner[1], corner[2]))
        min_x = min(min_x, world_coord.x)
        max_x = max(max_x, world_coord.x)
        min_y = min(min_y, world_coord.y)
        max_y = max(max_y, world_coord.y)
    print(f"Object: {obj.name}")
    print(f"min_x: {min_x}, max_x: {max_x}")
    print(f"min_y: {min_y}, max_y: {max_y}")
    return min_x, max_x, min_y, max_y

def compute_distance(coord1, coord2):
    """
    Compute Haversine distance between two coordinates in meters.
    
    Args:
        coord1 (tuple): (latitude, longitude) of first point
        coord2 (tuple): (latitude, longitude) of second point
    Returns:
        float: Distance in meters
    Note:
        Error is ~1m at 10km, negligible for this use case. For higher precision,
        consider using GeoPy: geopy.distance.geodesic(coord1, coord2).meters
    """
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # Convert to meters