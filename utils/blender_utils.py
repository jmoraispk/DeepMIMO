# utils/blender_utils.py
import bpy
import math
from config.materials import MATERIAL_COLORS, WORLD_EMITTER_COLOR

def compute_distance(coord1, coord2):
    """Compute Haversine distance between two coordinates in meters."""
    R = 6371.0
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000

def get_objs_with_material(mat):
    """Return objects using a specific material."""
    return [obj for obj in bpy.data.objects if any(slot.material == mat for slot in obj.material_slots)]

def get_obj_by_name(name):
    """Retrieve an object by name."""
    return next((obj for obj in bpy.data.objects if obj.name == name), None)

def get_slot_of_material(obj, mat):
    """Get the material slot index for a given material."""
    return next((i for i, slot in enumerate(obj.material_slots) if slot.material == mat), -1)

def clear_blender():
    """Clear all data blocks in Blender."""
    block_lists = [
        bpy.data.collections, bpy.data.objects, bpy.data.meshes,
        bpy.data.materials, bpy.data.textures, bpy.data.images,
        bpy.data.curves, bpy.data.cameras
    ]
    for block_list in block_lists:
        for block in block_list:
            block_list.remove(block, do_unlink=True)

def create_material(name, color=None):
    """Create a new material with an optional color."""
    mat = bpy.data.materials.new(name=name)
    if color:
        mat.diffuse_color = color
    return mat

def set_world_emitter():
    """Set the world emitter with a specific color."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    background_node = nodes.new('ShaderNodeBackground')
    output_node = nodes.new('ShaderNodeOutputWorld')
    background_node.inputs['Color'].default_value = WORLD_EMITTER_COLOR
    background_node.inputs['Strength'].default_value = 1.0
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])