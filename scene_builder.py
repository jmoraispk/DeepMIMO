import os
import sys 

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import csv
import bmesh
from mathutils import Vector
from constants import ADDONS, BUILDING_MATERIAL, ROAD_MATERIAL, FLOOR_MATERIAL
from utils.addon_utils import install_blender_addon
from utils.blender_utils import get_obj_by_name, get_bounding_box, clear_blender, compute_distance

def configure_osm_import(scene_folder, min_lat, max_lat, min_lon, max_lon):
    """Configure blosm add-on for OSM data import."""
    prefs = bpy.context.preferences.addons["blosm"].preferences
    prefs.dataDir = scene_folder
    scene = bpy.context.scene.blosm
    scene.mode = '3Dsimple'
    scene.minLat, scene.maxLat = min_lat, max_lat
    scene.minLon, scene.maxLon = min_lon, max_lon
    scene.buildings, scene.highways = True, True
    scene.water, scene.forests, scene.vegetation, scene.railways = False, False, False, False
    scene.singleObject, scene.ignoreGeoreferencing = True, True

def create_ground_plane(min_lat, max_lat, min_lon, max_lon):
    """Create and size a ground plane with FLOOR_MATERIAL."""
    bpy.ops.mesh.primitive_plane_add(size=1)
    x_size = compute_distance([min_lat, min_lon], [min_lat, max_lon]) * 1.2
    y_size = compute_distance([min_lat, min_lon], [max_lat, min_lon]) * 1.2
    print(f'Creating plane of size [{x_size}, {y_size}]')
    plane = get_obj_by_name("Plane")
    plane.scale = (x_size, y_size, 1)
    plane.name = 'terrain'
    floor_material = bpy.data.materials.new(name=FLOOR_MATERIAL)
    plane.data.materials.append(floor_material)
    return plane

def join_and_materialize_objects(name_pattern, target_name, material):
    """Join objects matching a name pattern and apply a material."""
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if name_pattern in o.name.lower() and o.type == 'MESH':
            o.select_set(True)
    selected = bpy.context.selected_objects
    print(f"Number of selected objects ({target_name}): {len(selected)}")
    if len(selected) > 1:
        bpy.context.view_layer.objects.active = selected[-1]
        bpy.ops.object.join()
    elif len(selected) == 1:
        bpy.context.view_layer.objects.active = selected[0]
    elif not selected:
        return None
    obj = bpy.context.active_object
    obj.name = target_name
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj

def trim_faces_outside_bounds(obj, min_x, max_x, min_y, max_y):
    """Remove faces of an object outside a bounding box in world space."""
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    matrix_world = obj.matrix_world
    faces_to_delete = []
    for face in bm.faces:
        center = Vector((0, 0, 0))
        for vert in face.verts:
            center += vert.co
        center /= len(face.verts)
        world_center = matrix_world @ center
        x, y = world_center.x, world_center.y
        if (x < min_x or x > max_x) or (y < min_y or y > max_y):
            faces_to_delete.append(face)
    bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

def setup_world_lighting():
    """Configure world lighting with a basic emitter."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    background_node = nodes.new('ShaderNodeBackground')
    output_node = nodes.new('ShaderNodeOutputWorld')
    background_node.inputs['Color'].default_value = (0.517334, 0.517334, 0.517334, 1.0)
    background_node.inputs['Strength'].default_value = 1.0
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

def create_camera_and_render(scene, output_path, location=(0, 0, 1000), rotation=(0, 0, 0)):
    """Add a camera, render the scene, and delete the camera."""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object
    scene.camera = camera
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.select_all(action='DESELECT')
    camera.select_set(True)
    bpy.ops.object.delete()

def save_osm_origin(scene_folder, origin_lat, origin_lon):
    """Save OSM origin coordinates to a text file."""
    with open(os.path.join(scene_folder, 'osm_gps_origin.txt'), 'w') as f:
        f.write(f"{origin_lat}\n{origin_lon}\n")

def export_scene(scene_folder, scene_name):
    """Export scene to Mitsuba and save .blend file."""
    bpy.ops.export_scene.mitsuba(
        filepath=os.path.join(scene_folder, 'scene.xml'),
        export_ids=True, axis_forward='Y', axis_up='Z'
    )
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(scene_folder, 'scene.blend'))

def create_scene(scene_name, positions, osm_folder, time_str):
    """Create scenes from CSV positions with OSM data and export."""
    scene_folder = os.path.join(osm_folder, scene_name)
    os.makedirs(scene_folder, exist_ok=True)
    output_fig_folder = os.path.join(scene_folder, 'figs')
    os.makedirs(output_fig_folder, exist_ok=True)

    min_lat, min_lon, max_lat, max_lon = positions

    # Initialize scene
    clear_blender()
    setup_world_lighting()

    # Import OSM data
    configure_osm_import(scene_folder, min_lat, max_lat, min_lon, max_lon)
    bpy.ops.blosm.import_data()
    origin_lat = bpy.data.scenes["Scene"]["lat"]
    origin_lon = bpy.data.scenes["Scene"]["lon"]
    save_osm_origin(scene_folder, origin_lat, origin_lon)
    
    # Create ground plane
    terrain = create_ground_plane(min_lat, max_lat, min_lon, max_lon)
    terrain_bounds = get_bounding_box(terrain)

    # Create materials
    building_material = bpy.data.materials.new(name=BUILDING_MATERIAL)
    building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige
    road_material = bpy.data.materials.new(name=ROAD_MATERIAL)
    road_material.diffuse_color = (0.29, 0.25, 0.21, 1)  # Dark grey

    # Convert all to meshes
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.convert(target='MESH', keep_original=False)

    # Render original scene
    scene = bpy.context.scene
    create_camera_and_render(scene, os.path.join(output_fig_folder, f"{scene_name}_org.png"))

    # Process buildings
    buildings = join_and_materialize_objects('building', 'buildings', building_material)
    if buildings and buildings.type == 'MESH':
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')

    # Process roads
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if 'terrain' in o.name.lower() or 'buildings' in o.name.lower():
            o.select_set(True)
    bpy.ops.object.select_all(action='INVERT')
    road_objs = bpy.context.selected_objects
    if road_objs:
        for obj in road_objs:
            if terrain_bounds:
                trim_faces_outside_bounds(obj, *terrain_bounds)
            bpy.context.view_layer.objects.active = obj
            road = bpy.context.active_object
            road.data.materials.clear()
            road.data.materials.append(road_material)
        print(f"Number of selected objects (Roads): {len(road_objs)}")       

    # Render processed scene
    create_camera_and_render(scene, os.path.join(output_fig_folder, f"{scene_name}_processed.png"))

    # Export scene
    export_scene(scene_folder, scene_name)

if __name__ == '__main__':
    """Main entry point for automated scene creation."""
    # Get command line arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # Get all args after "--"
    print(f"Received arguments: {argv}")
    scene_name = argv[0]
    min_lat, min_lon, max_lat, max_lon = map(float, argv[1:5])
    positions = [min_lat, min_lon, max_lat, max_lon]
    osm_folder = argv[5]
    time_str = argv[6]

    # Install required add-ons
    for addon_name in ADDONS:
        install_blender_addon(addon_name)

    # Create scenes
    create_scene(scene_name, positions, osm_folder, time_str)