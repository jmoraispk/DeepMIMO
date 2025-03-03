import os
import bpy
import csv
import bmesh
from mathutils import Vector
from constants import PROJ_ROOT, BUILDING_MATERIAL, ROAD_MATERIAL, FLOOR_MATERIAL
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

def join_and_materialize_objects(name_pattern, target_name, material):
    """
    Join objects matching a name pattern and apply a material.
    
    Args:
        name_pattern (str): Pattern to match object names
        target_name (str): Name for the joined object
        material (bpy.types.Material): Material to apply
    Returns:
        bpy.types.Object: The joined object
    """
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if name_pattern in o.name.lower() and o.type == 'MESH':
            o.select_set(True)
    selected = bpy.context.selected_objects
    print(f"Number of selected objects ({target_name}): {len(selected)}")
    if len(selected) > 1:
        bpy.context.view_layer.objects.active = selected[-1]
        bpy.ops.object.join()
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

def create_scene(positions, osm_folder, time_str):
    """Create scenes from CSV positions with OSM data and export."""
    for i, row in enumerate(positions):
        scene_folder = os.path.join(osm_folder, f'scene_{i}')
        os.makedirs(scene_folder, exist_ok=True)
        mesh_folder = os.path.join(scene_folder, 'meshes')
        os.makedirs(mesh_folder, exist_ok=True)

        min_lat, max_lat = float(row['min_lat']), float(row['max_lat'])
        min_lon, max_lon = float(row['min_lon']), float(row['max_lon'])

        # Clear Blender
        clear_blender()

        # Configure and import OSM data
        configure_osm_import(scene_folder, min_lat, max_lat, min_lon, max_lon)
        bpy.ops.blosm.import_data()

        # Save OSM origin
        origin_lat = bpy.data.scenes["Scene"]["lat"]
        origin_lon = bpy.data.scenes["Scene"]["lon"]
        with open(os.path.join(scene_folder, 'osm_gps_origin.txt'), 'w') as f:
            f.write(f"{origin_lat}\n{origin_lon}\n")

        # Create ground plane
        create_ground_plane(min_lat, max_lat, min_lon, max_lon)
        bpy.ops.object.select_all(action='DESELECT')
        objs = bpy.data.objects
        for o in objs:
            if o.name == 'terrain':
                o.select_set(True)
        obj = bpy.context.selected_objects[0]
        terrain_bounds = get_bounding_box(obj)

        # Create materials
        building_material = bpy.data.materials.new(name=BUILDING_MATERIAL)
        building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige
        road_material = bpy.data.materials.new(name=ROAD_MATERIAL)
        road_material.diffuse_color = (0.29, 0.25, 0.21, 1)  # Dark grey

        # Convert all to meshes
        bpy.ops.object.select_all(action='SELECT')
        bpy.context.view_layer.objects.active = bpy.data.objects[0]
        bpy.ops.object.convert(target='MESH', keep_original=False)

        # Process buildings
        buildings = join_and_materialize_objects('building', 'buildings', building_material)

        # Process roads
        bpy.ops.object.select_all(action='DESELECT')
        for o in bpy.data.objects:
            if 'terrain' in o.name.lower() or 'buildings' in o.name.lower():
                o.select_set(True)
        bpy.ops.object.select_all(action='INVERT')
        road_objs = bpy.context.selected_objects
        for obj in road_objs:
            if terrain_bounds:
                trim_faces_outside_bounds(obj, *terrain_bounds)
        if road_objs:
            bpy.context.view_layer.objects.active = road_objs[-1]
            bpy.ops.object.join()
            roads = bpy.context.active_object
            roads.name = 'roads'
            roads.data.materials.clear()
            roads.data.materials.append(road_material)
            print(f"Number of selected objects (Roads): {len(road_objs)}")

        # Set world emitter
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

        # Export to Mitsuba and save .blend file
        bpy.ops.export_scene.mitsuba(
            filepath=os.path.join(scene_folder, 'scene.xml'),
            export_ids=True, axis_forward='Y', axis_up='Z'
        )
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(scene_folder, 'scene.blend'))