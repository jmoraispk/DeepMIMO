"""
This file contains utility functions for Blender.
Many of them will only work inside Blender.
"""

import math
import os
import subprocess
import time
import sys
import requests
import logging
from typing import Optional, List, Any

# Blender imports
import bpy # type: ignore
import bmesh # type: ignore
import mathutils # type: ignore (comes with blender)  # noqa: E402

ADDONS = {
    "blosm": "blosm_2.7.11.zip",
    "mitsuba-blender": "mitsuba-blender.zip",
}

ADDON_URLS = {
    "blosm": "https://www.dropbox.com/scl/fi/cka3yriyrjppnfy2ztjq9/blosm_2.7.11.zip?rlkey=9ak7vnf4h13beqd4hpwt9e3ws&st=znk7icsq&dl=1",
    # blosm link is hosted on dropbox because it is not hosted. 
    # The original link is: https://github.com/vvoovv/blosm (which forwards to gumroad)
    "mitsuba-blender": "https://github.com/mitsuba-renderer/mitsuba-blender/releases/download/v0.4.0/mitsuba-blender.zip",
}

# Material names for scene objects
BUILDING_MATERIAL = 'itu_concrete'
ROAD_MATERIAL = 'itu_brick'  # Note: Manually changed to asphalt in Sionna
FLOOR_MATERIAL = 'itu_wet_ground'
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# LOGGER UTILS
LOGGER: Optional[Any] = None

def log_local_setup(name: str) -> logging.Logger:
    """Set up local logging configuration for both console and file output, 
    putting the log file in the same directory as the script.
    
    Args:
        name (str): Name for the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    log_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_file_path = os.path.join(log_dir, name)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(log_file_path, mode='w')  # File handler
        ]
    )

    return logging.getLogger(name)

def set_LOGGER(logger: Any) -> None:
    """Set the logger for the BlenderUtils class."""
    global LOGGER
    LOGGER = logger

# ADD ON INSTALL UTILS

# BLOSM UTILS

# MISC UTILS OF BOTH PIPELINES

# SIONNA PIPELINE UTILS

# WIRELESS INSITE UTILS


def get_obj_by_name(name: str) -> bpy.types.Object | None:
    """
    Get an object by its name.
    
    Args:
        name (str): Name of the object
    Returns:
        bpy.types.Object | None: The object if found, else None
    """
    return bpy.data.objects.get(name)

def clear_blender() -> None:
    """Remove all datablocks from Blender to start with a clean slate."""
    block_lists: List[Any] = [
        bpy.data.collections, bpy.data.objects, bpy.data.meshes, bpy.data.materials,
        bpy.data.textures, bpy.data.images, bpy.data.curves, bpy.data.cameras
    ]
    for block_list in block_lists:
        for block in list(block_list):
            block_list.remove(block, do_unlink=True)

def get_bounding_box(obj: bpy.types.Object) -> tuple[float, float, float, float] | None:
    """
    Calculate the world-space bounding box of an object.
    
    Args:
        obj (bpy.types.Object): Mesh object
    Returns:
        tuple[float, float, float, float] | None: (min_x, max_x, min_y, max_y) in world coordinates
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
        world_coord = matrix_world @ mathutils.Vector((corner[0], corner[1], corner[2]))
        min_x = min(min_x, world_coord.x)
        max_x = max(max_x, world_coord.x)
        min_y = min(min_y, world_coord.y)
        max_y = max(max_y, world_coord.y)
    print(f"Object: {obj.name}")
    print(f"min_x: {min_x}, max_x: {max_x}")
    print(f"min_y: {min_y}, max_y: {max_y}")
    return min_x, max_x, min_y, max_y

def compute_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """
    Compute Haversine distance between two coordinates in meters.
    
    Args:
        coord1 (tuple[float, float]): (latitude, longitude) of first point
        coord2 (tuple[float, float]): (latitude, longitude) of second point
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

def install_all_addons() -> None:
    """Install all addons from the ADDON_URLS dictionary."""
    for addon_name, _ in ADDONS.items():
        install_blender_addon(addon_name)

def download_addon(addon_name: str) -> str:
    """Download a file from a URL and save it to a local path."""
    output_path = os.path.join(PROJ_ROOT, "blender_addons", ADDONS[addon_name])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    url = ADDON_URLS[addon_name]
    LOGGER.info(f"📥 Downloading file from {url} to {output_path}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        error_msg = f"❌ Failed to download file from {url}: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)
    
    return output_path

def install_python_package(pckg_name: str) -> None:
    """
    Install a Python package using Blender's Python executable.
    
    Args:
        pckg_name (str): Name of the package to install (e.g., 'mitsuba==3.5.0')
    """
    LOGGER.info(f"📦 Installing Python package: {pckg_name}")
    python_exe = sys.executable
    LOGGER.debug(f"Using Python executable: {python_exe}")
    
    try:
        subprocess.call([python_exe, "-m", "ensurepip"])
        subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.call([python_exe, "-m", "pip", "install", pckg_name])
        LOGGER.info(f"✅ Successfully installed {pckg_name}")
    except Exception as e:
        error_msg = f"❌ Failed to install {pckg_name}: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def install_blender_addon(addon_name: str) -> None:
    """
    Install and enable a Blender add-on from a zip file if not already installed.
    
    Args:
        addon_name (str): Name of the add-on to install (e.g., 'blosm')
    """
    LOGGER.info(f"🔧 Processing Blender add-on: {addon_name}")
    zip_name = ADDONS.get(addon_name)
    if not zip_name:
        LOGGER.error(f"❌ No zip file defined for add-on '{addon_name}'")
        return
    
    LOGGER.debug(f"Currently installed add-ons: {list(bpy.context.preferences.addons.keys())}")
    
    # Check if add-on is already installed
    if addon_name in bpy.context.preferences.addons.keys():
        LOGGER.info(f"📌 Add-on '{addon_name}' is already installed")
        if bpy.context.preferences.addons[addon_name].module:
            LOGGER.info(f"✅ Add-on '{addon_name}' is enabled")
        else:
            LOGGER.info(f"  Enabling add-on '{addon_name}'")
            bpy.ops.preferences.addon_enable(module=addon_name)
            bpy.ops.wm.save_userpref()
            LOGGER.info(f"✅ Add-on '{addon_name}' has been enabled")
    else:
        LOGGER.info(f"📥 Installing new add-on '{addon_name}'")
        addon_zip_path = os.path.join(PROJ_ROOT, "blender_addons", zip_name)
        if not os.path.exists(addon_zip_path):
            LOGGER.warning(f"⚠ Add-on zip file not found: {addon_zip_path}")
            LOGGER.info(f"Attempting to download {addon_zip_path}")
            
            pass
            return
        try:
            bpy.ops.preferences.addon_install(filepath=addon_zip_path)
            bpy.ops.preferences.addon_enable(module=addon_name)
            bpy.ops.wm.save_userpref()
            LOGGER.info(f"✅ Add-on '{addon_name}' installed and enabled")
        except Exception as e:
            LOGGER.error(f"❌ Failed to install/enable add-on '{addon_name}': {str(e)}")
            raise
    
    # Special handling for Mitsuba
    if addon_name == 'mitsuba-blender':
        try:
            import mitsuba
            LOGGER.info("✅ Mitsuba import successful")
        except ImportError:
            LOGGER.info("📦 Mitsuba not found, installing mitsuba package")
            install_python_package('mitsuba==3.5.0')
            LOGGER.warning("🔄 Packages installed! Restarting Blender to update imports")
            time.sleep(5)
            sys.exit()

# Configure OSM import
def configure_osm_import(output_folder: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> None:
    """Configure blosm add-on for OSM data import."""
    LOGGER.info(f"🗺️ Configuring OSM import for region: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]")
    try:
        prefs = bpy.context.preferences.addons["blosm"].preferences
        prefs.dataDir = output_folder
        LOGGER.debug(f"Set OSM data directory to: {output_folder}")
        
        scene = bpy.context.scene.blosm
        scene.mode = '3Dsimple'
        scene.minLat, scene.maxLat = min_lat, max_lat
        scene.minLon, scene.maxLon = min_lon, max_lon
        scene.buildings, scene.highways = True, True
        scene.water, scene.forests, scene.vegetation, scene.railways = False, False, False, False
        scene.singleObject, scene.ignoreGeoreferencing = True, True
        LOGGER.info("✅ OSM import configuration complete")
    except Exception as e:
        error_msg = f"❌ Failed to configure OSM import: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def create_ground_plane(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> bpy.types.Object:
    """Create and size a ground plane with FLOOR_MATERIAL."""
    LOGGER.info("🌍 Creating ground plane")
    try:
        bpy.ops.mesh.primitive_plane_add(size=1)
        x_size = compute_distance([min_lat, min_lon], [min_lat, max_lon]) * 1.2
        y_size = compute_distance([min_lat, min_lon], [max_lat, min_lon]) * 1.2
        LOGGER.debug(f"Ground plane dimensions: [{x_size}, {y_size}]")
        
        plane = get_obj_by_name("Plane")
        if plane is None:
            raise ValueError("Failed to create ground plane")
        plane.scale = (x_size, y_size, 1)
        plane.name = 'terrain'
        
        floor_material = bpy.data.materials.new(name=FLOOR_MATERIAL)
        plane.data.materials.append(floor_material)
        LOGGER.info("✅ Ground plane created and configured")
        return plane
    except Exception as e:
        error_msg = f"❌ Failed to create ground plane: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def join_and_materialize_objects(name_pattern: str, target_name: str, material: bpy.types.Material) -> bpy.types.Object | None:
    """Join objects matching a name pattern and apply a material."""
    LOGGER.info(f"🔄 Processing objects matching pattern: {name_pattern}")
    try:
        bpy.ops.object.select_all(action='DESELECT')
        for o in bpy.data.objects:
            if name_pattern in o.name.lower() and o.type == 'MESH':
                o.select_set(True)
        
        selected = bpy.context.selected_objects
        LOGGER.info(f"📊 Found {len(selected)} objects matching '{name_pattern}'")
        
        if len(selected) > 1:
            LOGGER.debug("Joining multiple objects")
            bpy.context.view_layer.objects.active = selected[-1]
            bpy.ops.object.join()
        elif len(selected) == 1:
            LOGGER.debug("Single object found, setting as active")
            bpy.context.view_layer.objects.active = selected[0]
        elif not selected:
            LOGGER.warning(f"⚠️ No objects found matching pattern: {name_pattern}")
            return None
        
        obj = bpy.context.active_object
        obj.name = target_name
        obj.data.materials.clear()
        obj.data.materials.append(material)
        LOGGER.info(f"✅ Successfully created and materialized {target_name}")
        return obj
    except Exception as e:
        error_msg = f"❌ Failed to process objects with pattern '{name_pattern}': {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def trim_faces_outside_bounds(obj: bpy.types.Object, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
    """Remove faces of an object outside a bounding box in world space."""
    LOGGER.info(f"✂️ Trimming faces outside bounds for object: {obj.name}")
    LOGGER.debug(f"Bounds: x[{min_x}, {max_x}], y[{min_y}, {max_y}]")
    
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        matrix_world = obj.matrix_world
        faces_to_delete = []
        
        total_faces = len(bm.faces)
        for face in bm.faces:
            center = mathutils.Vector((0, 0, 0))
            for vert in face.verts:
                center += vert.co
            center /= len(face.verts)
            world_center = matrix_world @ center
            x, y = world_center.x, world_center.y
            if (x < min_x or x > max_x) or (y < min_y or y > max_y):
                faces_to_delete.append(face)
        
        if faces_to_delete:
            LOGGER.info(f"🗑️ Removing {len(faces_to_delete)} faces out of {total_faces}")
            bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
            bmesh.update_edit_mesh(obj.data)
        else:
            LOGGER.info("✅ No faces needed trimming")
        
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        error_msg = f"❌ Failed to trim faces for {obj.name}: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def setup_world_lighting() -> None:
    """Configure world lighting with a basic emitter."""
    LOGGER.info("💡 Setting up world lighting")
    try:
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
        LOGGER.info("✅ World lighting configured")
    except Exception as e:
        error_msg = f"❌ Failed to setup world lighting: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def create_camera_and_render(scene: bpy.types.Scene, output_path: str, location: tuple[float, float, float] = (0, 0, 1000), rotation: tuple[float, float, float] = (0, 0, 0)) -> None:
    """Add a camera, render the scene, and delete the camera."""
    LOGGER.info(f"📸 Setting up camera for render at {output_path}")
    LOGGER.debug(f"Camera position: {location}, rotation: {rotation}")
    
    try:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.active_object
        scene.camera = camera
        
        scene.render.filepath = output_path
        LOGGER.info("🎬 Starting render")
        bpy.ops.render.render(write_still=True)
        LOGGER.info("✅ Render complete")
        
        bpy.ops.object.select_all(action='DESELECT')
        camera.select_set(True)
        bpy.ops.object.delete()
        LOGGER.debug("Temporary camera removed")
    except Exception as e:
        error_msg = f"❌ Failed to render scene: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def save_osm_origin(scene_folder: str, origin_lat: float, origin_lon: float) -> None:
    """Save OSM origin coordinates to a text file."""
    LOGGER.info(f"📍 Saving OSM origin coordinates: [{origin_lat}, {origin_lon}]")
    try:
        output_path = os.path.join(scene_folder, 'osm_gps_origin.txt')
        with open(output_path, 'w') as f:
            f.write(f"{origin_lat}\n{origin_lon}\n")
        LOGGER.info("✅ OSM origin saved")
    except Exception as e:
        error_msg = f"❌ Failed to save OSM origin: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def export_scene(scene_folder: str) -> None:
    """Export scene to Mitsuba and save .blend file."""
    LOGGER.info("📤 Exporting scene")
    try:
        mitsuba_path = os.path.join(scene_folder, 'scene.xml')
        blend_path = os.path.join(scene_folder, 'scene.blend')
        
        LOGGER.debug(f"Exporting Mitsuba scene to: {mitsuba_path}")
        bpy.ops.export_scene.mitsuba(
            filepath=mitsuba_path,
            export_ids=True, axis_forward='Y', axis_up='Z'
        )
        
        LOGGER.debug(f"Saving Blender file to: {blend_path}")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        LOGGER.info("✅ Scene export complete")
    except Exception as e:
        error_msg = f"❌ Failed to export scene: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def create_scene(positions: tuple[float, float, float, float], out_folder: str) -> None:
    """Create scenes from CSV positions with OSM data and export."""
    LOGGER.info(f"🎨 Creating new scene in: {out_folder}")
    scene_name = os.path.basename(out_folder)
    
    os.makedirs(out_folder, exist_ok=True)
    output_fig_folder = os.path.join(out_folder, 'figs')
    os.makedirs(output_fig_folder, exist_ok=True)
    LOGGER.debug(f"Created output directories: {out_folder}, {output_fig_folder}")

    min_lat, min_lon, max_lat, max_lon = positions
    LOGGER.info(f"📍 Scene bounds: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]")

    # Initialize scene
    LOGGER.info("🔄 Initializing scene")
    clear_blender()
    setup_world_lighting()

    # Import OSM data
    LOGGER.info("🗺️ Importing OSM data")
    configure_osm_import(out_folder, min_lat, max_lat, min_lon, max_lon)
    
    bpy.ops.blosm.import_data()
    origin_lat = bpy.data.scenes["Scene"]["lat"]
    origin_lon = bpy.data.scenes["Scene"]["lon"]
    save_osm_origin(out_folder, origin_lat, origin_lon)
    
    # Create ground plane
    LOGGER.info("🌍 Setting up terrain")
    terrain = create_ground_plane(min_lat, max_lat, min_lon, max_lon)
    terrain_bounds = get_bounding_box(terrain)
    # Create materials
    LOGGER.info("🎨 Creating materials")
    building_material = bpy.data.materials.new(name=BUILDING_MATERIAL)
    building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige
    road_material = bpy.data.materials.new(name=ROAD_MATERIAL)
    road_material.diffuse_color = (0.29, 0.25, 0.21, 1)  # Dark grey

    # Convert all to meshes
    LOGGER.info("🔄 Converting objects to meshes")
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.convert(target='MESH', keep_original=False)

    # Render original scene
    LOGGER.info("📸 Rendering original scene")
    scene = bpy.context.scene
    create_camera_and_render(scene, os.path.join(output_fig_folder, f"{scene_name}_org.png"))

    ################################ 
    # Process buildings
    LOGGER.info("🏢 Processing buildings")
    buildings = join_and_materialize_objects('building', 'buildings', building_material)
    if buildings and buildings.type == 'MESH':
        LOGGER.debug("Separating building meshes")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')

    # Process roads
    LOGGER.info("🛣️ Processing roads")
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if 'terrain' in o.name.lower() or 'buildings' in o.name.lower():
            o.select_set(True)
    bpy.ops.object.select_all(action='INVERT')
    road_objs = bpy.context.selected_objects
    if road_objs:
        LOGGER.info(f"Found {len(road_objs)} road objects to process")
        for obj in road_objs:
            if terrain_bounds:
                trim_faces_outside_bounds(obj, *terrain_bounds)
            bpy.context.view_layer.objects.active = obj
            road = bpy.context.active_object
            road.data.materials.clear()
            road.data.materials.append(road_material)
        LOGGER.debug(f"Processed {len(road_objs)} road objects")

    # Render processed scene
    LOGGER.info("📸 Rendering processed scene")
    create_camera_and_render(scene, os.path.join(output_fig_folder, f"{scene_name}_processed.png"))

    # Export scene
    LOGGER.info("📤 Exporting final scene")
    export_scene(out_folder)
    LOGGER.info("✅ Scene creation complete")

###############################################################################
# Functions for Wireless Insite
###############################################################################


def save_osm_gps_origin(output_folder: str) -> None:
    """Save OSM GPS origin coordinates to a file.
    
    Args:
        output_folder (str): Directory to save the origin file
    """
    LOGGER.info("📍 Saving OSM GPS origin coordinates")
    try:
        origin_lat = bpy.data.scenes["Scene"]["lat"]
        origin_lon = bpy.data.scenes["Scene"]["lon"]
        with open(os.path.join(output_folder, "osm_gps_origin.txt"), "w") as file:
            file.write(f"{origin_lat}\n{origin_lon}\n")
        LOGGER.info("✅ OSM GPS origin saved.")
    except Exception as e:
        error_msg = f"❌ Failed to save OSM GPS origin: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def save_scenario_metadata(output_folder: str, minlat: float, minlon: float, maxlat: float, maxlon: float, output_formats: list) -> None:
    """Save scenario properties to a metadata file.
    
    Args:
        output_folder (str): Directory to save the metadata file
        minlat (float): Minimum latitude of bounding box
        minlon (float): Minimum longitude of bounding box
        maxlat (float): Maximum latitude of bounding box
        maxlon (float): Maximum longitude of bounding box
        output_formats (list): List of output formats being used
    """
    LOGGER.info("📝 Saving scenario metadata")
    try:
        metadata_path = os.path.join(output_folder, "scenario_info.txt")
        with open(metadata_path, "w") as meta_file:
            meta_file.write(f"Bounding Box: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]\n")
            meta_file.write(f"Output Formats: {output_formats}\n")
        LOGGER.info("✅ Scenario metadata saved.")
    except Exception as e:
        error_msg = f"❌ Failed to save scenario metadata: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)

def convert_objects_to_mesh() -> None:
    """Convert all selected objects to mesh type."""
    LOGGER.info("🔄 Converting objects to mesh")
    try:
        bpy.ops.object.select_all(action="SELECT")
        if len(bpy.context.selected_objects) > 0:
            bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
            bpy.ops.object.convert(target="MESH", keep_original=False)
            LOGGER.info("✅ All objects successfully converted to mesh.")
        else:
            LOGGER.warning("⚠ No objects found for conversion. Skipping.")
    except Exception as e:
        error_msg = f"❌ Failed to convert objects to mesh: {str(e)}"
        LOGGER.error(error_msg)
        raise Exception(error_msg)
        
# Export Buildings
def export_mesh_obj_to_ply(object_type: str, output_folder: str) -> None:
    """Export objects of a given type to PLY format.
    
    Args:
        object_type (str): Type of objects to export ('building' or 'road')
        output_folder (str): Folder to save the PLY file
    """
    bpy.ops.object.select_all(action="DESELECT")
    objects = [o for o in bpy.data.objects if object_type in o.name.lower()]
    for obj in objects:
        obj.select_set(True)
        
    if objects:
        emoji = "🏗" if object_type == "building" else "🛣"
        LOGGER.info(f"{emoji} Exporting {len(objects)} {object_type}s to .ply")
        bpy.ops.export_mesh.ply(
            filepath=os.path.join(output_folder, f"{object_type}s.ply"),
            use_ascii=True
        )
    else:
        LOGGER.warning(f"⚠ No {object_type}s found for export.")
