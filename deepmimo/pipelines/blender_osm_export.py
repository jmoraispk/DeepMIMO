"""
Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box)
Each scenario's output is stored in a folder named after its bounding box.
"""

import bpy # type: ignore
import os
import sys
import logging

# Setup logging to both console and file (great for debugging)
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_blender_osm.txt")
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

logger = logging.getLogger(__name__)

# Parse command-line arguments
try:
    minlat = float(sys.argv[sys.argv.index("--minlat") + 1])
    minlon = float(sys.argv[sys.argv.index("--minlon") + 1])
    maxlat = float(sys.argv[sys.argv.index("--maxlat") + 1])
    maxlon = float(sys.argv[sys.argv.index("--maxlon") + 1])
    output_folder = sys.argv[sys.argv.index("--output") + 1]
    
    output_formats = ["insite"] # Default format
    if "--format" in sys.argv:
        output_fmt = sys.argv[sys.argv.index("--format") + 1]

        valid_formats = ["insite", "sionna", "both"]
        if output_fmt not in valid_formats:
            logger.error(f"❌ Invalid format: {output_fmt}. Valid formats are: {', '.join(valid_formats)}")
            exit(1)
        
        output_formats = ["insite", "sionna"] if output_fmt == "both" else [output_fmt]
    
except (ValueError, IndexError):
    logger.error("❌ Invalid input format. Provide explicit coordinates (--minlat --minlon --maxlat --maxlon) and --output for custom output directory.")
    exit(1)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

logger.info(f"📍 Processing Scenario: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]")
logger.info(f"📂 Saving output to: {output_folder}")
logger.info(f"📊 Output formats: {output_formats}")

# Export based on the selected format
if "insite" in output_formats:

    
    # Clear existing objects in Blender
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Configure OSM import
    bpy.context.preferences.addons["blosm"].preferences.dataDir = output_folder
    bpy.context.scene.blosm.minLon = minlon
    bpy.context.scene.blosm.maxLon = maxlon
    bpy.context.scene.blosm.minLat = minlat
    bpy.context.scene.blosm.maxLat = maxlat

    bpy.context.scene.blosm.buildings = True
    bpy.context.scene.blosm.highways = True
    bpy.context.scene.blosm.singleObject = True
    bpy.context.scene.blosm.ignoreGeoreferencing = True

    bpy.ops.blosm.import_data()
    logger.info("✅ OSM data imported successfully.")

    # Save OSM GPS origin
    origin_lat = bpy.data.scenes["Scene"]["lat"]
    origin_lon = bpy.data.scenes["Scene"]["lon"]
    with open(os.path.join(output_folder, "osm_gps_origin.txt"), "w") as file:
        file.write(f"{origin_lat}\n{origin_lon}\n")
    logger.info("📍 OSM GPS origin saved.")

    # Save scenario properties to a metadata file
    metadata_path = os.path.join(output_folder, "scenario_info.txt")
    with open(metadata_path, "w") as meta_file:
        meta_file.write(f"Bounding Box: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]\n")
        meta_file.write(f"Output Formats: {output_formats}\n")
    logger.info("📝 Scenario metadata saved.")

    # Convert objects to mesh
    bpy.ops.object.select_all(action="SELECT")
    if len(bpy.context.selected_objects) > 0:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.convert(target="MESH", keep_original=False)
        logger.info("✅ All objects successfully converted to mesh.")
    else:
        logger.warning("⚠ No objects found for conversion. Skipping.")


    # Export Buildings
    def export_mesh_obj_to_ply(object_type: str, output_folder: str, logger) -> None:
        """Export objects of a given type to PLY format.
        
        Args:
            object_type (str): Type of objects to export ('building' or 'road')
            output_folder (str): Folder to save the PLY file
            logger: Logger object for status messages
        """
        bpy.ops.object.select_all(action="DESELECT")
        objects = [o for o in bpy.data.objects if object_type in o.name.lower()]
        for obj in objects:
            obj.select_set(True)
            
        if objects:
            emoji = "🏗" if object_type == "building" else "🛣"
            logger.info(f"{emoji} Exporting {len(objects)} {object_type}s to .ply")
            bpy.ops.export_mesh.ply(
                filepath=os.path.join(output_folder, f"{object_type}s.ply"),
                use_ascii=True
            )
        else:
            logger.warning(f"⚠ No {object_type}s found for export.")

    # Export buildings and roads
    export_mesh_obj_to_ply("building", output_folder, logger)
    export_mesh_obj_to_ply("road", output_folder, logger)

if "sionna" in output_formats:
    logger.warning("LOG: ⚠ Nothing to DO HERE YET!!!!!")

    import sys 

    # Add project root to sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    import bpy # type: ignore
    import bmesh # type: ignore

    from .blender_utils import get_obj_by_name, get_bounding_box, clear_blender, compute_distance

    ADDONS = {
        "blosm": "blosm_2.7.11.zip",
        "mitsuba-blender": "mitsuba-blender.zip",
    }

    # Material names for scene objects
    BUILDING_MATERIAL = 'itu_concrete'
    ROAD_MATERIAL = 'itu_brick'  # Note: Manually changed to asphalt in Sionna
    FLOOR_MATERIAL = 'itu_wet_ground'
    PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

    import subprocess
    import time
    
    # Install required packages
    def install_python_package(pckg_name):
        """
        Install a Python package using Blender's Python executable.
        
        Args:
            pckg_name (str): Name of the package to install (e.g., 'mitsuba==3.5.0')
        """
        logger.info(f"📦 Installing Python package: {pckg_name}")
        python_exe = sys.executable
        logger.debug(f"Using Python executable: {python_exe}")
        
        try:
            subprocess.call([python_exe, "-m", "ensurepip"])
            subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.call([python_exe, "-m", "pip", "install", pckg_name])
            logger.info(f"✅ Successfully installed {pckg_name}")
        except Exception as e:
            logger.error(f"❌ Failed to install {pckg_name}: {str(e)}")
            raise

    def install_blender_addon(addon_name):
        """
        Install and enable a Blender add-on from a zip file if not already installed.
        
        Args:
            addon_name (str): Name of the add-on to install (e.g., 'blosm')
        """
        logger.info(f"🔧 Processing Blender add-on: {addon_name}")
        zip_name = ADDONS.get(addon_name)
        if not zip_name:
            logger.error(f"❌ No zip file defined for add-on '{addon_name}'")
            return
        
        logger.debug(f"Currently installed add-ons: {list(bpy.context.preferences.addons.keys())}")
        
        # Check if add-on is already installed
        if addon_name in bpy.context.preferences.addons.keys():
            logger.info(f"📌 Add-on '{addon_name}' is already installed")
            if bpy.context.preferences.addons[addon_name].module:
                logger.info(f"✅ Add-on '{addon_name}' is enabled")
            else:
                logger.info(f"�� Enabling add-on '{addon_name}'")
                bpy.ops.preferences.addon_enable(module=addon_name)
                bpy.ops.wm.save_userpref()
                logger.info(f"✅ Add-on '{addon_name}' has been enabled")
        else:
            logger.info(f"📥 Installing new add-on '{addon_name}'")
            addon_zip_path = os.path.join(PROJ_ROOT, "blender_addons", zip_name)
            try:
                bpy.ops.preferences.addon_install(filepath=addon_zip_path)
                bpy.ops.preferences.addon_enable(module=addon_name)
                bpy.ops.wm.save_userpref()
                logger.info(f"✅ Add-on '{addon_name}' installed and enabled")
            except Exception as e:
                logger.error(f"❌ Failed to install/enable add-on '{addon_name}': {str(e)}")
                raise
        
        # Special handling for Mitsuba
        if addon_name == 'mitsuba-blender':
            try:
                import mitsuba
                logger.info("✅ Mitsuba import successful")
            except ImportError:
                logger.info("📦 Mitsuba not found, installing mitsuba package")
                install_python_package('mitsuba==3.5.0')
                logger.warning("🔄 Packages installed! Restarting Blender to update imports")
                time.sleep(5)
                sys.exit()

    # Configure OSM import
    def configure_osm_import(scene_folder, min_lat, max_lat, min_lon, max_lon):
        """Configure blosm add-on for OSM data import."""
        logger.info(f"🗺️ Configuring OSM import for region: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]")
        try:
            prefs = bpy.context.preferences.addons["blosm"].preferences
            prefs.dataDir = scene_folder
            logger.debug(f"Set OSM data directory to: {scene_folder}")
            
            scene = bpy.context.scene.blosm
            scene.mode = '3Dsimple'
            scene.minLat, scene.maxLat = min_lat, max_lat
            scene.minLon, scene.maxLon = min_lon, max_lon
            scene.buildings, scene.highways = True, True
            scene.water, scene.forests, scene.vegetation, scene.railways = False, False, False, False
            scene.singleObject, scene.ignoreGeoreferencing = True, True
            logger.info("✅ OSM import configuration complete")
        except Exception as e:
            logger.error(f"❌ Failed to configure OSM import: {str(e)}")
            raise

    def create_ground_plane(min_lat, max_lat, min_lon, max_lon):
        """Create and size a ground plane with FLOOR_MATERIAL."""
        logger.info("🌍 Creating ground plane")
        try:
            bpy.ops.mesh.primitive_plane_add(size=1)
            x_size = compute_distance([min_lat, min_lon], [min_lat, max_lon]) * 1.2
            y_size = compute_distance([min_lat, min_lon], [max_lat, min_lon]) * 1.2
            logger.debug(f"Ground plane dimensions: [{x_size}, {y_size}]")
            
            plane = get_obj_by_name("Plane")
            plane.scale = (x_size, y_size, 1)
            plane.name = 'terrain'
            
            floor_material = bpy.data.materials.new(name=FLOOR_MATERIAL)
            plane.data.materials.append(floor_material)
            logger.info("✅ Ground plane created and configured")
            return plane
        except Exception as e:
            logger.error(f"❌ Failed to create ground plane: {str(e)}")
            raise

    def join_and_materialize_objects(name_pattern, target_name, material):
        """Join objects matching a name pattern and apply a material."""
        logger.info(f"🔄 Processing objects matching pattern: {name_pattern}")
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for o in bpy.data.objects:
                if name_pattern in o.name.lower() and o.type == 'MESH':
                    o.select_set(True)
            
            selected = bpy.context.selected_objects
            logger.info(f"📊 Found {len(selected)} objects matching '{name_pattern}'")
            
            if len(selected) > 1:
                logger.debug("Joining multiple objects")
                bpy.context.view_layer.objects.active = selected[-1]
                bpy.ops.object.join()
            elif len(selected) == 1:
                logger.debug("Single object found, setting as active")
                bpy.context.view_layer.objects.active = selected[0]
            elif not selected:
                logger.warning(f"⚠️ No objects found matching pattern: {name_pattern}")
                return None
            
            obj = bpy.context.active_object
            obj.name = target_name
            obj.data.materials.clear()
            obj.data.materials.append(material)
            logger.info(f"✅ Successfully created and materialized {target_name}")
            return obj
        except Exception as e:
            logger.error(f"❌ Failed to process objects with pattern '{name_pattern}': {str(e)}")
            raise

    def trim_faces_outside_bounds(obj, min_x, max_x, min_y, max_y):
        """Remove faces of an object outside a bounding box in world space."""
        logger.info(f"✂️ Trimming faces outside bounds for object: {obj.name}")
        logger.debug(f"Bounds: x[{min_x}, {max_x}], y[{min_y}, {max_y}]")
        
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            matrix_world = obj.matrix_world
            faces_to_delete = []
            
            total_faces = len(bm.faces)
            for face in bm.faces:
                center = bpy.mathutils.Vector((0, 0, 0))
                for vert in face.verts:
                    center += vert.co
                center /= len(face.verts)
                world_center = matrix_world @ center
                x, y = world_center.x, world_center.y
                if (x < min_x or x > max_x) or (y < min_y or y > max_y):
                    faces_to_delete.append(face)
            
            if faces_to_delete:
                logger.info(f"🗑️ Removing {len(faces_to_delete)} faces out of {total_faces}")
                bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
                bmesh.update_edit_mesh(obj.data)
            else:
                logger.info("✅ No faces needed trimming")
            
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception as e:
            logger.error(f"❌ Failed to trim faces for {obj.name}: {str(e)}")
            raise

    def setup_world_lighting():
        """Configure world lighting with a basic emitter."""
        logger.info("💡 Setting up world lighting")
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
            logger.info("✅ World lighting configured")
        except Exception as e:
            logger.error(f"❌ Failed to setup world lighting: {str(e)}")
            raise

    def create_camera_and_render(scene, output_path, location=(0, 0, 1000), rotation=(0, 0, 0)):
        """Add a camera, render the scene, and delete the camera."""
        logger.info(f"📸 Setting up camera for render at {output_path}")
        logger.debug(f"Camera position: {location}, rotation: {rotation}")
        
        try:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.camera_add(location=location, rotation=rotation)
            camera = bpy.context.active_object
            scene.camera = camera
            
            scene.render.filepath = output_path
            logger.info("🎬 Starting render")
            bpy.ops.render.render(write_still=True)
            logger.info("✅ Render complete")
            
            bpy.ops.object.select_all(action='DESELECT')
            camera.select_set(True)
            bpy.ops.object.delete()
            logger.debug("Temporary camera removed")
        except Exception as e:
            logger.error(f"❌ Failed to render scene: {str(e)}")
            raise

    def save_osm_origin(scene_folder, origin_lat, origin_lon):
        """Save OSM origin coordinates to a text file."""
        logger.info(f"📍 Saving OSM origin coordinates: [{origin_lat}, {origin_lon}]")
        try:
            output_path = os.path.join(scene_folder, 'osm_gps_origin.txt')
            with open(output_path, 'w') as f:
                f.write(f"{origin_lat}\n{origin_lon}\n")
            logger.info("✅ OSM origin saved")
        except Exception as e:
            logger.error(f"❌ Failed to save OSM origin: {str(e)}")
            raise

    def export_scene(scene_folder):
        """Export scene to Mitsuba and save .blend file."""
        logger.info("📤 Exporting scene")
        try:
            mitsuba_path = os.path.join(scene_folder, 'scene.xml')
            blend_path = os.path.join(scene_folder, 'scene.blend')
            
            logger.debug(f"Exporting Mitsuba scene to: {mitsuba_path}")
            bpy.ops.export_scene.mitsuba(
                filepath=mitsuba_path,
                export_ids=True, axis_forward='Y', axis_up='Z'
            )
            
            logger.debug(f"Saving Blender file to: {blend_path}")
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
            logger.info("✅ Scene export complete")
        except Exception as e:
            logger.error(f"❌ Failed to export scene: {str(e)}")
            raise

    def create_scene(positions, out_folder):
        """Create scenes from CSV positions with OSM data and export."""
        logger.info(f"🎨 Creating new scene in: {out_folder}")
        scene_name = os.path.basename(out_folder)
        
        try:
            os.makedirs(out_folder, exist_ok=True)
            output_fig_folder = os.path.join(out_folder, 'figs')
            os.makedirs(output_fig_folder, exist_ok=True)
            logger.debug(f"Created output directories: {out_folder}, {output_fig_folder}")

            min_lat, min_lon, max_lat, max_lon = positions
            logger.info(f"📍 Scene bounds: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]")

            # Initialize scene
            logger.info("🔄 Initializing scene")
            clear_blender()
            setup_world_lighting()

            # Import OSM data
            logger.info("🗺️ Importing OSM data")
            configure_osm_import(out_folder, min_lat, max_lat, min_lon, max_lon)
            bpy.ops.blosm.import_data()
            origin_lat = bpy.data.scenes["Scene"]["lat"]
            origin_lon = bpy.data.scenes["Scene"]["lon"]
            save_osm_origin(out_folder, origin_lat, origin_lon)
            
            # Create ground plane
            logger.info("🌍 Setting up terrain")
            terrain = create_ground_plane(min_lat, max_lat, min_lon, max_lon)
            terrain_bounds = get_bounding_box(terrain)

            # Create materials
            logger.info("🎨 Creating materials")
            building_material = bpy.data.materials.new(name=BUILDING_MATERIAL)
            building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige
            road_material = bpy.data.materials.new(name=ROAD_MATERIAL)
            road_material.diffuse_color = (0.29, 0.25, 0.21, 1)  # Dark grey

            # Convert all to meshes
            logger.info("🔄 Converting objects to meshes")
            bpy.ops.object.select_all(action='SELECT')
            bpy.context.view_layer.objects.active = bpy.data.objects[0]
            bpy.ops.object.convert(target='MESH', keep_original=False)

            # Render original scene
            logger.info("📸 Rendering original scene")
            scene = bpy.context.scene
            create_camera_and_render(scene, os.path.join(output_fig_folder, f"{scene_name}_org.png"))

            # Process buildings
            logger.info("🏢 Processing buildings")
            buildings = join_and_materialize_objects('building', 'buildings', building_material)
            if buildings and buildings.type == 'MESH':
                logger.debug("Separating building meshes")
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.separate(type='LOOSE')
                bpy.ops.object.mode_set(mode='OBJECT')

            # Process roads
            logger.info("🛣️ Processing roads")
            bpy.ops.object.select_all(action='DESELECT')
            for o in bpy.data.objects:
                if 'terrain' in o.name.lower() or 'buildings' in o.name.lower():
                    o.select_set(True)
            bpy.ops.object.select_all(action='INVERT')
            road_objs = bpy.context.selected_objects
            if road_objs:
                logger.info(f"Found {len(road_objs)} road objects to process")
                for obj in road_objs:
                    if terrain_bounds:
                        trim_faces_outside_bounds(obj, *terrain_bounds)
                    bpy.context.view_layer.objects.active = obj
                    road = bpy.context.active_object
                    road.data.materials.clear()
                    road.data.materials.append(road_material)
                logger.debug(f"Processed {len(road_objs)} road objects")

            # Render processed scene
            logger.info("📸 Rendering processed scene")
            create_camera_and_render(scene, os.path.join(output_fig_folder, f"{scene_name}_processed.png"))

            # Export scene
            logger.info("📤 Exporting final scene")
            export_scene(out_folder)
            logger.info("✅ Scene creation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to create scene: {str(e)}")
            raise

if False:
    # Install required add-ons
    for addon_name in ADDONS:
        install_blender_addon(addon_name)

    # Create scenes
    create_scene([minlat, minlon, maxlat, maxlon], output_folder)

    
bpy.ops.wm.quit_blender()
