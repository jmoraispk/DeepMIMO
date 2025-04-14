"""
Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box)
Each scenario's output is stored in a folder named after its bounding box.
"""

import bpy # type: ignore
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.blender_utils import (log_local_setup, install_all_addons,
                                 set_LOGGER, create_scene, save_osm_gps_origin,
                                 save_scenario_metadata, convert_objects_to_mesh,
                                 export_mesh_obj_to_ply, clear_blender, configure_osm_import)

# Setup logging to both console and file (great for debugging)
logger = log_local_setup('logging_blender_osm.txt')
set_LOGGER(logger)  # So the inner functions can log

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

# Clear existing objects in Blender
clear_blender()

# Configure OSM import
configure_osm_import(output_folder, minlat, maxlat, minlon, maxlon)

# Export based on the selected format
if "insite" in output_formats:
    
    bpy.ops.blosm.import_data()
    logger.info("✅ OSM data imported successfully.")

    # Save OSM GPS origin
    save_origin = True
    if save_origin:
        save_osm_gps_origin(output_folder)

    # Save scenario properties to a metadata file
    save_bbox_metadata = False
    if save_bbox_metadata:
        save_scenario_metadata(output_folder, minlat, minlon, maxlat, maxlon, output_formats)

    # Convert objects to mesh
    convert_objects_to_mesh()

    # Export buildings and roads
    export_mesh_obj_to_ply("building", output_folder, logger)
    export_mesh_obj_to_ply("road", output_folder, logger)

if "sionna" in output_formats:
    print('inside SIONNA')

    # Automatically install all addons
    install_all_addons()

    # Create scene
    try:
        create_scene([minlat, minlon, maxlat, maxlon], output_folder)

    except Exception as e:
        err = f"❌ Failed to create scene: {str(e)}"
        logger.error(err)
        raise Exception(err)

# Quit Blender   
bpy.ops.wm.quit_blender()
