"""
Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box)
Each scenario's output is stored in a folder named after its bounding box.
"""

import bpy # type: ignore
import os
import sys
import logging

# Setup logging to both console and file
log_file_path = "C:/Users/jmora/Downloads/osm_root/logging.txt"
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

# Export based on the selected format
if "insite" in output_formats:
    # Export Buildings
    bpy.ops.object.select_all(action="DESELECT")
    buildings = [o for o in bpy.data.objects if "building" in o.name.lower()]
    for b in buildings:
        b.select_set(True)
        
    if buildings:
        logger.info(f"🏗 Exporting {len(buildings)} buildings to .ply")
        bpy.ops.export_mesh.ply(filepath=os.path.join(output_folder, "buildings.ply"), use_ascii=True)
    else:
        logger.warning("⚠ No buildings found for export.")

    # Export Roads
    bpy.ops.object.select_all(action="DESELECT")
    roads = [o for o in bpy.data.objects if "road" in o.name.lower()]
    for r in roads:
        r.select_set(True)

    if roads:
        logger.info(f"🛣 Exporting {len(roads)} roads to .ply")
        bpy.ops.export_mesh.ply(filepath=os.path.join(output_folder, "roads.ply"), use_ascii=True)
    else:
        logger.warning("⚠ No roads found for export.")

if "sionna" in output_formats:
    logger.warning("LOG: ⚠ Nothing to DO HERE YET!!!!!")

bpy.ops.wm.quit_blender()
