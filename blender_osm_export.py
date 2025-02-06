"""
Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box)
Each scenario's output is stored in a folder named after its bounding box.
"""

import bpy
import os
import sys
import logging
import pandas as pd
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Detect Operating System
OS_TYPE = platform.system()

# Required Python packages
REQUIRED_PACKAGES = ["pandas"]

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Check for missing packages
missing_packages = [pkg for pkg in REQUIRED_PACKAGES if not is_package_installed(pkg)]

if missing_packages:
    logger.error(f"❌ Missing required Python packages: {', '.join(missing_packages)}")

    # Provide installation instructions based on OS
    logger.info("\n🔹 How to Install Missing Python Packages:\n")
    
    if OS_TYPE == "Darwin":  # MacOS
        logger.info("🔹 MacOS:")
        logger.info("Run the following command in the terminal:")
        logger.info('''  /Applications/Blender.app/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target=/Applications/Blender.app/Contents/Resources/4.3/python/lib/python3.11/site-packages pandas pytz python-dateutil matplotlib''')

    elif OS_TYPE == "Windows":  # Windows
        logger.info("🔹 Windows Machine (Standard Installation):")
        logger.info("Run the following command in Command Prompt (cmd):")
        logger.info('''  "C:\\Program Files\\Blender Foundation\\Blender 3.5\\3.5\\python\\bin\\python.exe" -m pip install --target="C:\\Program Files\\Blender Foundation\\Blender 3.5\\3.5\\python\\lib\\site-packages" pandas pytz python-dateutil matplotlib''')
        
        logger.info("\n🔹 Windows Machine (Portable Blender - If the above doesn't work):")
        logger.info("Run the following command in Command Prompt (cmd):")
        logger.info('''  "C:\\Users\\{user_name}\\blender-4.3.2-windows-x64\\4.3\\python\\bin\\python.exe" -m pip install --target="C:\\Users\\{user_name}\\blender-4.3.2-windows-x64\\4.3\\python\\lib\\site-packages" pandas pytz python-dateutil matplotlib''')
        logger.info("💡 Make sure to place Portable Blender somewhere in Drive C before running the command.")

    logger.info("🔄 After installation, restart Blender and try running the script again.")
    exit(1)

# Function to check if an add-on is enabled
def is_addon_enabled(addon_name):
    return addon_name in bpy.context.preferences.addons.keys()

# Ensure Blender OSM Add-on (Blosm) is available
if not is_addon_enabled("blosm"):
    logger.error("❌ Blender OSM addon ('blosm') is not enabled.")

    # Provide installation instructions
    logger.info("\n🔹 How to Install the Blender OSM Addon (Blosm):\n")
    logger.info("1️⃣ Download Blosm from:")
    logger.info("   🔗 https://prochitecture.gumroad.com/l/blender-osm")
    logger.info("2️⃣ Open Blender and go to: Edit > Preferences > Add-ons")
    logger.info("3️⃣ Click 'Install from disk' and select the downloaded .zip file.")
    logger.info("4️⃣ Restart Blender to complete the installation.")

    exit(1)

# Ensure PLY Import Add-on is available
if not is_addon_enabled("Addon-Build"):  # Use the actual module name
    logger.error("❌ PLY Import addon ('PLY_As_Verts') is not enabled.")

    # Provide installation instructions
    logger.info("\n🔹 How to Install the PLY Import Addon:\n")
    logger.info("1️⃣ Download the addon from:")
    logger.info("   🔗 https://github.com/TombstoneTumbleweedArt/import-ply-as-verts")
    logger.info("2️⃣ Open Blender and go to: Edit > Preferences > Add-ons")
    logger.info("3️⃣ Click 'Install' and select the downloaded .zip file.")
    logger.info("4️⃣ Enable the add-on by checking the box next to its name.")
    logger.info("5️⃣ Restart Blender to complete the installation.")

    exit(1)

logger.info("✅ All required packages and add-ons are installed. Continuing execution...")

# Parse command-line arguments
csv_path = None
try:
    if "--csv" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--csv") + 1]
    else:
        minlat = float(sys.argv[sys.argv.index("--minlat") + 1])
        minlon = float(sys.argv[sys.argv.index("--minlon") + 1])
        maxlat = float(sys.argv[sys.argv.index("--maxlat") + 1])
        maxlon = float(sys.argv[sys.argv.index("--maxlon") + 1])
except (ValueError, IndexError):
    logger.error("❌ Invalid input format. Provide explicit coordinates (--minlat --minlon --maxlat --maxlon) or a CSV file (--csv path).")
    exit(1)

# Output directory setup
OS_TYPE = platform.system()
if OS_TYPE == "Darwin":  # MacOS
    deepmimo_root = "/Users/sadjadalikhani/Desktop/deepmimo/scenario_generator"
elif OS_TYPE == "Windows":  # Windows
    deepmimo_root = "C:\\Users\\salikha4\\Desktop\\scenario_generator"
else:
    logger.error("❌ Unsupported operating system. This script only works on MacOS and Windows.")
    sys.exit(1)

# Load CSV if provided, else process the single manual input
if csv_path:
    logger.info(f"📂 Loading CSV file: {csv_path}")
    try:
        scenarios = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"❌ Failed to read CSV: {e}")
        exit(1)
else:
    scenarios = pd.DataFrame([{"minlat": minlat, "minlon": minlon, "maxlat": maxlat, "maxlon": maxlon}])

# Process each scenario one by one
for index, scenario in scenarios.iterrows():
    minlat, minlon, maxlat, maxlon = scenario["minlat"], scenario["minlon"], scenario["maxlat"], scenario["maxlon"]

    # Create folder name based on bbox coordinates
    bbox_str = f"bbox_{minlat}_{minlon}_{maxlat}_{maxlon}".replace(".", "-")
    scenario_folder = os.path.join(deepmimo_root, "osm_exports", bbox_str)
    os.makedirs(scenario_folder, exist_ok=True)

    logger.info(f"📍 Processing Scenario: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]")
    logger.info(f"📂 Saving output to: {scenario_folder}")

    # Clear existing objects in Blender
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Configure OSM import
    bpy.context.preferences.addons["blosm"].preferences.dataDir = scenario_folder
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
    with open(os.path.join(scenario_folder, "osm_gps_origin.txt"), "w") as file:
        file.write(f"{origin_lat}\n{origin_lon}\n")
    logger.info("📍 OSM GPS origin saved.")

    # Save scenario properties to a metadata file
    metadata_path = os.path.join(scenario_folder, "scenario_info.txt")
    with open(metadata_path, "w") as meta_file:
        meta_file.write(f"Bounding Box: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]\n")
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
    bpy.ops.object.select_all(action="DESELECT")
    buildings = [o for o in bpy.data.objects if "building" in o.name.lower()]
    for b in buildings:
        b.select_set(True)
        
    if buildings:
        logger.info(f"🏗 Exporting {len(buildings)} buildings to .ply")
        bpy.ops.export_mesh.ply(filepath=os.path.join(scenario_folder, "buildings.ply"), use_ascii=True)
    else:
        logger.warning("⚠ No buildings found for export.")

    # Export Roads
    bpy.ops.object.select_all(action="DESELECT")
    roads = [o for o in bpy.data.objects if "road" in o.name.lower()]
    for r in roads:
        r.select_set(True)

    if roads:
        logger.info(f"🛣 Exporting {len(roads)} roads to .ply")
        bpy.ops.export_mesh.ply(filepath=os.path.join(scenario_folder, "roads.ply"), use_ascii=True)
    else:
        logger.warning("⚠ No roads found for export.")

logger.info("✅ All scenarios processed successfully.")
bpy.ops.wm.quit_blender()
