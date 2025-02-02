import os
import bpy
import csv
import math
from datetime import datetime as dt

# Developer notes: 
# To stop blender execution, use raise Exception("Stop here!")
# To see blender output (with UI open), go to Window -> Toggle Console

PROJ_ROOT = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/' # use / not \

WALL_MATERIAL = 'itu_marble' # 'itu_marble', 'itu_plasterboard', 'itu_brick'
ROOF_MATERIAL = 'itu_metal'
FLOOR_MATERIAL = 'itu_concrete'
OTHERS_MATERIAL = 'itu_wood' # all the other materials (should be very unlikely!)
DEFAULT_MATERIALS = [WALL_MATERIAL, ROOF_MATERIAL, FLOOR_MATERIAL, OTHERS_MATERIAL]

# List walls/roofs materials in order of priority/popularity in OSM
KNOWN_WALL_OSM_MATERIALS = ['wall']
KNOWN_ROOF_OSM_MATERIALS = ['roof']


################## Automatic install of add-ons ########################
addons = {
    # From https://github.com/vvoovv/blosm
    "blosm": "blosm_2.7.11.zip",  
    
    # From https://github.com/mitsuba-renderer/mitsuba-blender/tree/latest
    # "mistuba-blender": "mistuba-blender.zip",
    # This doesn't work... Needs manual installation of the mitsuba exporter
    # 1. Download the zip in https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.4.0
    # 2. Follow the manual install instructions:
    #    2.1. In Blender, go to Edit -> Preferences -> Add-ons -> Install.
    #    2.2. Select the downloaded ZIP archive.
    #    2.3. Find the add-on using the search bar and enable it.
    #    2.4. Click on "Install dependencies using pip" to download the latest package
    }

def install_blender_addon(addon_name, zip_name):
    """ Installs a blender add-on from a zip file if not yet installed. """
    print("Installed add-ons:", list(bpy.context.preferences.addons.keys()))

    # Check if the add-on is already installed
    if addon_name in bpy.context.preferences.addons.keys():
        print(f"The add-on '{addon_name}' is already installed.")
        
        # Check if it's enabled
        if bpy.context.preferences.addons[addon_name].module:
            print(f"The add-on '{addon_name}' is enabled.")
        else:
            # Enable Add-on
            bpy.ops.preferences.addon_enable(module=addon_name)

            # Save the preferences so the add-on stays enabled after restarting Blender
            bpy.ops.wm.save_userpref()

            print(f"The add-on '{addon_name}' has been enabled.")
    else:
        print(f"The add-on '{addon_name}' is not installed or enabled.")
        
        # Replace with the path to the add-on folder
        addon_zip_path = PROJ_ROOT + 'blender_addons/' + zip_name

        # Install the add-on
        bpy.ops.preferences.addon_install(filepath=addon_zip_path)

        # Enable the add-on
        bpy.ops.preferences.addon_enable(module=addon_name)

        # Save user preferences
        bpy.ops.wm.save_userpref()

        print(f"Add-on '{addon_name}' installed and enabled.")
    
########################################################################

for addon_name, zip_name in addons.items():
    install_blender_addon(addon_name, zip_name)

# raise Exception("Stop here!")

def compute_distance(coord1, coord2):
    """
    Computes the Haversine distance between coordinates in meters.
    At a 10km distance, the error is ~1m which is negligible for this use case.
    For perflectly accurate distances on the face of the earth, consider:
        A) using the GeoPy package. This function would then change to just:
        return geopy.distance.geodesic(coord1, coord2).meters
        B) Implementing 
    # Example usage
    coord1 = (41.49008, -71.312796)  # Newport, RI
    coord2 = (41.499498, -81.695391)  # Cleveland, OH

    distance = compute_distance(point1, point2)
    print(f"Distance: {distance:.2f} km")
    """
    
    # Radius of Earth in kilometers
    R = 6371.0  
    
    # Unpack latitude and longitude, convert to radians
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in meters
    distance = R * c * 1000
    return distance


# Blender utils
def get_objs_with_material(mat):
    objs_with_mat = []
    for obj in bpy.data.objects:
        for slot in obj.material_slots:
            if slot.material == mat:
                objs_with_mat += [obj]
                break
    return objs_with_mat

def get_obj_by_name(name):
    for obj in bpy.data.objects:
        if obj.name == name:
            return obj
    return None

def get_slot_of_material(obj, mat):
    for slot_idx, slot in enumerate(obj.material_slots):
        if slot.material == mat:
            return slot_idx
    return -1

def clear_blender():
    block_lists = [ bpy.data.collections,
                    bpy.data.objects, 
                    bpy.data.meshes,
                    bpy.data.materials, 
                    bpy.data.textures, 
                    bpy.data.images,
                    bpy.data.curves, 
                    bpy.data.cameras
                    ]
    for block_list in block_lists:
        for block in block_list:
            block_list.remove(block, do_unlink=True)

time_str = dt.now().strftime("%m-%d-%Y_%HH%MM%SS")

osm_folder = PROJ_ROOT + f'all_runs/run_{time_str}/'
csv_path = PROJ_ROOT + 'params.csv'

# Read CSV using the built-in CSV module
positions = []
with open(csv_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        positions.append(row)

# Save a small file with the path of the folder containing all the maps
with open(PROJ_ROOT + 'scenes_folder.txt', 'w') as fp:
    fp.write(osm_folder + '\n')

# Loop through the rows and create scenario folders
for i, row in enumerate(positions):
    scen_folder = osm_folder + f"/scen_{i}/"

    os.makedirs(scen_folder, exist_ok=True)
    
    min_lat = float(row['min_lat'])
    max_lat = float(row['max_lat'])
    min_lon = float(row['min_lon'])
    max_lon = float(row['max_lon'])
    
    # 0- Clean Blender
    clear_blender()
    

    # 1- Configure OSM map fetching
    bpy.context.preferences.addons["blosm"].preferences.dataDir = scen_folder
    
    bpy.context.scene.blosm.mode = '3Dsimple'

    bpy.context.scene.blosm.minLat = min_lat
    bpy.context.scene.blosm.maxLat = max_lat
    bpy.context.scene.blosm.minLon = min_lon
    bpy.context.scene.blosm.maxLon = max_lon

    bpy.context.scene.blosm.buildings = True
    bpy.context.scene.blosm.water = False
    bpy.context.scene.blosm.forests = False
    bpy.context.scene.blosm.vegetation = False
    bpy.context.scene.blosm.highways = False
    bpy.context.scene.blosm.railways = False
    bpy.context.scene.blosm.singleObject = False
    bpy.context.scene.blosm.ignoreGeoreferencing = True


    # 2- Fetch map
    bpy.ops.blosm.import_data()
    
    # 3- Ground Plane
    # 3.1- Create ground plane 
    bpy.ops.mesh.primitive_plane_add(size=1)

    # 3.2- Resize plane to fit area fetched (bit bigger than fetched area)
    x_size = compute_distance([min_lat, min_lon], [min_lat, max_lon]) * 1.2
    y_size = compute_distance([min_lat, min_lon], [max_lat, min_lon]) * 1.2
    
    print(f'Creating plane of size [{x_size}, {y_size}')
    plane = get_obj_by_name("Plane")
    plane.scale = (x_size, y_size, 1)
    
    # 3.3- Create FLOOR_MATERIAL and assign to ground plane
    floor_material = bpy.data.materials.new(name=FLOOR_MATERIAL)
    plane.data.materials.append(floor_material)


    # 4- Walls and Roofs
    # 4.1- Create default materials for walls, roofs and others
    wall_material = bpy.data.materials.new(name=WALL_MATERIAL)
    roof_material = bpy.data.materials.new(name=ROOF_MATERIAL)
    others_material = bpy.data.materials.new(name=OTHERS_MATERIAL)
    
    # 4.2- Set colors
    roof_material.diffuse_color   = (0.29, 0.25, 0.21, 1) # dark grey
    wall_material.diffuse_color   = (0.75, 0.40, 0.16, 1) # beije
    others_material.diffuse_color = (0.17, 0.09, 0.02, 1) # brown

    # 4.3- Find all objects with each material
    wall_assigned = False
    roof_assigned = False
    others_assigned = False
    
    for mat in bpy.data.materials:
        if mat.name in DEFAULT_MATERIALS:
            continue
        
        print(f'Material name: {mat.name}')
        
        objs = get_objs_with_material(mat)
        
        # 4.4- Determine what the material is used for (walls/roofs/unknown)
        if mat.name in KNOWN_WALL_OSM_MATERIALS:
            replace_mat = wall_material
        
        elif mat.name in KNOWN_ROOF_OSM_MATERIALS:
            replace_mat = roof_material
        
        else:
            replace_mat = others_material
            print(f'Unknown material found! Material name: {mat.name}')
            for obj in objs:
                print(f'Material name: {mat.name} | Obj {obj.name}')
            with open(osm_folder + 'unknown_materials.txt', 'a') as fp:
                fp.write(mat.name+'\n')
            
        # 4.5- Replace with the corresponding pre-defined material
        for obj in objs:
            idx_of_mat = get_slot_of_material(obj, mat)
            obj.data.materials[idx_of_mat] = replace_mat
        
        # 4.6- Delete material (to have only pre-def. materials in the scene)
        bpy.data.materials.remove(mat)
        
    # 6- Save scene as Mitsbua (XML)
    bpy.ops.export_scene.mitsuba(
        filepath=scen_folder + "scene.xml",
        export_ids=True,
        axis_forward='Y',
        axis_up='Z')
