import os
import bpy
import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
from geopy.distance import geodesic

WALL_MATERIAL = 'itu_marble' # 'itu_marble', 'itu_plasterboard', 'itu_brick'
ROOF_MATERIAL = 'itu_metal'
FLOOR_MATERIAL = 'itu_concrete'
OTHERS_MATERIAL = 'itu_wood' # all the other materials (should be very unlikely!)
DEFAULT_MATERIALS = [WALL_MATERIAL, ROOF_MATERIAL, FLOOR_MATERIAL, OTHERS_MATERIAL]

# List walls/roofs materials in order of priority/popularity in OSM
KNOWN_WALL_OSM_MATERIALS = ['wall']
KNOWN_ROOF_OSM_MATERIALS = ['roof']


def compute_distance(coord1, coord2):
    " Returns distance between coordinates in meters."
    return  geodesic(coord1, coord2).meters

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

proj_root = '/home/joao/Documents/GitHub/SionnaProjects/AutoRayTracing/'
osm_folder = proj_root + f'all_runs/run_{time_str}/'
csv_path = proj_root + 'params.csv'

pos_df = pd.read_csv(csv_path)

# Save a small file with the path of the folder containing all the maps
with open(proj_root + 'scenes_folder.txt', 'w') as fp:
    fp.write(osm_folder + '\n')

for i in range(pos_df.index.stop):
    scen_folder = osm_folder + f"/scen_{i}/"

    os.makedirs(scen_folder, exist_ok=True)
    
    min_lat = pos_df['min_lat'][i]
    max_lat = pos_df['max_lat'][i]
    min_lon = pos_df['min_lon'][i]
    max_lon = pos_df['max_lon'][i]
    
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
            replace_mat = wall_material #WALL_MATERIAL ##############3
        
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
