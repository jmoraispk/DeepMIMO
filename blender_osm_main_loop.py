import bpy
import pandas as pd
import os

deepmimo1000_root = "/home/joao/Documents/DeepMIMO1000-master/osm_test"
csv_path = deepmimo1000_root + '/scenario_pos.csv'
scenario_pos = pd.read_csv(csv_path)
num_scenarios = scenario_pos.shape[0]
num_scenarios = 1
save_dae = False

for i in range(num_scenarios):
    project_root = deepmimo1000_root + "/scenarios/run_%d/" % i
    name = "DeepMIMO1000"

    if not os.path.exists(project_root):
        os.makedirs(project_root)

    minlat = scenario_pos.iloc[i]['minlat']
    minlon = scenario_pos.iloc[i]['minlon']
    maxlat = scenario_pos.iloc[i]['maxlat']
    maxlon = scenario_pos.iloc[i]['maxlon']

    # empty the blender scenario
    colls = bpy.data.collections
    for i in colls:
        colls.remove(colls[i.name], do_unlink=True)
    objs = bpy.data.objects
    for i in objs:
        objs.remove(objs[i.name], do_unlink=True)

    # download and import osm data to blender 
    bpy.context.preferences.addons["blosm"].preferences.dataDir = project_root
    
    bpy.context.scene.blosm.mode = '3Dsimple'

    bpy.context.scene.blosm.minLon = minlon
    bpy.context.scene.blosm.maxLon = maxlon
    bpy.context.scene.blosm.minLat = minlat
    bpy.context.scene.blosm.maxLat = maxlat

    bpy.context.scene.blosm.buildings = True
    bpy.context.scene.blosm.water = False
    bpy.context.scene.blosm.forests = False
    bpy.context.scene.blosm.vegetation = False
    bpy.context.scene.blosm.highways = False
    bpy.context.scene.blosm.railways = False
    bpy.context.scene.blosm.singleObject = False
    bpy.context.scene.blosm.ignoreGeoreferencing = True

    bpy.ops.blosm.import_data()
    
    # bpy.context.space_data.overlay.show_relationship_lines = False
    
    # change size and scale based on coordinates
    bpy.ops.mesh.primitive_plane_add(size=400, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    
    bpy.data.materials["itu_concrete"].node_tree.nodes["Diffuse BSDF"].inputs[0].default_value = (0.159729, 0.258736, 0.8, 1)

    
    
    
    
    
    
    bpy.context.object.active_material.name = "itu_concrete"

    
    
    # save osm origin
    origin_lat = bpy.data.scenes["Scene"]["lat"]
    origin_lon = bpy.data.scenes["Scene"]["lon"]

    with open(project_root+'osm_gps_origin.txt', 'w') as file:
        file.write(f"{origin_lat}\n{origin_lon}\n")
        
    # remove material texture images as they are not compatibale with WI
    for img in bpy.data.images:
        bpy.data.images.remove(img)
        
    # convert all objects to mesh
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.convert(target='MESH', keep_original=False)

    # extract and save buildings
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        if "building" in o.name:
            o.select_set(True)
    print(
        "Number of selected object groups (Building): %d"
        % (len(bpy.context.selected_objects))
    )
    bpy.ops.export_mesh.ply(
        filepath=project_root + name + "_building.ply",
        use_ascii=True,
        use_selection=True,
        use_normals=False,
        use_uv_coords=False,
        use_colors=False,
    )
    if save_dae:
        bpy.ops.wm.collada_export(
            filepath=project_root + name + "_building.dae",
            selected=True,
        )

    # extract and save roads
    bpy.ops.object.select_all(action='INVERT')

    print(
        "Number of selected object groups (Road): %d"
        % (len(bpy.context.selected_objects))
    )
    bpy.ops.export_mesh.ply(
        filepath=project_root + name + "_road.ply",
        use_ascii=True,
        use_selection=True,
        use_normals=False,
        use_uv_coords=False,
        use_colors=False,
    )
    if save_dae:
        bpy.ops.wm.collada_export(
            filepath=project_root + name + "_road.dae",
            selected=True,
        )

    


