# core/scene_builder.py
import os
import bpy
import csv
from datetime import datetime as dt
from config.materials import *
from config.simulation_params import PROJ_ROOT
from utils.blender_utils import *
from utils.xml_utils import generate_xml_from_blender

class SceneBuilder:
    def __init__(self):
        self.osm_folder = None

    def setup_folders(self):
        """Set up output folders."""
        time_str = dt.now().strftime("%m-%d-%Y_%HH%MM%SS")
        self.osm_folder = os.path.join(PROJ_ROOT, 'all_runs', f'run_{time_str}')
        os.makedirs(self.osm_folder, exist_ok=True)
        with open(os.path.join(PROJ_ROOT, 'scenes_folder.txt'), 'w') as fp:
            fp.write(self.osm_folder + '\n')

    def read_positions(self):
        """Read coordinates from CSV."""
        with open(os.path.join(PROJ_ROOT, 'params.csv'), 'r') as file:
            return list(csv.DictReader(file))

    def configure_osm_import(self, scene_folder, row):
        """Configure Blender OSM import settings."""
        bpy.context.preferences.addons["blosm"].preferences.dataDir = scene_folder
        bpy.context.scene.blosm.mode = '3Dsimple'
        bpy.context.scene.blosm.minLat = float(row['min_lat'])
        bpy.context.scene.blosm.maxLat = float(row['max_lat'])
        bpy.context.scene.blosm.minLon = float(row['min_lon'])
        bpy.context.scene.blosm.maxLon = float(row['max_lon'])
        bpy.context.scene.blosm.buildings = True
        bpy.context.scene.blosm.water = False
        bpy.context.scene.blosm.forests = False
        bpy.context.scene.blosm.vegetation = False
        bpy.context.scene.blosm.highways = True
        bpy.context.scene.blosm.railways = False
        bpy.context.scene.blosm.singleObject = True
        bpy.context.scene.blosm.ignoreGeoreferencing = True

    def create_ground_plane(self, row):
        """Create and size a ground plane."""
        bpy.ops.mesh.primitive_plane_add(size=1)
        plane = get_obj_by_name("Plane")
        min_lat, max_lat = float(row['min_lat']), float(row['max_lat'])
        min_lon, max_lon = float(row['min_lon']), float(row['max_lon'])
        x_size = compute_distance([min_lat, min_lon], [min_lat, max_lon]) * 1.2
        y_size = compute_distance([min_lat, min_lon], [max_lat, min_lon]) * 1.2
        plane.scale = (x_size, y_size, 1)
        floor_mat = create_material(FLOOR_MATERIAL, MATERIAL_COLORS[FLOOR_MATERIAL])
        plane.data.materials.append(floor_mat)

    def assign_materials(self):
        """Assign predefined materials to objects."""
        building_mat = create_material(BUILDING_MATERIAL, MATERIAL_COLORS[BUILDING_MATERIAL])
        road_mat = create_material(ROAD_MATERIAL, MATERIAL_COLORS[ROAD_MATERIAL])
        others_mat = create_material(OTHERS_MATERIAL, MATERIAL_COLORS[OTHERS_MATERIAL])

        for mat in bpy.data.materials:
            if mat.name in DEFAULT_MATERIALS:
                continue
            objs = get_objs_with_material(mat)
            replace_mat = (
                building_mat if mat.name in KNOWN_BUILDING_OSM_MATERIALS else
                road_mat if mat.name in KNOWN_ROAD_OSM_MATERIALS else
                others_mat
            )
            if replace_mat == others_mat:
                print(f'Unknown material: {mat.name}')
                with open(os.path.join(self.osm_folder, 'unknown_materials.txt'), 'a') as fp:
                    fp.write(mat.name + '\n')
            for obj in objs:
                idx = get_slot_of_material(obj, mat)
                if len(obj.material_slots) == 1:
                    obj.data.materials[idx] = replace_mat
                else:
                    obj.data.materials.pop(index=idx)
            bpy.data.materials.remove(mat)

    def export_ply(self, scene_folder):
        """Export meshes as PLY files."""
        mesh_folder = os.path.join(scene_folder, 'meshes')
        os.makedirs(mesh_folder, exist_ok=True)

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = bpy.data.objects[0]
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.convert(target='MESH', keep_original=False)

        def export_mesh(filename, name_filter=None):
            if name_filter:
                bpy.ops.object.select_all(action='DESELECT')
                for o in bpy.data.objects:
                    if name_filter in o.name:
                        o.select_set(True)
            if bpy.context.selected_objects:
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.quads_convert_to_tris()
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.export_mesh.ply(filepath=os.path.join(mesh_folder, f'{filename}.ply'),
                                       use_selection=True, use_normals=False, use_uv_coords=False,
                                       use_colors=False, axis_forward='Y', axis_up='Z')
                print(f"Exported {filename}: {len(bpy.context.selected_objects)} objects")

        export_mesh('buildings', 'building')
        export_mesh('terrain', 'Plane')
        bpy.ops.object.select_all(action='DESELECT')
        for o in bpy.data.objects:
            if 'Plane' in o.name or 'building' in o.name:
                o.select_set(True)
        bpy.ops.object.select_all(action='INVERT')
        export_mesh('roads')

    def build_scenes(self):
        """Main method to build all scenes."""
        self.setup_folders()
        positions = self.read_positions()

        for i, row in enumerate(positions):
            scene_folder = os.path.join(self.osm_folder, f'scene_{i}')
            os.makedirs(scene_folder, exist_ok=True)
            clear_blender()
            self.configure_osm_import(scene_folder, row)
            bpy.ops.blosm.import_data()

            origin_lat = bpy.data.scenes["Scene"]["lat"]
            origin_lon = bpy.data.scenes["Scene"]["lon"]
            with open(os.path.join(scene_folder, 'osm_gps_origin.txt'), 'w') as f:
                f.write(f"{origin_lat}\n{origin_lon}\n")

            self.create_ground_plane(row)
            self.assign_materials()
            self.export_ply(scene_folder)
            set_world_emitter()
            generate_xml_from_blender(os.path.join(scene_folder, 'scene.xml'))
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(scene_folder, 'scene.blend'))