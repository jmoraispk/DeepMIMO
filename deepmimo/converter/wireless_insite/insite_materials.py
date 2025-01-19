"""
Materials handling for Wireless Insite conversion.
"""
from dataclasses import dataclass
from typing import List
import numpy as np

from .setup_parser import parse_file
from .. import converter_utils as cu

@dataclass
class InsiteMaterial():
    """
    Materials in Wireless InSite.
    
    Notes:
    - Diffuse model implemented from [1] + extended with cross-polarization scattering terms
    - Diffuse scattering models explained in [2], slides 29-31. 
    
    - At present, all MATERIALS in Wireless InSite are nonmagnetic, 
      and the permeability for all materials is that of free space 
      (µ0 = 4π x 10e-7 H/m) [3]. 

    Sources:
        [1] A Diffuse Scattering Model for Urban Propagation Prediction - Vittorio Degli-Esposti 2001
            https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=933491
        [2] https://x.webdo.cc/userfiles/Qiwell/files/Remcom_Wireless%20InSite_5G_final.pdf
        [3] Wireless InSite 3.3.0 Reference Manual, section 10.5 - Dielectric Parameters
    """
    name: str = ''
    diffuse_scattering_model: str = ''    # 'labertian', 'directive', 'directive_w_backscatter'
    fields_diffusively_scattered: int = 0  # 0-1, fraction of incident fields that are scattered
    cross_polarized_power: str = 0        # 0-1, fraction of the scattered field that is cross pol
    directive_alpha: int = 0     # 1-10, defines how broad forward beam is
    directive_beta: int = 0      # 1-10, defines how broad backscatter beam is
    directive_lambda: int = 0    # 0-1, fraction of the scattered power in forward direction (vs back)
    conductivity: int = 0        # >=0, conductivity
    permittivity: int = 0        # >=0, permittivity
    roughness: int = 0           # >=0, roughness
    thickness: int = 0           # >=0, thickness [m]


def read_materials(files_in_sim_folder, verbose):
    city_files = cu.ext_in_list('.city', files_in_sim_folder)
    ter_files  = cu.ext_in_list('.ter', files_in_sim_folder)
    veg_files  = cu.ext_in_list('.veg', files_in_sim_folder)
    fpl_files  = cu.ext_in_list('.flp', files_in_sim_folder)
    obj_files  = cu.ext_in_list('.obj', files_in_sim_folder)
    
    city_materials = read_material_files(city_files, verbose)
    ter_materials = read_material_files(ter_files, verbose)
    veg_materials = read_material_files(veg_files, verbose)
    floor_plan_materials = read_material_files(fpl_files, verbose)
    obj_materials = read_material_files(obj_files, verbose)

    materials_dict = {'city': city_materials, 
                     'terrain': ter_materials,
                     'vegetation': veg_materials,
                     'floorplans': floor_plan_materials,
                     'obj_materials': obj_materials}
    if verbose:
        from pprint import pprint
        pprint(materials_dict)
    return materials_dict


def read_material_files(files: List[str], verbose: bool):
    if verbose:
        print(f'Reading materials in {[os.path.basename(f) for f in files]}')
    
    # Extract materials for each file
    material_list = []
    for file in files:
        material_list += read_single_material_file(file, verbose)

    # Filter the list of materials so they are unique
    unique_mat_list = make_mat_list_unique(material_list)
    
    return unique_mat_list


def read_single_material_file(file: str, verbose: bool):
    document = parse_file(file)
    direct_fields = ['diffuse_scattering_model', 'fields_diffusively_scattered', 
                     'cross_polarized_power', 'directive_alpha',
                     'directive_beta', 'directive_lambda']
    dielectric_fields = ['conductivity', 'permittivity', 'roughness', 'thickness']
    
    mat_objs = []
    for prim in document.keys():
        materials = document[prim].values['Material']
        materials = [materials] if type(materials) != list else materials
        for mat in materials:
            material_obj = InsiteMaterial()
            material_obj.name = mat.name
            for field in direct_fields:
                setattr(material_obj, field, mat.values[field])
            
            for field in dielectric_fields:
                setattr(material_obj, field, mat.values['DielectricLayer'].values[field])
            
            mat_objs += [material_obj]

    return mat_objs


def make_mat_list_unique(mat_list):
    n_mats = len(mat_list)
    idxs_to_discard = []
    for i1 in range(n_mats):
        for i2 in range(n_mats):
            if i1 == i2:
                continue
            if mat_list[i1].get_dict() == mat_list[i2].get_dict():
                idxs_to_discard.append(i2)

    for idx in sorted(np.unique(idxs_to_discard), reverse=True):
        del mat_list[idx]
    
    return mat_list 