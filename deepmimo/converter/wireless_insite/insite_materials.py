"""
Materials handling for Wireless Insite conversion.

This module provides functionality for parsing materials from Wireless Insite files
and converting them to the base Material format.
"""

import os
from typing import List, Dict
from dataclasses import dataclass

from .setup_parser import parse_file
from .. import converter_utils as cu
from ...materials import (
    Material,
    MaterialList,
    CATEGORY_BUILDINGS,
    CATEGORY_TERRAIN,
    CATEGORY_VEGETATION,
    CATEGORY_FLOORPLANS,
    CATEGORY_OBJECTS
)


@dataclass
class InsiteMaterial:
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
    id: int = -1
    name: str = ''
    diffuse_scattering_model: str = ''    # 'labertian', 'directive', 'directive_w_backscatter'
    fields_diffusively_scattered: float = 0.0  # 0-1, fraction of incident fields that are scattered
    cross_polarized_power: float = 0.0    # 0-1, fraction of the scattered field that is cross pol
    directive_alpha: float = 4.0  # 1-10, defines how broad forward beam is
    directive_beta: float = 4.0   # 1-10, defines how broad backscatter beam is
    directive_lambda: float = 0.5 # 0-1, fraction of the scattered power in forward direction (vs back)
    conductivity: float = 0.0     # >=0, conductivity
    permittivity: float = 1.0     # >=0, permittivity
    roughness: float = 0.0        # >=0, roughness
    thickness: float = 0.0        # >=0, thickness [m]
    
    def to_material(self) -> Material:
        """Convert InsiteMaterial to base Material."""
        # Map scattering model names
        model_mapping = {
            '': Material.SCATTERING_NONE,
            'lambertian': Material.SCATTERING_LAMBERTIAN,
            'directive': Material.SCATTERING_DIRECTIVE,
            'directive_w_backscatter': Material.SCATTERING_DIRECTIVE  # Map both directive models to same type
        }
        
        return Material(
            id=self.id,
            name=self.name,
            permittivity=self.permittivity,
            conductivity=self.conductivity,
            scattering_model=model_mapping.get(self.diffuse_scattering_model, Material.SCATTERING_NONE),
            scattering_coefficient=self.fields_diffusively_scattered,
            cross_polarization_coefficient=self.cross_polarized_power,
            alpha=self.directive_alpha,
            beta=self.directive_beta,
            lambda_param=self.directive_lambda,
            roughness=self.roughness,
            thickness=self.thickness
        )


def parse_materials_from_file(file: str) -> List[Material]:
    """Parse materials from a single Wireless Insite file.
    
    Args:
        file: Path to file to read
        
    Returns:
        List of Material objects
    """
    document = parse_file(file)
    materials = []
    
    for prim in document.keys():
        mat_entries = document[prim].values['Material']
        mat_entries = [mat_entries] if not isinstance(mat_entries, list) else mat_entries
        
        for mat in mat_entries:
            # Create InsiteMaterial object
            insite_mat = InsiteMaterial(
                name=mat.name,
                diffuse_scattering_model=mat.values['diffuse_scattering_model'],
                fields_diffusively_scattered=mat.values['fields_diffusively_scattered'],
                cross_polarized_power=mat.values['cross_polarized_power'],
                directive_alpha=mat.values['directive_alpha'],
                directive_beta=mat.values['directive_beta'],
                directive_lambda=mat.values['directive_lambda'],
                conductivity=mat.values['DielectricLayer'].values['conductivity'],
                permittivity=mat.values['DielectricLayer'].values['permittivity'],
                roughness=mat.values['DielectricLayer'].values['roughness'],
                thickness=mat.values['DielectricLayer'].values['thickness']
            )
            
            # Convert to base Material
            materials.append(insite_mat.to_material())
    
    return materials


def read_materials(files_in_sim_folder: List[str], verbose: bool = False) -> Dict:
    """Read materials from Wireless Insite files.
    
    Args:
        files_in_sim_folder: List of files in simulation folder
        verbose: Whether to print debug information
        
    Returns:
        Dict containing materials and their categorization
    """
    # Initialize material list
    material_list = MaterialList()
    
    # Get files by type
    file_types = {
        CATEGORY_BUILDINGS: cu.ext_in_list('.city', files_in_sim_folder),
        CATEGORY_TERRAIN: cu.ext_in_list('.ter', files_in_sim_folder),
        CATEGORY_VEGETATION: cu.ext_in_list('.veg', files_in_sim_folder),
        CATEGORY_FLOORPLANS: cu.ext_in_list('.flp', files_in_sim_folder),
        CATEGORY_OBJECTS: cu.ext_in_list('.obj', files_in_sim_folder)
    }
    
    # Parse materials from each file type
    for category, files in file_types.items():
        for file in files:
            materials = parse_materials_from_file(file)
            material_list.add_materials(materials, category)
            
    if verbose:
        from pprint import pprint
        pprint(material_list.get_materials_dict())
        
    return material_list.get_materials_dict() 