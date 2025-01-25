"""
Materials handling for Wireless Insite conversion.

This module provides functionality for parsing materials from Wireless Insite files
and converting them to the base Material format.
"""

import os
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

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


def parse_materials_from_file(file: Path) -> List[Material]:
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


def read_materials(sim_folder: str, verbose: bool = False) -> Dict:
    """Read materials from a Wireless Insite simulation folder.
    
    Args:
        sim_folder: Path to simulation folder containing material files (.city, .ter, .veg)
        verbose: Whether to print debug information
        
    Returns:
        Dict containing materials and their categorization
    """
    sim_folder = Path(sim_folder)
    if not sim_folder.exists():
        raise ValueError(f"Simulation folder does not exist: {sim_folder}")
    
    # Initialize material list
    material_list = MaterialList()
    
    # Get files by type
    file_types = {
        CATEGORY_BUILDINGS: list(sim_folder.glob("*.city")),
        CATEGORY_TERRAIN: list(sim_folder.glob("*.ter")),
        CATEGORY_VEGETATION: list(sim_folder.glob("*.veg")),
        CATEGORY_FLOORPLANS: list(sim_folder.glob("*.flp")),
        CATEGORY_OBJECTS: list(sim_folder.glob("*.obj"))
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


if __name__ == "__main__":
    # Test directory with material files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    
    # Get all files in test directory
    files = []
    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    print(f"\nTesting materials extraction from: {test_dir}")
    print("-" * 50)
    
    # Basic test
    materials_dict = read_materials(test_dir, verbose=True)
    print("\nCategories found:")
    for category, materials in materials_dict.items():
        if materials:
            print(f"{category}: {len(materials)} materials")
            