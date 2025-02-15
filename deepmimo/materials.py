"""
Core material representation module.

This module provides the base class for representing materials and their properties,
including electromagnetic and scattering characteristics.
"""

from dataclasses import dataclass, asdict, astuple
from typing import Dict, ClassVar, List, Set

@dataclass
class Material:
    """Base class for material representation.
    
    This class defines the common properties of materials used in electromagnetic
    simulations, including their electrical properties and scattering characteristics.

    Notes:
    - Modeling based on https://ieeexplore.ieee.org/document/4052607
    (common approach to backscattering in all ray tracing software)
    """
    
    # Scattering model types
    SCATTERING_NONE: ClassVar[str] = 'none'
    SCATTERING_LAMBERTIAN: ClassVar[str] = 'lambertian'
    SCATTERING_DIRECTIVE: ClassVar[str] = 'directive'
    
    # Identification
    id: int = -1
    name: str = ''
    
    # Basic properties
    permittivity: float = 0.0
    conductivity: float = 0.0
    
    # Scattering properties
    scattering_model: str = SCATTERING_NONE
    scattering_coefficient: float = 0.0  # Fraction of incident fields scattered (0-1)
    cross_polarization_coefficient: float = 0.0  # Fraction of scattered field cross-polarized (0-1)
    
    # Directive scattering parameters
    alpha_r: float = 4.0  # Forward scattering lobe width (1-10) (r ~ reflection)
    alpha_i: float = 4.0  # Backscattering lobe width (1-10) (i ~ incidence)
    lambda_param: float = 0.5  # Forward vs backward scattering ratio (0-1)
    
    # Physical properties
    roughness: float = -1.0  # Surface roughness (m)
    thickness: float = -1.0  # Material thickness (m)

class MaterialList:
    """Container for managing a collection of materials."""
    
    def __init__(self):
        """Initialize an empty material list."""
        self._materials: List[Material] = []
    
    def __getitem__(self, idx: int | List[int]) -> 'Material | MaterialList':
        """Get material(s) by index or indices.
        
        Args:
            idx: Single index or list of indices
            
        Returns:
            Single Material if idx is int, or MaterialList if idx is list
        """
        if isinstance(idx, int):
            return self._materials[idx]
        else:
            # Create new MaterialList with selected materials
            materials = MaterialList()
            materials.add_materials([self._materials[i] for i in idx])
            return materials
    
    def __len__(self) -> int:
        """Get number of materials."""
        return len(self._materials)
    
    def __iter__(self):
        """Iterate over materials."""
        return iter(self._materials)
        
    def add_materials(self, materials: List[Material]) -> None:
        """Add materials to the collection.
        
        Args:
            materials: List of Material objects to add
        """
        # Add to main list and filter duplicates
        self._materials.extend(materials)
        self._filter_duplicates()
        
        # Assign IDs after filtering
        for i, mat in enumerate(self._materials):
            mat.id = i
    
    def _filter_duplicates(self) -> None:
        """Remove duplicate materials based on their properties."""
        unique_materials = []
        seen: Set[tuple] = set()
        
        for mat in self._materials:
            # Create hashable key from properties (excluding id)
            mat_key = astuple(mat)[1:]  # Skip the id field
            
            if mat_key not in seen:
                seen.add(mat_key)
                unique_materials.append(mat)
        
        self._materials = unique_materials

    def to_dict(self) -> Dict:
        """Get dictionary representation of all materials.
        
        Returns:
            Dict mapping material IDs to their properties. Note that when saved
            to .mat format, numeric keys will be converted to strings (e.g., '0', '1', etc.)
        """
        return {f'material_{mat.id}': asdict(mat) for mat in self._materials}

    @classmethod
    def from_dict(cls, materials_dict: Dict) -> 'MaterialList':
        """Create MaterialList from dictionary representation.
        
        Args:
            materials_dict: Dictionary mapping material IDs to their properties
            
        Returns:
            MaterialList containing the materials from the dictionary
        """
        materials_list = cls()
        materials = []
        
        for _, mat_data in materials_dict.items():
            materials.append(Material(**mat_data))
        
        materials_list.add_materials(materials)
        return materials_list
