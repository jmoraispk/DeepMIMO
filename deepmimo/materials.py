"""
Core material representation module.

This module provides the base class for representing materials and their properties,
including electromagnetic and scattering characteristics.
"""

from dataclasses import dataclass, asdict, astuple
from typing import Dict, ClassVar, List, Set


# Material categories used across all converters
CATEGORY_BUILDINGS: str = 'buildings'      # Materials used in outdoor building structures
CATEGORY_TERRAIN: str = 'terrain'          # Materials used in ground/terrain surfaces
CATEGORY_VEGETATION: str = 'vegetation'     # Materials used in vegetation/foliage
CATEGORY_FLOORPLANS: str = 'floorplans'    # Materials used in indoor structures (walls, floors, etc.)
CATEGORY_OBJECTS: str = 'objects'          # Materials used in other scene objects

# All valid material categories
MATERIAL_CATEGORIES = [
    CATEGORY_BUILDINGS,
    CATEGORY_TERRAIN,
    CATEGORY_VEGETATION,
    CATEGORY_FLOORPLANS,
    CATEGORY_OBJECTS
]


@dataclass
class Material:
    """Base class for material representation.
    
    This class defines the common properties of materials used in electromagnetic
    simulations, including their electrical properties and scattering characteristics.
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
    alpha: float = 4.0  # Forward scattering lobe width (1-10)
    beta: float = 4.0   # Backscattering lobe width (1-10)
    lambda_param: float = 0.5  # Forward vs backward scattering ratio (0-1)
    
    # Physical properties
    roughness: float = 0.0  # Surface roughness (m)
    thickness: float = 0.0  # Material thickness (m)


class MaterialList:
    """Container for managing a collection of materials and their categorization."""
    
    def __init__(self):
        """Initialize an empty material list."""
        self._materials: List[Material] = []
        self._materials_by_type: Dict[str, Set[str]] = {cat: set() for cat in MATERIAL_CATEGORIES}
        
    def add_materials(self, materials: List[Material], category: str | None = None) -> None:
        """Add materials to the collection.
        
        Args:
            materials: List of Material objects to add
            category: Optional category to assign materials to
        """
        # Add to main list and filter duplicates
        self._materials.extend(materials)
        self._filter_duplicates()
        
        # Assign IDs after filtering
        for i, mat in enumerate(self._materials):
            mat.id = i
        
        # Add to category if specified
        if category and category in MATERIAL_CATEGORIES:
            self._materials_by_type[category].update(mat.name for mat in materials)
    
    def get_materials_dict(self) -> Dict:
        """Get dictionary representation of all materials.
        
        Returns:
            Dict containing:
            - materials: Dict mapping material IDs to their properties. Note that when saved
              to .mat format, numeric keys will be converted to strings (e.g., '0', '1', etc.)
            - buildings_materials: List of material names used in buildings
            - terrain_materials: List of material names used in terrain
            - vegetation_materials: List of material names used in vegetation
            - floorplans_materials: List of material names used in indoor structures
            - objects_materials: List of material names used in other scene objects
        """
        # Get material definitions
        materials_dict = {}
        for mat in self._materials:
            mat_dict = asdict(mat)
            materials_dict[mat.id] = mat_dict  # Use numeric ID as key
            
        # Add _materials suffix to category names
        materials_by_type = {
            f"{k}_materials": sorted(list(v)) 
            for k, v in self._materials_by_type.items()
        }
        
        # Combine both in final dictionary
        return {'materials': materials_dict, **materials_by_type}
    
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