"""Parser for Wireless InSite physical object files.

This module provides functionality to parse physical object files (.city, .ter, .veg)
from Wireless InSite into DeepMIMO's physical object representation.
"""

import re
from typing import List, Set, Tuple
from pathlib import Path
from ...buildings import Building, Terrain, Vegetation, Face, PhysicalObject, Scene
from .insite_buildings import extract_buildings2, get_building_shape

# Map file extensions to object types
FILE_TYPES = {
    '.city': ('buildings', Building),
    '.ter': ('terrain', Terrain),
    '.veg': ('vegetation', Vegetation)
}

def create_scene_from_folder(folder_path: str | Path) -> Scene:
    """Create a Scene from a folder containing Wireless InSite files.
    
    This function searches the given folder for .city, .ter, and .veg files
    and creates a Scene containing all the objects defined in those files.
    
    Args:
        folder_path: Path to folder containing Wireless InSite files
        
    Returns:
        Scene containing all objects from the files
        
    Raises:
        ValueError: If folder doesn't exist or no valid files found
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    
    scene = Scene(name='WirelessInsiteScene')
    
    # Find all files with matching extensions
    found_files = {ext: [] for ext in FILE_TYPES}
    for file in folder.glob("*"):
        suffix = file.suffix.lower()
        if suffix in FILE_TYPES:
            found_files[suffix].append(str(file))
    
    # Check if any valid files were found
    if not any(files for files in found_files.values()):
        raise ValueError(f"No valid files (.city, .ter, .veg) found in {folder}")
    
    # Parse each type of file and add to scene
    for suffix, type_files in found_files.items():
        if not type_files:
            continue
            
        # Parse all files of this type
        all_objects = []
        for file in type_files:
            parser = PhysicalObjectParser(file)
            objects = parser.parse()
            all_objects.extend(objects)
        
        # Get group name for this file type
        group_name = FILE_TYPES[suffix][0]
            
        # Add to scene with appropriate group name
        scene.add_objects(group_name, all_objects)
    
    return scene


class PhysicalObjectParser:
    """Parser for Wireless InSite physical object files (.city, .ter, .veg)."""
    
    # Map file extensions to object types
    OBJECT_TYPES = {
        '.city': Building,
        '.ter': Terrain,
        '.veg': Vegetation
    }
    
    def __init__(self, file_path: str):
        """Initialize parser with file path.
        
        Args:
            file_path: Path to the physical object file (.city, .ter, .veg)
        """
        self.file_path = Path(file_path)
        if self.file_path.suffix not in self.OBJECT_TYPES:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")
        
        self.object_class = self.OBJECT_TYPES[self.file_path.suffix]
    
    def parse(self) -> List[PhysicalObject]:
        """Parse the file and return a list of physical objects.
        
        Returns:
            List of physical objects (Building, Terrain, or Vegetation)
        """
        # Read file content
        with open(self.file_path, 'r') as f:
            content = f.read()
            
        # Extract buildings using extract_buildings2
        building_vertices = extract_buildings2(content)
        
        # Convert each set of vertices into a Building object
        objects = []
        for i, vertices in enumerate(building_vertices):
            # Get faces for this building
            building_faces, _ = get_building_shape(vertices)
            
            # Convert faces to Face objects
            faces = [Face(vertices=face) for face in building_faces]
            
            # Create Building object
            building = self.object_class(faces=faces, object_id=i)
            objects.append(building)
            
        return objects

if __name__ == "__main__":
    # Test parsing and matrix export
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    
    # Create scene from test directory
    scene = create_scene_from_folder(test_dir)

    # Visualize
    scene.plot_3d(show=True) 