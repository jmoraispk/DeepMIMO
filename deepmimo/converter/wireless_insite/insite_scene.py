"""Parser for Wireless InSite physical object files.

This module provides functionality to parse physical object files (.city, .ter, .veg)
from Wireless InSite into DeepMIMO's physical object representation.
"""

import re
from typing import List, Set, Tuple, Dict
from pathlib import Path
from ...buildings import Building, Terrain, Vegetation, Face, PhysicalObject, Scene

def create_scene_from_files(files: List[str], name: str = "unnamed_scene") -> Scene:
    """Create a Scene from Wireless InSite files.
    
    This is a factory function that creates a Scene by parsing multiple physical object files.
    
    Args:
        files: List of file paths (.city, .ter, .veg)
        name: Name for the scene
        
    Returns:
        Scene containing all objects from the files
    """
    scene = Scene(name=name)
    
    # Group files by type
    file_types: Dict[str, List[str]] = {
        '.city': [],
        '.ter': [],
        '.veg': []
    }
    
    for file in files:
        suffix = Path(file).suffix
        if suffix in file_types:
            file_types[suffix].append(file)
    
    # Parse each type of file and add to scene
    for suffix, type_files in file_types.items():
        if not type_files:
            continue
            
        # Parse all files of this type
        all_objects = []
        for file in type_files:
            parser = PhysicalObjectParser(file)
            objects = parser.parse()
            all_objects.extend(objects)
        
        # Add to scene with appropriate type name
        object_type = suffix[1:]  # Remove dot
        scene.add_objects(object_type, all_objects)
    
    return scene


class WirelessInsiteScene(Scene):
    """Scene subclass specifically for Wireless InSite files.
    
    This class extends Scene with methods to load directly from Wireless InSite files.
    """
    
    @classmethod
    def from_files(cls, files: List[str], name: str = "unnamed_scene") -> 'WirelessInsiteScene':
        """Create scene from Wireless InSite files.
        
        Args:
            files: List of file paths (.city, .ter, .veg)
            name: Name for the scene
            
        Returns:
            WirelessInsiteScene containing all objects from the files
        """
        scene = cls(name=name)
        
        # Group files by type
        file_types: Dict[str, List[str]] = {
            '.city': [],
            '.ter': [],
            '.veg': []
        }
        
        for file in files:
            suffix = Path(file).suffix
            if suffix in file_types:
                file_types[suffix].append(file)
        
        # Parse each type of file and add to scene
        for suffix, type_files in file_types.items():
            if not type_files:
                continue
                
            # Parse all files of this type
            all_objects = []
            for file in type_files:
                parser = PhysicalObjectParser(file)
                objects = parser.parse()
                all_objects.extend(objects)
            
            # Add to scene with appropriate type name
            object_type = suffix[1:]  # Remove dot
            scene.add_objects(object_type, all_objects)
        
        return scene
    
    def load_files(self, files: List[str]) -> None:
        """Load additional files into the scene.
        
        Args:
            files: List of file paths (.city, .ter, .veg)
        """
        # Group files by type
        file_types: Dict[str, List[str]] = {
            '.city': [],
            '.ter': [],
            '.veg': []
        }
        
        for file in files:
            suffix = Path(file).suffix
            if suffix in file_types:
                file_types[suffix].append(file)
        
        # Parse each type of file and add to scene
        for suffix, type_files in file_types.items():
            if not type_files:
                continue
                
            # Parse all files of this type
            all_objects = []
            for file in type_files:
                parser = PhysicalObjectParser(file)
                objects = parser.parse()
                all_objects.extend(objects)
            
            # Add to scene with appropriate type name
            object_type = suffix[1:]  # Remove dot
            
            # If group already exists, extend it
            if object_type in self.groups:
                self.groups[object_type].objects.extend(all_objects)
            else:
                # Otherwise create new group
                self.add_objects(object_type, all_objects)
        
        # Reset cached bounding box
        self._bounding_box = None

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
        self._faces: List[str] = []
        self._vertex_pattern = r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
        self._read_file()
    
    def _read_file(self) -> None:
        """Read and preprocess the input file."""
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Extract all face definitions
        face_pattern = r"begin_<face>(.*?)end_<face>"
        self._faces = re.findall(face_pattern, content, re.DOTALL)
    
    def _extract_connected_faces(self, start_idx: int, faces: List[str], 
                               processed_faces: Set[int]) -> Set[Tuple[float, float, float]]:
        """Extract all faces connected to the starting face.
        
        Args:
            start_idx: Index of the starting face
            faces: List of all face definitions
            processed_faces: Set of already processed face indices
            
        Returns:
            Set of vertices forming a connected object
        """
        object_vertices = set()
        face_stack = [start_idx]
        
        while face_stack:
            current_face_idx = face_stack.pop()
            if current_face_idx in processed_faces:
                continue
                
            current_face = faces[current_face_idx]
            processed_faces.add(current_face_idx)
            
            # Extract vertices from current face
            current_vertices = [(float(x), float(y), float(z)) 
                              for x, y, z in re.findall(self._vertex_pattern, current_face)]
            
            # Add vertices to object
            object_vertices.update(current_vertices)
            
            # Find connected faces
            for j, other_face in enumerate(faces):
                if j not in processed_faces:
                    other_vertices = [(float(x), float(y), float(z)) 
                                    for x, y, z in re.findall(self._vertex_pattern, other_face)]
                    
                    # If faces share any vertices, add to stack
                    if any(v in current_vertices for v in other_vertices):
                        face_stack.append(j)
        
        return object_vertices
    
    def _create_face(self, face_str: str) -> Face:
        """Create a Face object from face string.
        
        Args:
            face_str: String containing face definition
            
        Returns:
            Face object
        """
        # Extract material index
        material_match = re.search(r"material\s+(\d+)", face_str)
        material_idx = int(material_match.group(1)) if material_match else 0
        
        # Extract vertices
        vertices = [(float(x), float(y), float(z)) 
                   for x, y, z in re.findall(self._vertex_pattern, face_str)]
        
        return Face(vertices=vertices, material_idx=material_idx)
    
    def _get_object_faces(self, vertices: Set[Tuple[float, float, float]]) -> List[Face]:
        """Get all faces that contain only vertices from the given set.
        
        Args:
            vertices: Set of vertices that belong to the object
            
        Returns:
            List of Face objects
        """
        object_faces = []
        for face_str in self._faces:
            face_vertices = [(float(x), float(y), float(z)) 
                           for x, y, z in re.findall(self._vertex_pattern, face_str)]
            if all(v in vertices for v in face_vertices):
                object_faces.append(self._create_face(face_str))
        return object_faces
    
    def parse(self) -> List[PhysicalObject]:
        """Parse the file and return list of physical objects.
        
        Returns:
            List of physical objects (Building, Terrain, or Vegetation)
        """
        processed_faces = set()
        objects = []
        
        # Process each unprocessed face
        for i in range(len(self._faces)):
            if i not in processed_faces:
                # Get all faces connected to this one
                object_vertices = self._extract_connected_faces(i, self._faces, processed_faces)
                
                # Get all faces for this object
                object_faces = self._get_object_faces(object_vertices)
                
                # Create object
                obj = self.object_class(faces=object_faces)
                objects.append(obj)
        
        return objects

if __name__ == "__main__":
    # Test parsing and matrix export
    import os
    from pathlib import Path
    
    # Get test file paths
    test_dir = Path(__file__).parent.parent.parent.parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Create test files if they don't exist
    test_files = {
        'city': test_dir / "test.city",
        'terrain': test_dir / "test.ter",
        'vegetation': test_dir / "test.veg"
    }
    
    # Create simple test files if they don't exist
    for file_type, file_path in test_files.items():
        if not file_path.exists():
            with open(file_path, "w") as f:
                f.write("""begin_<face>
material 1
0.0 0.0 0.0
10.0 0.0 0.0
10.0 10.0 0.0
0.0 10.0 0.0
end_<face>
begin_<face>
material 1
0.0 0.0 10.0
10.0 0.0 10.0
10.0 10.0 10.0
0.0 10.0 10.0
end_<face>""")
    
    # Test both approaches:
    
    # 1. Using factory function
    print("\nTesting factory function:")
    scene1 = create_scene_from_files(list(test_files.values()), name="test_scene1")
    print("Created scene with:")
    for group_type, group in scene1.groups.items():
        print(f"  {len(group.objects)} {group_type}")
    
    # 2. Using WirelessInsiteScene
    print("\nTesting WirelessInsiteScene:")
    scene2 = WirelessInsiteScene.from_files(list(test_files.values()), name="test_scene2")
    print("Created scene with:")
    for group_type, group in scene2.groups.items():
        print(f"  {len(group.objects)} {group_type}")
    
    # Test loading additional files
    print("\nTesting loading additional files:")
    scene2.load_files([test_files['city']])  # Add another building
    print("Updated scene with:")
    for group_type, group in scene2.groups.items():
        print(f"  {len(group.objects)} {group_type}")
    
    # Export and visualize
    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    metadata = scene2.export_data(str(output_dir))
    print("\nExported metadata:", metadata)
    
    # Visualize
    scene2.plot_3d(show=True, save=True, 
                  filename=str(output_dir / "scene_3d.png")) 