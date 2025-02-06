"""
Parser for Wireless InSite physical object files.

This module provides functionality to parse physical object files (.city, .ter, .veg)
from Wireless InSite into DeepMIMO's physical object representation.
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from scipy.spatial import ConvexHull

from ...scene import PhysicalElement, Face, Scene

# Map file extensions to their corresponding labels
OBJECT_LABELS: Dict[str, str] = {
    '.city': 'building',
    '.ter': 'terrain',
    '.veg': 'vegetation'
}


def read_scene(folder_path: str | Path) -> Scene:
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
    
    scene = Scene()
    
    # Find all files with matching extensions
    found_files = {ext: [] for ext in OBJECT_LABELS}
    for file in folder.glob("*"):
        suffix = file.suffix.lower()
        if suffix in OBJECT_LABELS:
            found_files[suffix].append(str(file))
    
    # Check if any valid files were found
    if not any(files for files in found_files.values()):
        raise ValueError(f"No valid files (.city, .ter, .veg) found in {folder}")
    
    # Parse each type of file and add to scene
    for suffix, type_files in found_files.items():
        if not type_files:
            continue
            
        # Parse all files of this type
        for file in type_files:
            parser = PhysicalObjectParser(file)
            objects = parser.parse()
            scene.add_objects(objects)
    
    return scene


class PhysicalObjectParser:
    """Parser for Wireless InSite physical object files (.city, .ter, .veg)."""
    
    def __init__(self, file_path: str):
        """Initialize parser with file path.
        
        Args:
            file_path: Path to the physical object file (.city, .ter, .veg)
        """
        self.file_path = Path(file_path)
        if self.file_path.suffix not in OBJECT_LABELS:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")
        
        self.label = OBJECT_LABELS[self.file_path.suffix]
    
    def parse(self) -> List[PhysicalElement]:
        """Parse the file and return a list of physical objects.
        
        Returns:
            List of PhysicalElement objects with appropriate labels
        """
        # Read file content
        with open(self.file_path, 'r') as f:
            content = f.read()
            
        # Extract objects using extract_objects
        object_vertices = extract_objects(content)
        
        # Convert each set of vertices into a PhysicalElement object
        objects = []
        for i, vertices in enumerate(object_vertices):
            # Get faces for this object
            object_faces = get_object_faces(vertices)
            
            # Convert faces to Face objects
            faces = [Face(vertices=face) for face in object_faces]
            
            # Create PhysicalElement object with appropriate label
            obj = PhysicalElement(faces=faces, object_id=i, label=self.label)
            objects.append(obj)
            
        return objects


def extract_objects(content: str) -> List[List[Tuple[float, float, float]]]:
    """Extract physical objects from Wireless InSite file content.
    
    This function parses the file content to extract and group vertices that form 
    complete physical objects (buildings, terrain, etc). It uses face connectivity
    to determine which vertices belong to the same object.

    Args:
        content (str): Raw file content from Wireless InSite object file

    Returns:
        list of list of tuple: List of objects, where each object is a list of 
            (x,y,z) vertex coordinate tuples
    """
    # Split content into faces
    face_pattern = r'begin_<face>(.*?)end_<face>'
    faces = re.findall(face_pattern, content, re.DOTALL)
    
    # Pattern to match coordinates in face definitions
    vertex_pattern = r'-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+'
    
    # Pre-process all vertices for all faces
    face_vertices = []
    vertex_to_faces = {}  # Map vertices to the faces they belong to
    
    for i, face in enumerate(faces):
        # Extract and convert vertices once
        vertices = []
        for v in re.findall(vertex_pattern, face):
            x, y, z = map(float, v.split())
            vertex = (x, y, z)
            vertices.append(vertex)
            # Build reverse mapping of vertex -> faces
            if vertex not in vertex_to_faces:
                vertex_to_faces[vertex] = {i}
            else:
                vertex_to_faces[vertex].add(i)
        face_vertices.append(vertices)
    
    # Group faces that share vertices to form objects
    objects = []
    processed_faces = set()
    
    for i in range(len(faces)):
        if i in processed_faces:
            continue
            
        # Start a new object with this face
        object_vertices = set()
        face_stack = [i]
        
        while face_stack:
            current_face_idx = face_stack.pop()
            if current_face_idx in processed_faces:
                continue
                
            current_vertices = face_vertices[current_face_idx]
            processed_faces.add(current_face_idx)
            
            # Add vertices to object
            object_vertices.update(current_vertices)
            
            # Find connected faces using vertex_to_faces mapping
            connected_faces = set()
            for vertex in current_vertices:
                connected_faces.update(vertex_to_faces[vertex])
            
            # Add unprocessed connected faces to stack
            face_stack.extend(f for f in connected_faces if f not in processed_faces)
        
        if object_vertices:
            objects.append(list(object_vertices))
    
    return objects


def get_object_faces(vertices: List[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
    """Generate faces for a physical object from its vertices.
    
    This function takes a list of vertices and generates faces to form a complete 
    3D object. It creates a convex hull from the base points and generates top, 
    bottom and side faces.

    Args:
        vertices (list of tuple): List of (x,y,z) vertex coordinates for the object

    Returns:
        list of list of tuple: List of faces, where each face is a list of (x,y,z) 
            vertex coordinates defining the face polygon
    """
    # Extract base points (x,y coordinates)
    points_2d = np.array([(x, y) for x, y, z in vertices])
    
    # Get object height (assuming constant height)
    heights = [z for _, _, z in vertices]
    object_height = max(heights) - min(heights)
    base_height = min(heights)
    
    # Create convex hull for base shape
    hull = ConvexHull(points_2d)
    base_shape = points_2d[hull.vertices]
    
    # Create top and bottom faces
    bottom_face = [(x, y, base_height) for x, y in base_shape]
    top_face = [(x, y, base_height + object_height) for x, y in base_shape]
    
    # Create side faces
    side_faces = []
    for i in range(len(base_shape)):
        j = (i + 1) % len(base_shape)
        side = [
            bottom_face[i],
            bottom_face[j],
            top_face[j],
            top_face[i]
        ]
        side_faces.append(side)
    
    # Combine all faces
    faces = [bottom_face, top_face] + side_faces
    
    return faces


if __name__ == "__main__":
    # Test parsing and matrix export
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    
    # Create scene from test directory
    scene = read_scene(test_dir)

    # Visualize
    scene.plot_3d(show=True) 