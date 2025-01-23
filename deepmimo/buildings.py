"""
Physical world representation module.

This module provides core classes for representing physical objects in a wireless environment,
including buildings, terrain, vegetation, and other structures that affect wireless propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.io import savemat, loadmat
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class BoundingBox:
    """Represents a 3D bounding box with min/max coordinates."""
    # Store bounds as numpy array for min/max of x,y,z
    bounds: np.ndarray  # shape: (2, 3) for min/max of x,y,z
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float):
        """Initialize bounding box with min/max coordinates."""
        self.bounds = np.array([
            [x_min, y_min, z_min],  # mins
            [x_max, y_max, z_max]   # maxs
        ])
    
    @property
    def x_min(self) -> float:
        """Get minimum x coordinate."""
        return self.bounds[0, 0]
    
    @property
    def x_max(self) -> float:
        """Get maximum x coordinate."""
        return self.bounds[1, 0]
    
    @property
    def y_min(self) -> float:
        """Get minimum y coordinate."""
        return self.bounds[0, 1]
    
    @property
    def y_max(self) -> float:
        """Get maximum y coordinate."""
        return self.bounds[1, 1]
    
    @property
    def z_min(self) -> float:
        """Get minimum z coordinate."""
        return self.bounds[0, 2]
    
    @property
    def z_max(self) -> float:
        """Get maximum z coordinate."""
        return self.bounds[1, 2]
    
    @property
    def width(self) -> float:
        """Get the width (X dimension) of the bounding box."""
        return self.x_max - self.x_min
    
    @property
    def length(self) -> float:
        """Get the length (Y dimension) of the bounding box."""
        return self.y_max - self.y_min
    
    @property
    def height(self) -> float:
        """Get the height (Z dimension) of the bounding box."""
        return self.z_max - self.z_min

class Face:
    """Represents a single face (surface) of a physical object."""
    
    def __init__(self, vertices: List[Tuple[float, float, float]] | np.ndarray, material_idx: int = 0):
        """Initialize a face from its vertices.
        
        Args:
            vertices: List of (x, y, z) coordinates or numpy array of shape (N, 3)
                defining the face vertices in counter-clockwise order
            material_idx: Index of the material for this face (default: 0)
        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.material_idx = material_idx
        self._normal: np.ndarray | None = None
        self._area: float | None = None
        self._centroid: np.ndarray | None = None
        self._triangular_faces: List[np.ndarray] | None = None
        
    @property
    def normal(self) -> np.ndarray:
        """Get the normal vector of the face."""
        if self._normal is None:
            # Calculate normal using cross product of two edges
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            normal = np.cross(v1, v2)
            self._normal = normal / np.linalg.norm(normal)
        return self._normal
    
    @property
    def triangular_faces(self) -> List[np.ndarray]:
        """Get the triangular faces that make up this face."""
        if self._triangular_faces is None:
            # If face is already a triangle, return it as is
            if len(self.vertices) == 3:
                self._triangular_faces = [self.vertices]
            else:
                # Triangulate the face using fan triangulation
                # This assumes the face is convex and planar
                triangles = []
                for i in range(1, len(self.vertices) - 1):
                    triangle = np.array([
                        self.vertices[0],
                        self.vertices[i],
                        self.vertices[i + 1]
                    ])
                    triangles.append(triangle)
                self._triangular_faces = triangles
        return self._triangular_faces
    
    @property
    def num_triangular_faces(self) -> int:
        """Get the number of triangular faces."""
        return len(self.triangular_faces)
    
    @property
    def area(self) -> float:
        """Get the area of the face."""
        if self._area is None:
            # Project vertices onto the plane defined by the normal
            n = self.normal
            # Find the coordinate axis most aligned with the normal
            proj_axis = np.argmax(np.abs(n))
            # Get the other two axes for projection
            other_axes = [i for i in range(3) if i != proj_axis]
            
            # Project points onto the selected plane
            points = self.vertices[:, other_axes]
            
            # Calculate area using shoelace formula
            x = points[:, 0]
            y = points[:, 1]
            # Roll arrays for vectorized computation
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            
            self._area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
            
        return self._area
    
    @property
    def centroid(self) -> np.ndarray:
        """Get the centroid of the face."""
        if self._centroid is None:
            self._centroid = np.mean(self.vertices, axis=0)
        return self._centroid

class PhysicalObject:
    """Base class for all physical objects in the wireless environment."""
    
    def __init__(self, faces: List[Face], object_id: int = -1):
        """Initialize a physical object from its faces.
        
        Args:
            faces: List of Face objects defining the object
            object_id: Unique identifier for the object (default: -1)
        """
        self._faces = faces
        self.object_id = object_id
        # Extract all vertices from faces for bounding box computation
        all_vertices = np.vstack([face.vertices for face in faces])
        self.vertices = all_vertices
        self._bounding_box: BoundingBox | None = None
        self._footprint: np.ndarray | None = None
        
        # Compute bounding box immediately as it's used frequently
        self._compute_bounding_box()
    
    def _compute_bounding_box(self) -> None:
        """Compute the object's bounding box."""
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        self._bounding_box = BoundingBox(
            x_min=mins[0], x_max=maxs[0],
            y_min=mins[1], y_max=maxs[1],
            z_min=mins[2], z_max=maxs[2]
        )
    
    @property
    def bounding_box(self) -> BoundingBox:
        """Get the object's bounding box."""
        return self._bounding_box
    
    @property
    def height(self) -> float:
        """Get the height of the object."""
        return self.bounding_box.height

    @property
    def faces(self) -> List[Face]:
        """Get the faces of the object."""
        return self._faces
    
    @property
    def footprint_area(self) -> float:
        """Get the area of the object's footprint."""
        if self._footprint is None:
            # Project all vertices to 2D and compute convex hull
            points_2d = self.vertices[:, :2]
            hull = ConvexHull(points_2d)
            self._footprint = points_2d[hull.vertices]
        return ConvexHull(self._footprint).area
    
    @property
    def volume(self) -> float:
        """Get the approximate volume of the object."""
        return self.footprint_area * self.height
    
    def to_dict(self) -> Dict:
        """Convert object to dictionary format."""
        raise NotImplementedError("Subclasses must implement to_dict()")
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhysicalObject':
        """Create object from dictionary format."""
        raise NotImplementedError("Subclasses must implement from_dict()")

class Building(PhysicalObject):
    """Represents a building in the wireless environment."""
    
    def to_dict(self) -> Dict:
        """Convert building to dictionary format."""
        return {
            'type': 'building',
            'id': self.object_id,
            'faces': [
                {
                    'vertices': face.vertices.tolist(),
                    'material_idx': face.material_idx
                }
                for face in self.faces
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Building':
        """Create building from dictionary format."""
        faces = [
            Face(
                vertices=np.array(face['vertices']),
                material_idx=face['material_idx']
            )
            for face in data['faces']
        ]
        return cls(faces=faces, object_id=data['id'])

class Terrain(PhysicalObject):
    """Represents terrain in the wireless environment."""
    
    def to_dict(self) -> Dict:
        """Convert terrain to dictionary format."""
        return {
            'type': 'terrain',
            'id': self.object_id,
            'faces': [
                {
                    'vertices': face.vertices.tolist(),
                    'material_idx': face.material_idx
                }
                for face in self.faces
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Terrain':
        """Create terrain from dictionary format."""
        faces = [
            Face(
                vertices=np.array(face['vertices']),
                material_idx=face['material_idx']
            )
            for face in data['faces']
        ]
        return cls(faces=faces, object_id=data['id'])

class Vegetation(PhysicalObject):
    """Represents vegetation in the wireless environment."""
    
    def to_dict(self) -> Dict:
        """Convert vegetation to dictionary format."""
        return {
            'type': 'vegetation',
            'id': self.object_id,
            'faces': [
                {
                    'vertices': face.vertices.tolist(),
                    'material_idx': face.material_idx
                }
                for face in self.faces
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Vegetation':
        """Create vegetation from dictionary format."""
        faces = [
            Face(
                vertices=np.array(face['vertices']),
                material_idx=face['material_idx']
            )
            for face in data['faces']
        ]
        return cls(faces=faces, object_id=data['id'])

class ObjectGroup:
    """Base class for managing groups of physical objects that share matrix storage."""
    
    def __init__(self, objects: List[PhysicalObject], prefix: str):
        """Initialize object group.
        
        Args:
            objects: List of physical objects in this group
            prefix: Prefix for matrix files (e.g., 'building' for building_faces.mat)
        """
        self.objects = objects
        self.prefix = prefix
        
        # Assign object IDs and track face indices
        self.face_indices = []  # List[List[List[int]]] for [object][face][triangle_indices]
        self._current_index = 0
        
        for i, obj in enumerate(objects):
            if obj.object_id == -1:
                obj.object_id = i
            # Add faces to group and track indices
            obj_indices = []
            for face in obj.faces:
                face_indices = self._add_face(face)
                obj_indices.append(face_indices)
            self.face_indices.append(obj_indices)
    
    def _add_face(self, face: Face) -> List[int]:
        """Add a face and return indices of its triangular faces.
        
        Args:
            face: Face to add
            
        Returns:
            List of indices for the face's triangular faces
        """
        n_triangles = face.num_triangular_faces
        triangle_indices = list(range(self._current_index, self._current_index + n_triangles))
        self._current_index += n_triangles
        return triangle_indices
    
    def export_data(self, base_folder: str) -> Dict:
        """Export group data and return metadata dictionary.
        
        Args:
            base_folder: Base folder to store matrix files
        
        Returns:
            Dict containing metadata needed to reload the group
        """
        # Export face and material matrices
        faces = []
        materials = []
        
        for obj, obj_indices in zip(self.objects, self.face_indices):
            for face, face_indices in zip(obj.faces, obj_indices):
                for i, triangle in enumerate(face.triangular_faces):
                    faces.append(triangle.reshape(-1))
                    materials.append(face.material_idx)
        
        faces = np.array(faces)  # Shape: (N, 9)
        materials = np.array(materials)  # Shape: (N,)
        
        # Save matrices
        savemat(f"{base_folder}/{self.prefix}_faces.mat", {'faces': faces})
        savemat(f"{base_folder}/{self.prefix}_materials.mat", {'materials': materials})
        
        # Return metadata
        return {
            'type': self.prefix,
            'n_objects': len(self.objects),
            'files': {
                'faces': f'{self.prefix}_faces.mat',
                'materials': f'{self.prefix}_materials.mat'
            },
            'objects': [
                {
                    'id': obj.object_id,
                    'faces': [
                        {
                            'n_triangles': len(indices)
                        }
                        for indices in obj_indices
                    ]
                }
                for obj, obj_indices in zip(self.objects, self.face_indices)
            ]
        }
    
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str, object_class: type) -> 'ObjectGroup':
        """Create group from metadata and matrix files.
        
        Args:
            metadata: Dictionary containing metadata about the group
            base_folder: Base folder containing matrix files
            object_class: Class to use for creating objects (Building, Vegetation, etc.)
        """
        # Load matrices
        faces = loadmat(f"{base_folder}/{metadata['files']['faces']}")['faces']
        materials = loadmat(f"{base_folder}/{metadata['files']['materials']}")['materials'].flatten()
        
        # Create objects using face counts from metadata
        objects = []
        current_index = 0
        
        for object_data in metadata['objects']:
            object_faces = []
            for face_data in object_data['faces']:
                n_triangles = face_data['n_triangles']
                # Get triangles for this face
                triangles = faces[current_index:current_index + n_triangles]
                material_idx = materials[current_index]  # Use first material (should be same for all triangles)
                
                # Create face from triangles
                vertices = triangles.reshape(-1, 3)  # Reshape to (N*3, 3)
                face = Face(vertices=vertices, material_idx=material_idx)
                object_faces.append(face)
                
                current_index += n_triangles
            
            # Create object
            obj = object_class(faces=object_faces, object_id=object_data['id'])
            objects.append(obj)
        
        return cls(objects=objects, prefix=metadata['type'])

class BuildingsGroup(ObjectGroup):
    """Group of buildings that share matrix storage."""
    
    def __init__(self, buildings: List[Building]):
        """Initialize buildings group.
        
        Args:
            buildings: List of Building objects
        """
        super().__init__(objects=buildings, prefix='building')
    
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str) -> 'BuildingsGroup':
        """Create buildings group from metadata and matrix files."""
        return super().from_data(metadata, base_folder, Building)

class VegetationGroup(ObjectGroup):
    """Group of vegetation objects that share matrix storage."""
    
    def __init__(self, vegetation_objects: List[Vegetation]):
        """Initialize vegetation group.
        
        Args:
            vegetation_objects: List of Vegetation objects
        """
        super().__init__(objects=vegetation_objects, prefix='vegetation')
    
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str) -> 'VegetationGroup':
        """Create vegetation group from metadata and matrix files."""
        return super().from_data(metadata, base_folder, Vegetation)

class TerrainGroup(ObjectGroup):
    """Group of terrain objects that share matrix storage."""
    
    def __init__(self, terrain_objects: List[Terrain]):
        """Initialize terrain group.
        
        Args:
            terrain_objects: List of Terrain objects
        """
        super().__init__(objects=terrain_objects, prefix='terrain')
    
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str) -> 'TerrainGroup':
        """Create terrain group from metadata and matrix files."""
        return super().from_data(metadata, base_folder, Terrain)

class Scene:
    """Represents a physical scene with various objects affecting wireless propagation."""
    
    # Map object types to their group classes and visualization colors
    GROUP_TYPES = {
        'buildings': (BuildingsGroup, 'Reds'),
        'terrain': (TerrainGroup, 'Greys'),
        'vegetation': (VegetationGroup, 'Greens')
    }
    
    def __init__(self, name: str = "unnamed_scene"):
        """Initialize an empty scene.
        
        Args:
            name: Name identifier for the scene
        """
        self.name = name
        self.groups: Dict[str, ObjectGroup] = {}
        self._bounding_box: BoundingBox | None = None
    
    def add_objects(self, object_type: str, objects: List[PhysicalObject]) -> None:
        """Add objects of a specific type to the scene.
        
        Args:
            object_type: Type of objects (e.g., 'buildings', 'terrain', 'vegetation')
            objects: List of objects to add
        """
        if object_type not in self.GROUP_TYPES:
            raise ValueError(f"Unknown object type: {object_type}")
        
        group_class = self.GROUP_TYPES[object_type][0]
        self.groups[object_type] = group_class(objects)
        print(f'object_type = {object_type}')
        self._bounding_box = None  # Reset cached bounding box
    
    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box containing all objects."""
        if self._bounding_box is None:
            # Collect all object bounding boxes
            boxes = []
            for group in self.groups.values():
                boxes.extend(obj.bounding_box.bounds for obj in group.objects)
            
            if not boxes:
                raise ValueError("Scene is empty")
            
            # Compute global bounds
            boxes = np.array(boxes)  # Shape: (N, 2, 3)
            global_min = np.min(boxes[:, 0], axis=0)  # Min of mins
            global_max = np.max(boxes[:, 1], axis=0)  # Max of maxs
            
            self._bounding_box = BoundingBox(
                x_min=global_min[0], x_max=global_max[0],
                y_min=global_min[1], y_max=global_max[1],
                z_min=global_min[2], z_max=global_max[2]
            )
        return self._bounding_box
    
    def export_data(self, base_folder: str) -> Dict:
        """Export scene data to files and return metadata dictionary.
        
        Creates matrix files for faces and materials in the base folder.
        Returns a dictionary containing metadata needed to reload the scene.
        
        Args:
            base_folder: Base folder to store matrix files
        
        Returns:
            Dict containing metadata needed to reload the scene
        """
        # Create base folder if it doesn't exist
        Path(base_folder).mkdir(parents=True, exist_ok=True)
        
        metadata = {'name': self.name}
        
        # Export data for each group
        for object_type, group in self.groups.items():
            metadata[object_type] = group.export_data(base_folder)
        
        return metadata
    
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str) -> 'Scene':
        """Create scene from metadata dictionary and data files.
        
        Args:
            metadata: Dictionary containing metadata about the scene
            base_folder: Base folder containing matrix files
        """
        scene = cls(name=metadata['name'])
        
        # Load each group
        for object_type, group_data in metadata.items():
            if object_type == 'name' or object_type not in cls.GROUP_TYPES:
                continue
            group_class = cls.GROUP_TYPES[object_type][0]
            scene.groups[object_type] = group_class.from_data(group_data, base_folder)
        
        return scene
    
    def plot_3d(self, show: bool = True, save: bool = False, 
                filename: str | None = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a 3D visualization of the scene."""
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        group_zorder = {}
        # Plot each group
        for object_type, group in self.groups.items():
            # Use rainbow colormap for better distinction between objects
            colors = plt.cm.rainbow(np.linspace(0, 1, len(group.objects)))
            
            for obj, color in zip(group.objects, colors):
                # Get vertices (already in correct order from parser)
                vertices = [(v[0], v[1], v[2]) for v in obj.vertices]
                
                # Extract footprint points (x,y coordinates)
                points_2d = np.array([(x, y) for x, y, _ in vertices])
                
                # Get building height
                heights = [z for _, _, z in vertices]
                obj_height = max(heights) - min(heights)
                base_height = min(heights)
                
                # Create convex hull for footprint
                hull = ConvexHull(points_2d)
                footprint = points_2d[hull.vertices]
                
                # Create top and bottom faces
                bottom_face = [(x, y, base_height) for x, y in footprint]
                top_face = [(x, y, base_height + obj_height) for x, y in footprint]
                
                # Create walls (side faces)
                walls = []
                for i in range(len(footprint)):
                    j = (i + 1) % len(footprint)
                    wall = [
                        bottom_face[i],
                        bottom_face[j],
                        top_face[j],
                        top_face[i]
                    ]
                    walls.append(wall)
                
                # Combine all faces
                faces = [bottom_face, top_face] + walls
                
                # Create 3D polygons
                poly3d = Poly3DCollection(faces, alpha=0.6)#, zorder=group_zorder[object_type])
                poly3d.set_facecolor(color)
                poly3d.set_edgecolor('black')
                ax.add_collection3d(poly3d)
        
        # Set axis limits
        bb = self.bounding_box
        max_range = max(bb.width, bb.length, bb.height) / 2.0
        mid_x = (bb.x_max + bb.x_min) * 0.5
        mid_y = (bb.y_max + bb.y_min) * 0.5
        mid_z = (bb.z_max + bb.z_min) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*1.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Count objects in each group
        title = f"{self.name}\n"
        counts = []
        for object_type, group in self.groups.items():
            n_objects = len(group.objects)
            if n_objects > 0:
                type_name = object_type.capitalize()
                if n_objects == 1 and type_name.endswith('s'):
                    type_name = type_name[:-1]
                counts.append(f"{type_name}: {n_objects}")
        
        ax.set_title(title + ", ".join(counts))
        
        # Set the view angle for better perspective
        ax.view_init(elev=20, azim=45)
        
        if save:
            output_file = filename or f'{self.name}_3d.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved as '{output_file}'")
        
        if show:
            plt.show()
        
        return fig, ax 