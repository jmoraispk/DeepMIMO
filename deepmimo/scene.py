"""
Physical world representation module.

This module provides core classes for representing physical objects in a wireless environment,
including buildings, terrain, vegetation, and other structures that affect wireless propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.io import savemat, loadmat
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

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
    """Base class for physical objects in the wireless environment."""
    
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
        """Convert physical object to dictionary format."""
        return {
            'type': self.__class__.__name__.lower(),  # Get type from class name
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
    def from_dict(cls, data: Dict) -> 'PhysicalObject':
        """Create physical object from dictionary format."""
        faces = [
            Face(
                vertices=np.array(face['vertices']),
                material_idx=face['material_idx']
            )
            for face in data['faces']
        ]
        return cls(faces=faces, object_id=data['id'])

class Building(PhysicalObject):
    """Represents a building in the wireless environment."""
    pass

class Terrain(PhysicalObject):
    """Represents terrain in the wireless environment."""
    pass

class Vegetation(PhysicalObject):
    """Represents vegetation in the wireless environment."""
    pass

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
        for obj in self.objects:
            for face in obj.faces:
                for triangle in face.triangular_faces:
                    faces.append(triangle.reshape(-1))
                    materials.append(face.material_idx)
        
        faces = np.array(faces)  # Shape: (N, 9)
        materials = np.array(materials)  # Shape: (N,)
        
        # Save matrices
        savemat(f"{base_folder}/{self.prefix}_faces.mat", {'faces': faces})
        savemat(f"{base_folder}/{self.prefix}_materials.mat", {'materials': materials})
        
        # Return metadata
        objects_metadata = []
        for obj, obj_indices in zip(self.objects, self.face_indices):
            n_tri_faces = [len(indices) for indices in obj_indices]
            objects_metadata.append({
                'id': obj.object_id,
                'n_faces': len(obj.faces), 
                'n_tri_faces': sum(n_tri_faces),
                'n_tri_faces_per_face': np.array(n_tri_faces)
            })
            
        return {
            'type': self.prefix, # or type(self.objects[0]).__name__
            'n_objects': len(self.objects),
            'objects': objects_metadata
        }
        
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str, object_class: type) -> 'ObjectGroup':
        """Create group from metadata and matrix files.
        
        Args:
            metadata: Dictionary containing metadata about the group
            base_folder: Base folder containing matrix files
            object_class: Class to use for creating objects (Building, Vegetation, etc.)
        """
        # Load matrices using prefix pattern
        faces = loadmat(f"{base_folder}/{metadata['type']}_faces.mat")['faces']
        materials = loadmat(f"{base_folder}/{metadata['type']}_materials.mat")['materials'].flatten()
        
        # Create objects using face counts from metadata
        objects = []
        current_index = 0
        
        # Handle both list and single dictionary cases
        object_data_list = metadata['objects']
        if isinstance(object_data_list, dict):
            object_data_list = [object_data_list]  # Convert single dict to list
        
        for object_data in object_data_list:
            object_faces = []
            n_tri_faces = object_data['n_tri_faces_per_face']
            
            for n_triangles in n_tri_faces:
                # Get triangles for this face
                triangles = faces[current_index:current_index + n_triangles]
                
                # Get material index for this face and verify all triangles have same material
                material_idx = materials[current_index]
                if not np.all(materials[current_index:current_index + n_triangles] == material_idx):
                    # Silence this if mixed materials are okay
                    raise ValueError("All triangles in a face must have the same material index")
                
                # Create face from triangles
                vertices = triangles.reshape(-1, 3)  # Reshape to (N*3, 3)
                face = Face(vertices=vertices, material_idx=material_idx)
                object_faces.append(face)
                
                current_index += n_triangles
            
            # Create object
            obj = object_class(faces=object_faces, object_id=object_data['id'])
            objects.append(obj)
        
        return cls(objects)

    def print_metadata(self) -> None:
        """Print metadata about the object group."""
    
        objects_metadata = []
        for obj, obj_indices in zip(self.objects, self.face_indices):
            n_tri_faces = [len(indices) for indices in obj_indices]
            objects_metadata.append({
                'id': obj.object_id,
                'n_faces': len(obj.faces), 
                'n_tri_faces': sum(n_tri_faces),
                # Add physical properties
                'height': obj.height,
                'footprint_area': obj.footprint_area,
                'volume': obj.volume,
                'bounds': {
                    'x_min': obj.bounding_box.x_min,
                    'x_max': obj.bounding_box.x_max,
                    'y_min': obj.bounding_box.y_min,
                    'y_max': obj.bounding_box.y_max,
                    'z_min': obj.bounding_box.z_min,
                    'z_max': obj.bounding_box.z_max,
                }
            })
        
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
    
    # Map object types to their group classes
    GROUP_TYPES = {
        'buildings': BuildingsGroup,
        'vegetation': VegetationGroup,
        'terrain': TerrainGroup
    }
    
    # Visualization settings for each object type
    VISUALIZATION_SETTINGS = {
        'terrain': {'z_order': 1, 'alpha': 0.1, 'color': 'grey'},
        'vegetation': {'z_order': 2, 'alpha': 0.8, 'color': 'green'},
        'buildings': {'z_order': 3, 'alpha': 0.8, 'color': None}  # None = use rainbow colors
    }
    
    def __init__(self):
        """Initialize an empty scene."""
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
        
        group_class = self.GROUP_TYPES[object_type]
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
        
        metadata = {}
        
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
        scene = cls()
        
        # Load each group
        for group_type, group_data in metadata.items():
            # Skip if not a physical object group or no data
            if not isinstance(group_data, dict) or 'type' not in group_data:
                continue
                
            # Get the appropriate group class based on object type
            group_class = cls.GROUP_TYPES.get(group_type)
            
            if group_class:
                scene.groups[group_type] = group_class.from_data(group_data, base_folder)
        
        return scene
    
    def plot_3d(self, show: bool = True, save: bool = False, 
                filename: str | None = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a 3D visualization of the scene using pre-computed faces.
        
        This method uses the faces already stored in each object, making it more efficient
        than plot_3d() which recomputes the faces from vertices.
        
        Args:
            show: Whether to display the plot
            save: Whether to save the plot to a file
            filename: Name of the file to save the plot to (if save is True)
            
        Returns:
            Tuple of (Figure, Axes) for the plot
        """
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each group
        for object_type, group in self.groups.items():
            # Get visualization settings for this type
            vis_settings = self.VISUALIZATION_SETTINGS[object_type]
            
            # Use rainbow colormap for objects without fixed color
            colors = (plt.cm.rainbow(np.linspace(0, 1, len(group.objects))) 
                     if vis_settings['color'] is None else None)
            
            for obj_idx, obj in enumerate(group.objects):
                # Create 3D polygons for each face
                for face in obj.faces:
                    poly3d = Poly3DCollection([face.vertices], alpha=vis_settings['alpha'])
                    poly3d.set_facecolor(vis_settings['color'] or colors[obj_idx])
                    poly3d.set_edgecolor('black')
                    poly3d.set_zorder(vis_settings['z_order'])
                    ax.add_collection3d(poly3d)
        
        self._set_axes_lims_to_scale(ax)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_title(self._get_title_with_counts())
        
        # Set the view angle for better perspective
        ax.view_init(elev=40, azim=-45)
        
        if save:
            output_file = filename or '3d_scene.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved as '{output_file}'")
        
        if show:
            plt.show()
        
        return fig, ax
    
    def _set_axes_lims_to_scale(self, ax, zoom: float = 1.3):
        """Set axis limits based on scene bounding box with equal scaling.
        
        Args:
            ax: Matplotlib 3D axes to set limits on
            zoom: Zoom factor (>1 zooms out, <1 zooms in)
        """
        bb = self.bounding_box
        
        # Find center point
        center_x = (bb.x_max + bb.x_min) / 2
        center_y = (bb.y_max + bb.y_min) / 2
        center_z = (bb.z_max + bb.z_min) / 2
        
        # Use the largest dimension to ensure equal scaling
        max_range = max(bb.width, bb.length, bb.height) / 2 / zoom
        
        # Set limits equidistant from center
        ax.set_xlim3d([center_x - max_range, center_x + max_range])
        ax.set_ylim3d([center_y - max_range, center_y + max_range])
        ax.set_zlim3d([center_z - max_range, center_z + max_range])
        
        # Ensure equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def _get_title_with_counts(self) -> str:
        """Generate a title string with object counts for each group.
        
        Returns:
            Title string with object counts
        """
        counts = []
        for object_type, group in self.groups.items():
            n_objects = len(group.objects)
            if n_objects > 0:
                type_name = object_type.capitalize()
                if n_objects == 1 and type_name.endswith('s'):
                    type_name = type_name[:-1]
                counts.append(f"{type_name}: {n_objects}")
        
        return ", ".join(counts)
