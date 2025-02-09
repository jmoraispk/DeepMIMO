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
from typing import List, Dict, Tuple, Literal, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from .materials import MaterialList

# Physical element categories
CAT_BUILDINGS: str = 'buildings'      # Building structures
CAT_TERRAIN: str = 'terrain'          # Ground/terrain surfaces
CAT_VEGETATION: str = 'vegetation'    # Vegetation/foliage
CAT_FLOORPLANS: str = 'floorplans'    # Indoor floorplans
CAT_OBJECTS: str = 'objects'          # Other scene objects

# All valid categories
ELEMENT_CATEGORIES = [
    CAT_BUILDINGS,
    CAT_TERRAIN,
    CAT_VEGETATION,
    CAT_FLOORPLANS,
    CAT_OBJECTS
]

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
    
    def __init__(self, vertices: List[Tuple[float, float, float]] | np.ndarray, 
                 material_idx: int | np.integer = 0):
        """Initialize a face from its vertices.
        
        Args:
            vertices: List of (x, y, z) coordinates or numpy array of shape (N, 3)
                defining the face vertices in counter-clockwise order
            material_idx: Index of the material for this face (default: 0)
        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.material_idx = int(material_idx)  # Convert to Python int
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

class PhysicalElement:
    """Base class for physical objects in the wireless environment."""
    
    # Default labels that can be used (users may define their own - only used for search)
    DEFAULT_LABELS = {CAT_BUILDINGS, CAT_TERRAIN, CAT_VEGETATION, CAT_FLOORPLANS, CAT_OBJECTS}
    
    def __init__(self, faces: List[Face], object_id: int = -1, 
                 label: str = CAT_OBJECTS, color: str = '', speed: float = 0.0):
        """Initialize a physical object from its faces.
        
        Args:
            faces: List of Face objects defining the object
            object_id: Unique identifier for the object (default: -1)
            label: Label identifying the type of object (default: 'objects')
            color: Color for visualization (default: '', which means use default color)
            speed: Speed of the object (default: 0.0)
        """
        self._faces = faces
        self.object_id = object_id
        self.label = label if label in self.DEFAULT_LABELS else CAT_OBJECTS
        self.color = color
        self.speed = speed
        
        # Extract all vertices from faces for bounding box computation
        all_vertices = np.vstack([face.vertices for face in faces])
        self.vertices = all_vertices
        self.bounding_box: BoundingBox
        self._footprint_area: float | None = None
        self._position: np.ndarray | None = None
        self._hull: ConvexHull | None = None
        self._hull_volume: float | None = None
        self._hull_surface_area: float | None = None
        
        # Cache material indices
        self._material_indices: Optional[Set[int]] = None
        
        # Compute bounding box immediately
        self._compute_bounding_box()
    
    def _compute_bounding_box(self) -> None:
        """Compute the object's bounding box."""
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        self.bounding_box = BoundingBox(
            x_min=mins[0], x_max=maxs[0],
            y_min=mins[1], y_max=maxs[1],
            z_min=mins[2], z_max=maxs[2]
        )
    
    @property
    def height(self) -> float:
        """Get the height of the object."""
        return self.bounding_box.height

    @property
    def faces(self) -> List[Face]:
        """Get the faces of the object."""
        return self._faces
    
    @property
    def hull(self) -> ConvexHull:
        """Get the convex hull of the object."""
        if self._hull is None:
            self._hull = ConvexHull(self.vertices)
        return self._hull
    
    @property
    def hull_volume(self) -> float:
        """Get the volume of the object using its convex hull."""
        if self._hull_volume is None:
            self._hull_volume = self.hull.volume
        return self._hull_volume
    
    @property
    def hull_surface_area(self) -> float:
        """Get the surface area of the object using its convex hull."""
        if self._hull_surface_area is None:
            self._hull_surface_area = self.hull.area
        return self._hull_surface_area

    @property
    def footprint_area(self) -> float:
        """Get the area of the object's footprint using 2D convex hull."""
        if self._footprint_area is None:
            # Project all vertices to 2D and compute convex hull
            points_2d = self.vertices[:, :2]
            self._footprint_area = ConvexHull(points_2d).area
        return self._footprint_area
    
    @property
    def volume(self) -> float:
        """Get the volume of the object using its convex hull."""
        return self.hull_volume
    
    def to_dict(self) -> Dict:
        """Convert physical object to dictionary format."""
        return {
            'label': self.label,
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
    def from_dict(cls, data: Dict) -> 'PhysicalElement':
        """Create physical object from dictionary format."""
        faces = [
            Face(
                vertices=np.array(face['vertices']),
                material_idx=face['material_idx']
            )
            for face in data['faces']
        ]
        return cls(faces=faces, object_id=data['id'], label=data['label'])

    @property
    def position(self) -> np.ndarray:
        """Get the center of mass (position) of the object."""
        if self._position is None:
            bb = self.bounding_box
            # Calculate center as midpoint of bounding box
            self._position = np.array([
                (bb.x_max + bb.x_min) / 2,
                (bb.y_max + bb.y_min) / 2, 
                (bb.z_max + bb.z_min) / 2
            ])
        return self._position

    def plot(self, ax: Optional[plt.Axes] = None, mode: Literal['faces', 'hull'] = 'faces',
            alpha: float = 0.8, color: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the object using the specified visualization mode.
        
        Args:
            ax: Matplotlib 3D axes to plot on (if None, creates new figure)
            mode: Visualization mode - either 'faces' or 'hull' (default: 'faces')
            alpha: Transparency for visualization (default: 0.8)
            color: Color for visualization (default: None, uses object's color)
        """
        ax = ax or plt.subplots(1, 1, subplot_kw={'projection': '3d'})[1]
        
        # Get vertices based on mode
        if mode == 'faces':
            vertices_list = [face.vertices for face in self.faces]
        elif mode == 'hull':
            vertices_list = [self.vertices[simplex] for simplex in self.hull.simplices]
        
        # Plot all vertices
        for vertices in vertices_list:
            poly3d = Poly3DCollection([vertices], alpha=alpha)
            plot_color = self.color or color
            poly3d.set_facecolor(plot_color)
            poly3d.set_edgecolor('black')
            ax.add_collection3d(poly3d)
            
        return ax.get_figure(), ax

    @property
    def material_indices(self) -> Set[int]:
        """Get set of material indices used by this object."""
        if self._material_indices is None:
            self._material_indices = {face.material_idx for face in self._faces}
        return self._material_indices

class PhysicalElementGroup:
    """Represents a group of physical objects that can be queried and manipulated together."""
    
    def __init__(self, objects: List[PhysicalElement]):
        """Initialize a group of physical objects."""
        self._objects = objects
        self._bounding_box: Optional[BoundingBox] = None
        
    def __len__(self) -> int:
        """Get number of objects in group."""
        return len(self._objects)
        
    def __iter__(self):
        """Iterate over objects in group."""
        return iter(self._objects)
        
    def __getitem__(self, idx: int) -> PhysicalElement:
        """Get object by index."""
        return self._objects[idx]
    
    def get_materials(self) -> Set[int]:
        """Get set of material indices used by objects in this group."""
        materials = set()
        for obj in self._objects:
            materials.update(obj.material_indices)
        return list(materials)
    
    def filter_by_label(self, label: str) -> 'PhysicalElementGroup':
        """Get all objects with a specific label."""
        objects = [obj for obj in self._objects if obj.label == label]
        return PhysicalElementGroup(objects)
    
    def filter_by_material(self, material_idx: int) -> 'PhysicalElementGroup':
        """Get all objects that use a specific material."""
        objects = [obj for obj in self._objects if material_idx in obj.material_indices]
        return PhysicalElementGroup(objects)
    
    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box containing all objects."""
        if self._bounding_box is None:
            if not self._objects:
                raise ValueError("Group is empty")
            
            # Collect all object bounding boxes
            boxes = [obj.bounding_box.bounds for obj in self._objects]
            boxes = np.array(boxes)  # Shape: (N, 2, 3)
            
            # Compute global bounds
            global_min = np.min(boxes[:, 0], axis=0)  # Min of mins
            global_max = np.max(boxes[:, 1], axis=0)  # Max of maxs
            
            self._bounding_box = BoundingBox(
                x_min=global_min[0], x_max=global_max[0],
                y_min=global_min[1], y_max=global_max[1],
                z_min=global_min[2], z_max=global_max[2]
            )
        return self._bounding_box

class Scene:
    """Represents a physical scene with various objects affecting wireless propagation."""
    
    # Default visualization settings for different labels
    DEFAULT_VISUALIZATION_SETTINGS = {
        CAT_TERRAIN: {'z_order': 1, 'alpha': 0.1, 'color': 'grey'},
        CAT_VEGETATION: {'z_order': 2, 'alpha': 0.8, 'color': 'green'},
        CAT_BUILDINGS: {'z_order': 3, 'alpha': 0.8, 'color': None},  # use random color
        CAT_FLOORPLANS: {'z_order': 4, 'alpha': 0.8, 'color': 'blue'},
        CAT_OBJECTS: {'z_order': 5, 'alpha': 0.8, 'color': 'blue'}
    }
    
    def __init__(self):
        """Initialize an empty scene."""
        self.objects: List[PhysicalElement] = []
        self.visualization_settings = self.DEFAULT_VISUALIZATION_SETTINGS.copy()
        
        # Matrix storage tracking
        self.face_indices = []  # List[List[List[int]]] for [object][face][triangle_indices]
        self._current_index = 0
        
        # Initialize tracking dictionaries
        self._objects_by_category: Dict[str, List[PhysicalElement]] = {
            cat: [] for cat in ELEMENT_CATEGORIES
        }
        self._objects_by_material: Dict[int, List[PhysicalElement]] = {}
        self._materials: Optional[MaterialList] = None

    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box containing all objects."""
        return self.get_objects().bounding_box
    
    def set_visualization_settings(self, label: str, settings: Dict) -> None:
        """Set visualization settings for a specific label."""
        self.visualization_settings[label] = settings

    def add_object(self, obj: PhysicalElement) -> None:
        """Add a physical object to the scene.
        
        Args:
            obj: PhysicalElement to add
        """
        if obj.object_id == -1:
            obj.object_id = len(self.objects)
        
        # Add faces to scene and track indices
        obj_indices = []
        for face in obj.faces:
            face_indices = self._add_face(face)
            obj_indices.append(face_indices)
        
        # Track object by materials
        for material_idx in obj.material_indices:
            if material_idx not in self._objects_by_material:
                self._objects_by_material[material_idx] = []
            self._objects_by_material[material_idx].append(obj)
        
        # Track object by category
        category = obj.label if obj.label in ELEMENT_CATEGORIES else CAT_OBJECTS
        if category not in self._objects_by_category:
            self._objects_by_category[category] = []
        self._objects_by_category[category].append(obj)
        
        self.face_indices.append(obj_indices)
        self.objects.append(obj)
        self._bounding_box = None  # Reset cached bounding box
    
    def add_objects(self, objects: List[PhysicalElement]) -> None:
        """Add multiple physical objects to the scene.
        
        Args:
            objects: List of PhysicalElement objects to add
        """
        for obj in objects:
            self.add_object(obj)
    
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
    
    def get_objects(self, label: str | None = None) -> PhysicalElementGroup:
        """Get all objects, optionally filtered by label.
        
        Args:
            label: Optional label to filter objects by
            
        Returns:
            PhysicalElementGroup containing objects (filtered by label if specified)
        """
        if label is None:
            return PhysicalElementGroup(self.objects)
        return PhysicalElementGroup(self._objects_by_category.get(label, []))
    
    def get_objects_by_material(self, material_idx: int) -> PhysicalElementGroup:
        """Get all objects that use a specific material.
        
        Args:
            material_idx: Index of the material to filter by
            
        Returns:
            PhysicalElementGroup containing objects that use the specified material
        """
        objects = self._objects_by_material.get(material_idx, [])
        return PhysicalElementGroup(objects)
    
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
        savemat(f"{base_folder}/faces.mat", {'faces': faces})
        savemat(f"{base_folder}/materials.mat", {'materials': materials})
        
        # Return metadata
        objects_metadata = []
        for obj, obj_indices in zip(self.objects, self.face_indices):
            n_tri_faces = [len(indices) for indices in obj_indices]
            objects_metadata.append({
                'id': obj.object_id,
                'label': obj.label,
                'n_faces': len(obj.faces), 
                'n_tri_faces': sum(n_tri_faces),
                'n_tri_faces_per_face': np.array(n_tri_faces)
            })
            
        return {
            'n_objects': len(self.objects),
            'objects': objects_metadata,
        }
    
    @classmethod
    def from_data(cls, metadata: Dict, base_folder: str) -> 'Scene':
        """Create scene from metadata dictionary and data files.
        
        Args:
            metadata: Dictionary containing metadata about the scene
            base_folder: Base folder containing matrix files
        """
        # Load matrices
        faces = loadmat(f"{base_folder}/faces.mat")['faces']
        materials = loadmat(f"{base_folder}/materials.mat")['materials'].flatten()
        
        scene = cls()
        
        # Load visualization settings if present
        if 'visualization_settings' in metadata:
            scene.visualization_settings = metadata['visualization_settings']
        
        # Create objects using face counts from metadata
        current_index = 0
        
        for object_data in metadata['objects']:
            object_faces = []
            n_tri_faces = object_data['n_tri_faces_per_face']
            
            for n_triangles in n_tri_faces:
                # Get triangles for this face
                triangles = faces[current_index:current_index + n_triangles]
                
                # Get material index for this face and verify all triangles have same material
                material_idx = materials[current_index]
                if not np.all(materials[current_index:current_index + n_triangles] == material_idx):
                    raise ValueError("All triangles in a face must have the same material index")
                
                # Create face from triangles
                vertices = triangles.reshape(-1, 3)  # Reshape to (N*3, 3)
                face = Face(vertices=vertices, material_idx=material_idx)
                object_faces.append(face)
                
                current_index += n_triangles
            
            # Create object with appropriate label
            obj = PhysicalElement(
                faces=object_faces,
                object_id=object_data['id'],
                label=object_data['label']
            )
            scene.add_object(obj)
        
        return scene
    
    def plot(self, show: bool = True, save: bool = False, filename: str | None = None,
             mode: Literal['faces', 'hull'] = 'faces') -> Tuple[plt.Figure, plt.Axes]:
        """Create a 3D visualization of the scene.
        
        Args:
            show: Whether to display the plot
            save: Whether to save the plot to a file
            filename: Name of the file to save the plot to (if save is True)
            mode: Visualization mode - either 'faces' or 'hull' (default: 'faces')
            
        Returns:
            Tuple of (Figure, Axes) for the plot
        """
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Group objects by label
        label_groups = {}
        for obj in self.objects:
            if obj.label not in label_groups:
                label_groups[obj.label] = []
            label_groups[obj.label].append(obj)
        
        # Plot each label group
        for label, objects in label_groups.items():
            # Get visualization settings for this label
            vis_settings = self.visualization_settings.get(
                label,
                {'z_order': 3, 'alpha': 0.8, 'color': None}  # Default settings
            )
            
            # Use rainbow colormap for objects without fixed color
            n_objects = len(objects)
            if vis_settings['color'] is None:
                colors = plt.cm.rainbow(np.linspace(0, 1, n_objects))
            else:
                colors = [vis_settings['color']] * n_objects
            
            for obj_idx, obj in enumerate(objects):
                # Determine color (same for faces and hull)
                color = obj.color or colors[obj_idx]
                
                # Plot object with specified mode
                obj.plot(ax, mode=mode, alpha=vis_settings['alpha'],
                        color=color)
        
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
        """Generate a title string with object counts for each label.
        
        Returns:
            Title string with object counts
        """
        # Count objects by label
        label_counts = {}
        for obj in self.objects:
            label_counts[obj.label] = label_counts.get(obj.label, 0) + 1
        
        # Format counts
        counts = []
        for label, count in label_counts.items():
            label_name = label.capitalize()
            if count == 1 and label_name.endswith('s'):
                label_name = label_name[:-1]
            counts.append(f"{label_name}: {count}")
        
        return ", ".join(counts)
