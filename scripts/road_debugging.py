
#%% Import libraries
import numpy as np
from typing import Tuple, List, NamedTuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




#%% ROAD PROCESSING
# Test code for road processing


class RoadGeometry(NamedTuple):
    """Container for road geometry processing results."""
    face_vertices: List[Tuple[float, float, float]]  # 3D vertices of the face
    centroid_3d: np.ndarray  # 3D centroid
    centroid_2d: np.ndarray  # 2D centroid in the plane
    points_2d: np.ndarray    # 2D coordinates in the plane
    normal: np.ndarray       # Normal vector of the plane
    properties: dict         # Road-specific properties
    left_endpoints: np.ndarray   # Left endpoint coordinates (2D)
    right_endpoints: np.ndarray  # Right endpoint coordinates (2D)

def filter_close_points(points: np.ndarray, min_dist_threshold: float = 0.5) -> np.ndarray:
    """Filter out points that are too close to each other.
    
    Args:
        points: Array of points (N x 3 for 3D points)
        min_dist_threshold: Minimum distance between points in meters
        
    Returns:
        np.ndarray: Filtered points array
    """
    n_points = len(points)
    
    # Calculate all pairwise distances
    distances = np.zeros((n_points, n_points))
    for i in range(n_points):
        distances[i] = np.sqrt(np.sum((points - points[i])**2, axis=1))
    
    # Keep points that don't have any neighbors closer than threshold
    # (except themselves, hence the distances[i] > 0 condition)
    to_keep = []
    for i in range(n_points):
        min_dist = np.min(distances[i][distances[i] > 0])
        if min_dist >= min_dist_threshold:
            to_keep.append(i)
    
    return points[to_keep]

def detect_endpoints(points_2d: np.ndarray, n_endpoints: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Detect the endpoints of a road by finding extreme points in X direction.
    
    Args:
        points_2d: Array of 2D points (N x 2)
        n_endpoints: Number of points to consider as endpoints on each side
        
    Returns:
        tuple: (left_end_indices, right_end_indices) - Arrays of indices for left and right endpoints
    """
    # Find the extreme points in X direction
    x_sorted_indices = np.argsort(points_2d[:, 0])
    left_end_indices = x_sorted_indices[:n_endpoints]  # leftmost points
    right_end_indices = x_sorted_indices[-n_endpoints:]  # rightmost points
    
    # Print endpoint information
    print("\nEndpoint Analysis:")
    print("Left endpoints:", left_end_indices)
    print("Left endpoint coordinates:")
    print(points_2d[left_end_indices])
    print("\nRight endpoints:", right_end_indices)
    print("Right endpoint coordinates:")
    print(points_2d[right_end_indices])
    
    return left_end_indices, right_end_indices

def process_road_vertices(vertices: np.ndarray) -> RoadGeometry:
    """Process road vertices to find the best-fit plane and ordered vertices."""
    if len(vertices) < 3:
        return RoadGeometry([], None, None, None, None, {}, 
                          np.array([]), np.array([]))
        
    # Find the best-fit plane using SVD
    centroid_3d = np.mean(vertices, axis=0)
    centered_pts = vertices - centroid_3d
    _, _, vh = np.linalg.svd(centered_pts)
    
    # Normal vector is the last right singular vector
    normal = vh[2]
    
    # Get two orthogonal vectors in the plane
    basis1 = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    if np.allclose(basis1, 0):
        basis1 = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    
    # Project points onto their best-fit plane
    projected_vertices = vertices - np.outer(np.dot(centered_pts, normal), normal)
    
    # Get 2D coordinates in the plane
    centered_projected = projected_vertices - centroid_3d
    x_coords = np.dot(centered_projected, basis1)
    y_coords = np.dot(centered_projected, basis2)
    points_2d = np.column_stack([x_coords, y_coords])
    centroid_2d = np.mean(points_2d, axis=0)
    
    # Print point coordinates and distances
    print("\nPoint coordinates and distances:")
    print("Point ID  |     X     |     Y     | Min Distance")
    print("-" * 55)
    for i, point in enumerate(points_2d):
        # Calculate distances to all other points
        distances = np.sqrt(np.sum((points_2d - point)**2, axis=1))
        # Get minimum non-zero distance (exclude distance to self)
        min_dist = np.min(distances[distances > 0])
        print(f"{i:8d} | {point[0]:8.2f} | {point[1]:8.2f} | {min_dist:8.2f}")
    
    # Detect endpoints
    left_indices, right_indices = detect_endpoints(points_2d)
    left_endpoints = points_2d[left_indices]
    right_endpoints = points_2d[right_indices]
    
    def segment_intersects(p1, p2, q1, q2):
        """Check if line segments p1p2 and q1q2 intersect."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)
    
    def path_will_cross(new_point, path):
        """Check if adding new_point will create any crossing edges."""
        if len(path) < 2:
            return False
        # Check against all segments except the last one
        new_segment = (path[-1], new_point)
        for i in range(len(path)-2):
            if segment_intersects(new_segment[0], new_segment[1],
                                path[i], path[i+1]):
                return True
        return False
    
    # Find the two most extreme points in X direction for endpoints
    x_sorted_indices = np.argsort(points_2d[:, 0])
    left_end_indices = x_sorted_indices[:2]  # 2 leftmost points
    right_end_indices = x_sorted_indices[-2:]  # 2 rightmost points
    
    # Start with the leftmost point
    start_idx = left_end_indices[0]
    current_point = points_2d[start_idx]
    
    # Initialize path with the starting point
    n_points = len(points_2d)
    path = [current_point]
    used_points = np.zeros(n_points, dtype=bool)
    used_points[start_idx] = True
    
    # Force second point to be the other left endpoint
    second_idx = left_end_indices[1]
    path.append(points_2d[second_idx])
    used_points[second_idx] = True
    
    # Rest of the path finding remains similar, but with endpoint awareness
    while len(path) < n_points:
        current_point = path[-1]
        
        # Special case: if we have 2 points remaining and they're the right endpoints
        remaining_points = np.where(~used_points)[0]
        if len(remaining_points) == 2 and set(remaining_points) == set(right_end_indices):
            # Add the right endpoints in order of x-coordinate
            for idx in sorted(remaining_points, key=lambda i: points_2d[i][0]):
                path.append(points_2d[idx])
                used_points[idx] = True
            break
            
        # Normal path finding for middle points
        unused_mask = ~used_points
        candidates = points_2d[unused_mask]
        if len(candidates) == 0:
            break
            
        # Calculate vectors and distances
        vectors = candidates - current_point
        distances = np.sqrt(np.sum(vectors**2, axis=1))
        directions = vectors / distances[:, np.newaxis]
        
        # Get current direction
        if len(path) > 1:
            current_direction = path[-1] - path[-2]
            current_direction = current_direction / np.linalg.norm(current_direction)
        else:
            current_direction = directions[np.argmin(distances)]
        
        # Calculate angles
        dot_products = np.dot(directions, current_direction)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        
        # Score candidates
        scores = distances * (1.0 + angles)  # Simplified scoring
        
        # Select best candidate that doesn't create crossings
        candidate_indices = np.argsort(scores)
        next_point = None
        
        for idx in candidate_indices:
            candidate = candidates[idx]
            if not path_will_cross(candidate, path):
                next_point = candidate
                break
        
        if next_point is None:
            break
            
        # Add point to path
        path.append(next_point)
        point_idx = np.where(np.all(points_2d == next_point, axis=1))[0][0]
        used_points[point_idx] = True
    
    # Convert path back to indices
    ordered_indices = []
    for point in path:
        idx = np.where(np.all(points_2d == point, axis=1))[0][0]
        ordered_indices.append(idx)
    
    # Apply the ordering to the filtered points
    ordered_vertices = projected_vertices[ordered_indices]
    ordered_2d = points_2d[ordered_indices]
    
    # Convert to list of tuples format
    face_vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in ordered_vertices]
    
    # Calculate area and perimeter
    area = 0.0
    perimeter = 0.0
    for i in range(len(ordered_2d)):
        j = (i + 1) % len(ordered_2d)
        area += ordered_2d[i][0] * ordered_2d[j][1]
        area -= ordered_2d[j][0] * ordered_2d[i][1]
        perimeter += np.sqrt(np.sum((ordered_2d[j] - ordered_2d[i])**2))
    area = abs(area) / 2.0
    
    # Analyze road properties
    properties = validate_road_properties(vertices, ordered_2d)
    properties['area'] = area
    properties['perimeter'] = perimeter
    
    return RoadGeometry(face_vertices, centroid_3d, centroid_2d, ordered_2d, normal, properties, left_endpoints, right_endpoints)

def plot_3d_road(ax: plt.Axes, geometry: RoadGeometry, vertices: np.ndarray):
    """Plot 3D visualization of road processing.
    
    Args:
        ax: Matplotlib 3D axes to plot on
        geometry: Processed road geometry
        vertices: Original vertices
    """
    # Plot original vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c='blue', marker='o', label='Original Vertices')
    
    # Plot projected vertices and face
    face_array = np.array(geometry.face_vertices)
    ax.scatter(face_array[:, 0], face_array[:, 1], face_array[:, 2], 
              c='green', marker='s', label='Projected Vertices')
    
    # Plot centroid
    ax.scatter(geometry.centroid_3d[0], geometry.centroid_3d[1], geometry.centroid_3d[2], 
              c='red', marker='*', s=200, label='Centroid')
    
    # Plot face as a polygon
    face_poly = Poly3DCollection([face_array], alpha=0.2)
    face_poly.set_facecolor('blue')
    ax.add_collection3d(face_poly)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def plot_2d_road(ax: plt.Axes, geometry: RoadGeometry):
    """Plot 2D visualization of road processing with road properties."""
    points = geometry.points_2d
    
    # Plot regular points in blue
    ax.scatter(points[:, 0], points[:, 1], 
              c='blue', marker='o', label='Regular Points')
    
    # Add vertex numbers
    for i, (x, y) in enumerate(points):
        ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot endpoints in red
    ax.scatter(geometry.left_endpoints[:, 0], geometry.left_endpoints[:, 1],
              c='red', marker='o', label='Left Endpoints')
    ax.scatter(geometry.right_endpoints[:, 0], geometry.right_endpoints[:, 1],
              c='red', marker='o')
    
    # Plot 2D centroid
    ax.scatter(geometry.centroid_2d[0], geometry.centroid_2d[1], 
              c='green', marker='*', s=200, label='Centroid')
    
    # Plot edges with arrows to show direction
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        ax.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.9, (p2[1]-p1[1])*0.9,
                head_width=0.1, head_length=0.2, fc='g', ec='g', alpha=0.5)
    
    # Fill the enclosed area with semi-transparent color
    ax.fill(points[:, 0], points[:, 1], alpha=0.2, color='lightblue', label='Enclosed Area')
    
    ax.set_xlabel('X in plane')
    ax.set_ylabel('Y in plane')
    ax.axis('equal')
    
    ax.legend()

def validate_road_properties(vertices: np.ndarray, points_2d: np.ndarray) -> dict:
    """Analyze road-specific properties with enhanced curve detection."""
    properties = {}
    
    # Basic validation
    n_points = len(vertices)
    properties['valid_point_count'] = n_points >= 4
    properties['n_points'] = n_points
    
    if n_points < 4:
        return properties
    
    # Segment Analysis
    segments = np.diff(points_2d, axis=0)
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    
    # Calculate angles between consecutive segments
    segment_vectors = segments / segment_lengths[:, np.newaxis]
    dot_products = np.sum(segment_vectors[:-1] * segment_vectors[1:], axis=1)
    angle_changes = np.arccos(np.clip(dot_products, -1.0, 1.0))
    
    # Curvature Analysis (excluding endpoints)
    middle_angles = angle_changes[1:-1]  # Exclude first and last angle changes
    properties['max_angle_change'] = np.max(np.abs(middle_angles)) if len(middle_angles) > 0 else 0
    properties['total_curvature'] = np.sum(np.abs(middle_angles))
    properties['avg_curvature_per_segment'] = properties['total_curvature'] / max(1, len(middle_angles))
    
    # Classify road type
    properties['is_straight'] = properties['max_angle_change'] < np.pi/6  # < 30 degrees
    properties['is_curved'] = properties['total_curvature'] > np.pi/2  # > 90 degrees total
    properties['has_sharp_turns'] = properties['max_angle_change'] > np.pi/3  # > 60 degrees
    
    return properties

def check_path_intersections(points_2d: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Check if any non-adjacent segments in the path intersect.
    
    Args:
        points_2d: Array of 2D points ordered in a path
        
    Returns:
        list: List of tuples (i,j,k,l) where segment i->j intersects with k->l
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def segments_intersect(p1, p2, q1, q2):
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    
    intersections = []
    n_points = len(points_2d)
    
    # Check each pair of non-adjacent segments
    for i in range(n_points - 1):
        for j in range(i + 2, n_points - 1):  # Start from i+2 to skip adjacent segments
            if segments_intersect(points_2d[i], points_2d[i+1], 
                                points_2d[j], points_2d[j+1]):
                intersections.append((i, i+1, j, j+1))
    
    return intersections

def analyze_path_angles(points_2d: np.ndarray) -> np.ndarray:
    """Calculate angles at each point relative to the previous line segment.
    
    Args:
        points_2d: Array of 2D points ordered in a path
        
    Returns:
        np.ndarray: Array of angles in degrees at each point.
        Angle is measured as the deviation from a straight line:
        0° or 180° = straight line (no turn)
        90° = right angle turn
        Angles are in [0, 180].
    """
    n_points = len(points_2d)
    if n_points < 3:
        return np.zeros(n_points)
        
    # Calculate vectors between consecutive points, including from last to first
    vectors = np.zeros((n_points, 2))
    vectors[:-1] = np.diff(points_2d, axis=0)  # Regular vectors
    vectors[-1] = points_2d[0] - points_2d[-1]  # Vector from last to first point
    
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    # Calculate angles between consecutive vectors
    angles = np.zeros(n_points)
    for i in range(n_points):
        # For any point i, we want the vector arriving at i and the vector leaving i
        v1 = vectors[i-1]  # Vector arriving at point
        v2 = vectors[i]    # Vector leaving point
        
        # Calculate the angle between vectors using dot product
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)  # Clip to avoid numerical errors
        angle = np.degrees(np.arccos(dot_product))
        
        # The angle we want is either angle or 180-angle, whichever is smaller
        angles[i] = min(angle, 180 - angle)
    
    return angles

# Load road vertices
road_vertices = np.load('road_vertices_roads_3.npy')
print(f"\nLoaded road vertices: {len(road_vertices)} points")

# Filter close points
filtered_vertices = filter_close_points(road_vertices)
print(f"After filtering: {len(filtered_vertices)} points")

# Process the road geometry
geometry = process_road_vertices(filtered_vertices)

# Check for path intersections
intersections = check_path_intersections(geometry.points_2d)
if intersections:
    print("\nFound path intersections between segments:")
    for i, j, k, l in intersections:
        print(f"Segment {i}->{j} intersects with {k}->{l}")
else:
    print("\nNo path intersections found")

# Analyze angles
angles = analyze_path_angles(geometry.points_2d)
print("\nAngles at each vertex (deviation from straight line):")
print("Point |  Angle  | Note")
print("-" * 40)
for i, angle in enumerate(angles):
    note = ""
    if angle > 1:  # Only print angles > 1 degree
        if angle > 150:
            note = "hairpin turn!"
        elif angle > 120:
            note = "very sharp turn!"
        elif angle > 90:
            note = "sharp turn!"
        elif angle > 60:
            note = "moderate turn"
        elif angle > 30:
            note = "gentle turn"
        print(f"{i:5d} | {angle:7.2f}° | {note}")

# 2D visualization
f, ax = plt.subplots(dpi=100)
plot_2d_road(ax, geometry)
plt.tight_layout()
plt.show()

print("\nSuggestions for improvement:")
print("1. The path should be reordered to minimize deviations from straight lines")
print("2. Consider using a path optimization that:")
print("   - Starts from endpoints")
print("   - At each step, chooses the next point that creates the smallest angle deviation")
print("   - Avoids creating intersections")
print("3. Could use dynamic programming to find the optimal path that:")
print("   - Minimizes the angles at each vertex (prefers straight lines)")
print("   - Penalizes intersections heavily")
print("   - Enforces endpoint constraints")

# %% NEW CODE HERE!


