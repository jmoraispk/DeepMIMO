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
    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray, epsilon: float = 1e-10) -> bool:
        """Check if point q lies on segment pr."""
        if abs((q[1] - p[1]) * (r[0] - p[0]) - (r[1] - p[1]) * (q[0] - p[0])) > epsilon:
            return False
        return (min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon and
                min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon)
    
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray, epsilon: float = 1e-10) -> int:
        """Returns orientation of ordered triplet (p, q, r).
        Returns:
         0 --> p, q and r are collinear
         1 --> Clockwise
         2 --> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < epsilon:  # collinear
            return 0
        return 1 if val > 0 else 2
    
    def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
        """Check if segments p1p2 and q1q2 intersect."""
        # Find the four orientations needed for general and special cases
        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)
        
        # Skip if segments share an endpoint
        if np.allclose(p1, q1) or np.allclose(p1, q2) or \
           np.allclose(p2, q1) or np.allclose(p2, q2):
            return False
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
            
        # Special Cases: collinear segments
        if o1 == 0 and on_segment(p1, q1, p2): return True
        if o2 == 0 and on_segment(p1, q2, p2): return True
        if o3 == 0 and on_segment(q1, p1, q2): return True
        if o4 == 0 and on_segment(q1, p2, q2): return True
        
        return False
    
    intersections = []
    n_points = len(points_2d)
    
    # Check all regular segments
    for i in range(n_points - 1):
        # Check against other regular segments
        for j in range(i + 2, n_points - 1):
            if segments_intersect(points_2d[i], points_2d[i+1], 
                                points_2d[j], points_2d[j+1]):
                intersections.append((i, i+1, j, j+1))
        
        # Check against the closing segment (last point to first point)
        # Skip if this segment is adjacent to the closing segment
        if i > 0 and i < n_points - 2:  # Skip first and second-to-last segments
            if segments_intersect(points_2d[i], points_2d[i+1],
                                points_2d[-1], points_2d[0]):
                intersections.append((i, i+1, n_points-1, 0))
    
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

def load_geometry() -> RoadGeometry:
    """Load and process road geometry from the numpy file."""
    # Load road vertices
    road_vertices = np.load('road_vertices_roads_3.npy')
    print(f"\nLoaded road vertices: {len(road_vertices)} points")

    # Filter close points
    filtered_vertices = filter_close_points(road_vertices)
    print(f"After filtering: {len(filtered_vertices)} points")

    # Process the road geometry
    return process_road_vertices(filtered_vertices)

def _orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray, epsilon: float = 1e-10) -> int:
    """Calculate orientation of ordered triplet (p, q, r).
    Returns:
     0 --> p, q and r are collinear
     1 --> Clockwise
     2 --> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < epsilon:
        return 0
    return 1 if val > 0 else 2

def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray, epsilon: float = 1e-10) -> bool:
    """Check if point q lies on segment pr."""
    if abs((q[1] - p[1]) * (r[0] - p[0]) - (r[1] - p[1]) * (q[0] - p[0])) > epsilon:
        return False
    return (min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon and
            min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon)

def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, 
                       epsilon: float = 1e-10) -> bool:
    """Check if segments p1p2 and q1q2 intersect."""
    # Skip if segments share an endpoint
    if np.allclose(p1, q1) or np.allclose(p1, q2) or \
       np.allclose(p2, q1) or np.allclose(p2, q2):
        return False
        
    o1, o2 = _orientation(p1, p2, q1), _orientation(p1, p2, q2)
    o3, o4 = _orientation(q1, q2, p1), _orientation(q1, q2, p2)
    
    if o1 != o2 and o3 != o4:
        return True
        
    if o1 == 0 and _on_segment(p1, q1, p2): return True
    if o2 == 0 and _on_segment(p1, q2, p2): return True
    if o3 == 0 and _on_segment(q1, p1, q2): return True
    if o4 == 0 and _on_segment(q1, p2, q2): return True
    
    return False

def _point_inside_polygon(point: np.ndarray, polygon_points: list[np.ndarray]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    if len(polygon_points) < 3:
        return False
        
    x, y = point
    inside = False
    j = len(polygon_points) - 1
    
    for i in range(len(polygon_points)):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]
        
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
        
    return inside

def _would_trap_points(new_point: np.ndarray, current_path: list[np.ndarray], 
                      unused_points: list[np.ndarray]) -> bool:
    """Check if adding new_point would trap any unused points inside the path."""
    test_path = current_path + [new_point]
    
    for point in unused_points:
        if np.allclose(point, new_point):
            continue
        if _point_inside_polygon(point, test_path):
            return True
    return False

def _can_close_path(current_point: np.ndarray, start_point: np.ndarray, 
                    path_points: list[np.ndarray]) -> bool:
    """Check if we can close the path by connecting back to the start point."""
    return not would_create_intersection(start_point, path_points + [current_point], None)

def _get_endpoints(points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the leftmost and rightmost points as endpoints."""
    x_sorted_indices = np.argsort(points_2d[:, 0])
    left_endpoints = x_sorted_indices[:2]
    right_endpoints = x_sorted_indices[-2:]
    return left_endpoints, right_endpoints

def _calculate_max_distance(points_2d: np.ndarray) -> float:
    """Calculate maximum distance between any two points."""
    max_dist = 0
    for i in range(len(points_2d)):
        for j in range(i + 1, len(points_2d)):
            dist = np.linalg.norm(points_2d[i] - points_2d[j])
            max_dist = max(max_dist, dist)
    return max_dist

def _calculate_score(angle_dev: float, distance: float, max_distance: float) -> float:
    """Calculate score based on angle deviation and normalized distance.
    Lower scores are better (closer points with smaller angle deviations).
    """
    normalized_distance = distance / max_distance
    return angle_dev * (1 + normalized_distance)  # More weight on angle deviation

def find_optimized_path(points_2d: np.ndarray, 
                       start_idx: int = None) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    """Find an optimized path through points that minimizes angle deviations and avoids intersections."""
    n_points = len(points_2d)
    if n_points < 3:
        return list(range(n_points)), [], None
    
    print("\nStarting path optimization...")
    
    # Get endpoints and start from leftmost if not specified
    left_endpoints, right_endpoints = _get_endpoints(points_2d)
    if start_idx is None:
        start_idx = left_endpoints[0]
    
    print(f"Starting from endpoint {start_idx}")
    
    # Initialize path with starting point
    path_indices = [start_idx]
    path_points = [points_2d[start_idx]]
    used_points = np.zeros(n_points, dtype=bool)
    used_points[start_idx] = True
    
    # Calculate maximum distance for normalization
    max_distance = _calculate_max_distance(points_2d)
    print(f"Maximum distance between any two points: {max_distance:.2f}")
    
    # Add second point (closest endpoint)
    endpoint_distances = []
    all_endpoints = np.concatenate([left_endpoints, right_endpoints])
    for endpoint in all_endpoints:
        if endpoint != start_idx:
            dist = np.linalg.norm(points_2d[endpoint] - points_2d[start_idx])
            endpoint_distances.append((dist, endpoint))
    
    second_idx = min(endpoint_distances)[1]
    print(f"Connected to closest endpoint {second_idx}")
    path_indices.append(second_idx)
    path_points.append(points_2d[second_idx])
    used_points[second_idx] = True
    
    # Build rest of path
    while len(path_indices) < n_points:
        current_point = path_points[-1]
        prev_point = path_points[-2]
        points_remaining = n_points - len(path_indices)
        print(f"\nFinding next point... ({points_remaining} points remaining)")
        
        # Get unused points
        unused_mask = ~used_points
        unused_indices = np.where(unused_mask)[0]
        unused_points = points_2d[unused_mask]
        
        # Try all unused points
        candidates = []
        for idx, point in zip(unused_indices, unused_points):
            # Check for intersections
            if would_create_intersection(point, path_points, None):
                print(f"  Point {idx} rejected: Would create intersection")
                continue
            
            if _would_trap_points(point, path_points, unused_points):
                print(f"  Point {idx} rejected: Would trap unused points")
                continue
            
            # If this is the last point, check if we can close the path
            if points_remaining == 1 and not _can_close_path(point, points_2d[start_idx], path_points):
                print(f"  Point {idx} rejected: Would prevent path closure")
                continue
            
            # Calculate metrics
            angle_dev = calculate_angle_deviation(prev_point, current_point, point)
            distance = np.linalg.norm(point - current_point)
            norm_distance = distance / max_distance
            score = _calculate_score(angle_dev, distance, max_distance)
            
            print(f"  Point {idx}: angle={angle_dev:.2f}°, dist={distance:.2f} (norm={norm_distance:.2f}), score={score:.2f}")
            candidates.append((score, idx, point))
        
        if candidates:
            # Sort by score and take the best one
            candidates.sort()  # Will sort by score since it's first in tuple
            best_score, best_idx, best_point = candidates[0]
            print(f"Selected point {best_idx} with score {best_score:.2f}")
            
            path_indices.append(best_idx)
            path_points.append(best_point)
            used_points[best_idx] = True
        else:
            print(f"\nPath finding stuck after {len(path_indices)} points")
            # Create edge list for partial path
            edge_list = [(path_indices[i], path_indices[i + 1]) 
                        for i in range(len(path_indices) - 1)]
            print("\nPartial path found. Edge list:")
            for i, (start, end) in enumerate(edge_list):
                print(f"Edge {i}: ({start}, {end})")
            return None, edge_list, path_indices
    
    print("\nPath complete!")
    
    # Create final edge list
    edge_list = [(path_indices[i], path_indices[i + 1]) 
                 for i in range(len(path_indices) - 1)]
    
    print("\nFinal edge list:")
    for i, (start, end) in enumerate(edge_list):
        print(f"Edge {i}: ({start}, {end})")
    
    return path_indices, edge_list, None

def visualize_path(points_2d: np.ndarray, 
                  complete_path: list[int] = None, 
                  partial_path: list[int] = None,
                  edge_list: list[tuple[int, int]] = None,
                  title: str = "Path Visualization"):
    """Visualize the points and path with labeled edges."""
    # Get endpoints
    left_endpoints, right_endpoints = _get_endpoints(points_2d)
    all_endpoints = np.concatenate([left_endpoints, right_endpoints])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    regular_points = [i for i in range(len(points_2d)) if i not in all_endpoints]
    plt.scatter(points_2d[regular_points, 0], points_2d[regular_points, 1], 
                c='blue', s=50, label='Regular Points')
    plt.scatter(points_2d[all_endpoints, 0], points_2d[all_endpoints, 1], 
                c='red', s=100, label='Endpoints')
    
    # Annotate points with their indices
    for i, (x, y) in enumerate(points_2d):
        plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot edges with arrows and labels
    if edge_list:
        for i, (idx1, idx2) in enumerate(edge_list):
            p1, p2 = points_2d[idx1], points_2d[idx2]
            
            # Draw arrow
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            plt.arrow(p1[0], p1[1], dx, dy,
                     head_width=0.1, head_length=0.2, fc='green', ec='green',
                     length_includes_head=True, alpha=0.6)
            
            # Add edge label
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            x_offset = 20 if p1[1] < p2[1] else -20
            edge_label = f"({idx1},{idx2})"
            plt.annotate(edge_label, (mid_x + x_offset, mid_y), 
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    status = "Complete" if complete_path else "Incomplete"
    plt.title(f"{title} ({status})")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_angle_deviation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate the angle deviation from a straight line between three points."""
    if np.allclose(p1, p2) or np.allclose(p2, p3):
        return 180.0  # Maximum deviation for duplicate points
        
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    
    # Return the deviation from 180 degrees (straight line)
    return min(angle, 180.0 - angle)

def would_create_intersection(new_point: np.ndarray, current_path: list[np.ndarray], 
                            points_2d: np.ndarray) -> bool:
    """Check if adding new_point to the path would create any intersections."""
    if len(current_path) < 2:
        return False
        
    # Create the new segment
    new_segment = (current_path[-1], new_point)
    
    # Check against all existing segments except the last one
    for i in range(len(current_path) - 2):
        if _segments_intersect(new_segment[0], new_segment[1],
                           current_path[i], current_path[i + 1]):
            return True
    
    return False

def compute_edge_cost(source: np.ndarray, current: np.ndarray, next_hop: np.ndarray) -> float:
    """Compute the cost of adding next_hop after the edge (source, current).
    
    Args:
        source: The source vertex (previous point)
        current: The current vertex (already connected to source)
        next_hop: The candidate next vertex to evaluate
        
    Returns:
        float: Cost value where lower is better. Combines:
            - Angle deviation from straight line (0° is best)
            - Distance from current to next_hop (shorter is better)
    """
    # Calculate angle deviation from straight line
    angle_dev = calculate_angle_deviation(source, current, next_hop)
    
    # Calculate normalized distance (0 to 1 range)
    distance = np.linalg.norm(next_hop - current)
    
    # Combine metrics:
    # - angle_dev ranges from 0° (best) to 180° (worst)
    # - distance is actual distance in the space
    # We want both small angles and small distances
    cost = angle_dev * (1 + distance)
    
    return cost

def main():
    # Load geometry
    geometry = load_geometry()
    
    # Find optimized path and get edge list
    complete_path, edge_list, partial_path = find_optimized_path(geometry.points_2d)
    
    # Visualize the results
    if complete_path is not None:
        visualize_path(geometry.points_2d, 
                      complete_path=complete_path,
                      edge_list=edge_list,
                      title="Complete Optimized Path")
        
        print("\nFinal edge list:")
        for i, (start, end) in enumerate(edge_list):
            print(f"Edge {i}: ({start}, {end})")
    else:
        print("\nNo complete path found! Showing partial path.")
        visualize_path(geometry.points_2d, 
                      partial_path=partial_path,
                      edge_list=edge_list,
                      title="Incomplete Path")
        
        print("\nPartial edge list:")
        for i, (start, end) in enumerate(edge_list):
            print(f"Edge {i}: ({start}, {end})")

if __name__ == "__main__":
    main()

#%%

# This is like a shortest path, where cost is the angle deviation from the straight line
# and the distance between the points. 

# The algorithm is:
# 1. Find the two pairs of end points of the path (already done)
# 2. Start in one pair and connect the endpoints.
# 3. Build a tree of possible point orders.
# 4. Once all points are used, connect the last point to the start point.
# 5. The edge list is the list of edges in the tree.

# The tree is such that each node is a point and each edge is a possible point order.
# The root is the starting point.
# The leaves are the ending points.
# The internal nodes are the points in the middle.


#%%

import numpy as np
import pandas as pd

# Input data as a DataFrame for easier handling
data = {
    "Point ID": list(range(16)),
    "X": [26.87, 41.20, 27.76, 0.90, -4.58, -76.82, -67.64, 30.11, -85.58, 35.19, 24.14, 18.49, -62.15, 34.61, 23.92, 33.59],
    "Y": [-37.58, -77.20, -41.83, 12.30, 7.95, 107.70, 83.88, -26.84, 107.70, -77.20, -18.42, -22.55, 88.22, -40.39, -30.21, -35.54],
    "Min Distance": [4.35, 6.00, 4.35, 7.00, 7.00, 8.76, 7.00, 7.05, 8.76, 6.00, 7.00, 7.00, 7.00, 4.96, 7.05, 4.96]
}
df = pd.DataFrame(data)

# Convert points into a dictionary for fast access
points = {int(row["Point ID"]): np.array([row["X"], row["Y"]]) for _, row in df.iterrows()}

# Cost function
def calculate_angle_deviation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    if np.allclose(p1, p2) or np.allclose(p2, p3):
        return 180.0
    v1 = p2 - p1
    v2 = p3 - p2
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    return min(angle, 180.0 - angle)

# Graph: each node points to others within the min distance radius, computing cost using p1 -> p2 -> p3
graph = {i: [] for i in points}

# For each triple (p1, p2, p3), if p3 is within min distance of p2, add an edge from p2 to p3 with cost
for p1_id in points:
    for p2_id in points:
        if p1_id == p2_id:
            continue
        for p3_id in points:
            if p2_id == p3_id or p1_id == p3_id:
                continue
            p2 = points[p2_id]
            p3 = points[p3_id]
            min_dist = df.loc[df["Point ID"] == p2_id, "Min Distance"].values[0]
            if np.linalg.norm(p3 - p2) <= min_dist:
                cost = calculate_angle_deviation(points[p1_id], p2, p3)
                graph[p2_id].append((p3_id, cost))



#%%

import heapq

def dijkstra(graph, start):
    # graph: dict of node -> list of (neighbor, weight)
    # returns: dict of shortest distances from start
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        curr_dist, node = heapq.heappop(heap)
        if curr_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = curr_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return dist


print(dijkstra(graph, 0))


#%% TSP brute force

import itertools
import time
import random

def generate_random_dist_matrix(n, seed=42):
    random.seed(seed)
    return [[0 if i == j else random.randint(1, 100) for j in range(n)] for i in range(n)]

def tsp_brute_force(dist_matrix):
    n = len(dist_matrix)
    nodes = range(1, n)
    min_cost = float('inf')
    best_path = None

    for perm in itertools.permutations(nodes):
        path = (0,) + perm + (0,)
        cost = sum(dist_matrix[path[i]][path[i+1]] for i in range(n))
        if cost < min_cost:
            min_cost = cost
            best_path = path

    return best_path, min_cost

results = []

for n in range(3, 12):
    dist_matrix = generate_random_dist_matrix(n)
    start_time = time.time()
    tsp_brute_force(dist_matrix)
    duration = time.time() - start_time
    results.append((n, duration))

print(results)

# If I can discard all vertices of a road such that we keep only up to 10 points, 
# we can use brute force.

#%% TSP Held-Karp

def tsp_held_karp(dist_matrix):
    """
    Solves TSP using Held-Karp dynamic programming algorithm.
    Returns the minimum cost and the corresponding path.
    """
    n = len(dist_matrix)
    C = {}

    # Set initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dist_matrix[0][k], [0, k])

    # Iterate through subsets of increasing length and store optimal paths
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    prev_cost, prev_path = C[(prev_bits, m)]
                    cost = prev_cost + dist_matrix[m][k]
                    res.append((cost, prev_path + [k]))
                C[(bits, k)] = min(res)

    # Final step: return to start node
    bits = (1 << n) - 2
    res = []
    for k in range(1, n):
        cost, path = C[(bits, k)]
        res.append((cost + dist_matrix[k][0], path + [0]))

    return min(res)

# Time the Held-Karp algorithm for n=3 to n=10
results_held_karp = []
for n in range(3, 17):
    dist_matrix = generate_random_dist_matrix(n)
    start_time = time.time()
    _, _ = tsp_held_karp(dist_matrix)
    duration = time.time() - start_time
    results_held_karp.append((n, duration))

print(results_held_karp)

# 14 points is the first to cross the 0.1 second mark.

# beyond that we may start trimming points or partitioning roads. 
# If there are roads longer than 14 that we can't trim, we may just not do it. 
# I.e. Conversion without converting the roads. 
# Otherwise it becomes a whole new problem. And buildings + plane is enough. 
# Therefore, we may only support roads for a couple of locations. 
# Also, if we include the blender screenshot, we might not even have to HAVE roads.

# ------------------------------------------------------------------------------
# DeepMIMO is not supposed to be a perfect representation of a real city. 
# Just a pragmatic representation of it. The buildings don't have to be perfect. 
# The roads don't have to be perfect. (or even existent!)
# Better lightweight, simple, and fast conversion, than lossless, perfect, and slow.
# We care about the ray tracing, not a perfect representation of the scene.
# For that, people can use Blender. And we can include the blender file for the scene.

#%%




