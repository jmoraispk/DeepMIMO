import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

def extract_buildings(content):
    # Split content into faces
    face_pattern = r'begin_<face>(.*?)end_<face>'
    faces = re.findall(face_pattern, content, re.DOTALL)
    
    # Pattern to match coordinates in face definitions
    vertex_pattern = r'-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+'
    
    # Group faces that share vertices to form buildings
    buildings = []
    processed_faces = set()
    
    for i, face in enumerate(faces):
        if i in processed_faces:
            continue
            
        # Start a new building with this face
        building_vertices = set()
        face_stack = [i]
        
        while face_stack:
            current_face_idx = face_stack.pop()
            if current_face_idx in processed_faces:
                continue
                
            current_face = faces[current_face_idx]
            processed_faces.add(current_face_idx)
            
            # Extract vertices from current face
            current_vertices = [(float(x), float(y), float(z)) 
                              for x, y, z in [v.split() 
                              for v in re.findall(vertex_pattern, current_face)]]
            
            # Add vertices to building
            building_vertices.update(current_vertices)
            
            # Look for connected faces
            for j, other_face in enumerate(faces):
                if j not in processed_faces:
                    other_vertices = [(float(x), float(y), float(z)) 
                                    for x, y, z in [v.split() 
                                    for v in re.findall(vertex_pattern, other_face)]]
                    
                    # If faces share any vertices, add to stack
                    if any(v in current_vertices for v in other_vertices):
                        face_stack.append(j)
        
        if building_vertices:
            buildings.append(list(building_vertices))
    
    return buildings


def extract_buildings2(content):
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
    
    # Group faces that share vertices to form buildings
    buildings = []
    processed_faces = set()
    
    for i in range(len(faces)):
        if i in processed_faces:
            continue
            
        # Start a new building with this face
        building_vertices = set()
        face_stack = [i]
        
        while face_stack:
            current_face_idx = face_stack.pop()
            if current_face_idx in processed_faces:
                continue
                
            current_vertices = face_vertices[current_face_idx]
            processed_faces.add(current_face_idx)
            
            # Add vertices to building
            building_vertices.update(current_vertices)
            
            # Find connected faces using vertex_to_faces mapping
            connected_faces = set()
            for vertex in current_vertices:
                connected_faces.update(vertex_to_faces[vertex])
            
            # Add unprocessed connected faces to stack
            face_stack.extend(f for f in connected_faces if f not in processed_faces)
        
        if building_vertices:
            buildings.append(list(building_vertices))
    
    return buildings


def get_building_shape(vertices):
    # Extract footprint points (x,y coordinates)
    points_2d = np.array([(x, y) for x, y, z in vertices])
    
    # Get building height (assuming constant height)
    heights = [z for _, _, z in vertices]
    building_height = max(heights) - min(heights)
    base_height = min(heights)
    
    # Create convex hull for footprint
    hull = ConvexHull(points_2d)
    footprint = points_2d[hull.vertices]
    
    # Create top and bottom faces
    bottom_face = [(x, y, base_height) for x, y in footprint]
    top_face = [(x, y, base_height + building_height) for x, y in footprint]
    
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
    
    return faces, building_height



def plot_buildings_3d(buildings, save=False):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(buildings)))
    
    # Plot each building
    for building, color in zip(buildings, colors):
        faces, height = get_building_shape(building)
        
        # Create 3D polygons
        poly3d = Poly3DCollection(faces, alpha=0.6)
        poly3d.set_facecolor(color)
        poly3d.set_edgecolor('black')
        ax.add_collection3d(poly3d)
    
    # Set axis limits
    all_points = np.vstack([np.array(building) for building in buildings])
    min_x, max_x = np.min(all_points[:,0]), np.max(all_points[:,0])
    min_y, max_y = np.min(all_points[:,1]), np.max(all_points[:,1])
    min_z, max_z = np.min(all_points[:,2]), np.max(all_points[:,2])
    
    # Center the view and set equal aspect ratio
    max_range = max(max_x-min_x, max_y-min_y, max_z-min_z) / 2.0
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    mid_z = (max_z + min_z) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*1.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Building Shapes\nTotal Buildings: {len(buildings)}')
    
    # Save the plot
    if save:
        plt.savefig('building_3d_shapes.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'building_3d_shapes.png'")
    
    return fig, ax


def save_results(buildings):
    with open('building_shapes_results.txt', 'w') as f:
        f.write(f"Total number of buildings: {len(buildings)}\n\n")
        
        for i, building in enumerate(buildings):
            _, height = get_building_shape(building)
            points_2d = np.array([(x, y) for x, y, z in building])
            hull = ConvexHull(points_2d)
            
            f.write(f"Building {i+1}:\n")
            f.write(f"Number of vertices: {len(building)}\n")
            f.write(f"Height: {height:.2f}\n")
            f.write(f"Footprint area: {hull.area:.2f}\n")
            f.write(f"Volume (approximate): {hull.area * height:.2f}\n")
            f.write("\n")
    
    print("Results saved to 'building_shapes_results.txt'")
    
def city_vis(city_file, plot=False):
    
    # Read city file
    with open(city_file, 'r') as file:
        content = file.read()
    
    # Extract buildings
    # buildings = extract_buildings(content)
    buildings = extract_buildings2(content)

    
    if buildings:
        print(f"\nFound {len(buildings)} buildings")
        
        if plot:
            # Create visualization
            plot_buildings_3d(buildings)
        
            # Save detailed results
            # save_results(buildings)
    else:
        print("No buildings found in the file.")
    
if __name__ == "__main__":
    # For testing with the provided content
    # file = r'./P2Ms/simple_street_canyon_test/simple_street_canyon_buildings.city'
    file = r'./P2Ms/asu_campus/asu_buildings.city'
    import time
    t = time.time()
    city_vis(file, plot=True)
    print(f'{time.time() - t:.2f}s')