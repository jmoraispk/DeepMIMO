from generate_city.convert_ply2city import convert_ply2city
from pathlib import Path


def generate_city(project_root, insite_path, building_mtl_path, road_mtl_path):
    """Convert PLY files to Wireless InSite city feature files"""
    city_feature_list = []

    project_root = Path(project_root)  # Convert to Path object for cross-platform compatibility
    insite_path = Path(insite_path)

    # Convert buildings
    ply_path = project_root / "buildings.ply"
    material_path = building_mtl_path
    save_path = insite_path / "buildings.city"

    if ply_path.exists():
        num_vertex, num_faces = convert_ply2city(str(ply_path), material_path, str(save_path))
        print(f"Converted {num_vertex} vertices and {num_faces} faces for buildings")
        city_feature_list.append("buildings.city")
    else:
        print(f"Warning: {ply_path} not found. Skipping building conversion.")

    # Convert roads
    ply_path = project_root / "roads.ply"
    material_path = road_mtl_path
    save_path = insite_path / "roads.city"

    if ply_path.exists():
        num_vertex, num_faces = convert_ply2city(str(ply_path), material_path, str(save_path))
        print(f"Converted {num_vertex} vertices and {num_faces} faces for roads")
        city_feature_list.append("roads.city")
    else:
        print(f"Warning: {ply_path} not found. Skipping road conversion.")

    return city_feature_list

