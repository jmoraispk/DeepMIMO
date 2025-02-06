from plyfile import PlyData


def convert_ply2city(ply_path, material_path, save_path, object_name=None):
    if not object_name:
        object_name = ply_path.split(".")[0].split("/")[-1]

    ply_data = PlyData.read(ply_path)
    with open(material_path) as f:
        material_sec = f.readlines()

    with open(save_path, "w") as f:
        f.write("Format type:keyword version: 1.1.0\n")
        f.write("begin_<city> " + object_name + "\n")
        write_reference_sec(f)
        write_material_sec(f, material_sec)
        write_face_sec(f, ply_data)
        f.write("end_<city>\n")
    return (len(ply_data["vertex"]), len(ply_data["face"]))


def write_reference_sec(f):
    with open("resource/template/reference_section.txt") as f1:
        reference_sec = f1.readlines()
    return f.writelines(reference_sec)


def write_material_sec(f, material_sec):
    return f.writelines(material_sec)


def write_face_sec(f, ply_data):
    f.write("begin_<structure_group> \n")
    f.write("begin_<structure> \n")
    f.write("begin_<sub_structure> \n")

    for face in ply_data["face"]:
        vertex_idx = face[0]
        num_vertex = vertex_idx.size
        f.write("begin_<face> \n")
        f.write("Material 1\n")
        f.write("nVertices %d\n" % num_vertex)
        for v in vertex_idx:
            x = ply_data["vertex"][v][0]
            y = ply_data["vertex"][v][1]
            z = ply_data["vertex"][v][2]
            f.write("%.10f " % x)
            f.write("%.10f " % y)
            f.write("%.10f\n" % z)
        f.write("end_<face>\n")

    f.write("end_<sub_structure>\n")
    f.write("end_<structure>\n")
    f.write("end_<structure_group>\n")
    return


if __name__ == "__main__":
    ply_path = "scenario/city_models/scenario_0/gwc_building.ply"
    material_path = "resource/material/ITU Concrete 2.4 GHz.mtl"
    save_path = "scenario/city_models/scenario_0/gwc_building.city"

    (num_vertex, num_faces) = convert_ply2city(ply_path, material_path, save_path)

    print("Converted %d vertexes and %d faces" % (num_vertex, num_faces))

    ply_path = "scenario/city_models/scenario_0/gwc_road.ply"
    material_path = "resource/material/Asphalt_1GHz.mtl"
    save_path = "scenario/city_models/scenario_0/gwc_road.city"

    (num_vertex, num_faces) = convert_ply2city(ply_path, material_path, save_path)

    print("Converted %d vertexes and %d faces" % (num_vertex, num_faces))
