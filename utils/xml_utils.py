# utils/xml_utils.py
import xml.etree.ElementTree as ET
import bpy

def get_material_rgb(material):
    """Extract RGB from a material."""
    return f"{material.diffuse_color[0]:.6f} {material.diffuse_color[1]:.6f} {material.diffuse_color[2]:.6f}"

def generate_xml_from_blender(output_path):
    """Generate XML from Blender scene data."""
    scene = bpy.context.scene
    xml_scene = ET.Element("scene", version="2.1.0")

    # Integrator
    integrator = ET.SubElement(xml_scene, "integrator", type="path", id="elm__0", name="elm__0")
    ET.SubElement(integrator, "integer", name="max_depth", value=str(scene.cycles.max_bounces))

    # BSDF: Materials
    for mat in bpy.data.materials:
        if mat.users > 0:
            bsdf_id = f"mat-{mat.name.replace('.', '_')}"
            bsdf = ET.SubElement(xml_scene, "bsdf", type="diffuse", id=bsdf_id, name=bsdf_id)
            ET.SubElement(bsdf, "rgb", name="reflectance", value=get_material_rgb(mat))

    # Emitter: World
    world = scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                color = node.inputs['Color'].default_value[:3]
                emitter = ET.SubElement(xml_scene, "emitter", type="constant", id="World", name="World")
                ET.SubElement(emitter, "rgb", name="radiance", value=f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}")
                break

    # Shape: Mesh objects
    for o, mat in [('buildings', 'itu_concrete'), ('roads', 'itu_brick'), ('terrain', 'itu_wet_ground')]:
        shape_id = f"mesh-{o}"
        shape = ET.SubElement(xml_scene, "shape", type="ply", id=shape_id, name=shape_id)
        ET.SubElement(shape, "string", name="filename", value=f"meshes/{o}.ply")
        ET.SubElement(shape, "boolean", name="face_normals", value="true")
        ET.SubElement(shape, "ref", id=f"mat-{mat}", name="bsdf")

    # Pretty print
    def indent(elem, level=0):
        indent_str = "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = "\n" + indent_str * (level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = "\n" + indent_str * level
            for child in elem:
                indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = "\n" + indent_str * level
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = "\n" + indent_str * level

    indent(xml_scene)
    xml_str = ET.tostring(xml_scene, encoding='utf-8', method='xml').decode('utf-8')
    if xml_str.startswith('<?xml'):
        xml_str = xml_str.split('\n', 1)[1]

    with open(output_path, "w") as f:
        f.write(xml_str)
    print(f"XML file generated: {output_path}")