# utils/sionna_utils.py
from sionna.rt import (
    load_scene,
    PlanarArray,
    RadioMaterial,
    BackscatteringPattern,
    Scene
)

def set_materials(scene: Scene) -> Scene:
    """Set radio material properties for Sionna."""
    for obj in scene.objects.values():
        mat_name = scene.objects[obj.name].radio_material.name
        if mat_name == 'itu_concrete':
            scene.objects[obj.name].radio_material.scattering_coefficient = 0.4
            scene.objects[obj.name].radio_material.xpd_coefficient = 0.4
            pattern = BackscatteringPattern(alpha_r=4, alpha_i=4, lambda_=0.75)
            scene.objects[obj.name].radio_material.scattering_pattern = pattern
        elif mat_name in ['itu_wet_ground', 'itu_brick']:
            continue
        else:
            print(f"Unknown material: {mat_name}")
            exit()

    asphalt_pattern = BackscatteringPattern(alpha_r=4, alpha_i=4, lambda_=0.75)
    asphalt_material = RadioMaterial(
        "asphalt", 
        relative_permittivity=5.72, 
        conductivity=5e-4,
        scattering_coefficient=0.4, 
        xpd_coefficient=0.4,
        scattering_pattern=asphalt_pattern)
    scene.add(asphalt_material)

    for obj in scene.objects.keys():
        if 'road' in obj or 'path' in obj:
            scene.objects[obj].radio_material = asphalt_material
            print(f"Set asphalt material for {obj}")

    return scene

def create_base_scene(scene_path: str, center_frequency: float) -> Scene:
    """Create a base Sionna scene."""
    scene = load_scene(scene_path)
    scene.frequency = center_frequency
    scene.tx_array = PlanarArray(
        num_rows=1, 
        num_cols=1, 
        vertical_spacing=0.5,
        horizontal_spacing=0.5, 
        pattern="iso", 
        polarization="V")
    scene.rx_array = scene.tx_array
    scene.synthetic_array = True
    return scene