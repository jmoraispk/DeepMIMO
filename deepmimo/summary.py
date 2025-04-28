"""
Summarizes dataset characteristics.

This module is used by the Database API to send summaries to the server. 
As such, the information displayed here will match the information
displayed on the DeepMIMO website. 

The module is also leveraged by users to understand a dataset during development.

Usage:
    summary(scen_name, print_summary=True)


Three functions:

1. summary(scen_name, print_summary=True)
    - If print_summary is True, prints a summary of the dataset.
    - If print_summary is False, returns a string summary of the dataset.
    - Used for printing summaries to the console.
    - *Provides* the information for each DeepMIMO scenario page.

2. plot_summary(scen_name)
    - Plots several figures representing the dataset.
    - Plot 1: LOS image
    - Plot 2: 3D view of the scene (buildings, roads, trees, etc.)
    - Plot 3: 2D view of the scene with BSs and users
    - Returns None
    - *Provides* the figures for each DeepMIMO scenario page.

3. stats(scen_name)
    - (coming soon)
    - Returns a dictionary of statistics about the dataset.

"""
from .general_utils import (
    get_params_path,
    load_dict_from_json
)
from . import consts as c
from typing import Optional


def summary(scen_name: str, print_summary: bool = True) -> Optional[str]:
    """Print a summary of the dataset."""
    # Initialize empty string to collect output
    summary_str = ""

    # Read params.mat and provide TXRX summary, total number of tx & rx, scene size,
    # and other relevant parameters, computed/extracted from the all dicts, not just rt_params

    params_json_path = get_params_path(scen_name)

    params_dict = load_dict_from_json(params_json_path)
    rt_params = params_dict[c.RT_PARAMS_PARAM_NAME]
    scene_params = params_dict[c.SCENE_PARAM_NAME]
    material_params = params_dict[c.MATERIALS_PARAM_NAME]
    txrx_params = params_dict[c.TXRX_PARAM_NAME]

    summary_str += "\n" + "=" * 50 + "\n"
    summary_str += f"DeepMIMO {scen_name} Scenario Summary\n"
    summary_str += "=" * 50 + "\n"

    summary_str += "\n[Ray-Tracing Configuration]\n"
    summary_str += (
        f"- Ray-tracer: {rt_params[c.RT_PARAM_RAYTRACER]} "
        f"v{rt_params[c.RT_PARAM_RAYTRACER_VERSION]}\n"
    )
    summary_str += f"- Frequency: {rt_params[c.RT_PARAM_FREQUENCY]/1e9:.1f} GHz\n"

    summary_str += "\n[Ray-tracing parameters]\n"

    # Interaction limits
    summary_str += "\nMain interaction limits\n"
    summary_str += f"- Max path depth: {rt_params[c.RT_PARAM_PATH_DEPTH]}\n"
    summary_str += f"- Max reflections: {rt_params[c.RT_PARAM_MAX_REFLECTIONS]}\n"
    summary_str += f"- Max diffractions: {rt_params[c.RT_PARAM_MAX_DIFFRACTIONS]}\n"
    summary_str += f"- Max scatterings: {rt_params[c.RT_PARAM_MAX_SCATTERING]}\n"
    summary_str += f"- Max transmissions: {rt_params[c.RT_PARAM_MAX_TRANSMISSIONS]}\n"

    # Diffuse scattering settings
    summary_str += "\nDiffuse Scattering\n"
    is_diffuse_enabled = rt_params[c.RT_PARAM_MAX_SCATTERING] > 0
    summary_str += f"- Diffuse scattering: {'Enabled' if is_diffuse_enabled else 'Disabled'}\n"
    if is_diffuse_enabled:
        summary_str += f"- Diffuse reflections: {rt_params[c.RT_PARAM_DIFFUSE_REFLECTIONS]}\n"
        summary_str += f"- Diffuse diffractions: {rt_params[c.RT_PARAM_DIFFUSE_DIFFRACTIONS]}\n"
        summary_str += f"- Diffuse transmissions: {rt_params[c.RT_PARAM_DIFFUSE_TRANSMISSIONS]}\n"
        summary_str += f"- Final interaction only: {rt_params[c.RT_PARAM_DIFFUSE_FINAL_ONLY]}\n"
        summary_str += f"- Random phases: {rt_params[c.RT_PARAM_DIFFUSE_RANDOM_PHASES]}\n"

    # Terrain settings
    summary_str += "\nTerrain\n"
    summary_str += f"- Terrain reflection: {rt_params[c.RT_PARAM_TERRAIN_REFLECTION]}\n"
    summary_str += f"- Terrain diffraction: {rt_params[c.RT_PARAM_TERRAIN_DIFFRACTION]}\n"
    summary_str += f"- Terrain scattering: {rt_params[c.RT_PARAM_TERRAIN_SCATTERING]}\n"

    # Ray casting settings
    summary_str += "\nRay Casting Settings\n"
    summary_str += f"- Number of rays: {rt_params[c.RT_PARAM_NUM_RAYS]:,}\n"
    summary_str += f"- Casting method: {rt_params[c.RT_PARAM_RAY_CASTING_METHOD]}\n"
    summary_str += f"- Casting range (az): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_AZ]:.1f}°\n"
    summary_str += f"- Casting range (el): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_EL]:.1f}°\n"
    summary_str += f"- Synthetic array: {rt_params[c.RT_PARAM_SYNTHETIC_ARRAY]}\n"

    # Scene
    summary_str += "\n[Scene]\n"
    summary_str += f"- Number of scenes: {scene_params[c.SCENE_PARAM_NUMBER_SCENES]}\n"
    summary_str += f"- Total objects: {scene_params[c.SCENE_PARAM_N_OBJECTS]:,}\n"
    summary_str += f"- Vertices: {scene_params[c.SCENE_PARAM_N_VERTICES]:,}\n"
    summary_str += f"- Faces: {scene_params[c.SCENE_PARAM_N_FACES]:,}\n"
    summary_str += f"- Triangular faces: {scene_params[c.SCENE_PARAM_N_TRIANGULAR_FACES]:,}\n"

    # Materials
    summary_str += "\n[Materials]\n"
    summary_str += f"Total materials: {len(material_params)}\n"
    for _, mat_props in material_params.items():
        summary_str += f"\n{mat_props[c.MATERIALS_PARAM_NAME_FIELD]}:\n"
        summary_str += f"- Permittivity: {mat_props[c.MATERIALS_PARAM_PERMITTIVITY]:.2f}\n"
        summary_str += f"- Conductivity: {mat_props[c.MATERIALS_PARAM_CONDUCTIVITY]:.2f} S/m\n"
        summary_str += f"- Scattering model: {mat_props[c.MATERIALS_PARAM_SCATTERING_MODEL]}\n"
        summary_str += f"- Scattering coefficient: {mat_props[c.MATERIALS_PARAM_SCATTERING_COEF]:.2f}\n"
        summary_str += f"- Cross-polarization coefficient: {mat_props[c.MATERIALS_PARAM_CROSS_POL_COEF]:.2f}\n"

    # TX/RX
    summary_str += "\n[TX/RX Configuration]\n"

    # Sum total number of receivers and transmitters
    n_rx = sum(
        set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]
        for set_info in txrx_params.values()
        if set_info[c.TXRX_PARAM_IS_RX]
    )
    n_tx = sum(
        set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]
        for set_info in txrx_params.values()
        if set_info[c.TXRX_PARAM_IS_TX]
    )
    summary_str += f"Total number of receivers: {n_rx}\n"
    summary_str += f"Total number of transmitters: {n_tx}\n"

    for set_name, set_info in txrx_params.items():
        summary_str += f"\n{set_name} ({set_info[c.TXRX_PARAM_NAME_FIELD]}):\n"
        role = []
        if set_info[c.TXRX_PARAM_IS_TX]:
            role.append("TX")
        if set_info[c.TXRX_PARAM_IS_RX]:
            role.append("RX")
        summary_str += f"- Role: {' & '.join(role)}\n"
        summary_str += f"- Total points: {set_info[c.TXRX_PARAM_NUM_POINTS]:,}\n"
        summary_str += f"- Active points: {set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]:,}\n"
        summary_str += f"- Antennas per point: {set_info[c.TXRX_PARAM_NUM_ANT]}\n"
        summary_str += f"- Dual polarization: {set_info[c.TXRX_PARAM_DUAL_POL]}\n"

    # GPS Bounding Box
    if rt_params.get(c.RT_PARAM_GPS_BBOX, (0,0,0,0)) != (0,0,0,0):
        summary_str += "\n[GPS Bounding Box]\n"
        summary_str += f"- Min latitude: {rt_params[c.RT_PARAM_GPS_BBOX][0]:.2f}\n"
        summary_str += f"- Min longitude: {rt_params[c.RT_PARAM_GPS_BBOX][1]:.2f}\n"
        summary_str += f"- Max latitude: {rt_params[c.RT_PARAM_GPS_BBOX][2]:.2f}\n"
        summary_str += f"- Max longitude: {rt_params[c.RT_PARAM_GPS_BBOX][3]:.2f}\n"

    # Print summary
    if print_summary:
        print(summary_str)
        return None
    
    return summary_str