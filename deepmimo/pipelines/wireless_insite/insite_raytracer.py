"""
Wireless InSite Ray Tracing Pipeline.

This module provides a complete pipeline for generating electromagnetic simulation scenarios
using OpenStreetMap data and running ray tracing simulations with Wireless InSite.
It handles the entire process from OSM data extraction to DeepMIMO dataset generation.
"""

#%% Imports

# Standard library imports
import os
import subprocess
from dataclasses import fields
from typing import Dict, Tuple, Any

# Third-party imports
import numpy as np

# Local application imports
from .WI_interface.XmlGenerator import XmlGenerator
from .WI_interface.SetupEditor import SetupEditor, RayTracingParam
from .WI_interface.TxRxEditor import TxRxEditor
from .WI_interface.TerrainEditor import TerrainEditor
from .convert_ply2city import convert_to_city_file

# Project-specific imports
from ..pipeline_consts import (WI_EXE, WI_LIC, WI_VERSION, BUILDING_MATERIAL_PATH,
                               ROAD_MATERIAL_PATH, TERRAIN_MATERIAL_PATH)
from ..geo_utils import convert_GpsBBox2CartesianBBox

TERRAIN_TEMPLATE = "newTerrain.ter"

def create_directory_structure(osm_folder: str, rt_params: Dict[str, Any]) -> Tuple[str, str]:
    """Create folders for the scenario generations with a names based on parameters.
    
    Args:
        base_path (str): Base path for the scenario
        rt_params (Dict[str, Any]): Ray tracing parameters
        
    Returns:
        Tuple[str, str]: Paths to the insite directory and study area directory
    """
    
    # Format folder name with key parameters
    folder_name = (f"insite2_{rt_params['carrier_freq']/1e9:.1f}GHz_"
                   f"{rt_params['max_reflections']}R_{rt_params['max_diffractions']}D_"
                   f"{1 if rt_params['ds_enable'] else 0}S")
    insite_path = os.path.join(osm_folder, folder_name)
    os.makedirs(insite_path, exist_ok=True)

    insite_path = os.path.join(osm_folder, folder_name)
    study_area_path = os.path.join(insite_path, "study_area")

    # Create directories
    for path in [insite_path, study_area_path]:
        os.makedirs(path, exist_ok=True)

    return insite_path, study_area_path

def raytrace_insite(osm_folder: str, tx_pos: np.ndarray, rx_pos: np.ndarray, **rt_params: Any) -> str:
    """Run Wireless InSite ray tracing simulation.
    
    This function sets up the simulation environment, generates the necessary files,
    and runs the ray tracing simulation. It creates both human-readable text files
    (.setup, .txrx, .ter, .city) and the XML file that is actually used by Wireless InSite.
    
    The text files are saved for reference and compatibility with the converter, but
    the XML file is what is actually used in the simulation. The text files are important
    because they are human-readable and can be used with the Wireless InSite UI for
    verification and debugging.
    
    Args:
        osm_folder (str): Path to the OSM folder
        tx_pos (np.ndarray): Transmitter positions
        rx_pos (np.ndarray): Receiver positions
        **rt_params (Any): Ray tracing parameters
        
    Returns:
        str: Path to the insite directory
    """
    insite_path, study_area_path = create_directory_structure(osm_folder, rt_params)
    
    # Create buildings.city & roads.city files
    bldgs_city = convert_to_city_file(osm_folder, insite_path, "buildings", BUILDING_MATERIAL_PATH)
    roads_city = convert_to_city_file(osm_folder, insite_path, "roads", ROAD_MATERIAL_PATH)

    PAD = 30
    xmin_pad, ymin_pad, xmax_pad, ymax_pad = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
        rt_params['origin_lat'], rt_params['origin_lon'], pad=PAD
    ) # pad makes the box larger

    # Create terrain file (.ter)
    terrain_editor = TerrainEditor()
    terrain_editor.set_vertex(xmin=xmin_pad, ymin=ymin_pad, xmax=xmax_pad, ymax=ymax_pad)
    terrain_editor.set_material(TERRAIN_MATERIAL_PATH)
    terrain_editor.save(os.path.join(insite_path, TERRAIN_TEMPLATE))

    # Configure Tx/Rx (.txrx)
    txrx_editor = TxRxEditor()

    #   TX (BS)
    for b_idx, pos in enumerate(tx_pos):
        txrx_editor.add_txrx(
            txrx_type="points",
            is_transmitter=True,
            is_receiver=True,
            pos=pos,
            name=f"BS{b_idx+1}",
            conform_to_terrain=True)

    #   RX (UEs)
    if False:
        txrx_editor.add_txrx(
                txrx_type="points",
                is_transmitter=False,
                is_receiver=True,
                pos=rx_pos,
                name="user_grid",
                conform_to_terrain=rt_params['conform_to_terrain'])
    grid_side = [xmax_pad - xmin_pad - 2 * PAD + rt_params['grid_spacing'], 
                 ymax_pad - ymin_pad - 2 * PAD + rt_params['grid_spacing']]
    txrx_editor.add_txrx(
        txrx_type="grid",
        is_transmitter=False,
        is_receiver=True,
        pos=[xmin_pad + PAD + 1e-3, ymin_pad + PAD, rt_params['ue_height']],
        name="UE_grid",
        grid_side=grid_side,
        grid_spacing=rt_params['grid_spacing'],
        conform_to_terrain=rt_params['conform_to_terrain']
    )
    txrx_editor.save(os.path.join(insite_path, "insite.txrx"))

    # Get ray tracing parameter names from the dataclass
    rt_param_names = {field.name for field in fields(RayTracingParam)}
    rt_params_filtered = {k: v for k, v in rt_params.items() if k in rt_param_names}

    # Define study area bbox in Cartesian coordinates
    study_area_vertex = np.array([[xmin_pad, ymin_pad, 0],
                                  [xmax_pad, ymin_pad, 0],
                                  [xmax_pad, ymax_pad, 0],
                                  [xmin_pad, ymax_pad, 0]])

    # Create setup file (.setup)
    scenario = SetupEditor(insite_path)
    scenario.set_carrierFreq(rt_params['carrier_freq'])
    scenario.set_bandwidth(rt_params['bandwidth'])
    scenario.set_study_area(zmin=-3, zmax=20, all_vertex=study_area_vertex)
    scenario.set_ray_tracing_param(rt_params_filtered)
    scenario.set_txrx("insite.txrx")
    scenario.add_feature(TERRAIN_TEMPLATE, "terrain")
    scenario.add_feature(bldgs_city, "city")
    scenario.add_feature(roads_city, "road")
    scenario.save("insite") # insite.setup

    # Generate XML file (.xml) - What Wireless InSite executable actually uses
    xml_generator = XmlGenerator(insite_path, scenario, txrx_editor, version=int(WI_VERSION[0]))
    xml_generator.update()
    xml_path = os.path.join(insite_path, "insite.study_area.xml")
    xml_generator.save(xml_path)

    license_info = ["-set_licenses", WI_LIC] if WI_VERSION.startswith("4") else []
    
    # Run Wireless InSite using the XML file
    command = [WI_EXE, "-f", xml_path, "-out", study_area_path, "-p", "insite"] + license_info
    print('running command: ', ' '.join(command))
    subprocess.run(command, check=True)
    
    return insite_path
