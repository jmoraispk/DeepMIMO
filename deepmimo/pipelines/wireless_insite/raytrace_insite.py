import numpy as np
import os
import subprocess
import argparse
import scipy.io
from pathlib import Path
import pandas as pd 

# Import existing modules
from WI_interface.XmlGenerator import XmlGenerator
from WI_interface.SetupEditor import SetupEditor
from WI_interface.TxRxEditor import TxRxEditor
from WI_interface.TerrainEditor import TerrainEditor
from generate_city.generate_city import generate_city
from DeepMIMO_p2m_converter.ChannelDataLoader import WIChannelConverter
from DeepMIMO_p2m_converter.ChannelDataFormatter import DeepMIMODataFormatter
from utils.convert_GpsBBox2CartesianBBox import convert_GpsBBox2CartesianBBox, convert_Gps2RelativeCartesian

def create_directory_structure(base_path, args):
    """Create necessary directories for the scenario with a professional folder name based on parameters."""
    
    # Format folder name with key parameters
    folder_name = (f"insite_{args.carrier_frequency/1e9:.1f}GHz_{args.bandwidth/1e6:.0f}MHz_"
                   f"{args.max_paths}paths_{args.max_reflections}ref_{args.max_transmissions}trans_{args.max_diffractions}diff")
    
    insite_path = base_path / folder_name
    intermediate_path = insite_path / "intermediate_files"
    mat_path = insite_path / "study_area_mat"
    study_area_path = insite_path / "study_area"

    # Create directories
    for path in [insite_path, intermediate_path, mat_path, study_area_path]:
        path.mkdir(parents=True, exist_ok=True)

    return insite_path, intermediate_path, mat_path, study_area_path

def parse_bs_positions(bs_str):
    """Parse base station positions from string format"""
    bs_positions = []
    for pair in bs_str.split(';'):
        lat, lon = map(float, pair.strip().split(','))
        bs_positions.append({'lat': lat, 'lon': lon})
    return bs_positions

def get_grid_info(xmin, ymin, xmax, ymax, grid_spacing):
    """Calculate the grid layout and extract available rows and users per row."""
    # Create grid
    x_coords = np.arange(xmin, xmax + grid_spacing, grid_spacing)
    y_coords = np.arange(ymin, ymax + grid_spacing, grid_spacing)
    # Indices of rows and number of users per row
    row_indices = np.arange(len(y_coords) - 1)
    users_per_row = len(x_coords) - 1  # Each row has the same number of users
    return row_indices, users_per_row

def run_raytracing_simulation(args):
    # Set up base paths
    root_dir = Path("C:/Users/namhyunk/Desktop/osm2dt")
    bbox_folder = f"bbox_{args.minlat}_{args.minlon}_{args.maxlat}_{args.maxlon}".replace('.', '-')
    osm_export_path = root_dir / "osm_exports" / bbox_folder
    insite_path, intermediate_path, mat_path, study_area_path = create_directory_structure(osm_export_path, args)

    # Load GPS origin
    origin_file = osm_export_path / "osm_gps_origin.txt"
    with open(origin_file) as f:
        origin_lat = float(f.readline().strip())
        origin_lon = float(f.readline().strip())

    # Parse base station positions
    print(args.bs)
    bs_gps_pos = parse_bs_positions(args.bs)

    # Generate city features
    city_feature_list = generate_city(
        str(osm_export_path) + os.sep,  # Add trailing separator explicitly
        str(insite_path) + os.sep,
        minlat=args.minlat,
        minlon=args.minlon,
        maxlat=args.maxlat,
        maxlon=args.maxlon,
        building_mtl_path=str(root_dir / "resource/material/ITU Concrete 3.5 GHz.mtl"),
        road_mtl_path=str(root_dir / "resource/material/Asphalt_1GHz.mtl"),
    )

    # Coordinate conversions
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        args.minlat, args.minlon, args.maxlat, args.maxlon,
        origin_lat, origin_lon, pad=0
    )
    xmin_pad, ymin_pad, xmax_pad, ymax_pad = convert_GpsBBox2CartesianBBox(
        args.minlat, args.minlon, args.maxlat, args.maxlon,
        origin_lat, origin_lon, pad=30
    )

    # Process BS positions
    bs_pos_xyz = []
    for bs in bs_gps_pos:
        bs_x, bs_y = convert_Gps2RelativeCartesian(bs['lat'], bs['lon'], origin_lat, origin_lon)
        bs_pos_xyz.append([bs_x, bs_y, args.bs_height]) 

    # Create terrain
    terrain_editor = TerrainEditor()
    terrain_editor.set_vertex(xmin=xmin_pad, ymin=ymin_pad, xmax=xmax_pad, ymax=ymax_pad)
    terrain_editor.set_material(str(root_dir / "resource/material/ITU Wet earth 3.5 GHz.mtl"))
    terrain_editor.save(str(insite_path / "newTerrain.ter"))

    # Configure Tx/Rx
    txrx_editor = TxRxEditor()
    for b_idx, pos in enumerate(bs_pos_xyz):
        txrx_editor.add_txrx(
            txrx_type="points",
            is_transmitter=True,
            is_receiver=True,
            pos=pos,
            name=f"BS{b_idx+1}"
        )

    grid_side = [xmax - xmin, ymax - ymin]
    grid_spacing = args.grid_spacing 
    txrx_editor.add_txrx(
        txrx_type="grid",
        is_transmitter=False,
        is_receiver=True,
        pos=[xmin, ymin, args.ue_height],
        name="UE_grid",
        grid_side=grid_side,
        grid_spacing=grid_spacing
    )
    txrx_editor.save(str(insite_path / "insite.txrx"))

    # Calculate grid info
    row_indices, users_per_row = get_grid_info(xmin, ymin, xmax, ymax, grid_spacing)
    
    # Create setup file
    scenario = SetupEditor(str(insite_path))
    scenario.set_carrierFreq_and_bandwidth(carrier_frequency=args.carrier_frequency, bandwidth=args.bandwidth)
    scenario.set_study_area(
        zmin=-3,
        zmax=17.5,
        all_vertex=np.array([
            [xmin_pad, ymin_pad, 0],
            [xmax_pad, ymin_pad, 0],
            [xmax_pad, ymax_pad, 0],
            [xmin_pad, ymax_pad, 0]
        ])
    )
    scenario.set_ray_tracing_param(
        args.max_paths,
        args.ray_spacing,
        args.max_reflections,
        args.max_transmissions,
        args.max_diffractions,
        args.ds_enable,
        args.ds_max_reflections,
        args.ds_max_transmissions,
        args.ds_max_diffractions,
        args.ds_final_interaction_only
    )
    scenario.set_txrx("/insite.txrx")
    scenario.add_feature("newTerrain.ter", "terrain")
    for city_feature in city_feature_list:
        scenario.add_feature(city_feature, "city")
    scenario.save("/insite") # insite

    # Generate XML and run simulation
    xml_generator = XmlGenerator(str(insite_path), "\\insite.setup") # insite.setup
    xml_generator.update()
    xml_path = insite_path / "insite.study_area.xml"
    xml_generator.save(str(xml_path))

    # Run Wireless InSite
    wi_path = "C:\\Program Files\\Remcom\\Wireless InSite 3.3.0.4\\bin\\calc\\wibatch.exe"
    subprocess.run([
        wi_path,
        "-f", str(xml_path),
        "-out", str(study_area_path),
        "-p", "insite"
    ], check=True)

    # Convert P2M to MAT format
    WIChannelConverter(str(study_area_path), str(intermediate_path))

    DeepMIMODataFormatter(
        str(intermediate_path),
        str(mat_path),
        max_channels=30000,
        TX_order=range(1, len(bs_pos_xyz) + 1),
        RX_order=[len(bs_pos_xyz) + 1],
    )

    # Save parameters
    scipy.io.savemat(
        str(mat_path / "params.mat"),
        {
            "version": 2,
            "carrier_freq": args.carrier_frequency,
            "transmit_power": 0.0,
            "user_grids": np.array([[1, int(grid_side[1] // grid_spacing + 1), int(grid_side[0] // grid_spacing + 1)]], dtype=float),
            "num_BS": len(bs_pos_xyz),
            "dual_polar_available": 0,
            "doppler_available": 0
        }
    )
    
    # Save arguments and grid details to a text file
    param_file = insite_path / "parameters.txt"
    with open(param_file, "w") as f:
        # Save all input arguments
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"n_rows: {len(row_indices)}\n")
        f.write(f"user_per_row: {users_per_row}\n")
        f.write(f"n_users: {len(row_indices) * users_per_row}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wireless InSite ray tracing simulation")
    parser.add_argument("--csv", type=str, help="CSV file containing multiple scenarios")
    parser.add_argument("--minlat", type=float, help="Minimum latitude for a single scenario")
    parser.add_argument("--minlon", type=float)
    parser.add_argument("--maxlat", type=float)
    parser.add_argument("--maxlon", type=float)
    parser.add_argument("--bs", type=str)

    parser.add_argument("--carrier_frequency", type=float, default=3.5e9)
    parser.add_argument("--bandwidth", type=float, default=10e6)
    parser.add_argument("--ue_height", type=float, default=2)
    parser.add_argument("--bs_height", type=float, default=6)
    parser.add_argument("--grid_spacing", type=float, default=2.5)
    parser.add_argument("--max_paths", type=int, default=25)
    parser.add_argument("--ray_spacing", type=float, default=0.25)
    parser.add_argument("--max_reflections", type=int, default=3)
    parser.add_argument("--max_transmissions", type=int, default=0)
    parser.add_argument("--max_diffractions", type=int, default=0)
    parser.add_argument("--ds_enable", action="store_true")
    parser.add_argument("--ds_max_reflections", type=int, default=2)
    parser.add_argument("--ds_max_diffractions", type=int, default=0)
    parser.add_argument("--ds_max_transmissions", type=int, default=0)
    parser.add_argument("--ds_final_interaction_only", action="store_true", default=True)

    args = parser.parse_args()

    if args.csv:
        scenarios = pd.read_csv(args.csv)
        for _, row in scenarios.iterrows():
            args.minlat, args.minlon, args.maxlat, args.maxlon, args.bs = row["minlat"], row["minlon"], row["maxlat"], row["maxlon"], row["bs"]
            run_raytracing_simulation(args)
    else:
        run_raytracing_simulation(args)