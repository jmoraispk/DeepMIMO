import subprocess
import argparse
import pandas as pd

def run_command(command, description):
    """Run a shell command and stream output in real-time."""
    print(f"\n🚀 Starting: {description}...\n")
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")

def process_single_scenario(args):
    """Process a single scenario."""
    # Step 1: Run OSM Extraction
    osm_command = [
        "python", "run_osm_extraction.py",
        "--minlat", str(args.minlat), "--minlon", str(args.minlon),
        "--maxlat", str(args.maxlat), "--maxlon", str(args.maxlon)
    ]
    run_command(osm_command, "OSM Extraction")

    # Step 2: Run Ray-Tracing
    raytrace_command = [
        "python", "raytrace_insite.py",
        "--minlat", str(args.minlat), "--minlon", str(args.minlon),
        "--maxlat", str(args.maxlat), "--maxlon", str(args.maxlon),
        "--bs", args.bs,
        "--carrier_frequency", str(args.carrier_frequency),
        "--bandwidth", str(args.bandwidth),
        "--max_paths", str(args.max_paths),
        "--ray_spacing", str(args.ray_spacing),
        "--max_reflections", str(args.max_reflections),
        "--max_transmissions", str(args.max_transmissions),
        "--max_diffractions", str(args.max_diffractions),
        "--ds_max_reflections", str(args.ds_max_reflections),
        "--ds_max_diffractions", str(args.ds_max_diffractions),
        "--ds_max_transmissions", str(args.ds_max_transmissions),
    ]
    
    if args.ds_enable:
        raytrace_command.append("--ds_enable")
    if args.ds_final_interaction_only:
        raytrace_command.append("--ds_final_interaction_only")

    run_command(raytrace_command, "Ray-Tracing Simulation")

def process_csv_scenarios(csv_path, args):
    """Process multiple scenarios from a CSV file."""
    scenarios = pd.read_csv(csv_path)

    for index, scenario in scenarios.iterrows():
        print(f"\n📍 Processing Scenario {index + 1}...")
        # Extract parameters for each scenario
        minlat = scenario["minlat"]
        minlon = scenario["minlon"]
        maxlat = scenario["maxlat"]
        maxlon = scenario["maxlon"]
        bs = scenario["bs"]

        # Run OSM Extraction
        osm_command = [
            "python", "run_osm_extraction.py",
            "--minlat", str(minlat), "--minlon", str(minlon),
            "--maxlat", str(maxlat), "--maxlon", str(maxlon)
        ]
        run_command(osm_command, f"OSM Extraction for Scenario {index + 1}")

        # Run Ray-Tracing
        raytrace_command = [
            "python", "raytrace_insite.py",
            "--minlat", str(minlat), "--minlon", str(minlon),
            "--maxlat", str(maxlat), "--maxlon", str(maxlon),
            "--bs", bs,
            "--carrier_frequency", str(args.carrier_frequency),
            "--bandwidth", str(args.bandwidth),
            "--ue_height", str(args.ue_height),
            "--bs_height", str(args.bs_height),
            "--grid_spacing", str(args.grid_spacing),
            "--max_paths", str(args.max_paths),
            "--ray_spacing", str(args.ray_spacing),
            "--max_reflections", str(args.max_reflections),
            "--max_transmissions", str(args.max_transmissions),
            "--max_diffractions", str(args.max_diffractions),
            "--ds_max_reflections", str(args.ds_max_reflections),
            "--ds_max_diffractions", str(args.ds_max_diffractions),
            "--ds_max_transmissions", str(args.ds_max_transmissions),
        ]
        
        if args.ds_enable:
            raytrace_command.append("--ds_enable")
        if args.ds_final_interaction_only:
            raytrace_command.append("--ds_final_interaction_only")

        run_command(raytrace_command, f"Ray-Tracing Simulation for Scenario {index + 1}")

def main():
    parser = argparse.ArgumentParser(description="Run OSM Extraction and Ray-Tracing Sequentially")
    parser.add_argument("--csv", type=str, help="CSV file containing multiple scenarios")
    parser.add_argument("--minlat", type=float, help="Minimum latitude for a single scenario")
    parser.add_argument("--minlon", type=float, help="Minimum longitude for a single scenario")
    parser.add_argument("--maxlat", type=float, help="Maximum latitude for a single scenario")
    parser.add_argument("--maxlon", type=float, help="Maximum longitude for a single scenario")
    parser.add_argument("--bs", type=str, help="Base station coordinates for a single scenario")

    # Carrier frequency and bandwidth
    parser.add_argument("--carrier_frequency", type=float, default=3.5e9, help="Carrier frequency in Hz")
    parser.add_argument("--bandwidth", type=float, default=10e6, help="Bandwidth in Hz")
    
    parser.add_argument("--ue_height", type=float, default=2)
    parser.add_argument("--bs_height", type=float, default=6)
    parser.add_argument("--grid_spacing", type=float, default=2.5)
    
    # Ray-tracing parameters
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

    # Check if processing a CSV file or a single scenario
    if args.csv:
        process_csv_scenarios(args.csv, args)
    elif all([args.minlat, args.minlon, args.maxlat, args.maxlon, args.bs]):
        process_single_scenario(args)
    else:
        print("❌ Invalid input. Provide either a CSV file (--csv) or bounding box and base station parameters for a single scenario.")
        exit(1)

if __name__ == "__main__":
    main()
