"""
Run Blender OSM to PLY Converter (Cross-Platform)
Supports:
  - MacOS & Windows (Automatically detects OS)
  - Single scenario (User inputs bounding box coordinates)
  - Multiple scenarios (User provides a CSV file)
"""

import subprocess
import logging
import argparse
import os
import platform
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Detect Operating System
OS_TYPE = platform.system()

# Set Blender Path Based on OS
if OS_TYPE == "Darwin":  # MacOS
    DEFAULT_BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
elif OS_TYPE == "Windows":  # Windows
    DEFAULT_BLENDER_PATH = "C:\\Users\\salikha4\\Desktop\\blender-4.3.2-windows-x64\\blender.exe" 
else:
    logger.error("❌ Unsupported operating system. This script only works on MacOS and Windows.")
    sys.exit(1)

# Ask for Blender path if not found
if not os.path.exists(DEFAULT_BLENDER_PATH):
    logger.warning(f"⚠ Blender not found at {DEFAULT_BLENDER_PATH}. Please enter the correct path:")
    DEFAULT_BLENDER_PATH = input("Enter Blender executable path: ").strip()
    if not os.path.exists(DEFAULT_BLENDER_PATH):
        logger.error(f"❌ Blender not found at {DEFAULT_BLENDER_PATH}. Exiting.")
        sys.exit(1)

# Path to the Blender script
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "blender_osm_export.py")

# Ensure the Blender script exists
if not os.path.exists(SCRIPT_PATH):
    logger.error(f"❌ Blender script not found: {SCRIPT_PATH}")
    sys.exit(1)

# Define DeepMIMO Root Path
if OS_TYPE == "Darwin":  # MacOS
    DEEPMIMO_ROOT = "/Users/sadjadalikhani/Desktop/deepmimo/scenario_generator"
elif OS_TYPE == "Windows":  # Windows
    DEEPMIMO_ROOT = "C:\\Users\\salikha4\\Desktop\\scenario_generator"

# Argument parser for bounding box or CSV input
parser = argparse.ArgumentParser(description="Run Blender OSM to PLY converter for multiple scenarios.")
parser.add_argument("--csv", type=str, help="Path to CSV file containing multiple scenarios")
parser.add_argument("--minlat", type=float, help="Minimum latitude")
parser.add_argument("--minlon", type=float, help="Minimum longitude")
parser.add_argument("--maxlat", type=float, help="Maximum latitude")
parser.add_argument("--maxlon", type=float, help="Maximum longitude")

args = parser.parse_args()

# If CSV is provided, process multiple scenarios
if args.csv:
    logger.info(f"📂 Processing multiple scenarios from CSV: {args.csv}")
    
    import pandas as pd
    try:
        scenarios = pd.read_csv(args.csv)
    except Exception as e:
        logger.error(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    for index, scenario in scenarios.iterrows():
        minlat, minlon, maxlat, maxlon = scenario["minlat"], scenario["minlon"], scenario["maxlat"], scenario["maxlon"]

        # Generate bbox folder name
        bbox_folder = f"bbox_{minlat}_{minlon}_{maxlat}_{maxlon}".replace(".", "-")
        scenario_path = os.path.join(DEEPMIMO_ROOT, "osm_exports", bbox_folder)

        # **Check if the folder already exists**
        if os.path.exists(scenario_path):
            logger.info(f"⏩ Folder '{bbox_folder}' already exists. Skipping scenario.")
            continue  # Skip processing this scenario

        # Build command to run Blender
        command = [DEFAULT_BLENDER_PATH, "--background", "--python", SCRIPT_PATH, "--", "--minlat", str(minlat),
                   "--minlon", str(minlon), "--maxlat", str(maxlat), "--maxlon", str(maxlon)]

        # Run Blender and process scenario
        try:
            logger.info(f"🚀 Running Blender for scenario {bbox_folder}...")
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            
            logger.info(process.stdout)
            if process.stderr:
                logger.error(f"\n{process.stderr}")  # ❌ Blender Error:
            
            logger.info("✅ Blender execution completed successfully for this scenario.")

        except Exception as e:
            logger.error(f"❌ Error running Blender for scenario {bbox_folder}: {e}")

# **Single Scenario Processing**
elif all([args.minlat, args.minlon, args.maxlat, args.maxlon]):
    minlat, minlon, maxlat, maxlon = args.minlat, args.minlon, args.maxlat, args.maxlon

    # Generate bbox folder name
    bbox_folder = f"bbox_{minlat}_{minlon}_{maxlat}_{maxlon}".replace(".", "-")
    scenario_path = os.path.join(DEEPMIMO_ROOT, "osm_exports", bbox_folder)

    # **Check if the folder already exists**
    if os.path.exists(scenario_path):
        logger.info(f"⏩ Folder '{bbox_folder}' already exists. Skipping scenario.")
        sys.exit(0)  # Exit without processing

    # Build command to run Blender
    command = [DEFAULT_BLENDER_PATH, "--background", "--python", SCRIPT_PATH, "--", "--minlat", str(minlat),
               "--minlon", str(minlon), "--maxlat", str(maxlat), "--maxlon", str(maxlon)]

    # Run Blender and process scenario
    try:
        logger.info(f"🚀 Running Blender for scenario {bbox_folder}...")
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        
        logger.info(process.stdout)
        if process.stderr:
            logger.error(f"\n{process.stderr}")  # ❌ Blender Error:
        
        logger.info("✅ Blender execution completed successfully for this scenario.")

    except Exception as e:
        logger.error(f"❌ Error running Blender for scenario {bbox_folder}: {e}")

else:
    logger.error("❌ Invalid input. Provide either a CSV file (--csv) or explicit coordinates (--minlat, --minlon, --maxlat, --maxlon).")
    sys.exit(1)
