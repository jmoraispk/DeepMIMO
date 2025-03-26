# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:37:06 2025

@author: salikha4
"""

import pandas as pd
import subprocess
import random
import shutil
import requests
import os
import json
from input_preprocess import DeepMIMO_data_gen, create_labels

# -----------------------------------------------------------------------------
# 1. MAKE SURE YOUR "Geocoding API" AND "Maps Static API" ARE ENABLED.
# 2. DON'T FORGET TO ASSIGN YOUR GOOGLE API KEY TO THE "API_KEY" VARIABLE.
# -----------------------------------------------------------------------------

def generate_bounding_boxes(city_coords, API_KEY, n=200):
    """
    Generate n small bounding boxes in urban areas with a base station in each.
    
    Args:
        n (int): Number of bounding boxes to generate.
        city_coords (numpy.ndarray): Array of [lat, lon] pairs for urban cities.
    
    Returns:
        list: List of dictionaries containing minlat, minlon, maxlat, maxlon, and bs.
    """
    bounding_boxes = []
    city_names = []
    
    for _ in range(n):
        # Select a random city
        city_lat, city_lon = random.choice(city_coords)
        
        # Offset the center from the city center by up to 0.02° (~2.2 km)
        offset_lat = random.uniform(0, 0) # -0.02, 0.02
        offset_lon = random.uniform(0, 0) # -0.02, 0.02
        center_lat = city_lat + offset_lat
        center_lon = city_lon + offset_lon
        
        # Define small bounding box size (differences < 0.005°)
        lat_diff = .003 # random.uniform(0.001, 0.002) 
        lon_diff = .003 # random.uniform(0.001, 0.002)  
        minlat = center_lat - lat_diff / 2
        maxlat = center_lat + lat_diff / 2
        minlon = center_lon - lon_diff / 2
        maxlon = center_lon + lon_diff / 2
        
        # # Place base station randomly within the bounding box
        # bs_lat = random.uniform(minlat, maxlat)
        # bs_lon = random.uniform(minlon, maxlon)
        
        # Use the city coordinates as the base station location
        bs_lat = city_lat
        bs_lon = city_lon
        bs = f"{bs_lat:.6f}, {bs_lon:.6f}"
        
        # Store the bounding box and base station
        bounding_boxes.append({
            'minlat': minlat,
            'minlon': minlon,
            'maxlat': maxlat,
            'maxlon': maxlon,
            'bs': bs
        })
        
        city_name = fetch_satellite_view(minlat, minlon, maxlat, maxlon, API_KEY)
        city_names.append(city_name)
        
    return bounding_boxes, city_names

def get_city_from_coordinates(lat, lon, api_key):
    """Fetch the city name from coordinates using Google Maps Geocoding API."""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            # Look for the city in the address components
            for result in data["results"]:
                for component in result["address_components"]:
                    if "locality" in component["types"]:  # 'locality' typically means city
                        return component["long_name"]
            return "unknown"  # Fallback if no city is found
        else:
            print(f"Geocoding error: {data['status']}")
            return "unknown"
    else:
        print(f"Geocoding request failed: {response.status_code}")
        return "unknown"

def fetch_satellite_view(minlat, minlon, maxlat, maxlon, API_KEY):
    # Create a unique folder name based on coordinates (replace dots with dashes)
    bbox_folder = f"bbox_{minlat}_{minlon}_{maxlat}_{maxlon}".replace(".", "-")
    
    # Define the save directory
    save_dir = os.path.join("scenarios", bbox_folder)
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate the center of the bounding box
    center_lat = (minlat + maxlat) / 2
    center_lon = (minlon + maxlon) / 2

    # Get the city name from the center coordinates
    city_name = get_city_from_coordinates(center_lat, center_lon, API_KEY)
    # Replace spaces or special characters in city name for safe filename
    city_name = city_name.replace(" ", "_").replace(",", "").lower()

    # Parameters for the Static Maps API
    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": 18,  # Adjust zoom level (higher = more detailed)
        "size": "640x640",  # Image size in pixels (max 640x640 for free tier)
        "maptype": "satellite",  # Options: roadmap, satellite, hybrid, terrain
        "key": API_KEY
    }

    # API endpoint
    url = "https://maps.googleapis.com/maps/api/staticmap"

    # Make the request
    response = requests.get(url, params=params)

    # Save the image in the specified directory with city name
    if response.status_code == 200:
        image_path = os.path.join(save_dir, "satellite_view.png")
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Satellite view saved as '{image_path}'")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    return city_name

def plot_propagation(scenario_name, task='los', n_beams=16):
    labels = []
    data = DeepMIMO_data_gen(scenario_name)
    labels = create_labels(data, task, scenario_name, n_beams=n_beams)

def worldwide_scenario_gen(coordinates, API_KEY):
    # -------------------------------------------------------------------------
    # MANUAL COORDINATES: IF YOU DO NOT WANT TO USE THE WORLDCITIES.CSV, INCLUDE 
    # YOUR COORDINATES HERE
    # -------------------------------------------------------------------------
    
    # Create a DataFrame
    df_cities = pd.DataFrame(coordinates, columns=['lat', 'lng'])
    
    # Add a placeholder population column (optional, to mimic worldcities.csv)
    df_cities['population'] = 10000000
    
    # Save to CSV
    df_cities.to_csv('custom_cities.csv', index=False)
    print(f"Generated 'custom_cities.csv' with {len(coordinates)} coordinates.")
    
    # -------------------------------------------------------------------------
    # EXTRACT n BOUNDING BOXES FROM THE WORLDCITIES.CSV FILE AND SAVE THEM IN
    # A INSITE PIPELINE-COMPATIBLE FORMAT IN ANOTHER CSV FILE
    # -------------------------------------------------------------------------
    
    random.seed(42)

    # Load city coordinates from worldcities.csv
    try:
        cities = pd.read_csv("custom_cities.csv") # worldcities
        # Filter for urban areas (population > 100,000)
        urban_cities = cities[cities['population'] > 100000]
        city_coords = urban_cities[['lat', 'lng']].values
    except FileNotFoundError:
        raise FileNotFoundError("Please download 'worldcities.csv' from https://simplemaps.com/data/world-cities and place it in the same directory.")

    # Generate 400 bounding boxes (adjust n as needed)
    n = len(city_coords)
    bounding_boxes, city_names = generate_bounding_boxes(city_coords, API_KEY, n=n)

    # Convert to DataFrame and round to 6 decimal places
    df = pd.DataFrame(bounding_boxes)
    df[['minlat', 'minlon', 'maxlat', 'maxlon']] = df[['minlat', 'minlon', 'maxlat', 'maxlon']].round(6)

    # Save to CSV
    df.to_csv("bounding_boxes.csv", index=False)
    print(f"Generated {n} bounding boxes and saved to 'bounding_boxes.csv'.")
    
    # -------------------------------------------------------------------------
    # NOW IT'S TIME TO DO RAY-TRACING USING THE PIPELINE
    # -------------------------------------------------------------------------
    
    # Define the command as a list of arguments
    command = [
        "python", 
        "scenario_generator.py", 
        "--ue_height", "1.5", 
        "--bs_height", "15", 
        "--max_reflections", "3", 
        "--max_diffractions", "1", 
        "--grid_spacing", "2", 
        "--csv", "bounding_boxes.csv"
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)  # check=True raises an error if the command fails
        print("Command executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        
    # -------------------------------------------------------------------------  
    # THIS SCRIPT GOES INTO THE RAY-TRACED SCENARIOS' FOLDERS AND COPIES THE 
    # DEEPMIMO VERSION OF THAT SCENARIO INTO A FOLDER IN THE ROOT DIR, CALLED
    # "SCENARIOS" IN ANOTHER FOLDER WITH THE SCENARIO NAME. THE PARAMETERS.TXT
    # FILE CAN ALSO BE FOUND IN EVERY FINAL FOLDER
    # -------------------------------------------------------------------------

    # Define paths relative to the root directory (assumed to be the current working directory)
    root_dir = '.'  # Change this if your root directory is elsewhere
    osm_exports_path = os.path.join(root_dir, 'osm_exports')
    scenarios_path = os.path.join(root_dir, 'scenarios')

    # Create the "scenarios" folder if it doesn’t exist
    if not os.path.exists(scenarios_path):
        os.makedirs(scenarios_path)

    # Get all scenario folders in "osm_exports"
    scenario_folders = [f for f in os.listdir(osm_exports_path) 
                        if os.path.isdir(os.path.join(osm_exports_path, f))]

    # Process each scenario folder
    for scenario_idx, scenario in enumerate(scenario_folders):
        scenario_path = os.path.join(osm_exports_path, scenario)
        scenario_dest_path = os.path.join(scenarios_path, scenario)
        
        # Create the scenario subfolder in "scenarios" if it doesn’t exist
        if not os.path.exists(scenario_dest_path):
            os.makedirs(scenario_dest_path)
        
        # Find the "insite" subfolder within the scenario folder
        insite_folders = [f for f in os.listdir(scenario_path) 
                          if f.startswith('insite') and os.path.isdir(os.path.join(scenario_path, f))]
        
        if insite_folders:
            # Assume there’s only one "insite" subfolder; take the first one
            insite_path = os.path.join(scenario_path, insite_folders[0])
            
            # Copy all files from "study_area_mat" to the scenario’s subfolder in "scenarios"
            study_area_mat_path = os.path.join(insite_path, 'study_area_mat')
            if os.path.exists(study_area_mat_path):
                for file_name in os.listdir(study_area_mat_path):
                    file_path = os.path.join(study_area_mat_path, file_name)
                    if os.path.isfile(file_path):  # Only copy files, not subfolders
                        shutil.copy2(file_path, scenario_dest_path)
            else:
                print(f"Warning: 'study_area_mat' not found in {insite_path}")
            
            # Copy "parameters.txt" to the scenario’s subfolder in "scenarios"
            parameters_path = os.path.join(insite_path, 'parameters.txt')
            if os.path.exists(parameters_path):
                shutil.copy2(parameters_path, scenario_dest_path)
            else:
                print(f"Warning: 'parameters.txt' not found in {insite_path}")
        else:
            print(f"Warning: No 'insite' folder found in {scenario}")
        
        # Rename the scenario folder to the corresponding city name from x
        new_scenario_dest_path = os.path.join(scenarios_path, city_names[scenario_idx])
        if os.path.exists(scenario_dest_path) and scenario_dest_path != new_scenario_dest_path:
            os.rename(scenario_dest_path, new_scenario_dest_path)
            print(f"Renamed '{scenario_dest_path}' to '{new_scenario_dest_path}'")
        
        # plot_propagation(city_names[scenario_idx], task='bp', n_beams=16)
        plot_propagation(city_names[scenario_idx], task='los')
        
    print("File copying completed.")
    
    
if __name__ == "__main__":
    
    root_dir = '.'  # Current working directory
    osm_exports_path = os.path.join(root_dir, 'osm_exports')
    scenarios_path = os.path.join(root_dir, 'scenarios')

    # Remove osm_exports and scenarios folders if they exist
    for folder in [osm_exports_path, scenarios_path]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed folder: {folder}")
        else:
            print(f"Folder not found, skipping: {folder}")
            
    # Provided coordinates
    coordinates = [((45.4347492,	12.3390049))]
    # Google Maps Static API key
    API_KEY = "X"
    worldwide_scenario_gen(coordinates, API_KEY)
