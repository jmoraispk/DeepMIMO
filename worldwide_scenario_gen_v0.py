#%% MANUAL COORDINATES: IF YOU DO NOT WANT TO USE THE WORLDCITIES.CSV, INCLUDE 
# YOUR COORDINATES HERE
import pandas as pd

# Provided coordinates
# coordinates = [
#     (39.992018, -75.170006),
#     (40.047111, -75.098202),
#     (40.080172, -75.068675),
#     (29.775077, -95.381668),
#     (29.740068, -95.409171),
#     (37.437555, -122.175751),
#     (41.486478, -81.739122),
#     (41.481435, -81.711297),
#     (45.566746, -122.711675),
#     (45.578029, -122.616719),
#     (33.421865, -111.930211),
#     (33.444921, -112.069778),
#     (48.857356, 2.291226),
#     (50.918719, 6.952315),
#     (51.511699, -0.147493),
#     (53.405388, -2.932635),
#     (41.406719, 2.147751),
#     (52.357382, 4.927952),
#     (52.358719, 4.860124),
#     (64.131628, -21.913133),
#     (63.931308, -21.001432),
#     (59.928143, 10.728233),
#     (59.941220, 10.855732),
#     (60.172059, 24.937645),
#     (55.661318, 12.540542),
#     (35.701042, 139.747607),
#     (35.659366, 139.700669),
#     (37.556109, 126.967621),
#     (25.065068, 121.478012),
#     (31.226975, 121.476557),
#     (39.907881, 116.405381),
#     (36.058138, -91.906120),
#     (-33.927865, 18.459795),
#     (-33.910097, 18.603038),
#     (33.576772, -7.625860)
# ]

coordinates = [
    ((41.404332, 2.174500))
]

# Create a DataFrame
df_cities = pd.DataFrame(coordinates, columns=['lat', 'lng'])

# Add a placeholder population column (optional, to mimic worldcities.csv)
df_cities['population'] = 10000000

# Save to CSV
df_cities.to_csv('custom_cities.csv', index=False)
print(f"Generated 'custom_cities.csv' with {len(coordinates)} coordinates.")
#%% EXTRACT n BOUNDING BOXES FROM THE WORLDCITIES.CSV FILE AND SAVE THEM IN
# A INSITE PIPELINE-COMPATIBLE FORMAT IN ANOTHER CSV FILE
import pandas as pd
import random

random.seed(42)

# Load city coordinates from worldcities.csv
try:
    cities = pd.read_csv("custom_cities.csv") # worldcities
    # Filter for urban areas (population > 100,000)
    urban_cities = cities[cities['population'] > 100000]
    city_coords = urban_cities[['lat', 'lng']].values
except FileNotFoundError:
    raise FileNotFoundError("Please download 'worldcities.csv' from https://simplemaps.com/data/world-cities and place it in the same directory.")

def generate_bounding_boxes(n, city_coords):
    """
    Generate n small bounding boxes in urban areas with a base station in each.
    
    Args:
        n (int): Number of bounding boxes to generate.
        city_coords (numpy.ndarray): Array of [lat, lon] pairs for urban cities.
    
    Returns:
        list: List of dictionaries containing minlat, minlon, maxlat, maxlon, and bs.
    """
    bounding_boxes = []
    
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
    
    return bounding_boxes

# Generate 400 bounding boxes (adjust n as needed)
n = len(city_coords)
bounding_boxes = generate_bounding_boxes(n, city_coords)

# Convert to DataFrame and round to 6 decimal places
df = pd.DataFrame(bounding_boxes)
df[['minlat', 'minlon', 'maxlat', 'maxlon']] = df[['minlat', 'minlon', 'maxlat', 'maxlon']].round(6)

# Save to CSV
df.to_csv("bounding_boxes.csv", index=False)
print(f"Generated {n} bounding boxes and saved to 'bounding_boxes.csv'.")
#%% NOW IT'S TIME TO DO RAY-TRACING USING THE PIPELINE
import subprocess

# Define the command as a list of arguments
command = [
    "python", 
    "scenario_generator.py", 
    "--ue_height", "1.5", 
    "--bs_height", "15", 
    "--max_reflections", "4", 
    "--max_diffractions", "1", 
    "--grid_spacing", "1", 
    "--csv", "bounding_boxes.csv"
]

# Execute the command
try:
    subprocess.run(command, check=True)  # check=True raises an error if the command fails
    print("Command executed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")
#%% AFTER DOING RAY-TRACING, THIS SCRIPT GOES INTO EVERY SCENARIO'S FOLDER 
# AND GENERATES A DICTIONARY OF #ROWS AND #USERS PER ROW, COMPATIBLE WITH 
# THE LWM-BASED DEEPMIMO DATA GENERATION. THE NAME OF SCENARIOS ARE BASED ON
# THEIR BOUNDING BOX COORDINATES
import os

# Define the path to the osm_exports folder
osm_exports_path = 'osm_exports'

# Get a list of all scenario folders in osm_exports
scenario_folders = [
    folder for folder in os.listdir(osm_exports_path)
    if os.path.isdir(os.path.join(osm_exports_path, folder))
]

# Initialize the dictionary to store the results
row_column_users = {}

# Process each scenario folder
for scenario in scenario_folders:
    # Construct the full path to the scenario folder
    scenario_path = os.path.join(osm_exports_path, scenario)
    
    # Find the subfolder starting with 'insite'
    insite_folders = [
        folder for folder in os.listdir(scenario_path)
        if folder.startswith('insite') and os.path.isdir(os.path.join(scenario_path, folder))
    ]
    
    # Proceed if an insite folder is found
    if insite_folders:
        insite_folder = insite_folders[0]  # Take the first matching folder
        # Construct the path to parameters.txt
        parameters_path = os.path.join(scenario_path, insite_folder, 'parameters.txt')
        
        # Read the parameters.txt file
        with open(parameters_path, 'r') as file:
            for line in file:
                # Split the line into key and value at the colon
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    # Extract n_rows
                    if key == 'n_rows':
                        n_rows = int(value)
                    # Extract user_per_row and map to n_per_row
                    elif key == 'user_per_row':
                        n_per_row = int(value)
        
        # Store the extracted values in the dictionary
        row_column_users[scenario] = {
            'n_rows': n_rows,
            'n_per_row': n_per_row
        }

# Optional: Verify the number of scenarios processed
print(f"Total scenarios processed: {len(row_column_users)}")
#%% THIS SCRIPT GOES INTO THE RAY-TRACED SCENARIOS' FOLDERS AND COPIES THE 
# DEEPMIMO VERSION OF THAT SCENARIO INTO A FOLDER IN THE ROOT DIR, CALLED
# "SCENARIOS" IN ANOTHER FOLDER WITH THE SCENARIO NAME. THE PARAMETERS.TXT
# FILE CAN ALSO BE FOUND IN EVERY FINAL FOLDER
import os
import shutil

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
for scenario in scenario_folders:
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

print("File copying completed.")
#%% THIS SCRIPT GOES INTO EVERY SCENARIO'S FOLDER AND GENERATES A DICTIONARY
# OF #ROWS AND #USERS PER ROW, COMPATIBLE WITH THE LWM-BASED DEEPMIMO DATA 
# GENERATION. THE NAME OF SCENARIOS ARE BASED ON THEIR BOUNDING BOX COORDINATES
import os

# Define the path to the scenarios folder
scenarios_path = 'scenarios'

# Get a list of all scenario folders in scenarios
scenario_folders = [
    folder for folder in os.listdir(scenarios_path)
    if os.path.isdir(os.path.join(scenarios_path, folder))
]

# Initialize the dictionary to store the results
row_column_users = {}

# Process each scenario folder
for scenario in scenario_folders:
    # Construct the full path to the parameters.txt file
    parameters_path = os.path.join(scenarios_path, scenario, 'parameters.txt')
    
    # Check if parameters.txt exists
    if os.path.exists(parameters_path):
        # Read the parameters.txt file
        with open(parameters_path, 'r') as file:
            for line in file:
                # Split the line into key and value at the colon
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    # Extract n_rows
                    if key == 'n_rows':
                        n_rows = int(value)
                    # Extract user_per_row and map to n_per_row
                    elif key == 'user_per_row':
                        n_per_row = int(value)
        
        # Store the extracted values in the dictionary
        row_column_users[scenario] = {
            'n_rows': n_rows,
            'n_per_row': n_per_row
        }
    else:
        print(f"Warning: 'parameters.txt' not found in {scenario}")

# Optional: Verify the number of scenarios processed
print(f"Total scenarios processed: {len(row_column_users)}")
#%% THIS SCRIPTS GIVES AN ARRAY OF VALID SCENARIO NAMES IN CASE YOU WANT TO 
# CALL THEM IN YOUR CODE
import os
import numpy as np

# Define the path to the scenarios folder
scenarios_path = 'scenarios'

# Initialize an empty list to store scenario names
scenario_names_list = []

# Check if the scenarios folder exists
if os.path.exists(scenarios_path):
    # Iterate over each entry in the scenarios folder
    for folder in os.listdir(scenarios_path):
        folder_path = os.path.join(scenarios_path, folder)
        # Check if the entry is a directory
        if os.path.isdir(folder_path):
            parameters_file = os.path.join(folder_path, 'parameters.txt')
            # Check if parameters.txt exists in the directory
            if os.path.exists(parameters_file):
                scenario_names_list.append(folder)
else:
    print(f"Warning: The '{scenarios_path}' folder does not exist.")

# Convert the list to a NumPy array
scenario_names = np.array(scenario_names_list)

# Optional: Print the number of scenarios found for verification
print(f"Total scenarios with parameters.txt: {len(scenario_names)}")
