import os

def create_scenario(path_to_scenario):
    # Example logic to determine generator type based on files found
    if 'aodt' in os.listdir(path_to_scenario):
        print("Using AODT generator")
        # You can add more complex logic here, e.g., importing submodules
    elif 'sionna_rt' in os.listdir(path_to_scenario):
        print("Using Sionna_RT generator")
    elif 'wireless_insite' in os.listdir(path_to_scenario):
        print("Using Wireless Insite generator")
    else:
        print("Unknown scenario type")
