import os

def create_scenario(path_to_rt_folder):
    print('here3!')
    
    # Example logic to determine generator type based on files found
    files_in_dir = os.listdir('.')
    if 'aodt' in files_in_dir:
        print("Using AODT generator")
        # ...
    elif '.path' in files_in_dir:
        print("Using Sionna_RT generator")
        # ...
    elif '.p2m' in files_in_dir:
        print("Using Wireless Insite generator")
        # ...
    else:
        print("Unknown raytracer type")
        # ...
