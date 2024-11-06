#%%

# The directives below are for ipykernel to auto reload updated modules (avoid importlib)
%reload_ext autoreload
%autoreload 2

import deepmimo as dm

#%%

dm.create_scenario('asu_campus2')

# params = dm.default_params('asu_campus')
# dataset = dm.generate(params)

#%%
import os
import requests
from tqdm import tqdm

# Headers to mimic a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Mapping of current scenarios names and their dropbox links:
NAME_TO_LINK = {
    'asu_campus1': 'https://www.dropbox.com/scl/fi/unldvnar22cuxjh7db2rf/ASU_Campus1.zip?rlkey=rs2ofv3pt4ctafs2zi3vwogrh&dl=1', 

} # In the future, this dictionary will be fetched from a server. 

SCENARIOS_FOLDER = 'deepmimo_scenarios'
os.makedirs(SCENARIOS_FOLDER)

def download_scenario(name):
    url = NAME_TO_LINK[name]
    output_path = os.path.join(SCENARIOS_FOLDER, name + '.zip')
    response = requests.get(url, stream=True, headers=HEADERS)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192  # 8 KB
        with open(output_path, 'wb') as file:
            with tqdm(
                desc=f"Downloading '{name}' scenario",
                total=total_size / (1024 * 1024),  # Convert total size to MB
                unit='MB',
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
                        progress_bar.update(len(chunk) / (1024 * 1024))  # Update progress in MB
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

download_scenario('asu_campus1')










