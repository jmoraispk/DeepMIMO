"""
DeepMIMO Scenario Downloader Module.

This module provides functionality for downloading and managing DeepMIMO scenario files,
including:
- Automated scenario downloading from remote storage
- Progress tracking and status reporting
- Retry mechanisms for failed downloads
- Scenario extraction and verification
- Support for multiple scenario types and variants

The module maintains a mapping of scenario names to their download URLs and handles
the complete download and extraction process.
"""

# Standard library imports
import os
import zipfile
from typing import Optional, Dict

# Third-party imports
import requests
from tqdm import tqdm

# Local imports
from ... import consts as c


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Mapping of current scenarios names and their dropbox links
NAME_TO_LINK: Dict[str, str] = {
    # ASU Campus
    #'asu_campus1': 'https://www.dropbox.com/scl/fi/unldvnar22cuxjh7db2rf/ASU_Campus1.zip?rlkey=rs2ofv3pt4ctafs2zi3vwogrh&dl=1', 
    'asu_campus': 'https://www.dropbox.com/scl/fi/ujeyo5fg6akbl4ic0gghd/asu_campus.zip?rlkey=8bctahqn1rdinsnljh3urdi5x&st=vlri3krf&dl=1',
    
    # O1 variants
    'O1_3p4': 'https://www.dropbox.com/s/2zj4xk3oo9lh07n/O1_3p4.zip?dl=1',
    'O1_3p5': 'https://www.dropbox.com/s/38t2fhduzvamy9l/O1_3p5.zip?dl=1',
    'O1_28': 'https://www.dropbox.com/s/7fo0em7fbmd8xd2/O1_28.zip?dl=1',
    'O1_60': 'https://www.dropbox.com/s/g667xpu1x96853e/O1_60.zip?dl=1',
    'O1_140': 'https://www.dropbox.com/s/g667xpu1x96853e/O1_60.zip?dl=1',
    'O1_3p5B': 'https://www.dropbox.com/s/33lux1kxpf8czmc/O1_3p5B.zip?dl=1',
    'O1_28B': 'https://www.dropbox.com/s/9jmbqums3siw2f3/O1_28B.zip?dl=1',
    'O1_drone_200': 'https://www.dropbox.com/s/4su29iyfl4d3c7f/O1_drone_200.zip?dl=1',

    # Boston5G variants
    'Boston5G_3p5': 'https://www.dropbox.com/s/ntwbregpbyuroxm/Boston5G_3p5.zip?dl=1', 
    'Boston5G_28': 'https://www.dropbox.com/s/vyum64dq8z5xjzg/Boston5G_28.zip?dl=1',
    'Boston5G_3p5_RIS': 'https://www.dropbox.com/scl/fi/klye9ieoh3mtq5k3xiosl/Boston5G_3p5_RIS.zip?rlkey=z7mbupe44minah7pzoq3m1ab2&dl=1',
    'Boston5G_3p5_polar': 'https://www.dropbox.com/scl/fi/lfipjfwvyjeqvhocle0lh/Boston5G_3p5_polar.zip?rlkey=4plendv0r5s6cmgwmsppt359o&dl=1',
    'Boston5G_28': 'https://www.dropbox.com/scl/fi/t4megnywkvoeejapvb190/Boston5G_28_RIS.zip?rlkey=o7nrrqqy68qguin99vbsimtdk&dl=1',
    
    # Dynamic Scenarios
    'O1_dyn_3p4': 'https://www.dropbox.com/s/3tii4zcqhmle9z9/O2_dyn_3p4.zip?dl=1',
    'O1_dyn_3p5': 'https://www.dropbox.com/s/11o71u0gmdmuozh/O2_dyn_3p5.zip?dl=1',
    'dyn_doppler_DD1_28': 'https://www.dropbox.com/scl/fi/keyxa6htaz6ulxtms2545/dyn_doppler_DD1_28.zip?rlkey=zw3egzi1jddy5j6r8hjzvbkzv&dl=1',

    # Indoor
    'I1_2p4': 'https://www.dropbox.com/s/68fk8i9xe0sy6hx/I1_2p4.zip?dl=1',
    'I1_2p5': 'https://www.dropbox.com/s/gytyb31dr73kctm/I1_2p5.zip?dl=1',
    'I2_28B': 'https://www.dropbox.com/s/ej47p7ooac90m9e/I2_28B%20scenario.zip?dl=1',
    'I3_2p4': 'https://www.dropbox.com/s/qjk4l9jjtml567z/I3_2p4.zip?dl=1',
    'I4_60': 'https://www.dropbox.com/s/vjp4kelu8fi2da8/I3_60.zip?dl=1',
    'officefloor1_28': 'https://www.dropbox.com/scl/fi/f5vimggavya2jj9o3u0t5/officefloor1.zip?rlkey=fnxgrx0i6a2sxuouwfmvlsbvn&dl=1',

    # DeepMIMO city
    'city_0_newyork': 'https://www.dropbox.com/scl/fi/0vw6i3ho7a8wi95w9atmq/city_0_newyork.zip?rlkey=ikuvis7zmsovhwbdln69iyepl&dl=1',
    'city_1_losangeles': 'https://www.dropbox.com/scl/fi/5s0tep8tq4ptkau07fmox/city_1_losangeles.zip?rlkey=8z014jku27e9r3vm8mws1a4hk&dl=1',
    'city_2_chicago': 'https://www.dropbox.com/scl/fi/ipmyctnsyavqpqtjn98za/city_2_chicago.zip?rlkey=oyyncczh8fo19hy6cmrqfv5wl&dl=1',
    'city_3_houston': 'https://www.dropbox.com/scl/fi/npgu5czuf7rpr409oj1rf/city_3_houston.zip?rlkey=qc7tw91mrxu147yqz2lgysm9l&dl=1',
    'city_4_phoenix': 'https://www.dropbox.com/scl/fi/ayxf6ek1teco8eghc1cik/city_4_phoenix.zip?rlkey=xotulfyki2igosofij0gyxtrg&dl=1',
    'city_5_philadelphia': 'https://www.dropbox.com/scl/fi/mvww486fr4eqgg9e0x7r4/city_5_philadelphia.zip?rlkey=n3n05wm716fz70v4amex9pub6&dl=1',
    'city_6_miami': 'https://www.dropbox.com/scl/fi/75sk9q9q6osdih57x13n6/city_6_miami.zip?rlkey=2efo5z7h5l76zm82kodn0jr5b&dl=1',
    'city_7_sandiego': 'https://www.dropbox.com/scl/fi/lmhnama4ulfet590kvxkr/city_7_sandiego.zip?rlkey=5c67b43k0p79i4qg8odwjegrj&dl=1',
    'city_8_dallas': 'https://www.dropbox.com/scl/fi/xx5o57689fpryxmowgc71/city_8_dallas.zip?rlkey=sgdra1yiow21szh5bld2llo5h&dl=1',
    'city_9_sanfrancisco': 'https://www.dropbox.com/scl/fi/z2bytmy8j1l9bpoheefwi/city_9_sanfrancisco.zip?rlkey=61cmkhez1df53k8h028fom3gd&dl=1',
    'city_10_austin': 'https://www.dropbox.com/scl/fi/eymupp9nrufgyhbhqzjtf/city_10_austin.zip?rlkey=jcbtxaim2zzg3nhxd8g5c63fz&dl=1',
    'city_11_santaclara': 'https://www.dropbox.com/scl/fi/y4p9ygz7ycztud4ycbml1/city_11_santaclara.zip?rlkey=zhhz3u4eimlcfht1h3imbw2ni&dl=1',
    'city_12_fortworth': 'https://www.dropbox.com/scl/fi/cb2m3nikqn7ovl2e6rj6y/city_12_fortworth.zip?rlkey=9z8lr1vi2nwv9u9pebdyaf5aq&dl=1',
    'city_13_columbus': 'https://www.dropbox.com/scl/fi/kbo9jntwvccb2zbpd70t5/city_13_columbus.zip?rlkey=xzqrgx3qaynhqab5syv3adjzo&dl=1',
    'city_14_charlotte': 'https://www.dropbox.com/scl/fi/pwb8kpna2d358b2ynyzr3/city_14_charlotte.zip?rlkey=7sx3p06a5vvnogyrni2a5v05m&dl=1',
    'city_15_indianapolis': 'https://www.dropbox.com/scl/fi/jrt5x9uecslefge3aoidz/city_15_indianapolis.zip?rlkey=3btgccmiha652z08vbniwfqk2&dl=1',
    'city_16_sanfrancisco': 'https://www.dropbox.com/scl/fi/zgams3dergwyvs1z16j54/city_16_sanfrancisco.zip?rlkey=vul0pt09tkhchwwz972qyci4t&dl=1',
    'city_17_seatle': 'https://www.dropbox.com/scl/fi/1jqvqmijfgtdoqg0hdfq4/city_17_seattle.zip?rlkey=xllzpe22huags0aou2hws2vnu&dl=1',
    'city_18_denver': 'https://www.dropbox.com/scl/fi/fpttaunc7j3gla0lrqp63/city_18_denver.zip?rlkey=zd4zs4qhwdpzjf329ozr45r4k&dl=1',
    'city_19_oklaoma': 'https://www.dropbox.com/scl/fi/yexjytvahsf27x9sunmfi/city_19_oklahoma.zip?rlkey=vgz7drqyjudmxqyepocu05b5u&dl=1',
}


def download_scenario(name: str) -> Optional[str]:
    """Download a specific DeepMIMO scenario file.
    
    This function downloads a scenario file from remote storage, creating the
    necessary directories and tracking download progress.

    Args:
        name (str): Name of the scenario to download.

    Returns:
        Optional[str]: Path to the downloaded zip file if successful, None otherwise.
        
    Raises:
        KeyError: If scenario name is not found in NAME_TO_LINK.
        ConnectionError: If download fails due to network issues.
    """
    os.makedirs(c.SCENARIOS_FOLDER, exist_ok=True)
    url = NAME_TO_LINK[name]
    output_path = os.path.join(c.SCENARIOS_FOLDER, name + '.zip')

    if os.path.exists(output_path):
        print(f'output path "{output_path}" already exists')
        return output_path

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
                        progress_bar.update(len(chunk) / (1024 * 1024))
        return output_path
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None


def download_scenario_handler(name: str) -> str:
    """Handle scenario download with retry mechanism.
    
    This function attempts to download a scenario multiple times in case of
    network failures, implementing a simple retry mechanism.

    Args:
        name (str): Name of the scenario to download.

    Returns:
        str: Path to the downloaded zip file.
        
    Raises:
        RuntimeError: If all download attempts fail.
    """
    zip_path = ''
    attempt = 0
    while attempt < 3:
        attempt += 1
        try: 
            zip_path = download_scenario(name)
            break
        except ConnectionError:
            print(f'Attempt {attempt}/3 failed..')
    
    return zip_path


def test_all_scen_download() -> None:
    """Test downloading all available scenarios.
    
    This function attempts to download all scenarios listed in NAME_TO_LINK
    to verify their availability and accessibility.
    """
    for scen in NAME_TO_LINK.keys():
        download_scenario_handler(scen)