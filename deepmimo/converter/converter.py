import os

import converter_utils as cu

from .aodt.aodt_converter import aodt_rt_converter
from .sionna_rt.sionna_converter import sionna_rt_converter
from .wireless_insite.insite_converter import insite_rt_converter


def create_scenario(path_to_rt_folder):
    print('Determining converter...')
    
    # Example logic to determine generator type based on files found
    files_in_dir = os.listdir(path_to_rt_folder)
    if cu.ext_in_list('.aodt', files_in_dir):
        print("Using AODT generator")
        rt_converter = aodt_rt_converter
    elif cu.ext_in_list('.path', files_in_dir):
        print("Using Sionna_RT generator")
        rt_converter = sionna_rt_converter
    elif cu.ext_in_list('.p2m', files_in_dir):
        print("Using Wireless Insite generator")
        rt_converter = insite_rt_converter
    else:
        print("Unknown raytracer type")
        return
    
    rt_converter(path_to_rt_folder)
