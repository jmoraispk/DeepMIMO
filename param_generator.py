#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:21:08 2023

@author: joao

Parameters expander will take the parameters and decompose each row in as 
many rows as it requires to obtain scenes with some maximum dimensions. 

This script is meant to be used to speed up the population of scenes. One can
go to some big cities and get very large areas, and by using this script, 
split them in smaller chunks for computation.

With the default settings, this script will create boxes of 500 x 500 meters
from the min_corner = [min_lat, min_lon] to the max corner. 

If strick = True, then it's possible that some area will not be covered, as the
chunk size will be enforced strictly. 


"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic

def compute_distance(coord1, coord2):
    " Returns distance in meters, more precisely than the other methods."
    
    return geodesic(coord1, coord2).meters


def dist_to_angle(origin_coord, dist, lat_or_lon='lat'):
    """
    Returns the angle correspondent to a certain distance from the origin
    coordinate, along latitude or longitude.
    """
    small_angle = 0.001
    
    if 'lat' in lat_or_lon:
        new_coord = origin_coord + [small_angle,0]
    elif 'lon' in lat_or_lon:
        new_coord = origin_coord + [0,small_angle]
    else:
        raise Exception('Invalid "lat_or_lon". Input "lat" or "lon".')
    
    # Measure the distance caused by that angle
    dist_small_ang = compute_distance(origin_coord, new_coord)
    
    # Scale the angle proportionally to the distance desired
    # small_angle/dist_small ang = UNKNOWN ANGLE/ dist
    
    return small_angle / dist_small_ang * dist


def compute_array_combinations(arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))


# INPUT
chunk_size = [500, 500] # [m] along [vertical, horizontal]
strict = True


df = pd.read_csv('raw_params.csv')

# Setup array to copy the values of the CSV
n_cols = len(df.columns)
copy_values = {col:[] for col in df.columns}


n_rows = df.index.stop
for row_idx in range(n_rows):
    min_lat = df['min_lat'][row_idx]
    min_lon = df['min_lon'][row_idx]
    max_lat = df['max_lat'][row_idx]
    max_lon = df['max_lon'][row_idx]
    
    min_corner = np.array([min_lat, min_lon])
    max_corner = np.array([max_lat, max_lon])
    
    lat_delta = dist_to_angle(min_corner, chunk_size[0], lat_or_lon='lat')
    lon_delta = dist_to_angle(min_corner, chunk_size[1], lat_or_lon='lon')
    
    lat_corners = np.arange(min_lat, max_lat, lat_delta)
    lon_corners = np.arange(min_lon, max_lon, lon_delta)
    
    if not strict:
        lat_corners[-1] = max_lat
        lon_corners[-1] = max_lon
    
    # Make pairs of corners (these will be the new lat/lon min and max)
    lat_minmax = np.array([[lat_corners[i], lat_corners[i+1]] for i in range(len(lat_corners)-1)])
    lon_minmax = np.array([[lon_corners[i], lon_corners[i+1]] for i in range(len(lon_corners)-1)])
    
    # Compute combinations of corners to generate all boxes
    idxs = compute_array_combinations( [np.arange(len(lat_minmax)), np.arange(len(lon_minmax))])
    
    new_lat_mins = lat_minmax[idxs[:,0], 0]
    new_lat_maxs = lat_minmax[idxs[:,0], 1]
    new_lon_mins = lon_minmax[idxs[:,1], 0]
    new_lon_maxs = lon_minmax[idxs[:,1], 1]
    
    copy_values['min_lat'] = new_lat_mins.tolist()
    copy_values['min_lon'] = new_lon_mins.tolist()
    copy_values['max_lat'] = new_lat_maxs.tolist()
    copy_values['max_lon'] = new_lon_maxs.tolist()
    
    n_boxes = len(new_lat_mins)
    
    # copy other values in the row
    for col_idx in range(n_cols-4):
        col_name = df.columns[col_idx+4]
        copy_values[col_name] += [df.loc[row_idx, col_name]] * n_boxes
    
# Write lines to new CSV
new_df = pd.DataFrame(copy_values)
new_df.to_csv('gen_params.csv', index=False)
