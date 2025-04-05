import utm
import numpy as np

def xy_from_latlong(lat, long):
    """Assumes lat and long along row. Returns same row vec/matrix on
    cartesian coords."""
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat, long)
    return x, y

def convert_GpsBBox2CartesianBBox(minlat, minlon, maxlat, maxlon, origin_lat, origin_lon, pad=0):
    xmin, ymin = xy_from_latlong(minlat, minlon)
    xmax, ymax = xy_from_latlong(maxlat, maxlon)
    x_origin, y_origin = xy_from_latlong(origin_lat, origin_lon)

    xmin = xmin - x_origin
    xmax = xmax - x_origin
    ymin = ymin - y_origin
    ymax = ymax - y_origin
    
    return xmin-pad, ymin-pad, xmax+pad, ymax+pad

def convert_Gps2RelativeCartesian(lat, lon, origin_lat, origin_lon):
    x_origin, y_origin = xy_from_latlong(origin_lat, origin_lon)
    x, y = xy_from_latlong(lat, lon)
    
    return x - x_origin, y - y_origin


if __name__ == "__main__":
    minlat = 33.4196500
    minlon = -111.9326400
    maxlat = 33.4218200
    maxlon = -111.9288200
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(minlat, minlon, maxlat, maxlon)
    print('done')

