import pandas as pd


def parse_scenario_csv(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    num_scenario = df.shape[0]

    # Filter the column names that match 'bs[i]_lat' and 'bs[i]_lon'
    lat_columns = [col for col in df.columns if "bs" in col and "_lat" in col]
    lon_columns = [col for col in df.columns if "bs" in col and "_lon" in col]

    # Create an empty list to store the rows
    bs_gps_pos = []
    # Create an empty list to store the min/max lat/lon values
    gps_bbox = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        gps_bs_pos_row = []

        # Fetch minlat, minlon, maxlat, maxlon for each row
        minlat = row["minlat"]
        minlon = row["minlon"]
        maxlat = row["maxlat"]
        maxlon = row["maxlon"]

        gps_bbox_row = {
            "minlat": minlat,
            "minlon": minlon,
            "maxlat": maxlat,
            "maxlon": maxlon,
        }

        # Populate the row_data list only if both latitude and longitude are non-empty
        for lat_col, lon_col in zip(lat_columns, lon_columns):
            lat_val = row[lat_col]
            lon_val = row[lon_col]

            if pd.notna(lat_val) and pd.notna(lon_val):
                lat_lon = {"lat": lat_val, "lon": lon_val}
                gps_bs_pos_row.append(lat_lon)

        # Append the row_data to rows_data only if it's not empty
        if gps_bs_pos_row:
            bs_gps_pos.append(gps_bs_pos_row)
            gps_bbox.append(gps_bbox_row)


    return num_scenario, gps_bbox, bs_gps_pos

if __name__ == "__main__":  
    num_scenario, gps_bbox, bs_gps_pos = parse_scenario_csv("scenario_pos_multi_bs.csv")
    print('done')