# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/annual/catalog.html

# +
# I'm using python 3.9
# # !pip install jupyter jupytext xarray pandas scipy cftime matplotlib seaborn rasterio

# +
# Standard libraries
import os
import glob

# Dependencies
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import requests
import seaborn as sns
import matplotlib.pyplot as plt


# +
common_names = {
    "Tmax":"Maximum_Temperature",
    "Tmin":"Minimum_Temperature",
    "Pg":"Annual_Rainfall" 
}
aggregations = {
    "Tmax":"AnnualMaxima",
    "Tmin":"AnnualMinima",
    "Pg":"AnnualSums"
}

# Tmax and Tmin have 500m resolution, rainfall has 5km resolution
pixel_sizes = {
    "Tmax":0.005,
    "Tmin":0.005,
    "Pg":0.05
}


# -

def ozwald_yearly(var="Tmax", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021", stub="test", tmp_dir=""):
    
    # For the buffer size, 0.001 degrees is roughly 1km
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    # base_url = "https://thredds.nci.org.au"  # This is the new url (dapds00 is supposedly deprecated), but LAI only works with the old url
    base_url = "https://dapds00.nci.org.au"

    aggregation = aggregations[var]
    prefix = ".annual" if var == "Pg" else ""
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/annual/OzWALD{prefix}.{var}.{aggregation}.nc?var={aggregation}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
    
    response = requests.get(url)
    filename = os.path.join(tmp_dir, f'{stub}_{var}_{year}.nc')
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    ds = xr.open_dataset(filename)
    df = ds.to_dataframe().reset_index()

    return df


def ozwald_multiyear(var="Tmax", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], stub="test", tmp_dir="", cleanup=True):
    dfs = []
    for year in years:
        df_year = ozwald_yearly(var, latitude, longitude, buffer, year, stub, tmp_dir)
        dfs.append(df_year)
    df_combined = pd.concat(dfs, ignore_index=True)
    df_notime = df_combined.drop(columns=['time'])
    df_average = df_notime.groupby(['latitude', 'longitude'], as_index=False)[aggregations[var]].mean()

    if cleanup:
        nc_files = glob.glob('*.nc')
        for file in nc_files:
            os.remove(file)
        
    return df_average


def write_tiff(df, variable="Tmax", filename="output.tiff", pixel_size=0.05):
    # Assumes the df has columns: latitude, longitude and variable 
    
    df_pivot = df.pivot(index='latitude', columns='longitude', values=aggregations[variable])
    grid_array = df_pivot.values    
    transform = rasterio.transform.from_origin(df['longitude'].min(), df['latitude'].min(), pixel_size, -pixel_size) 
    
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=grid_array.shape[0],
        width=grid_array.shape[1],
        count=1,
        dtype=grid_array.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(grid_array, 1)
    print("Saved:", filename)


def ozwald_yearly_average(variables=["Tmax", "Pg"], lat=-34.3890427, lon=148.469499, buffer=0.005, start_year='2020', end_year='2021', outdir="", stub="test", tmp_dir=""):
    """Download yearly averages from SILO stored on Ozwald"""
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        df = ozwald_multiyear(var=variable, years=years, stub=stub, tmp_dir=tmp_dir)
        pixel_size = pixel_sizes[variable]
        filename = os.path.join(outdir, f'{stub}_{common_names[variable]}_{start_year}_{end_year}_average.tif')
        write_tiff(df, variable, filename, pixel_size)



# %%time
if __name__ == '__main__':
    ozwald_yearly_average()
