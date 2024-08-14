# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/annual/catalog.html

# +
# I'm using python 3.9
# # !pip install jupyter jupytext xarray pandas scipy cftime matplotlib seaborn rasterio
# -

import requests
import os
import xarray as xr
import pandas as pd
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin


# +
common_names = {
    "Tmax":"Maximum Temperature",
    "Tmin":"Minimum Temperature",
    "Pg":"Annual Rainfall" 
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

def ozwald_yearly(var="Tmax", latitude=-34.3890427, longitude=148.469499, year="2021", buffer=0.1):
    
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
    print(url)
    
    response = requests.get(url)
    filename = f'{var}_{year}.nc' 
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    ds = xr.open_dataset(filename)
    df = ds.to_dataframe().reset_index()

    return df
    


def ozwald_multiyear(var="Tmax", latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"], buffer=0.1, cleanup=True):
    dfs = []
    for year in years:
        df_year = ozwald_yearly(var, latitude, longitude, year, buffer)
        dfs.append(df_year)
    df_combined = pd.concat(dfs, ignore_index=True)
    df_notime = df_combined.drop(columns=['time'])
    df_average = df_notime.groupby(['latitude', 'longitude'], as_index=False)[aggregations[variable]].mean()
    
    nc_files = glob.glob('*.nc')
    for file in nc_files:
        os.remove(file)
        
    return df_average


def write_tiff(df, variable="Tmax", filename="output.tiff", pixel_size=0.05):
    # Assumes the df has columns: latitude, longitude and variable 
    # For the pixel_size, 0.001 degrees is roughly 1km
    
    df_pivot = df.pivot(index='latitude', columns='longitude', values=aggregations[variable])
    grid_array = df_pivot.values    
    transform = from_origin(df['longitude'].min(), df['latitude'].min(), pixel_size, -pixel_size) 
    
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


def visualise_tiff(filename="output.tiff", title="Maximum Temperature"):
    with rasterio.open(filename) as src:
        data = src.read(1)  
        
        # Flip the image to match the orientation in QGIS
        flipped_data = np.flip(data, axis=0)

        plt.figure(figsize=(8, 6))
        img = plt.imshow(flipped_data, cmap='viridis', extent=(
            src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        cbar = plt.colorbar(img, ax=plt.gca())
        plt.show()


# %%time
if __name__ == '__main__':

    years = [str(year) for year in list(range(2014, 2024))]
    for variable in common_names.keys():
        df = ozwald_multiyear(var=variable, years=years)
        pixel_size = pixel_sizes[variable]
        filename = f'{common_names[variable]}.tiff'
        write_tiff(df, variable, filename, pixel_size)
        visualise_tiff(filename, common_names[variable])



