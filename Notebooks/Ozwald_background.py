import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import animation

# Region of interest
lat = -34.044803
lon = 148.602840
buffer = 1

# Create a list of Ozwald filepaths from 2017 to 2023 (stored yearly)
# example_filepath = "/g/data/ub8/au/OzWALD/8day/Ssoil/OzWALD.Ssoil.2020.nc"
variables = ["Ssoil", "LAI", "NDVI", "GPP"]
filepaths = {}
for variable in variables:
    filepaths[variable] = []
    for year in range(2017,2023):
        filepath = f"/g/data/ub8/au/OzWALD/8day/{variable}/OzWALD.{variable}.{year}.nc"
        filepaths[variable].append(filepath)
filepaths.keys()


# Save the data for just the variables, years, and region of interest 
def select_ozwald_data():
    data = {}
    datasets = []
    variables = ["Ssoil", "LAI", "NDVI", "GPP"]
    for variable in variables:
        start = datetime.now()
        print(datetime.now(), "Loading variable: ", variable)
        for i, filepath in enumerate(filepaths[variable]):
            print(f"{datetime.now()} {i}/{len(filepaths[variable])}", end='\r')
            ds = xr.open_dataset(filepath)

            # When slicing the latitude anuclim needs the order (north, south) whereas silo needs (south, north).
            ds_region = ds.sel(latitude=slice(lat + buffer, lat - buffer), longitude=slice(lon - buffer, lon + buffer))
            datasets.append(ds_region)
            combined_ds = xr.concat(datasets, dim='time')
            data[variable] = combined_ds[variable]

            filepath = f'/scratch/xe2/cb8590/MILG/OzWALD_midpoint_{variable}_2017-2023.nc'
            data[variable].to_netcdf(filepath)

        end = datetime.now()
        print(f"Time taken to load {variable}: ", end - start)
# select_ozwald_data()


# Load in the presaved data
variables = ["Ssoil", "LAI", "NDVI", "GPP"]
data = {}
for variable in variables:
    filepath = f'/scratch/xe2/cb8590/MILG/OzWALD_midpoint_{variable}_2017-2023.nc'
    data[variable] = xr.open_dataset(filepath)

# !ls

# +

import matplotlib.image as mpimg

background_image_path = 'region_500km.png'
background_img = mpimg.imread(background_image_path)
background_extent = [145.709428, 152.443774, -35.923749, -31.888347]


# +
# Small example for creating an animation with a background image
variable = 'NDVI'
filepath = f'/scratch/xe2/cb8590/MILG/OzWALD_midpoint_{variable}_2017-2023.nc'
ds = xr.open_dataset(filepath)
ds = ds.dropna(dim='time', how='all')
ds = ds.ffill(dim='time')
time_series = ds[variable].sel(time=slice('2022-10-01', '2022-10-31'))
times = time_series['time']

fig, ax = plt.subplots(figsize=(10, 6))
def animate(time_index):
    ax.clear()
    ax.imshow(background_img, extent=background_extent, aspect='auto')

    time_slice = time_series.isel(time=time_index)
    c = ax.pcolormesh(time_slice['longitude'], time_slice['latitude'], time_slice, shading='auto', cmap='YlGn')
    if not hasattr(ax, 'colorbar'):
        ax.colorbar = fig.colorbar(c, ax=ax, label=variable)
    ax.set_title(f"{variable} on {str(times[time_index].values)[:10]}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

start = datetime.now()
ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=33)
ani.save(f'{variable}_animation.mp4', writer='ffmpeg')
end = datetime.now()

print(f"Time taken to create animation for {variable} {end - start}")

# -



