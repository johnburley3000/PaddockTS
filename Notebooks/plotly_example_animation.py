import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
from matplotlib.colors import LogNorm
import numpy as np

# +
# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 1  # 1 degree ~= 100km

# Load the silo data for 2017-2023 for just the region of interest
variables = ["daily_rain", "max_temp", "min_temp", "radiation", "vp"]
print("Loading SILO variables: ", variables)

start = datetime.now()
silo_data = {}
for variable in variables:
    silo_filepaths = [
        f"/g/data/ub8/au/SILO/{variable}/2017.{variable}.nc",
        f"/g/data/ub8/au/SILO/{variable}/2018.{variable}.nc",
        f"/g/data/ub8/au/SILO/{variable}/2019.{variable}.nc",
        f"/g/data/ub8/au/SILO/{variable}/2020.{variable}.nc",
        f"/g/data/ub8/au/SILO/{variable}/2021.{variable}.nc",
        f"/g/data/ub8/au/SILO/{variable}/2022.{variable}.nc",
        f"/g/data/ub8/au/SILO/{variable}/2023.{variable}.nc"
    ]
    
    # Load the data for each year and select just the region of interest 
    silo_datasets = []
    for filepath in silo_filepaths:
        ds = xr.open_dataset(filepath)
        ds_region = ds.sel(lat=slice(lat - buffer, lat + buffer), lon=slice(lon - buffer, lon + buffer))
        silo_datasets.append(ds_region)
    combined_ds = xr.concat(silo_datasets, dim='time')
    silo_data[variable] = combined_ds[variable]
    
end = datetime.now()
print("Time taken to load SILO datasets: ", end - start)

# +
variables = ["daily_rain", "max_temp", "min_temp", "radiation", "vp"]
colours = ["Blues", "Reds", "Greens", "Oranges", "Purples"] 

for variable, colour in zip(variables, colours):
    ds = silo_data[variable]
    time_series = ds 
    times = ds['time']

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    def animate(time_index):
        ax.clear()
        time_slice = time_series.isel(time=time_index)
        c = ax.pcolormesh(time_slice['lon'], time_slice['lat'], time_slice, shading='auto', cmap=colour)
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
# +
# Fixing the colour scale

variables = ["daily_rain", "max_temp", "min_temp", "radiation", "vp"]
colours = ["Blues", "Reds", "Purples", "Oranges", "Greens"]

for variable, colour in zip(variables, colours):
    ds = silo_data[variable]
    time_series = ds
    times = ds['time']

    # Determine the global min and max for the color scale
    vmin = float(time_series.min().values)
    vmax = float(time_series.max().values)

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(time_index):
        ax.clear()
        time_slice = time_series.isel(time=time_index)
        c = ax.pcolormesh(time_slice['lon'], time_slice['lat'], time_slice, shading='auto', cmap=colour, vmin=vmin, vmax=vmax)
        if not hasattr(ax, 'colorbar') or ax.colorbar is None:
            ax.colorbar = fig.colorbar(c, ax=ax, label=variable)
        ax.set_title(f"{variable} on {str(times[time_index].values)[:10]}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    start = datetime.now()
    ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=33)
    ani.save(f'{variable}_animation.mp4', writer='ffmpeg')
    end = datetime.now()

    print(f"Time taken to create animation for {variable}: {end - start}")


# +
# Using a log scale for the rain
variables = ["daily_rain"]
colours = ["Blues"]

for variable, colour in zip(variables, colours):
    ds = silo_data[variable]
    time_series = ds
    times = ds['time']

    # Determine the global min and max for the color scale
    vmin = float(time_series.min().values)
    vmax = float(time_series.max().values)

    # Ensure vmin is positive for LogNorm
    if variable == "daily_rain" and vmin <= 0:
        vmin = 1e-3  # Set to a small positive value if the minimum is zero or negative

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(time_index):
        ax.clear()
        time_slice = time_series.isel(time=time_index)
        if variable == "daily_rain":
            # Ensure all values in the time_slice are positive
            time_slice = time_slice.where(time_slice > 0, 1e-3)
            c = ax.pcolormesh(time_slice['lon'], time_slice['lat'], time_slice, shading='auto', cmap=colour, norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            c = ax.pcolormesh(time_slice['lon'], time_slice['lat'], time_slice, shading='auto', cmap=colour, vmin=vmin, vmax=vmax)
        if not hasattr(ax, 'colorbar') or ax.colorbar is None:
            ax.colorbar = fig.colorbar(c, ax=ax, label=variable)
        ax.set_title(f"{variable} on {str(times[time_index].values)[:10]}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    start = datetime.now()
    ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=33)
    ani.save(f'{variable}_animation.mp4', writer='ffmpeg')
    end = datetime.now()

    print(f"Time taken to create animation for {variable}: {end - start}")


# +
# Small example for testing colours

anuclim_rain = silo_data['daily_rain']
date_range = slice('2023-06-22', '2023-06-24')
anuclim_ts = anuclim_rain.sel(time=date_range)

variable = 'max_temp'
colour = 'Reds'

ds = anuclim_ts
time_series = ds 
times = ds['time']

# Create the animation
fig, ax = plt.subplots(figsize=(10, 6))
def animate(time_index):
    ax.clear()
    time_slice = time_series.isel(time=time_index)
    c = ax.pcolormesh(time_slice['lon'], time_slice['lat'], time_slice, shading='auto', cmap=colour)
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


