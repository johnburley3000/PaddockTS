# Make sure to include these paths when creating the interactive session on NCI ARE
# gdata/xe2+gdata/v10+gdata/ka08+gdata/ub8+gdata/gh70
# !ls /g/data/ub8/au/SILO/daily_rain/2023.daily_rain.nc
# !ls /g/data/gh70/ANUClimate/v2-0/stable/day/rain/2023/ANUClimate_v2-0_rain_daily_202306.nc
# !ls /g/data/ub8/au/OzWALD/8day/Ssoil/OzWALD.Ssoil.2020.nc

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import animation

# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 1 # 20km

# +
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

# +
# Comparing soil moisture variation between pixels 5km away from each other
filepath = filepaths['Ssoil'][0]
ds = xr.open_dataset(filepath)
ds_region = ds.sel(latitude=slice(lat + buffer, lat - buffer), longitude=slice(lon - buffer, lon + buffer))
years = [2017]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

buffer = 0.05
for i, year in enumerate(years):
    date_range = slice(f'{year}-01-01', f'{year}-12-31')

    ts = ds_region['Ssoil'].sel(time=date_range)
    pixel1_ts = ts.sel(latitude=lat - buffer, longitude=lon - buffer, method='nearest').cumsum(dim='time')
    pixel2_ts = ts.sel(latitude=lat - buffer, longitude=lon + buffer, method='nearest').cumsum(dim='time')
    pixel3_ts = ts.sel(latitude=lat + buffer, longitude=lon - buffer, method='nearest').cumsum(dim='time')
    pixel4_ts = ts.sel(latitude=lat + buffer, longitude=lon + buffer, method='nearest').cumsum(dim='time')

    ax = axes[i]  # Get the current subplot
    ax.plot(pixel1_ts.time, pixel1_ts, label='northwest pixel')
    ax.plot(pixel2_ts.time, pixel2_ts, label='northeast pixel')
    ax.plot(pixel3_ts.time, pixel3_ts, label='southwest pixel')
    ax.plot(pixel4_ts.time, pixel4_ts, label='southeast pixel')
    ax.set_xlabel('Time')
    ax.set_ylabel('Soil Moisture (mm)')
    ax.set_title(f'Time Series for {year}')
    ax.legend()

if len(years) < len(axes):
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# -

# Load the data for each year and select just the region of interest 
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

        filepath = f'/scratch/xe2/cb8590/MILG/OzWALD_{variable}_2017-2023.nc'
        data[variable].to_netcdf(filepath)

    end = datetime.now()
    print(f"Time taken to load {variable}: ", end - start)

data.keys()

data['Ssoil']

data

# +
# Comparing cumulative rainfall between different years
years = [2017, 2018, 2019, 2020, 2021, 2022]
variables = ["Ssoil", "LAI", "NDVI",  "GPP"]
summed = ["Ssoil", "LAI", "NDVI", "GPP"]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
axes = axes.flatten()

for i, variable in enumerate(variables):

    ax = axes[i]
    data[variable] = data[variable].sortby('time')

    total = {}
    for year in years:
        date_range = slice(f'{year}-01-01', f'{year}-12-31')
        ts = data[variable].sel(time=date_range)
        pixel1_ts = ts.sel(latitude=lat, longitude=lon, method='nearest').cumsum(dim='time')
        total[year] = 0 if len(pixel1_ts) == 0 else pixel1_ts[-1].item()
    sorted_years = sorted(total, key=total.get, reverse=True)

    for year in sorted_years:
        date_range = slice(f'{year}-01-01', f'{year}-12-31')
                
        ts = data[variable].sel(time=date_range)
        pixel1_ts = ts.sel(latitude=lat, longitude=lon, method='nearest')
        
        if variable in summed:
            pixel1_ts = pixel1_ts.cumsum(dim='time')

        # Convert time to an arbitrary year (2020) so they can be overlayed on the one plot
        time_values = pd.to_datetime(pixel1_ts.time.values)
        normalized_time_values = time_values.map(lambda x: x.replace(year=2020))

        ax.plot(normalized_time_values, pixel1_ts, label=year)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')) 
           
    # Format x-axis to show month names and ensure all months appear
    ax.set_xlabel('Month') 
    ax.set_ylabel(f'{variable}') 
    
    if variable in summed:
        title = f'8day cumulative sum of {variable} time series at centre pixel'
    else:
        title = f'8day {variable} time series at centre pixel'
    ax.set_title(title) 
    
    if variable in summed:
        # Moving the legend outside the plot to the right
        ax.legend(title=f'Year', bbox_to_anchor=(1, 1), loc='upper left') 
    else:
        ax.legend(title=f'Year') 

if len(variables) < len(axes):
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# +
variables = ["Ssoil", "LAI", "GPP", "NDVI",]
colours = ["Blues", "YlGn", "BuGn", "Greens"]

variables = ["LAI", "GPP"]
colours = ["Greens", "YlGn"]


for variable, colour in zip(variables, colours):
    ds = data[variable]
    ds = ds.dropna(dim='time', how='all')
    
    # Forward fill missing values along the time dimension
    ds = ds.ffill(dim='time')
    
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
        c = ax.pcolormesh(time_slice['longitude'], time_slice['latitude'], time_slice, shading='auto', cmap=colour, vmin=vmin, vmax=vmax)
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
date_range = slice('2022-10-01', '2022-12-31')
ts = data[variable].sel(time=date_range)

variable = 'LAI'
colour = 'Greens'

ds = ts
ds= ds.dropna(dim='time', how='all')
time_series = ds 
times = ds['time']

# + endofcell="--"
# Create the animation
fig, ax = plt.subplots(figsize=(10, 6))
def animate(time_index):
    ax.clear()
    time_slice = time_series.isel(time=time_index)
    c = ax.pcolormesh(time_slice['longitude'], time_slice['latitude'], time_slice, shading='auto', cmap=colour)
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

# --


