# Make sure to include these paths when creating the interactive session on NCI ARE
# gdata/xe2+gdata/v10+gdata/ka08+gdata/ub8+gdata/gh70
# !ls /g/data/gh70/ANUClimate/v2-0/stable/day/rain/2023/ANUClimate_v2-0_rain_daily_202306.nc
# !ls /g/data/ub8/au/SILO/daily_rain/2023.daily_rain.nc

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime

# filepaths
anuclim_filepath = "/g/data/gh70/ANUClimate/v2-0/stable/day/rain/2023/ANUClimate_v2-0_rain_daily_202306.nc"
silo_filepath = "/g/data/ub8/au/SILO/daily_rain/2023.daily_rain.nc"
anuclim_full = xr.open_dataset(anuclim_filepath)
silo_full = xr.open_dataset(silo_filepath)

# +
# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 0.1

# When slicing the latitude anuclim needs the order (north, south) whereas silo needs (south, north).
anuclim_region = anuclim_full.sel(lat=slice(lat + buffer, lat - buffer), lon=slice(lon - buffer, lon + buffer))
silo_region = silo_full.sel(lat=slice(lat - buffer, lat + buffer), lon=slice(lon - buffer, lon + buffer))

# Choose the rain variable
anuclim_rain = anuclim_region['rain']
silo_rain = silo_region['daily_rain']

# +
# Visualising a map at a single time point
time_point = '2023-06-13'

fig, axes = plt.subplots(1, 2, figsize=(21, 7))

# ANUClim Rainfall plot
anuclim_subset = anuclim_rain.sel(time=time_point)
anuclim_subset.plot(ax=axes[0])
axes[0].set_title('ANUClim Rainfall on ' + time_point)

# SILO Rainfall plot
silo_subset = silo_rain.sel(time=time_point)
silo_subset.plot(ax=axes[1])
axes[1].set_title('SILO Rainfall on ' + time_point)

# Fixing x axis to have a consistent naming format
lon_formatter = mticker.ScalarFormatter()
lon_formatter.set_useOffset(False)
axes[0].xaxis.set_major_formatter(lon_formatter)

plt.show()

# +
# Plotting time series for a single pixel
date_range = slice('2023-06-01', '2023-06-30')
anuclim_ts = anuclim_rain.sel(time=date_range)
silo_ts = silo_rain.sel(time=date_range)
anuclim_pixel_ts = anuclim_ts.sel(lat=lat, lon=lon, method='nearest')
silo_pixel_ts = silo_ts.sel(lat=lat, lon=lon, method='nearest')

plt.figure(figsize=(10, 5))
plt.plot(anuclim_pixel_ts.time, anuclim_pixel_ts, label='ANUClim')
plt.plot(silo_pixel_ts.time, silo_pixel_ts, label='SILO')
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
plt.title('Rainfall Time Series at ({}, {})'.format(lat, lon))
plt.legend()
plt.show()
# -






