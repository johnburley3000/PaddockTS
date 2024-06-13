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

# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 0.1

# +
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
# Comparing variation between pixels 5km away from each other
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

buffer = 0.05
for i, year in enumerate(years):
    date_range = slice(f'{year}-01-01', f'{year}-12-31')

    silo_ts = silo_data['daily_rain'].sel(time=date_range)
    silo_pixel1_ts = silo_ts.sel(lat=lat - buffer, lon=lon - buffer, method='nearest').cumsum(dim='time')
    silo_pixel2_ts = silo_ts.sel(lat=lat - buffer, lon=lon + buffer, method='nearest').cumsum(dim='time')
    silo_pixel3_ts = silo_ts.sel(lat=lat + buffer, lon=lon - buffer, method='nearest').cumsum(dim='time')
    silo_pixel4_ts = silo_ts.sel(lat=lat + buffer, lon=lon + buffer, method='nearest').cumsum(dim='time')

    ax = axes[i]  # Get the current subplot
    ax.plot(silo_pixel1_ts.time, silo_pixel1_ts, label='SILO northwest pixel')
    ax.plot(silo_pixel2_ts.time, silo_pixel2_ts, label='SILO northeast pixel')
    ax.plot(silo_pixel3_ts.time, silo_pixel3_ts, label='SILO southwest pixel')
    ax.plot(silo_pixel4_ts.time, silo_pixel4_ts, label='SILO southeast pixel')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rainfall (mm)')
    ax.set_title(f'SILO Rainfall Time Series for {year}')
    ax.legend()

if len(years) < len(axes):
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# -

# ### Notable features SILO rain comparison between pixels
# - Mostly similar rainfall between nearby pixels
# - 50mm less rain in the nNotableorthwest pixel in 2018

# +
# Comparing variables for different years
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
variables = ["daily_rain", "max_temp", "min_temp", "radiation", "vp"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
axes = axes.flatten()

# Find the year with the highest rainfall to sort the legend
total = {}
variable = 'daily_rain'
for year in years:
    date_range = slice(f'{year}-01-01', f'{year}-12-31')
    silo_ts = silo_data[variable].sel(time=date_range)
    silo_pixel1_ts = silo_ts.sel(lat=lat, lon=lon, method='nearest').cumsum(dim='time')
    total[year] = silo_pixel1_ts[-1].item()
sorted_years = sorted(total, key=total.get, reverse=True)

# Plot each variable
for i, variable in enumerate(variables):
    ax = axes[i]
    for year in sorted_years:
        date_range = slice(f'{year}-01-01', f'{year}-12-31')
        silo_ts = silo_data[variable].sel(time=date_range)
        silo_pixel1_ts = silo_ts.sel(lat=lat, lon=lon, method='nearest')
        
        # Aggregate by month showing a cumulative sum of rainfall, and a monthly average for everything else
        if variable == 'daily_rain':
            monthly_values = silo_pixel1_ts.resample(time='M').sum().cumsum()
            label=f'{year} ({total[year]:.1f}mm)'
        else:
            monthly_values = silo_pixel1_ts.resample(time='M').mean()
            label = f'{year}'

        monthly_start_dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')
        monthly_values['time'] = monthly_start_dates
        
        ax.plot(monthly_values, label=label)

    # Format x-axis to show month names and ensure all months appear
    ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
           labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel('Month') 
    ax.set_ylabel(f'{variable}') 
    
    metric = "cumulative sum" if variable == 'daily_rain' else "average"
    ax.set_title(f'{variable} monthly {metric} at centre pixel') \
    
    # Moving the legend outside the plot to the right
    ax.legend(title=f'Year', bbox_to_anchor=(1.05, 1), loc='upper left') 

# Remove the empty plot
if len(variables) < len(axes):
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])
    
plt.tight_layout()
plt.show()
# -

# ### Notable features in the above plots
# - Low rainfall in 2017-2019 and the start of 2020 (& 2023)
# - High summer temperature in 2019
# - High winter temperature in 2023
# - Low spring temperature in 2021-2022
# - Low vapour pressure in December 2019

# +
# Create a list of all the ANUClim filepaths from 2017 to 2023 (stored monthly)
# example_filepath = "/g/data/gh70/ANUClimate/v2-0/stable/day/rain/2023/ANUClimate_v2-0_rain_daily_202306.nc"

variable = "rain"
anuclim_filepaths = []
for year in range(2017,2023):
    for month in range(1,13):
        filepath = f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/{year}/ANUClimate_v2-0_{variable}_daily_{year}{month:02}.nc"
        anuclim_filepaths.append(filepath)

# This ANUClim directory only has data up to June 2023
anucim_2023_filepaths = [
        f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202301.nc",
        f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202302.nc",
        f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202303.nc",
        f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202304.nc",
        f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202305.nc",
        f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202306.nc",
    ]

anuclim_filepaths.extend(anucim_2023_filepaths)
anuclim_filepaths[0]

# +
variables = ["rain"]
variable = "rain"

# Load the data for each year and select just the region of interest 
print("Loading ANUClim variables: ", variables)
start = datetime.now()

datasets = []
for filepath in anuclim_filepaths:
    ds = xr.open_dataset(filepath)

    # When slicing the latitude anuclim needs the order (north, south) whereas silo needs (south, north).
    ds_region = ds.sel(lat=slice(lat + buffer, lat - buffer), lon=slice(lon - buffer, lon + buffer))
    datasets.append(ds_region)
ds_anuclim_rain = xr.concat(datasets, dim='time')

end = datetime.now()
print("Time taken to load ANUClim datasets: ", end - start)

# +
# Comparing ANUClim rain variation between pixels 5km away from each other
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

buffer = 0.05
for i, year in enumerate(years):
    date_range = slice(f'{year}-01-01', f'{year}-12-31')

    ts = ds_anuclim_rain['rain'].sel(time=date_range)
    pixel1_ts = ts.sel(lat=lat - buffer, lon=lon - buffer, method='nearest').cumsum(dim='time')
    pixel2_ts = ts.sel(lat=lat - buffer, lon=lon + buffer, method='nearest').cumsum(dim='time')
    pixel3_ts = ts.sel(lat=lat + buffer, lon=lon - buffer, method='nearest').cumsum(dim='time')
    pixel4_ts = ts.sel(lat=lat + buffer, lon=lon + buffer, method='nearest').cumsum(dim='time')

    ax = axes[i]  # Get the current subplot
    ax.plot(pixel1_ts.time, pixel1_ts, label='northwest pixel')
    ax.plot(pixel2_ts.time, pixel2_ts, label='northeast pixel')
    ax.plot(pixel3_ts.time, pixel3_ts, label='southwest pixel')
    ax.plot(pixel4_ts.time, pixel4_ts, label='southeast pixel')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rainfall (mm)')
    ax.set_title(f'ANUClim Rainfall Time Series for {year}')
    ax.legend()

if len(years) < len(axes):
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# -

# ### Notable features in ANUClim rain comparison between pixels
# - Consistently higher rainfall in the northeast pixel

monthly_values

xr.DataArray(
    np.append(monthly_values.values, [0, 0, 0, 0, 0, 0]),
    coords={"lat": monthly_values.lat, "lon": monthly_values.lon},
)

xr.DataArray(
    np.append(monthly_values.values, [0, 0, 0, 0, 0, 0]),
    coords={"lat": monthly_values.lat, "lon": monthly_values.lon},
    dims=['rain']
)

np.append(monthly_values.values, [0, 0, 0, 0, 0, 0])

monthly_values

xr.DataArray(
        np.append(monthly_values.values, [0, 0, 0, 0, 0, 0]),
        coords={"time": monthly_start_dates, "lat": monthly_values.lat, "lon": monthly_values.lon},
        dims=["time"]
    )

# +
# Comparing variables for different years
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
variables = ["rain"]
variable = "rain"

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
axes = axes.flatten()

# Find the year with the highest rainfall to sort the legend
total = {}
variable = 'rain'
for year in years:
    date_range = slice(f'{year}-01-01', f'{year}-12-31')
    ts = ds_anuclim_rain['rain'].sel(time=date_range)
    pixel1_ts = ts.sel(lat=lat, lon=lon, method='nearest').cumsum(dim='time')
    total[year] = pixel1_ts[-1].item()
sorted_years = sorted(total, key=total.get, reverse=True)

for year in sorted_years:
    date_range = slice(f'{year}-01-01', f'{year}-12-31')
    ts = ds_anuclim_rain[variable].sel(time=date_range)
    pixel1_ts = ts.sel(lat=lat, lon=lon, method='nearest')

    # Aggregate by month showing a cumulative sum of rainfall, and a monthly average for everything else
    monthly_values = pixel1_ts.resample(time='M').sum().cumsum()
    label=f'{year} ({total[year]:.1f}mm)'
    monthly_start_dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')

    # ANUClim only has rainfall up to June for 2023 so append 6 months of no rain for visualisation
    if year == 2023:
        xr.DataArray(
            np.append(monthly_values.values, [0, 0, 0, 0, 0, 0]),
            coords={"time": monthly_start_dates, "lat": monthly_values.lat, "lon": monthly_values.lon},
            dims=['time']
        ) 
    else: 
        monthly_values['time'] = monthly_start_dates
    plt.plot(monthly_values, label=label)

# Format x-axis to show month names and ensure all months appear
# plt.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
#        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
# plt.set_xlabel('Month') 
# plt.set_ylabel(f'{variable}') 

metric = "cumulative sum" if variable == 'rain' else "average"
# plt.set_title(f'{variable} monthly {metric} at centre pixel') \

# Moving the legend outside the plot to the right
plt.legend(title=f'Year', bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.show()
# -

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


