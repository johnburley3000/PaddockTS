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
import gc

# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 0.1

# +
# Create a list of all the ANUClim filepaths from 2017 to 2023 (stored monthly)
# example_filepath = "/g/data/gh70/ANUClimate/v2-0/stable/day/rain/2023/ANUClimate_v2-0_rain_daily_202306.nc"

variables = ["evap", "rain", "srad", "tavg", "tmax", "tmin", "vp", "vpd"]
anuclim_filepaths = {}
for variable in variables:
    anuclim_filepaths[variable] = []
    for year in range(2017,2023):
        for month in range(1,13):
            filepath = f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/{year}/ANUClimate_v2-0_{variable}_daily_{year}{month:02}.nc"
            anuclim_filepaths[variable].append(filepath)

    # This ANUClim directory only has data up to June 2023
    anuclim_2023_filepaths = [
            f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202301.nc",
            f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202302.nc",
            f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202303.nc",
            f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202304.nc",
            f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202305.nc",
            f"/g/data/gh70/ANUClimate/v2-0/stable/day/{variable}/2023/ANUClimate_v2-0_{variable}_daily_202306.nc",
        ]
    anuclim_filepaths[variable].extend(anuclim_2023_filepaths)
    
anuclim_filepaths.keys()
# -

# Load the data for each year and select just the region of interest 
anuclim_data = {}
datasets = []
variables = ["tmax", "tmin", "vp", "vpd"]
for variable in variables:
    start = datetime.now()
    print(datetime.now(), "Loading ANUClim variable: ", variable)
    for i, filepath in enumerate(anuclim_filepaths[variable]):
        print(f"{datetime.now()} {i}/{len(anuclim_filepaths[variable])}", end='\r')
        ds = xr.open_dataset(filepath)

        # When slicing the latitude anuclim needs the order (north, south) whereas silo needs (south, north).
        ds_region = ds.sel(lat=slice(lat + buffer, lat - buffer), lon=slice(lon - buffer, lon + buffer))
        datasets.append(ds_region)
        combined_ds = xr.concat(datasets, dim='time')
        anuclim_data[variable] = combined_ds[variable]

        filepath = f'/scratch/xe2/cb8590/MILG/anuclim_{variable}_2017-2023.nc'
        anuclim_data[variable].to_netcdf(filepath)

    end = datetime.now()
    print(f"Time taken to load {variable}: ", end - start)

# Clean up the memory usage
gc.collect()

filepath = '/scratch/xe2/cb8590/MILG/anuclim_rain_2017-2023.nc'
start = datetime.now()
ds = xr.open_dataset(filepath)
print(datetime.now() - start)

# +
# Comparing ANUClim rain variation between pixels 5km away from each other
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

buffer = 0.05
for i, year in enumerate(years):
    date_range = slice(f'{year}-01-01', f'{year}-12-31')

    ts = anuclim_data[variable].sel(time=date_range)
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
    ts = anuclim_data[variable].sel(time=date_range)
    pixel1_ts = ts.sel(lat=lat, lon=lon, method='nearest').cumsum(dim='time')
    total[year] = pixel1_ts[-1].item()
sorted_years = sorted(total, key=total.get, reverse=True)

# Plot each variable
for i, variable in enumerate(variables):
    ax = axes[i]
    for year in sorted_years:
        date_range = slice(f'{year}-01-01', f'{year}-12-31')
        ts = anuclim_data[variable].sel(time=date_range)
        pixel1_ts = ts.sel(lat=lat, lon=lon, method='nearest')

        # Aggregate by month showing a cumulative sum of rainfall, and a monthly average for everything else
        if variable == 'rain':
            monthly_values = pixel1_ts.resample(time='M').sum().cumsum()
            label=f'{year} ({total[year]:.1f}mm)'
        else:
            monthly_values = pixel1_ts.resample(time='M').mean()
            label = f'{year}'
        
        monthly_start_dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')

        # ANUClim only has data up to June for 2023 so append 6 months of zeros for visualisation
        if year == 2023:
            xr.DataArray(
                np.append(monthly_values.values, [0, 0, 0, 0, 0, 0]),
                coords={"time": monthly_start_dates, "lat": monthly_values.lat, "lon": monthly_values.lon},
                dims=['time']
            ) 
        else: 
            monthly_values['time'] = monthly_start_dates
        ax.plot(monthly_values, label=label)
        
        # Moving the legend outside the plot to the right
        ax.legend(title=f'Year', bbox_to_anchor=(1, 1), loc='upper left') 

    # Format x-axis to show month names and ensure all months appear
    ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
           labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel('Month') 
    ax.set_ylabel(f'{variable}') 

    metric = "cumulative sum" if variable == 'rain' else "average"
    ax.set_title(f'{variable} monthly {metric} at centre pixel') \

if len(years) < len(axes):
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
