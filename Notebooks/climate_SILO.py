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
# lat, lon = -34.38904277303204, 148.46949938279096   # Ronda Milgdara
lat, lon = -34.755489, 139.295124  # Titiana South Australia
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
# Comparing cumulative rainfall between different years
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
# variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain"]
# summed = "max_temp", "min_temp", "daily_rain"

variables = ["daily_rain", "max_temp", "min_temp"]
summed = "daily_rain"

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = axes.flatten()

for i, variable in enumerate(variables):

    ax = axes[i]  # Get the current subplot

    total = {}
    for year in years:
        date_range = slice(f'{year}-01-01', f'{year}-12-31')
        silo_ts = silo_data[variable].sel(time=date_range)
        silo_pixel1_ts = silo_ts.sel(lat=lat, lon=lon, method='nearest').cumsum(dim='time')
        total[year] = silo_pixel1_ts[-1].item()  # Get the total rainfall for the year
    sorted_years = sorted(total, key=total.get, reverse=True)

    ordered_years = sorted_years if variable in summed else years
    for year in ordered_years:
        date_range = slice(f'{year}-01-01', f'{year}-12-31')
        silo_ts = silo_data[variable].sel(time=date_range)
        silo_pixel1_ts = silo_ts.sel(lat=lat, lon=lon, method='nearest')
        
        if variable in summed:
            silo_pixel1_ts = silo_pixel1_ts.cumsum(dim='time')

        # Convert time to an arbitrary year (2020) so they can be overlayed on the one plot
        time_values = pd.to_datetime(silo_pixel1_ts.time.values)
        normalized_time_values = time_values.map(lambda x: x.replace(year=2020))

        label = f'{year} ({total[year]:.1f}mm)' if variable in summed else year 
        ax.plot(normalized_time_values, silo_pixel1_ts, label=label)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')) 
           
    # Format x-axis to show month names and ensure all months appear
    ax.set_xlabel('Month') 
    ax.set_ylabel(f'{variable}') 
    
    if variable in summed:
        title = f'daily cumulative sum of {variable}'
    else:
        title = f'daily {variable}'
    ax.set_title(title) 
    
    if variable in summed:
        # Moving the legend outside the plot to the right
        ax.legend(title=f'Year (total {variable})', bbox_to_anchor=(1, 1), loc='upper left') 
    else:
        ax.legend(title=f'Year') 



if len(variables) < len(axes):
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# -

