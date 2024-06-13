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
variables = ["max_temp", "min_temp", "radiation", "vp", "daily_rain"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
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
    ax.set_title(f'SILO {variable} monthly {metric} at centre pixel') \
    
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
# Playing with video creation

# +
import numpy as np
import pandas as pd
import plotly.express as px

# Example data
times = pd.date_range('2023-01-01', periods=10, freq='D')
latitudes = np.linspace(-90, 90, 20)
longitudes = np.linspace(-180, 180, 20)
rainfall_data = np.random.rand(len(times), len(latitudes), len(longitudes))

data = []
for t, time in enumerate(times):
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            data.append([time, lat, lon, rainfall_data[t, i, j]])

df = pd.DataFrame(data, columns=['time', 'lat', 'lon', 'rainfall'])

# Create the plotly figure
fig = px.scatter_geo(df, 
                     lon='lon', 
                     lat='lat', 
                     color='rainfall', 
                     animation_frame='time',
                     color_continuous_scale='Blues',
                     projection='natural earth',
                     title='Time Series Rainfall Data')

fig.update_layout(
    geo=dict(
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="lightblue"
    ),
    coloraxis_colorbar=dict(
        title="Rainfall"
    )
)

# Save the animation as an HTML file
fig.write_html("rainfall_animation.html")
# -


