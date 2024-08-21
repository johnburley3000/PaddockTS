# +
# Just downloading enough data to make the rgb and fractional calendar plots

# +
# Project: xe2
# Storage: gdata/+gdata/xe2+gdata/v10+gdata
# Module Directories: /g/data/v10/public/modules/modulefiles
# Modules: dea/20231204
# +
# Standard libraries
import pickle
import sys
import os
import calendar
import datetime
from IPython.core.display import Video

# Dependencies
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(1, '../Tools/')
import datacube
from dea_tools.datahandling import load_ard
from dea_tools.plotting import xr_animation
from dea_tools.plotting import xr_animation

# -
import resource
def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return f"Memory usage: {usage.ru_maxrss / 1024} MB"
print(memory_usage())

# +
# dc = datacube.Datacube(app='Sentinel')
# -


# Filenames
stub = "MILG_1km_all_years_90p"
scratch_dir = "/scratch/xe2/cb8590/"
gdata_dir = "/g/data/xe2/cb8590"
chris_outdir = os.path.join(gdata_dir, "Data/PadSeg/")
filename = os.path.join(chris_outdir, f"{stub}.nc")

# +
# Specify parameters
lat = -34.389042
lon = 148.469499
buffer = 0.033    # 0.01 is 1km in each direction to 2kmx2km total 
stub = "MILG_6km"

start_year = 2010  # This automatically gets the earlist timepoint (late 2015)
end_year = 2030    # This automatically gets the most recent timepoint


# +
def define_query(lat=-34.389042, lon=148.469499, buffer=0.005 , start_year=2020, end_year=2021):
    lat_range = (lat-buffer, lat+buffer)
    lon_range = (lon-buffer, lon+buffer)
    query = {
        'y': lat_range,
        'x': lon_range,
        'time': (start_year, end_year),
        'resolution': (-10, 10),
        'output_crs': 'epsg:6933',
        'group_by': 'solar_day',
        'measurements': ['nbart_red', 'nbart_green', 'nbart_blue', 'nbart_nir_1', 'nbart_swir_2','nbart_swir_3'],
        'min_gooddata':0.9
    }
    return query

query = define_query(lat, lon, buffer, start_year, end_year)


# -

# %%time
def load_and_process_data(dc, query):
    ds = load_ard(
        dc=dc,
        products=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        cloud_mask='s2cloudless',
        **query
    )
    return ds
ds = load_and_process_data(dc, query)


# # Sometimes need to remove these for the netcdf save to work
ds['time'].attrs.pop('units', None)
if 'flags_definition' in ds.attrs:
    ds.attrs.pop('flags_definition')
for var in ds.variables:
    if 'flags_definition' in ds[var].attrs:
        ds[var].attrs.pop('flags_definition')

# %%time
# ds.to_netcdf(filename)


ds = xr.open_dataset(filename)

# +
# Heatmap of available imagery dates
# ds = ds_full

# Create an empty heatmap with the dimensions: years, weeks
dates = pd.to_datetime(ds['time'].values)
years = np.arange(dates.year.min(), dates.year.max() + 1)
weeks = np.arange(1, 53)
heatmap_data = pd.DataFrame(0, index=years, columns=weeks)

# Calculate the week of the year for each date (using date.week has some funny behaviour leading to 53 weeks which I don't like)
weeks_years_dates = [(date.strftime('%W'), date.year, date) for date in dates]
weeks_years_dates = [(1 if wyd[0] == '00' else int(wyd[0]), wyd[1], wyd[2]) for wyd in weeks_years_dates]

# Fill the DataFrame with 0s (no image), 1 (cloudy image), 2 (good image)
threshold = ds['x'].size * ds['y'].size * 0.01
for week_year_date in weeks_years_dates:
    week, year, date = week_year_date
    data_array = ds.sel(time=date)
    nan_count = data_array['nbart_red'].isnull().sum()
    
    if nan_count > threshold:
        heatmap_data.loc[year, week] = 1  
    else:
        heatmap_data.loc[year, week] = 2 
        
# Plotting the heatmap
plt.figure(figsize=(15, 10))
cmap = plt.get_cmap('Greens', 3)  # Use 3 levels of green
plt.imshow(heatmap_data, cmap=cmap, aspect='equal', interpolation='none')

# Change the xticks to use months instead of weeks
month_start_weeks = [pd.Timestamp(f'{years[0]}-{month:02d}-01').week for month in range(1, 13)]
month_labels = [calendar.month_name[month] for month in range(1, 13)]
plt.xticks(ticks=month_start_weeks, labels=month_labels)
plt.yticks(ticks=np.arange(len(years)), labels=years)

# Cloud cover categories
labels = ['< 90%', '90-99%', '> 99%']
cbar = plt.colorbar(ticks=[0.33, 1, 1.66], shrink=0.3)
cbar.ax.set_yticklabels(labels)
cbar.set_label('Clear pixels')

plt.title('Available Sentinel Imagery')
plt.tight_layout()

output = os.path.join(scratch_dir,f"{stub}_cloud_heatmap.png")
plt.savefig(output)
print("Saved:", output)
# -

ds = ds.fillna(0)

# Set the image size to at least 1 (100x100 pixels per image), but at most 5 or else the filesize goes over 100MB
image_size = ds.x.values.shape / 100  
image_size = max(image_size, 1)
image_size = min(image_size, 5)

# +
red = ds['nbart_red']
green = ds['nbart_green']
blue = ds['nbart_blue']

# Using a percent clip with the same default parameters asd dea_tools.plotting.rgb
# Note: to get more photorealistic images, use p_low=0 and p_high = 1, but some images may look very dark.
p_low, p_high = 2, 98
red_clip = np.clip(red, np.percentile(red, p_low), np.percentile(red, p_high))
green_clip = np.clip(green, np.percentile(green, p_low), np.percentile(green, p_high))
blue_clip = np.clip(blue, np.percentile(blue, p_low), np.percentile(blue, p_high))

red_normalized = (red_clip - np.min(red_clip)) / (np.max(red_clip) - np.min(red_clip))
green_normalized = (green_clip - np.min(green_clip)) / (np.max(green_clip) - np.min(green_clip))
blue_normalized = (blue_clip - np.min(blue_clip)) / (np.max(blue_clip) - np.min(blue_clip))


# +
# %%time
# Create a calendar plot, spaced nicely across each year

# Setup the dimensions of the calendar plot
weeks = list(range(1,53))
years = sorted(np.unique([pd.Timestamp(time).year for time in ds.time.values]))
week_year_dict = {date:[week, year] for week, year, date in weeks_years_dates}
year_minimum = min(years)

# Create the subplot grid
image_size = 6
rows = len(years)
cols = len(weeks)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*image_size, rows*image_size))

# Create the labels for the x axis
row_labels = years
month_labels = []
previous_month = None
for week in range(1, 53):
    
    # Find the month
    date = datetime.datetime.strptime(f'2024-W{week}-1', "%Y-W%U-%w").date()
    month = date.strftime("%b")
    
    # Add the month if it's a new month
    if month != previous_month:
        month_labels.append(month)
        previous_month = month
    else:
        month_labels.append("")

# Move the label to the second week of the month, because some years don't start on Monday 
month_labels = [""] + month_labels[:-1]         
        
# Cleaning up the axes
label_fontsize = 9 * size
for row in range(rows):
    for col in range(cols):
        ax = axes[row, col]
        
        # Remove all the ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])  
        ax.set_yticklabels([]) 
        
        # Add row and column labels if on the very left or very bottom
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=label_fontsize, ha='center')
        if row == rows - 1:
            ax.set_xlabel(month_labels[col], fontsize=label_fontsize, ha='center')


# Plot each of the timestamps in the correct location           
for i in range(len(ds.time)):
    time = ds.time[i].values
    
    # Extract the week and year index of this timestamp
    timestamp = pd.Timestamp(time)
    date = str(timestamp)[:10]
    week, year = week_year_dict[timestamp]
    row = year - year_minimum
    col = week - 1
    
    # Extract the normalized RGB image
    r = red_normalized[i]
    g = green_normalized[i]
    b = blue_normalized[i]
    
    rgb_image = np.stack([r, g, b], axis=-1)

    # Plot the image
    ax = axes[row, col]
    ax.imshow(rgb_image, aspect='auto')
    ax.set_title(date, fontsize=8, pad=1)
        
plt.tight_layout(pad=0.2)

output = os.path.join(scratch_dir,f"{stub}_calendar_plot.png")
plt.savefig(output)
print(f"Saved:", output)
# -


# !du -sh {output}

# +
# # %%time
# Video

# Resampling takes a while and adds data that isn't there, but makes the video much more pleasant to watch
ds_weekly = ds.resample(time="1W").interpolate("linear").interpolate_na(dim = 'time', method = 'linear')


output = os.path.join(scratch_dir,f"{stub}_video.mp4")
num_frames = len(ds_weekly.time) # get total length from ds_weekly
xr_animation(ds_weekly, 
             bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
             output_path = output, 
             limit=num_frames)
plt.close()
Video(output, embed=True)
# -


