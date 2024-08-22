# +
# Planet CLI refernce is here: https://planet-sdk-for-python.readthedocs.io/en/latest/cli/reference.html
# Planet band info is here: https://developers.planet.com/docs/apis/data/sensors/

# +
# Assumes tiff files have already been downloaded using John's bash script

# +
# Standard library
import os
import sys
import json
import pickle
from datetime import datetime
import calendar

# Dependencies
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt

# DEA modules
sys.path.insert(1, '../Tools/')
from dea_tools.plotting import rgb

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, memory_usage
from DAESIM_preprocess.sentinel import time_lapse

# -


print(memory_usage())

# +
base_dir = "/g/data/xe2/datasets/Planet/Farms"
stub = "MILG"
outdir = os.path.join(gdata_dir, "Data/PadSeg")

order_ids = "7f4b89d0-f25b-4743-9e4f-37a45eb66764", "844e2072-2266-40ac-be00-ba5bba69078c" # 8bands


# +
def find_timestamps(base_dir, stub, order_id):
    """Find all the timestamps in a planetscope order folder"""
    timestamps = set()
    dir_name = os.path.join(base_dir, stub, order_id, "PSScene")
    for s in os.listdir(dir_name):
        timestamp = s[:23]
        timestamps.add(timestamp)
    if 'PSScene_collection.json' in timestamps:
        timestamps.remove('PSScene_collection.json')
    timestamps = sorted(timestamps)
    return timestamps

timestamps = find_timestamps(base_dir, stub, order_ids[0])
timestamps[:1]


# +
def load_image(base_dir, stub, order_id, timestamp, rgb_only=True):
    """Load a single planetscope image into an xarray"""
    
    # Open the tiff file
    bands_tiff_prefix = "_3B_AnalyticMS_SR_8b_clip"
    filename = os.path.join(base_dir, stub, order_id, "PSScene", f"{timestamp}{bands_tiff_prefix}.tif")
    da = rxr.open_rasterio(filename)
    
    # Mask the clouds
    udm_suffix="_3B_udm2_clip"
    filemask = os.path.join(base_dir, stub, order_id, "PSScene", f"{timestamp}{udm_suffix}.tif")
    da_udm = rxr.open_rasterio(filemask)
    cloud_mask = da_udm.sel(band=1) # udm band 1 is "Regions of a scene that are free of cloud, haze, cloud shadow and/or snow"
    da_bands = da.where(cloud_mask != 0, other=np.nan)

    # Extract the bands into their own variables to match sentinel
    ds = da_bands.to_dataset(dim='band')
    planetscope_bands = {1: "nbart_coastal_aerosol", 2: 'nbart_blue', 3: 'planet_green_1', 4: 'nbart_green', 5: 'planetscope_yellow', 6: 'nbart_red', 7: 'nbart_red_edge_1', 8: 'nbart_nir_2'}
    ds_named = ds.rename(planetscope_bands)
    
    if rgb_only:
        ds_named = ds_named[['nbart_red', 'nbart_green', 'nbart_blue']]

    return ds_named

ds = load_image(base_dir, stub, order_ids[0], timestamps[0])
print(ds)
rgb(ds)


# +
# %%time
def load_directory(base_dir, stub, order_id, limit=None):
    """Load all the in a single planetscope order folder into a single xarray"""
    
    timestamps = find_timestamps(base_dir, stub, order_id)
    
    if limit:
        timestamps = timestamps[:limit]
    
    dss = []
    for timestamp in timestamps:
        
        # Skip timepoints that failed to download
        bands_tiff_prefix = "_3B_AnalyticMS_SR_8b_clip"
        filename = os.path.join(base_dir, stub, order_id, "PSScene", f"{timestamp}{bands_tiff_prefix}.tif")
        if not os.path.exists(filename):
            continue

        ds = load_image(base_dir, stub, order_id, timestamp)
        time = pd.to_datetime(ds.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
        ds_timed = ds.expand_dims(time=[time])
        dss.append(ds_timed)
        
    combined_ds = xr.concat(dss, dim='time')
    return combined_ds

# ds = load_directory(base_dir, stub, order_ids[0], limit=None)
# print(ds)
# +
# %%time
def load_directories(base_dir, stub, order_ids, limit=None):
    """Load images from multiple directories"""
    dss = []
    for order_id in order_ids:
        ds = load_directory(base_dir, stub, order_id, limit)
        dss.append(ds)    
    combined_ds = xr.concat(dss, dim='time')
    return combined_ds

ds = load_directories(base_dir, stub, order_ids, limit=None)
# -

print(memory_usage())

outdir

# %%time
filename = os.path.join(outdir, f"{stub}_planet.ds")
ds.to_netcdf(filename)


# +
# %%time
def available_imagery_planet(ds, outdir="", stub=""):
    """Create a heatmap showing the available imagery and cloud cover percentage"""    
    # Create an empty heatmap with the dimensions: years, weeks
    dates = pd.to_datetime(ds['time'].values)
    years = np.arange(dates.year.min(), dates.year.max() + 1)
    weeks = np.arange(1, 53)
    heatmap_data = pd.DataFrame(0, index=years, columns=weeks)

    # Calculate the week of the year for each date (using date.week has some funny behaviour leading to 53 weeks which I don't like)
    weeks_years_dates = [(date.strftime('%W'), date.year, date) for date in dates]
    weeks_years_dates = [(1 if wyd[0] == '00' else int(wyd[0]), wyd[1], wyd[2]) for wyd in weeks_years_dates]

    # Find the percent of clear pixels for each timestamp
    total_pixels = ds['x'].size * ds['y'].size
    weeks_years_dates_percents = []
    for week, year, date in weeks_years_dates:
        data_array = ds.sel(time=date)
        nan_count = data_array['nbart_red'].isnull().sum()
        percent_clear = 1 - nan_count/total_pixels
        percent_rounded = percent_clear.values.round(2)
        weeks_years_dates_percents.append((week, year, date, percent_rounded))

    # Find the timestamp with the clearest imagery for each week
    best_pixels = dict() # Mapping of {(week, year): {date, percent_clear_pixels}}
    for week, year, date, percent in weeks_years_dates_percents:
        key = (week, year)
        if key not in best_pixels:
            best_pixels[key] = (date, percent)
            continue
        if percent > best_pixels[key][1]:
            best_pixels[key] = (date, percent)

    # Fill the DataFrame with 0s (no image), 1 (cloudy image), 2 (good image)
    for week, year in best_pixels:
        date, percent_clear = best_pixels[(week, year)]
        data_array = ds.sel(time=date)

        # Assumes pixels covered by cloud are represented by NaN
        if percent_clear > 0.5:
            heatmap_data.loc[year, week] = 1  
        if percent_clear > 0.95:
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
    labels = ['< 50%', '50-95%', '> 95%']
    cbar = plt.colorbar(ticks=[0.33, 1, 1.66], shrink=0.3)
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Clear pixels')

    plt.title('Planetscope Available Imagery')
    plt.tight_layout()

    # Save the image
    filename = os.path.join(outdir, f"{stub}_available_imagery_planet.png")
    plt.savefig(filename)
    print("Saved:", filename)

available_imagery_planet(ds, outdir, stub)


# +
# %%time
def calendar_plot_planet(ds, image_size=1, outdir="", stub=""):
    """Create a calendar plot showing the image with the most clear pixels for each week"""
    
    bands=['nbart_red', 'nbart_green', 'nbart_blue']

    # Copying code from the available imagery heatmap to find the best_pixels
    dates = pd.to_datetime(ds['time'].values)
    years = np.arange(dates.year.min(), dates.year.max() + 1)
    weeks = np.arange(1, 53)
    heatmap_data = pd.DataFrame(0, index=years, columns=weeks)

    # Calculate the week of the year for each date (using date.week has some funny behaviour leading to 53 weeks which I don't like)
    weeks_years_dates = [(date.strftime('%W'), date.year, date) for date in dates]
    weeks_years_dates = [(1 if wyd[0] == '00' else int(wyd[0]), wyd[1], wyd[2]) for wyd in weeks_years_dates]

    # Find the percent of clear pixels for each timestamp
    total_pixels = ds['x'].size * ds['y'].size
    weeks_years_dates_percents = []
    for week, year, date in weeks_years_dates:
        data_array = ds.sel(time=date)
        nan_count = data_array['nbart_red'].isnull().sum()
        percent_clear = 1 - nan_count/total_pixels
        percent_rounded = percent_clear.values.round(2)
        weeks_years_dates_percents.append((week, year, date, percent_rounded))

    # Find the timestamp with the clearest imagery for each week
    best_pixels = dict() # Mapping of {(week, year): {date, percent_clear_pixels}}
    for week, year, date, percent in weeks_years_dates_percents:
        key = (week, year)
        if key not in best_pixels:
            best_pixels[key] = (date, percent)
            continue
        if percent > best_pixels[key][1]:
            best_pixels[key] = (date, percent)

    # Reduce to just the best timestamp per week
    best_timestamps = [best_pixels[key][0] for key in best_pixels]
    ds_best = ds.sel(time=best_timestamps)

    # NaN values mess up the normalization
    ds = ds_best.fillna(0)

    # Flip the image for xr_animation and plt.imshow to work correctly 
    # Note this means that dea_tools.plotting.rgb will now be flipped, I think because it uses the coordinate system
    ds = ds.isel(y=slice(None, None, -1))

    # Percent clip normlization with the same default parameters as dea_tools.plotting.rgb, to make the images look brighter
    p_low, p_high = 2, 98

    red = ds[bands[0]]
    green = ds[bands[1]]
    blue = ds[bands[2]]

    red_clip = np.clip(red, np.percentile(red, p_low), np.percentile(red, p_high))
    green_clip = np.clip(green, np.percentile(green, p_low), np.percentile(green, p_high))
    blue_clip = np.clip(blue, np.percentile(blue, p_low), np.percentile(blue, p_high))

    red_normalized = (red_clip - np.min(red_clip)) / (np.max(red_clip) - np.min(red_clip))
    green_normalized = (green_clip - np.min(green_clip)) / (np.max(green_clip) - np.min(green_clip))
    blue_normalized = (blue_clip - np.min(blue_clip)) / (np.max(blue_clip) - np.min(blue_clip))

    # Setup the dimensions of the calendar plot
    weeks = list(range(1,53))
    years = sorted(np.unique([pd.Timestamp(time).year for time in ds.time.values]))
    dates = pd.to_datetime(ds['time'].values)
    weeks_years_dates = [(date.strftime('%W'), date.year, date) for date in dates]
    weeks_years_dates = [(1 if wyd[0] == '00' else int(wyd[0]), wyd[1], wyd[2]) for wyd in weeks_years_dates]
    week_year_dict = {date:[week, year] for week, year, date in weeks_years_dates}
    year_minimum = min(years)

    # Create the subplot grid
    rows = len(years)
    cols = len(weeks)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*image_size, rows*image_size))

    # Create the labels for the x axis
    row_labels = years
    month_labels = []
    previous_month = None
    for week in range(1, 53):

        # Find the month
        date = datetime.strptime(f'2024-W{week}-1', "%Y-W%U-%w").date()
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
    label_fontsize = 9 * image_size
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

    filename = os.path.join(outdir,f"{stub}_calendar_plot_planet_{image_size}.png")
    plt.savefig(filename)
    print(f"Saved:", filename)   
    
# calendar_plot_planet(ds, 1, outdir, stub)
calendar_plot_planet(ds, 5, outdir, stub)
# -

# %%time
# Flip the image for xr_animation and plt.imshow to work correctly (Note: Don't flip for rgb or merging, because the coordinates handle this)
time_lapse(ds.isel(y=slice(None, None, -1)), interpolate=True, outdir=outdir, stub=f"{stub}_planet")

if __name__ == '__main__':
    print("hi")


