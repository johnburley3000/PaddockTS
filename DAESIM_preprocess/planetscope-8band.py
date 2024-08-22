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

# Dependencies
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr

# DEA modules
sys.path.insert(1, '../Tools/')
from dea_tools.plotting import rgb

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir
from DAESIM_preprocess.sentinel import available_imagery, calendar_plot, time_lapse



# +
orderid = ""
outpath = ""
stub = "MILG"
base_dir = "/g/data/xe2/datasets/Planet/Farms"

order_ids = "7f4b89d0-f25b-4743-9e4f-37a45eb66764", "844e2072-2266-40ac-be00-ba5bba69078c" # 8bands
# order_ids = "5534c295-c405-4e10-8d51-92c7c7c02427", "570f6f33-9471-4dcb-a553-2c97928c53aa" # 3bands

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
timestamps[:5]


# +
def load_image(base_dir, stub, order_id, timestamp):
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
        
        # Some of the downloads failed for MILG 2023. This skips those failed ones.
        filename = os.path.join(base_dir, stub, order_id, "PSScene", f"{timestamp}{bands_tiff_prefix}.tif")
        if not os.path.exists(filename):
            continue

        ds = load_image(base_dir, stub, order_id, timestamp)
        time = pd.to_datetime(ds.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
        ds_timed = ds.expand_dims(time=[time])
        dss.append(ds_timed)
        
    combined_ds = xr.concat(dss, dim='time')
    return combined_ds

ds = load_directory(base_dir, stub, order_ids[1], limit=3)
print(ds)
rgb(ds, col="time")
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

ds = load_directories(base_dir, stub, order_ids, limit=3)
rgb(ds, col='time')
# -

available_imagery(ds)

# +
outdir=""
stub=""

"""Create a heatmap showing the available imagery and cloud cover percentage"""    
# Create an empty heatmap with the dimensions: years, weeks
dates = pd.to_datetime(ds['time'].values)
years = np.arange(dates.year.min(), dates.year.max() + 1)
days = np.arange(1, 365)
heatmap_data = pd.DataFrame(0, index=years, columns=days)

# Calculate the week of the year for each date (using date.week has some funny behaviour leading to 53 weeks which I don't like)
days_years_dates = [(date.strftime('%D'), date.year, date) for date in dates]
days_years_dates = [(1 if dyd[0] == '00' else int(dyd[0]), dyd[1], dyd[2]) for dyd in days_years_dates]

# Fill the DataFrame with 0s (no image), 1 (cloudy image), 2 (good image)
threshold = ds['x'].size * ds['y'].size * 0.01
for days_years_date in days_years_dates:
    day, year, date = days_years_date
    data_array = ds.sel(time=date)

    # Assumes pixels covered by cloud are represented by NaN
    nan_count = data_array['nbart_red'].isnull().sum()
    if nan_count > threshold:
        heatmap_data.loc[year, day] = 1  
    else:
        heatmap_data.loc[year, day] = 2 

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

plt.title('Available Imagery')
plt.tight_layout()

# Save the image
filename = os.path.join(outdir, f"{stub}_available_imagery.png")
plt.savefig(filename)
print("Saved:", filename)
# -

calendar_plot(ds)

# %%time
time_lapse(ds, interpolate=False)

if __name__ == '__main__':

    print("Hi")
