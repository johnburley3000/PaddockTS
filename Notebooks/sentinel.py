# +
# Just downloading enough data to make the rgb and fractional calendar plots

# +
# Project: xe2
# Storage: gdata/+gdata/xe2+gdata/v10+gdata
# Module Directories: /g/data/v10/public/modules/modulefiles
# Modules: dea/20231204
# -



# +
# Standard libraries
import pickle
import sys
import os

# Dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


sys.path.insert(1, '../Tools/')
import datacube
from dea_tools.datahandling import load_ard
from dea_tools.plotting import rgb
# -

import resource
def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return f"Memory usage: {usage.ru_maxrss / 1024} MB"
print(memory_usage())

dc = datacube.Datacube(app='Sentinel')


# +
# Specify parameters
lat = -34.389042
lon = 148.469499
buffer = 0.005    # 0.01 is 1km in each direction to 2kmx2km total 
stub = "MILG_1km"

start_year = 2010  # This automatically gets the earlist timepoint (late 2015)
end_year = 2030    # This automatically gets the most recent timepoint


# +
def define_query(lat=-34.389042, lon=148.469499, buffer=0.005 , start_year=2020, end_year=2021):
    lat_range = (lat-buffer, lat+buffer)
    lon_range = (lon-buffer, lon+buffer)
    query = {
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'output_crs': 'epsg:6933',
        'group_by': 'solar_day',
        'measurements': ['nbart_red', 'nbart_green', 'nbart_blue', 'nbart_nir_1', 'nbart_swir_2','nbart_swir_3'],
        'min_gooddata':0.5
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


ds

stub = "MILG_1km_all_years"
scratch_dir = "/scratch/xe2/cb8590/"
gdata_dir = "/g/data/xe2/cb8590"
chris_outdir = os.path.join(gdata_dir, "Data/PadSeg/")
filename = os.path.join(chris_outdir, f"{stub}.nc")

# # Sometimes need to remove these for the netcdf save to work
ds['time'].attrs.pop('units', None)
if 'flags_definition' in ds.attrs:
    ds.attrs.pop('flags_definition')
for var in ds.variables:
    if 'flags_definition' in ds[var].attrs:
        ds[var].attrs.pop('flags_definition')

# %%time
ds.to_netcdf(filename)


# !ls /g/data/xe2/cb8590/Data/PadSeg/*.pkl


# +
# %%time
filename = "/g/data/xe2/cb8590/Data/PadSeg/AO_b02_y20-22_ds2.pkl"
with open(filename, 'rb') as handle:
    ds = pickle.load(handle)
    
    

# +
# %%time
# John's calendar plot
# ds_resamp = ds.resample(time="1W").interpolate("linear").interpolate_na(dim = 'time', method = 'linear')

rgb(ds_resamp, 
    bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
    robust = True, 
    size = 4,
    col="time", 
    #col_wrap=36,  # 10-day
    col_wrap=52, # weekly
    savefig_path = path_out+stub+'_calendar_plot.png')


# +
# %%time
# Calendar plot (takes about the same amount of time as a video, but more space)
output = os.path.join(scratch_dir,f"{stub}_calendar_plot.png")
rgb(ds, 
    bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
    size = 1,
    col="time", 
    col_wrap=52,  # 7 is roughly monthly, 52 is roughly yearly (depends on how many get missed)
    savefig_path = output)


# -

# !ls /scratch/xe2/cb8590/MILG_1km_all_years_calendar_plot.png

# !du -sh /scratch/xe2/cb8590/MILG_1km_all_years_calendar_plot.png

# +
# # %%time
# # Video



# # RGB actual time series
# output = os.path.join(scratch_dir,"calendar_plot.mp4")
# num_frames = len(ds.time) # get total length from ds_weekly
# xr_animation(ds, 
#              bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
#              output_path = output, 
#              limit=num_frames)
# plt.close()
# Video(output, embed=True)


# -

heatmap_data.shape

heatmap_data

weeks



# +
# Create an empty heatmap with the dimensions: years, weeks
dates = pd.to_datetime(ds['time'].values)
years = np.arange(dates.year.min(), dates.year.max() + 1)
weeks = np.arange(1, 53)
heatmap_data = pd.DataFrame(0, index=years, columns=weeks)

# Fill the DataFrame with 1s where dates exist
weeks_years_dates = [(date.strftime('%W'), date.year, date) for date in dates]
weeks_years = [(1 if wyd[0] == '00' else int(wyd[0]), wyd[1]) for wyd in weeks_years_dates]
for week_year in weeks_years:
    heatmap_data.loc[week_year[1], week_year[0]] = 1

# Plotting the heatmap
plt.figure(figsize=(15, 10))
plt.imshow(heatmap_data, cmap='Greens', aspect='equal', interpolation='none')

# Setting labels
plt.xlabel('Week')
plt.ylabel('Year')
plt.title('Heatmap of Dates')
plt.xticks(ticks=np.arange(len(weeks)), labels=weeks)
plt.yticks(ticks=np.arange(len(years)), labels=years)
plt.title("Available Sentinel Imagery")

plt.show()

