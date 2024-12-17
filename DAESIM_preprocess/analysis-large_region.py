# +
# Sentinel loading examples are here: https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Sentinel2_Surface_Reflectance/
# Sentinel band info is here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/

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

# DEA modules
sys.path.insert(1, '../Tools/')
import datacube
from dea_tools.datahandling import load_ard
from dea_tools.plotting import xr_animation

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir
# -
time = '2020-01-08'
stub = f"MILG_100km"

query = {'y': (-33.8, -34.8),
         'x': (148.0, 149.0),
         'time': ("2019-12-01", "2020-01-31"),
         'resolution': (-10, 10),
         'measurements': ['nbart_red', 'nbart_green', 'nbart_blue', 'nbart_nir_1']
}
query

# %%time
"""Just download the data from sentinel"""
dc = datacube.Datacube(app='Sentinel')
ds = load_ard(
    dc,
    products=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
    cloud_mask='s2cloudless',
    group_by= 'solar_day',
    output_crs= 'epsg:6933',
    min_gooddata=0,
    **query
)


B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))

ds

# +


time = ds.time[0]

for time in ds.time:
    time_string = str(time.dt.date.values)
    ds_time = ds.isel(time=0)

    # Export the EVI
    filename = os.path.join(scratch_dir, f"EVI_{stub}_{time_string}.tif")
    ds_time['EVI'].rio.to_raster(filename)
    print(filename)

    # Export the true colour image
    filename = os.path.join(scratch_dir, f"RGB_{stub}_{time_string}.tif")
    ds_time.attrs = {}
    rgb_stack = ds_time[['nbart_red', 'nbart_green', 'nbart_blue']]
    rgb_stack.rio.to_raster(filename)
    print(filename)
