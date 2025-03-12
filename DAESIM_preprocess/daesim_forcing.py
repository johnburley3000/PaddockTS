# +
# This script merges SILO and OzWald data into a csv for DAESim
# -

# I'm using python 3.9
# !pip install jupyter jupytext xarray pandas scipy cftime

# Standard Libraries
import os

# Dependencies
import pandas as pd
import xarray as xr

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.ozwald_8day import ozwald_8day, ozwald_8day_abbreviations
from DAESIM_preprocess.ozwald_daily import ozwald_daily, ozwald_daily_abbreviations
from DAESIM_preprocess.silo_daily import silo_daily, silo_abbreviations

# +
# DAESim requirements:
# VPeff (vapour pressure), Uavg (wind), min temperature, max temperature, precipitation, solar radiation
# Soil moisture, runoff, leaf area, growth

# Ozwald has everything except radiation
# SILO has everything except wind, soil moisture, runoff, leaf area, growth
# ANUClim has everything except wind, soil moisture, runoff, leaf area, growth
# Terraclim has everything except leaf area, growth
# -

# Specify the location and years
latitude = -34.52194
longitude=148.30472
buffer = 0.0000000001 # Single point
start_year = 2000
end_year = 2019



# +
# %%time  

# Some of the OzWald variables use different grids, so we load separate xarrays and merge
ds_ozwald_daily1 = ozwald_daily(["VPeff", "Uavg"], latitude, longitude, buffer, start_year, end_year)
ds_ozwald_daily2 = ozwald_daily(["Tmax", "Tmin"], latitude, longitude, buffer, start_year, end_year)
ds_ozwald_daily3 = ozwald_daily(["Pg"], latitude, longitude, buffer, start_year, end_year)

# Took 1 minute (15 seconds per variable per year)
# -

ds_ozwald_daily3['Pg'].plot()

# +
# %%time  

# Fetch 8day variables from ozwald
ds_ozwald_8day = ozwald_8day(["Ssoil", "Qtot", "LAI", "GPP"], latitude, longitude, buffer, start_year, end_year)
ds_ozwald_8day

# Took 11 seconds (1 second per variable per year)

# +
# # %%time
# ds_silo_daily = silo_daily(["radiation", "et_morton_actual", "et_morton_potential", "et_short_crop", "et_tall_crop"], latitude, longitude, buffer, start_year, end_year)
ds_silo_daily = silo_daily(["radiation"], latitude, longitude, buffer, start_year, end_year)
ds_silo_daily

# Took about 10 secs (because its pre-downloaded to gdata)
# -

# Remove the coordinates before merging, since we are looking at a single point location
ds1 = ds_ozwald_daily1.max(dim=['latitude','longitude'])
ds2 = ds_ozwald_daily2.max(dim=['latitude','longitude'])
ds3 = ds_ozwald_daily3.max(dim=['latitude','longitude'])
ds4 = ds_ozwald_8day.drop_vars(['latitude', 'longitude'])
ds5 = ds_silo_daily.drop_vars(['lat', 'lon'])

# Combine the datasets along the 'time' dimension
ds_merged = xr.merge([ds1, ds2, ds3, ds4, ds5])

df = ds_merged.to_dataframe().reset_index()
df = df.drop(columns=["crs"])
df = df.set_index('time')

# Rename the columns to match DAESim_forcing.csv
abbreviations = {
    "Pg" : "Precipitation",
    "Tmax" : "Maximum temperature",
    "Tmin" : "Minimum temperature",
    "Ssoil":"Soil moisture",
    "Qtot":"Runoff",
    "LAI":"Vegetation leaf area",
    "GPP":"Vegetation growth",
    "radiation":"SRAD",
    }
df.rename(columns=abbreviations, inplace=True)
df.rename_axis("date", inplace=True)
df_ordered = df[["Precipitation", "Runoff", "Minimum temperature", "Maximum temperature", "Soil moisture", "Vegetation growth", "Vegetation leaf area", "VPeff",	"Uavg", "SRAD"]] 
df.to_csv("DAESim_forcing_Harden_2000-2019.csv")
df.head()
