# +
# This script merges SILO and OzWald data into a csv for DAESim

# +
# I'm using python 3.9
# # !pip install jupyter jupytext xarray pandas scipy cftime
# -

# Standard Libraries
import os

# Dependencies
import pandas as pd
import xarray as xr

os.chdir(os.path.dirname(os.getcwd()))
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

# Input parameters
lat=-37.1856746323413
lon=143.8202752762509
buffer = 0.000001
stub = "DSIM"
start_year = "2021"
end_year = "2022"
outdir = '/scratch/xe2/cb8590'
tmpdir = '/scratch/xe2/cb8590'
thredds=False

# %%time
# Download all the variables we need (notebook version of environmental.py)
ozwald_daily(["Uavg", "VPeff"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
ozwald_daily(["Tmax", "Tmin"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
ozwald_daily(["Pg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)


# %%time
variables = ["Ssoil", "Qtot", "LAI", "GPP"]
ozwald_8day(variables, lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)


# %%time
variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"]
ds_silo_daily = silo_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub)

# +
# Make sure this works for a large area or a small one
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
