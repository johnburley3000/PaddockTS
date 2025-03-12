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

paddockTS_path = os.path.dirname(os.getcwd())
os.chdir(paddockTS_path)
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

# +

# Input parameters
# lat=-37.1856746323413
# lon=143.8202752762509
# buffer = 0.000001
# start_year = "2021"
# end_year = "2022"
stub = "DSIM"
outdir = os.path.join(paddockTS_path,'data')
tmpdir = os.path.join(paddockTS_path,'tmp')
# thredds=False

# # Come back to this to check the downloads all work locally as well as on NCI. Probably need to auto-create directories "data" and "tmp"

# -

outdir

# +
# # %%time
# # Download all the variables we need (notebook version of environmental.py)
# ozwald_daily(["Uavg", "VPeff"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
# ozwald_daily(["Tmax", "Tmin"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
# ozwald_daily(["Pg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)


# +
# # %%time
# variables = ["Ssoil", "Qtot", "LAI", "GPP"]
# ozwald_8day(variables, lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)


# +
# # %%time
# variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"]
# ds_silo_daily = silo_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub)

# +
# Make sure this works for a large area or a small one
# -
# %%time
ds_silo_daily = xr.open_dataset(os.path.join(outdir, stub+'_silo_daily.nc'))
ds_ozwald_8day = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_8day.nc'))
ds_ozwald_daily_Pg = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_daily_Pg.nc'))
ds_ozwald_daily_Tmax = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_daily_Tmax.nc'))
ds_ozwald_daily_Uavg = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_daily_Uavg.nc'))

ds_silo_daily

# Should probably do this drop earlier during the SILO download
if 'crs' in ds_silo_daily.data_vars:
    ds_silo_daily = ds_silo_daily.drop_vars(['crs'])

ds_silo_daily_median = ds_silo_daily.median(dim=["lat", "lon"])
ds_ozwald_8day_median = ds_ozwald_8day.median(dim=["latitude", "longitude"])
ds_ozwald_daily_Pg_median = ds_ozwald_daily_Pg.median(dim=["latitude", "longitude"])
ds_ozwald_daily_Tmax_median = ds_ozwald_daily_Tmax.median(dim=["latitude", "longitude"])
ds_ozwald_daily_Uavg_median = ds_ozwald_daily_Uavg.median(dim=["latitude", "longitude"])


# Even though they have overlapping variables it's fine to merge ozwald and SILO, because the variables all have different names (e.g. 'Pg' and 'daily_rain')
ds_merged = xr.merge([ds_silo_daily_median, ds_ozwald_8day_median, ds_ozwald_daily_Pg_median, ds_ozwald_daily_Tmax_median, ds_ozwald_daily_Uavg_median])

# +
# Rename the columns to match DAESim_forcing.csv

# Use this dict to primarily use SILO
# abbreviations = {
#     "radiation":"SRAD",  # SILO
#     "daily_rain" : "Precipitation",  # SILO
#     "max_temp" : "Maximum temperature",  # SILO
#     "min_temp" : "Minimum temperature",  # SILO
#     "vp":"VPeff"  # SILO
#     "Uavg":"Uavg"  # OzWald
#     "Ssoil":"Soil moisture",  # OzWald
#     "Qtot":"Runoff",  # OzWald
#     "LAI":"Vegetation leaf area",  # OzWald
#     "GPP":"Vegetation growth",  # OzWald
#     }

# Use this dict to primarily use OzWald
abbreviations = {
    "radiation":"SRAD",  # SILO
    "Pg" : "Precipitation",  # OzWald
    "Tmax" : "Maximum temperature",  # OzWald
    "Tmin" : "Minimum temperature",  # OzWald
    "VPeff":"VPeff",  # OzWald
    "Uavg":"Uavg",  # OzWald
    "Ssoil":"Soil moisture",  # OzWald
    "Qtot":"Runoff",  # OzWald
    "LAI":"Vegetation leaf area",  # OzWald
    "GPP":"Vegetation growth"  # OzWald
    }

df = ds_merged.to_dataframe().reset_index()
df = df.set_index('time')
df.rename(columns=abbreviations, inplace=True)
df.rename_axis("date", inplace=True)

daesim_ordering = ["Precipitation", "Runoff", "Minimum temperature", "Maximum temperature", "Soil moisture", "Vegetation growth", "Vegetation leaf area", "VPeff",	"Uavg", "SRAD"]
df_ordered = df[daesim_ordering] 

filepath = os.path.join(outdir, stub + "_DAESim_forcing.csv")
df.to_csv(filepath)
print(filepath)
