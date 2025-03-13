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

# Find the paddockTS repo on gadi or locally
if os.path.expanduser("~").startswith("/home/"):
    paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/PaddockTS")
else:
    paddockTS_dir = os.path.dirname(os.getcwd())
os.chdir(paddockTS_dir)
from DAESIM_preprocess.ozwald_8day import ozwald_8day, ozwald_8day_abbreviations
from DAESIM_preprocess.ozwald_daily import ozwald_daily, ozwald_daily_abbreviations
from DAESIM_preprocess.silo_daily import silo_daily, silo_abbreviations

# +
# DAESim requirements:
# VPeff (vapour pressure), Uavg (wind), min temperature, max temperature, precipitation, solar radiation
# Soil moisture, runoff, leaf area, growth

# Ozwald doesn't have radiation
# SILO doesn't have wind, soil moisture, runoff, leaf area, GPP
# ANUClim doesn't have wind, soil moisture, runoff, leaf area, GPP
# Terraclim doesn't have leaf area, GPP

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

# Should probably do this earlier during the SILO download
if 'crs' in ds_silo_daily.data_vars:
    ds_silo_daily = ds_silo_daily.drop_vars(['crs'])
ds_silo_daily = ds_silo_daily.rename({"lat": "latitude", "lon": "longitude"})


def aggregate_pixels(ds):
    """Find the median of all pixels for each timepoint, or drop that dimension if it only has one coordinate"""
    dims = ds.dims
    coords = ds.coords
    if "latitude" in dims and "longitude" in dims:
        ds = ds.median(dim=["latitude", "longitude"])
    elif "latitude" in dims:  
        ds = ds.median(dim=["latitude"])
    elif "longitude" in dims: 
        ds = ds.median(dim=["longitude"])
        
    if "latitude" in coords:
        ds = ds.drop_vars(["latitude"])
    if "longitude" in coords:
        ds = ds.drop_vars(["longitude"])
    return ds


ds_silo_daily = aggregate_pixels(ds_silo_daily)
ds_ozwald_8day = aggregate_pixels(ds_ozwald_8day)
ds_ozwald_daily_Pg = aggregate_pixels(ds_ozwald_daily_Pg)
ds_ozwald_daily_Tmax = aggregate_pixels(ds_ozwald_daily_Tmax)
ds_ozwald_daily_Uavg = aggregate_pixels(ds_ozwald_daily_Uavg)

# Even though they have overlapping variables it's fine to merge ozwald and SILO, because the overlapping variables all have different names (e.g. 'Pg' and 'daily_rain')
ds_merged = xr.merge([ds_silo_daily, ds_ozwald_8day, ds_ozwald_daily_Pg, ds_ozwald_daily_Tmax, ds_ozwald_daily_Uavg])

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
