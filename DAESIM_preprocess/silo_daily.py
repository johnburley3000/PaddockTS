# +
# Documentation is here: https://www.longpaddock.qld.gov.au/silo/gridded-data/

# +
# Standard Libraries
import os
import glob
import shutil

# Dependencies
import requests
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -

# Find the paddockTS repo on gadi or locally
if os.path.expanduser("~").startswith("/home/"):
    paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/PaddockTS")
else:
    paddockTS_dir = os.path.dirname(os.getcwd())
os.chdir(paddockTS_dir)
from DAESIM_preprocess.util import create_bbox, scratch_dir

# Taken from https://github.com/Sydney-Informatics-Hub/geodata-harvester/blob/main/src/geodata_harvester/getdata_silo.py
silo_abbreviations = {
        "daily_rain": "Daily rainfall, mm",
        "monthly_rain": "Monthly rainfall, mm",
        "max_temp": "Maximum temperature, degrees Celsius",
        "min_temp": "Minimum temperature, degrees Celsius",
        "vp": "Vapour pressure, hPa",
        "vp_deficit": "Vapour pressure deficit, hPa",
        "evap_pan": "Class A pan evaporation, mm",
        "evap_syn": "Synthetic estimate, mm",
        "evap_comb": "Combination: synthetic estimate pre-1970, class A pan 1970 onwards, mm",
        "evap_morton_lake": "Morton's shallow lake evaporation, mm",
        "radiation": "Solar radiation: Solar exposure, consisting of both direct and diffuse components, MJ/m2",
        "rh_tmax": "Relative humidity:	Relative humidity at the time of maximum temperature, %",
        "rh_tmin": "Relative humidity at the time of minimum temperature, %",
        "et_short_crop": "Evapotranspiration FAO564 short crop, mm",
        "et_tall_crop": "ASCE5 tall crop6, mm",
        "et_morton_actual": "Morton's areal actual evapotranspiration, mm",
        "et_morton_potential": "Morton's point potential evapotranspiration, mm",
        "et_morton_wet": "Morton's wet-environment areal potential evapotranspiration over land, mm",
        "mslp": "Mean sea level pressure Mean sea level pressure, hPa",
    }


SILO_FOLDER="/g/data/xe2/datasets/Climate_SILO"


def download_from_SILO(silo_folder=SILO_FOLDER, var="radiation", year="2021"):
    """Download a NetCDF for the whole of Australia, for a given year and variable"""
    # Note: I haven't found a way to download only the region of interest from SILO, hence downloading all of Australia
    silo_baseurl = "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/"
    url = silo_baseurl + var + "/" + str(year) + "." + var + ".nc"
    filename = os.path.join(silo_folder, f"{year}.{var}.nc")

    with requests.get(url, stream=True) as stream:
        with open(filename, "wb") as file:
            shutil.copyfileobj(stream.raw, file)
    print(f"Downloaded {filename}")
    # Takes about 400MB and 5 mins per file


def silo_daily_singleyear(var="radiation", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021", silo_folder=SILO_FOLDER):
    """Select the region of interest from the Australia wide NetCDF file"""
    filename = os.path.join(silo_folder, f"{year}.{var}.nc")
    
    if not os.path.exists(filename):
        print(f"Downloading from SILO: {var} {year} ~400MB, estimated time: 5 mins")
        download_from_SILO(silo_folder, var, year)
    
    ds = xr.open_dataset(filename)
    bbox = create_bbox(latitude, longitude, buffer)
    ds_region = ds.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

    # If the region is too small, then just find a single point
    if ds_region[var].shape[1] == 0:
        ds_region = ds.sel(lat=latitude, lon=longitude, method="nearest")
        
    return ds_region


def silo_daily_multiyear(var="radiation", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], silo_folder=SILO_FOLDER):
    dss = []
    for year in years:
        ds = silo_daily_singleyear(var, latitude, longitude, buffer, year, silo_folder)
        dss.append(ds)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def silo_daily(variables=["radiation", "et_morton_actual"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2021", outdir="data", stub="Test", tmp_dir=scratch_dir, silo_folder=SILO_FOLDER):
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds = silo_daily_multiyear(variable, lat, lon, buffer, years)
        dss.append(ds)
    ds_concat = xr.merge(dss)
    
    filename = os.path.join(outdir, f'{stub}_silo_daily.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
    return ds_concat


# %%time
if __name__ == '__main__':
    ds = silo_daily()
    print(ds)
