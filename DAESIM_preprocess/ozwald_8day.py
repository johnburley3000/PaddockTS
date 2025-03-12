# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/8day/catalog.html

# +
# Standard Libraries
import os
import glob

# Dependencies
import numpy as np
import requests
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import create_bbox, scratch_dir
# -

ozwald_8day_abbreviations = {
    "Alb": "Albedo",
    "BS": "Bare Surface",
    "EVI": "Enhanced Vegetation Index",
    "FMC": "Fuel Moisture Content",
    "GPP": "Gross Primary Productivity",
    "LAI": "Leaf Area Index",
    "NDVI": "Normalised Difference Vegetation Index",
    "NPV": "Non Photosynthetic Vegetation",
    "OW": "Open Water",
    "PV": "Photosynthetic Vegetation",
    "Qtot": "Streamflow",
    "SN": "Snow",
    "Ssoil": "Soil profile moisture change"
}


# This function uses the public facing Thredds API, so does not need to be run on NCI
# However it doesn't work in a PBS script from my tests
def ozwald_8day_singleyear_thredds(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2021", stub="Test", tmp_dir=scratch_dir):
    
    # buffer = 0.0000000001    # Using a buffer less than the grid size of 500m (0.005 degrees) gives you a single point
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    # base_url = "https://thredds.nci.org.au"  # This is the new url (dapds00 is supposedly deprecated), but LAI only works with the old url
    base_url = "https://dapds00.nci.org.au"
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
    
    response = requests.get(url)
    filename = os.path.join(tmp_dir, f"{stub}_{var}_{year}.nc")
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    print("Downloaded from Thredds", filename)

    ds = xr.open_dataset(filename, engine='netcdf4')
    return ds


# This function accesses files directly, so is much faster but needs to be run on NCI with access to the ub8 project
def ozwald_8day_singleyear_gdata(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021"):
    """Select the region of interest from the Australia wide NetCDF file"""
    filename = os.path.join("/g/data/ub8/au/OzWALD/8day", var, f"OzWALD.{var}.{year}.nc")

    # OzWald doesn't have 2024 data in this folder yet.
    if not os.path.exists(filename):
        return None
    
    ds = xr.open_dataset(filename)
    bbox = create_bbox(latitude, longitude, buffer)
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))

    # If the region is too small, then just find a single point
    if ds_region[var].shape[1] == 0:
        ds_region = ds.sel(latitude=latitude, longitude=longitude, method="nearest")

    print("Loaded from gdata", filename)
        
    return ds_region


def ozwald_8day_multiyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, years=["2020", "2021"], stub="Test", tmp_dir=scratch_dir, thredds=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = ozwald_8day_singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmp_dir)
        else:
            ds_year = ozwald_8day_singleyear_gdata(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def ozwald_8day(variables=["Ssoil", "GPP"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir=scratch_dir, stub="Test", tmp_dir=scratch_dir, thredds=True):
    """Download 8day variables from OzWald"""
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = ozwald_8day_multiyear(variable, lat, lon, buffer, years, stub, tmp_dir, thredds=thredds)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)
    
    filename = os.path.join(outdir, f'{stub}_ozwald_8day.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat


# %%time
if __name__ == '__main__':  
    ds = ozwald_8day(thredds=False)
    print(ds)

    ds = ozwald_8day(thredds=True)
    print(ds)
