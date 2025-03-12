# +
# Documentation is here: https://www.longpaddock.qld.gov.au/silo/gridded-data/

# +
# Standard Libraries
import os
import glob

# Dependencies
import requests
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.getcwd()))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, create_bbox
# -

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


silo_path = "/g/data/xe2/datasets/Climate_SILO"

# I used this script to download data from SILO into gdata
def download_from_SILO():
    from geodata_harvester.getdata_silo import download_file
    
    silo_baseurl = "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/"
    layernames = ["radiation", "et_morton_actual", "et_morton_potential", "et_short_crop", "et_tall_crop"]
    years = ["2017","2018","2019","2020", "2021", "2022", "2023", "2024"]
    outpath = os.path.join(gdata_dir, "SILO")
    for layername in layernames:
        for year in years:
            url = silo_baseurl + layername + "/" + str(year) + "." + layername + ".nc"
            download_file(url, layername, year, outpath)
            print("Downloaded:", os.path.join(outpath, f"{year}.{layername}.nc"))
            
    
    # Takes about 5 mins per file, so about 40 mins per variable, so about 3 hours for these 5 variables
    # Uses about 400MB per file, so about 3GB per variable, or 15GB for these 5 variables
def silo_daily_singleyear(var="radiation", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021"):
    """Select the region of interest from the Australia wide NetCDF file"""
    filename = os.path.join(silo_path, f"{year}.{var}.nc")
    ds = xr.open_dataset(filename)
    bbox = create_bbox(latitude, longitude, buffer)
    ds_region = ds.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

    # If the region is too small, then just find a single point
    if ds_region[var].shape[1] == 0:
        ds_region = ds.sel(lat=latitude, lon=longitude, method="nearest")
        
    return ds_region


def silo_daily_multiyear(var="radiation", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"]):
    dss = []
    for year in years:
        ds = silo_daily_singleyear(var, latitude, longitude, buffer, year)
        dss.append(ds)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def silo_daily(variables=["radiation", "et_morton_actual"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2021", outdir="", stub=""):
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


def merge_ozwald_silo(ds_ozwald, ds_silo):
    """Combine soil moisture data from ozwald with temp, rainfall, evaporation from SILO"""

    # Find the maximum and minimum soil moisture to see some variation
    ssoil_sum = ds_ozwald['Ssoil'].sum(dim='time')
    
    max_row = int(ssoil_sum.argmax() // ssoil_sum.shape[0])
    max_col = int(ssoil_sum.argmax() % ssoil_sum.shape[0])
    min_row = int(ssoil_sum.argmin() // ssoil_sum.shape[0])
    min_col = int(ssoil_sum.argmin() % ssoil_sum.shape[0])
    
    ssoil_maxpoint = ds_ozwald.isel(longitude=max_row, latitude=max_col)
    ssoil_minpoint = ds_ozwald.isel(longitude=min_row, latitude=min_col)
    
    ssoil_maxpoint = ssoil_maxpoint.rename({'Ssoil': 'Maximum Soil Moisture'})
    ssoil_minpoint = ssoil_minpoint.rename({'Ssoil': 'Minimum Soil Moisture'})
    
    if (ds_silo['lat'].values.size == 1 and ds_silo['lon'].values.size == 1):
        ds_silo_point = ds_silo
    # Making the assumption that the four 5km SILO pixels are similar, so just choosing the first one. Later will want to be more precise using ANU Climate at 1km resolution
    else:
        ds_silo_point = ds_silo.isel(lat=0, lon=0)
    
    # Combine the datasets along the time dimension
    ds_merged = xr.merge([ds_silo_point, ssoil_maxpoint, ssoil_minpoint], compat='override')
    df = ds_merged.to_dataframe().reset_index()
    df = df.drop(columns=["lat", "lon", "crs", "latitude", "longitude"])
    df = df.set_index('time')
    
    # Rename the columns 
    abbreviations = {
        "daily_rain" : "Rainfall",
        "max_temp" : "Maximum temperature",
        "min_temp" : "Minimum temperature",
        "et_morton_actual": "Actual Evapotranspiration",
        "et_morton_potential":"Potential Evapotranspiration"
        }
    df.rename(columns=abbreviations, inplace=True)
    df.rename_axis("date", inplace=True)

    return df

def resample_weekly(df):
    weekly_df = df.resample('W').agg({
        "Rainfall":"sum",
        "Maximum temperature": "mean",
        "Minimum temperature": "mean",
        "Maximum Soil Moisture": "mean",
        "Minimum Soil Moisture": "mean",
        "Actual Evapotranspiration": "sum",
        "Potential Evapotranspiration": "sum"
    })
    weekly_df = weekly_df.interpolate('linear')
    
    # We don't have soil moisture data for 2024 right now
    weekly_df.loc[weekly_df.index > '2024-01-01', 'Maximum Soil Moisture'] = np.nan
    weekly_df.loc[weekly_df.index > '2024-01-01', 'Minimum Soil Moisture'] = np.nan

    return weekly_df

def visualise_water(df, outdir=scratch_dir, stub="Test"):
    """Create a plot comparing Rainfall, evapotranspiration, and soil moisture"""
    plt.figure(figsize=(50, 20))
    
    # Plot the data
    rainfall_plot = plt.bar(df.index, df['Rainfall'], color='skyblue', width=5)
    et_actual_plot = plt.bar(df.index, df['Potential Evapotranspiration'], color='orange')
    et_potential_plot = plt.plot(df.index, df['Actual Evapotranspiration'], color='green')
    moisture_max_plot = plt.plot(df.index, df['Maximum Soil Moisture']/10, color='darkorchid')
    moisture_min_plot = plt.plot(df.index, df['Minimum Soil Moisture']/10, color='blue')
    
    # Adjust the size of the tick labels on the x-axis and y-axis
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    
    # Reorder the legend items
    handles = [rainfall_plot, et_actual_plot[0], et_potential_plot[0], moisture_max_plot[0], moisture_min_plot[0]]
    labels = ['Total Rainfall (mm)', "Potential Evapotranspiration (mm)", "Actual Evapotranspiration (mm)", "Maximum Soil Moisture (mm/10)", "Minimum Soil Moisture (mm/10)"]
    plt.legend(handles=handles, labels=labels, fontsize=30, loc='upper left')
    plt.title("Weather", fontsize=50)
    plt.tight_layout()
    
    filename = os.path.join(outdir, f"{stub}_weather.png")
    plt.savefig(filename)
    print("Saved", filename)
    plt.show()

def visualise_temp(df, outpath=scratch_dir, stub="Test"):
    """Create a plot showing min and max temp over time"""
    
    plt.figure(figsize=(50, 10))
    maxtemp_plot = plt.plot(df.index, df["Maximum temperature"], color='red')
    mintemp_plot = plt.plot(df.index, df["Minimum temperature"], color='blue')
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    handles = [maxtemp_plot[0], mintemp_plot[0]]
    labels = ['Maximum Temperature (°C)', 'Minimum Temperature (°C)']
    plt.legend(handles=handles, labels=labels, fontsize=30, loc='upper left')
    plt.title("Temperature", fontsize=50)
    plt.tight_layout()
    
    filename = os.path.join(outpath, f"{stub}_temperature.png")
    plt.savefig(filename)
    print("Saved", filename)
    plt.show()

# %%time
if __name__ == '__main__':
    ds = silo_daily()
    print(ds)
