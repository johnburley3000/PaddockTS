# Testing/demo-ing all of the main functions in DAESim_preprocess

# Change directory to this repo - this should work on gadi or locally via python or jupyter.
# Unfortunately, this needs to be in all files that can be run directly & use local imports.
import os, sys
repo_name = "PaddockTS"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)

from DAESIM_preprocess.terrain_tiles import terrain_tiles
from DAESIM_preprocess.slga_soils import slga_soils
from DAESIM_preprocess.ozwald_8day import ozwald_8day
from DAESIM_preprocess.ozwald_daily import ozwald_daily
from DAESIM_preprocess.silo_daily import silo_daily
from DAESIM_preprocess.daesim_forcing import daesim_forcing, daesim_soils

# Create a tmpdir and outdir in this repo for testing
if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')


# Basic tests
ds = ozwald_daily(variables=['Uavg'], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {'Uavg'}
assert os.path.exists("outdir/TEST_ozwald_daily_Uavg.nc")
assert os.path.exists("outdir/TEST_ozwald_daily_Uavg.png")

ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Ssoil"}
assert os.path.exists("outdir/TEST_ozwald_8day.nc")
assert os.path.exists("outdir/TEST_ozwald_8day.png")

ds = silo_daily(variables=["radiation"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="/g/data/xe2/datasets/Climate_SILO", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'radiation'}
assert os.path.exists("outdir/TEST_silo_daily.nc")
assert os.path.exists("outdir/TEST_silo_daily.png")


# More comprehensive tests for OzWald daily: All variables, 3x buffers, all years, with or without netcdf & plotting
ds = ozwald_daily(variables=["Tmax", "Tmin"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Tmax", "Tmin"}

ds = ozwald_daily(variables=["Pg"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Pg"}

ds = ozwald_daily(variables=["Uavg", "VPeff"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_daily(variables=["Ueff", "kTavg", "kTeff"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Ueff", "kTavg", "kTeff"}

ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2000", end_year="2030", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

if os.path.exists("outdir/TEST_ozwald_daily_Ueff.nc"):
    os.remove("outdir/TEST_ozwald_daily_Ueff.nc")
ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=False, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert not os.path.exists("outdir/TEST_ozwald_daily_Ueff.nc")

if os.path.exists("outdir/TEST_ozwald_daily_Ueff.png"):
    os.remove("outdir/TEST_ozwald_daily_Ueff.png")
ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=False, save_netcdf=True, plot=False)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert not os.path.exists("TEST_ozwald_daily_Ueff.png")

# Should also test (and handle) larger buffer sizes, and locations outside Australia


# More comprehensive tests for ozwald_8day: All variables, 2x buffers, all years, with or without netcdf & plotting
ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir="outdir", stub="TEST", tmpdir=None, thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2000", end_year="2030", outdir="outdir", stub="TEST", tmpdir=None, thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

if os.path.exists("outdir/TEST_ozwald_8day.nc"):
    os.remove("outdir/TEST_ozwald_8day.nc")
ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=False, save_netcdf=False, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert not os.path.exists("outdir/TEST_ozwald_8day.nc")

ds = ozwald_8day(variables=["BS", "EVI", "FMC", "GPP", "LAI", "NDVI", "NPV", "OW", "PV", "Qtot", "SN", "Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=False, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"BS", "EVI", "FMC", "GPP", "LAI", "NDVI", "NPV", "OW", "PV", "Qtot", "SN", "Ssoil"}


# More comprehensive tests for SILO: all variables, multiple years
ds = silo_daily(variables=["min_temp"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2000", end_year="2030", outdir="outdir", stub="TEST", tmpdir="/g/data/xe2/datasets/Climate_SILO", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'min_temp'}

ds = silo_daily(variables=["monthly_rain"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="/g/data/xe2/datasets/Climate_SILO", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'min_monthly_raintemp'}

ds = silo_daily(variables=['daily_rain', 'min_temp', "max_temp", "et_morton_actual", "et_morton_potential"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="/g/data/xe2/datasets/Climate_SILO", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'daily_rain', 'min_temp', "max_temp", "et_morton_actual", "et_morton_potential"}

ds = silo_daily(variables=['daily_rain', 'min_temp', "max_temp", "et_morton_actual", "et_morton_potential"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="/g/data/xe2/datasets/Climate_SILO", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'daily_rain', 'min_temp', "max_temp", "et_morton_actual", "et_morton_potential"}


silo_abbreviations = {

        "vp": "Vapour pressure, hPa",
        "vp_deficit": "Vapour pressure deficit, hPa",
        "evap_pan": "Class A pan evaporation, mm",
        "evap_syn": "Synthetic estimate, mm",
        "evap_comb": "Combination: synthetic estimate pre-1970, class A pan 1970 onwards, mm",
        "evap_morton_lake": "Morton's shallow lake evaporation, mm",
        "rh_tmax": "Relative humidity:	Relative humidity at the time of maximum temperature, %",
        "rh_tmin": "Relative humidity at the time of minimum temperature, %",
        "et_short_crop": "Evapotranspiration FAO564 short crop, mm",
        "et_tall_crop": "ASCE5 tall crop6, mm",
        "et_morton_wet": "Morton's wet-environment areal potential evapotranspiration over land, mm",
        "mslp": "Mean sea level pressure Mean sea level pressure, hPa",
    }