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


# Basic tests for quickly checking all the API's
ds = ozwald_daily(variables=['Uavg'], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {'Uavg'}
assert os.path.exists("outdir/TEST_ozwald_daily_Uavg.nc")
assert os.path.exists("outdir/TEST_ozwald_daily_Uavg.png")

ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Ssoil"}
assert os.path.exists("outdir/TEST_ozwald_8day.nc")
assert os.path.exists("outdir/TEST_ozwald_8day.png")

ds = terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="outdir", stub="TEST", tmpdir="tmpdir", tile_level=14, interpolate=True)
assert set(ds.coords) == {'x', 'y'}  
assert set(ds.data_vars) == {'terrain'}
assert os.path.exists("tmpdir/TEST_terrain_original.tif")
assert os.path.exists("outdir/TEST_terrain.tif")

slga_soils(variables=["Clay"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="tmpdir", stub="TEST",  depths=["5-15cm"])
assert os.path.exists("tmpdir/TEST_Clay_5-15cm.tif")

ds = silo_daily(variables=["radiation"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'radiation'}
assert os.path.exists("outdir/TEST_silo_daily.nc")
assert os.path.exists("outdir/TEST_silo_daily.png")


# More comprehensive tests for OzWald: All variables, 3x buffers, all years, with or without netcdf & plotting
ds = ozwald_daily(variables=["Tmax", "Tmin"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Tmax", "Tmin"}

ds = ozwald_daily(variables=["Pg"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Pg"}

ds = ozwald_daily(variables=["Uavg", "VPeff"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_daily(variables=["Ueff", "kTavg", "kTeff"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"Ueff", "kTavg", "kTeff"}

ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2000", end_year="2030", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

if os.path.exists("outdir/TEST_ozwald_daily_Ueff.nc"):
    os.remove("outdir/TEST_ozwald_daily_Ueff.nc")
ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=False, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert not os.path.exists("outdir/TEST_ozwald_daily_Ueff.nc")

if os.path.exists("outdir/TEST_ozwald_daily_Ueff.png"):
    os.remove("outdir/TEST_ozwald_daily_Ueff.png")
ds = ozwald_daily(variables=["Ueff"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=True, save_netcdf=True, plot=False)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert not os.path.exists("TEST_ozwald_daily_Ueff.png")

# Should also test (and handle) larger buffer sizes, and locations outside Australia


# More comprehensive tests for ozwald_8day: All variables, 2x buffers, all years, with or without netcdf & plotting
ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2024", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  

if os.path.exists("outdir/TEST_ozwald_8day.nc"):
    os.remove("outdir/TEST_ozwald_8day.nc")
ds = ozwald_8day(variables=["Ssoil"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=False, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert not os.path.exists("outdir/TEST_ozwald_8day.nc")

ds = ozwald_8day(variables=["BS", "EVI", "FMC", "NPV"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"BS", "EVI", "FMC", "NPV"}

ds = ozwald_8day(variables=["OW", "PV",  "SN"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"OW", "PV", "SN"}

ds = ozwald_8day(variables=["GPP", "LAI", "NDVI", "Ssoil", "Qtot"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
assert set(ds.data_vars) == {"GPP", "LAI", "NDVI", "Ssoil", "Qtot"}


# More comprehensive tests for terrain tiles: 3x buffers, 3x tile levels, with or without interpolation
ds = terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0, outdir="outdir", stub="TEST", tmpdir="tmpdir", tile_level=14, interpolate=True)
assert set(ds.coords) == {'x', 'y'}  
assert set(ds.data_vars) == {'terrain'}

ds = terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0.1, outdir="outdir", stub="TEST", tmpdir="tmpdir", tile_level=9, interpolate=True)
assert set(ds.coords) == {'x', 'y'}  
assert set(ds.data_vars) == {'terrain'}

ds = terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=1, outdir="outdir", stub="TEST", tmpdir="tmpdir", tile_level=4, interpolate=True)
assert set(ds.coords) == {'x', 'y'}  
assert set(ds.data_vars) == {'terrain'}

if os.path.exists("outdir/TEST_terrain.tif"):
    os.remove("outdir/TEST_terrain.tif")
if os.path.exists("tmpdir/TEST_terrain_original.tif"):
    os.remove("tmpdir/TEST_terrain_original.tif")
terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="outdir", stub="TEST", tmpdir="tmpdir", tile_level=14, interpolate=False)
assert not os.path.exists("outdir/TEST_terrain.tif")
assert os.path.exists("tmpdir/TEST_terrain_original.tif")


# More comprehensive tests for slga soils: 2x buffers, all variables, all depths 
slga_soils(variables=["Clay"], lat=-34.3890427, lon=148.469499, buffer=0, outdir="tmpdir", stub="TEST",  depths=["5-15cm"])
assert os.path.exists("tmpdir/TEST_Clay_5-15cm.tif")

slga_soils(variables=["Clay"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="tmpdir", stub="TEST",  depths=["5-15cm", "15-30cm", "30-60cm", "60-100cm"])
assert os.path.exists("tmpdir/TEST_Clay_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Clay_15-30cm.tif")
assert os.path.exists("tmpdir/TEST_Clay_30-60cm.tif")
assert os.path.exists("tmpdir/TEST_Clay_60-100cm.tif")

slga_soils(variables=["Clay", "Silt", "Sand", "pH_CaCl2", "Bulk_Density", "Available_Water_Capacity", "Effective_Cation_Exchange_Capacity",  "Total_Nitrogen", "Total_Phosphorus"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="tmpdir", stub="TEST",  depths=["5-15cm"])
assert os.path.exists("tmpdir/TEST_Clay_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Silt_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Sand_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_pH_CaCl2_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Bulk_Density_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Available_Water_Capacity_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Effective_Cation_Exchange_Capacity_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Total_Nitrogen_5-15cm.tif")
assert os.path.exists("tmpdir/TEST_Total_Phosphorus_5-15cm.tif")


# More comprehensive tests for SILO: multiple buffers, with or without saving netcdf and plotting 
# (not testing multiple variables or years locally because it takes too long)
ds = silo_daily(variables=["radiation"], lat=-34.3890427, lon=148.469499, buffer=0, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=None, save_netcdf=True, plot=True)
assert set(ds.coords) == {'time', 'lat', 'lon'}  
assert set(ds.data_vars) == {'radiation'}

if os.path.exists("outdir/TEST_silo_daily.nc"):
    os.remove("outdir/TEST_silo_daily.nc")
ds = silo_daily(variables=["radiation"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=None, save_netcdf=False, plot=True)
assert not os.path.exists("outdir/TEST_silo_daily.nc")

if os.path.exists("outdir/TEST_silo_daily.png"):
    os.remove("outdir/TEST_silo_daily.png")
ds = silo_daily(variables=["radiation"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2020", outdir="outdir", stub="TEST", tmpdir="tmpdir", thredds=None, save_netcdf=True, plot=False)
assert not os.path.exists("outdir/TEST_silo_daily.png")


# Testing creation of the daesim_forcing csv (Doing this last so the necessary files are all pre-downloaded)
df = daesim_forcing(outdir="outdir", stub="TEST")
assert set(df.columns) == {"Precipitation", "Runoff", "Minimum temperature", "Maximum temperature", "Soil moisture", "Vegetation growth", "Vegetation leaf area", "VPeff",	"Uavg", "SRAD"}
assert os.path.exists("outdir/TEST_DAESim_forcing.csv")

slga_soils(variables=["Clay", "Silt", "Sand", "pH_CaCl2", "Bulk_Density", "Available_Water_Capacity", "Effective_Cation_Exchange_Capacity",  "Total_Nitrogen", "Total_Phosphorus"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="tmpdir", stub="TEST",  depths=["15-30cm"])
slga_soils(variables=["Clay", "Silt", "Sand", "pH_CaCl2", "Bulk_Density", "Available_Water_Capacity", "Effective_Cation_Exchange_Capacity",  "Total_Nitrogen", "Total_Phosphorus"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="tmpdir", stub="TEST",  depths=["30-60cm"])
slga_soils(variables=["Clay", "Silt", "Sand", "pH_CaCl2", "Bulk_Density", "Available_Water_Capacity", "Effective_Cation_Exchange_Capacity",  "Total_Nitrogen", "Total_Phosphorus"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="tmpdir", stub="TEST",  depths=["60-100cm"])

df = daesim_soils(outdir="outdir", stub="TEST", tmpdir="tmpdir")
assert os.path.exists("outdir/TEST_DAESim_Soils.csv")
