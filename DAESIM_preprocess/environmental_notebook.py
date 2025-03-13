# +
# This is a replica of Code/environmental that can be run as a jupyter notebook
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
from DAESIM_preprocess.slga_soils import slga_soils, slga_soils_abbrevations
from DAESIM_preprocess.util import scratch_dir

# +
# Input parameters
lat=-37.1856746323413
lon=143.8202752762509
buffer = 0.000001
start_year = "2021"
end_year = "2022"
stub = "DSIM"
# outdir = os.path.join(paddockTS_dir,'data')
# tmpdir = os.path.join(paddockTS_dir,'tmp')
outdir = scratch_dir
tmpdir = scratch_dir
thredds=False

# # Come back to this to check the downloads all work locally as well as on NCI. Probably need to auto-create directories "data" and "tmp"

# -

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

# %%time
variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
slga_soils(variables, lat, lon, buffer, tmpdir, stub, depths)

# +
# Load the tiff files we just downloaded (each should just have a single pixel)
values = []
stub = "Harden"
depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
for variable in variables:
    for depth in depths:
        filename = os.path.join(scratch_dir, f"{stub}_{variable}_{depth}.tif")
        ds = rxr.open_rasterio(filename)
        value = float(ds.isel(band=0, x=0, y=0).values)
        values.append({
            "variable":variable,
            "depth":depth,
            "value":value
        })

# Pivot
df = pd.DataFrame(values)
pivot_df = df.pivot(index='depth', columns='variable', values='value')
pivot_df = pivot_df.reset_index() 

# Sort by depth
df = pivot_df
depth_order = ['5-15cm', '15-30cm', '30-60cm', '60-100cm']
df['depth'] = pd.Categorical(df['depth'], categories=depth_order, ordered=True)
sorted_df = df.sort_values(by='depth')

# Save
sorted_df.to_csv("Harden_Soils_sorted.csv", index=False)

