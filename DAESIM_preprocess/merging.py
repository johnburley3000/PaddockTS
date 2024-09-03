# +
# Aim of this notebook is to combine all the datasources into a single xarray

# +
# Standard library
import os
import pickle

# Dependencies
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import box, Polygon
from rasterio.enums import Resampling

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, transform_bbox
from DAESIM_preprocess.topography import pysheds_accumulation, calculate_slope

# -

stubs = {
    "MULL": "Mulloon",
    "CRGM": "Craig Moritz Farm",
    "MILG": "Milgadara",
    "ARBO": "Arboreturm",
    "KOWN": "Kowen Forest",
    "ADAM": "Canowindra"
}

# Filepaths
outdir = os.path.join(gdata_dir, "Data/PadSeg/")
stub = "MILG"

# Sentinel imagery
filename = os.path.join(outdir, f"{stub}_ds2.pkl")
with open(filename, 'rb') as file:
    ds = pickle.load(file)

# Terrain calculations
filename = os.path.join(outdir, f"{stub}_terrain.tif")
grid, dem, fdir, acc = pysheds_accumulation(filename)
slope = calculate_slope(filename)


# +
def add_tiff_band(ds, variable, resampling_method, outdir, stub):
    """Add a new band to the xarray from a tiff file using the given resampling method"""
    filename = os.path.join(outdir, f"{stub}_{variable}.tif")
    array = rxr.open_rasterio(filename)
    reprojected = array.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected.isel(band=0).drop_vars('band')
    return ds

# Add the soil bands
ds = add_tiff_band(ds, "Clay", Resampling.average, outdir, stub)
ds = add_tiff_band(ds, "Silt", Resampling.average, outdir, stub)
ds = add_tiff_band(ds, "Sand", Resampling.average, outdir, stub)
ds = add_tiff_band(ds, "pH_CaCl2", Resampling.average, outdir, stub)
# -

# Add the height bands
ds = add_tiff_band(ds, "terrain", Resampling.average, outdir, stub)
ds = add_tiff_band(ds, "canopy_height", Resampling.max, outdir, stub)


# +
def add_numpy_band(ds, variable, array, affine, resampling_method):
    """Add a new band to the xarray from a numpy array and affine using the given resampling method"""
    da = xr.DataArray(
        array, 
        dims=["y", "x"], 
        attrs={
            "transform": affine,
            "crs": "EPSG:3857"
        }
    )
    da.rio.write_crs("EPSG:3857", inplace=True)
    reprojected = da.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected
    return ds

# Add the terrain bands
ds = add_numpy_band(ds, "slope", slope, grid.affine, Resampling.average)
ds = add_numpy_band(ds, "topographic_index", acc, grid.affine, Resampling.max)
ds = add_numpy_band(ds, "aspect", fdir, grid.affine, Resampling.nearest)
# -

# The resampling often messes up the boundary, so we trim the outside pixels after adding all the resampled bounds
ds_trimmed = ds.isel(
    y=slice(1, -1),
    x=slice(1, -1) 
)

ds_trimmed
