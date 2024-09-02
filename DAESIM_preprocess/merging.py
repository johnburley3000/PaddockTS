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

# +
# Canopy height
filename = os.path.join(outdir, f"{stub}_canopy_height.tif")
canopy_height = rxr.open_rasterio(filename)

# We might not need any of this cropping since the previous script should have gathered data for the exact region anyway.
# Convert the satellite imagery bbox (EPSG:6933) to match the canopy height coordinates (EPSG:3857)
min_lat = ds.y.min().item()
max_lat = ds.y.max().item()
min_lon = ds.x.min().item()
max_lon = ds.x.max().item()
bbox = [min_lat, min_lon, max_lat, max_lon]
bbox_3857 = transform_bbox(bbox, inputEPSG="EPSG:6933", outputEPSG="EPSG:3857")
roi_coords_3857 = box(*bbox_3857)
roi_polygon_3857 = Polygon(roi_coords_3857)

# Clip the canopy height to the region of interest
cropped_canopy_height = canopy_height.rio.clip([roi_polygon_3857])

# Rescale to canopy height to match the satellite imagery, taking the maximum canopy height for each pixel
canopy_height_reprojected = cropped_canopy_height.rio.reproject_match(ds, resampling=Resampling.max)

# The resampling on the boundary sometimes doesn't work. We should remove all cells on the edge to fix this.
canopy_height_reprojected[np.where(canopy_height_reprojected == 255)] = 0

# Save the reprojected canopy height
filename = os.path.join(outdir, f"{stub}_canopy_height_reprojected.tif")
canopy_height_reprojected.rio.to_raster(filename)
print("Saved:", filename)

# Attach the canopy height to the satellite imagery xarray
canopy_height_band = canopy_height_reprojected.isel(band=0)
ds['canopy_height'] = canopy_height_band
# -

# Terrain
filename = os.path.join(outdir, f"{stub}_terrain.tif")
grid, dem, fdir, acc = pysheds_accumulation(filename)
slope = calculate_slope(filename)

# +
# Elevation
elevation = rxr.open_rasterio(filename)

# Clip the variable to the region of interest
cropped = elevation.rio.clip([roi_polygon_3857])

# The resampling on the boundary sometimes doesn't work. We should remove all cells on the edge to fix this.
reprojected = cropped.rio.reproject_match(ds, resampling=Resampling.average)

# Attach to the satellite imagery xarray
band = reprojected.isel(band=0)
ds['elevation'] = band

# +
# Accumulation
acc_da = xr.DataArray(
    acc, 
    dims=["y", "x"], 
    attrs={
        "transform": grid.affine,
        "crs": "EPSG:3857"
    }
)
acc_da.rio.write_crs("EPSG:3857", inplace=True)

# Using Resampling.max for accumulation and canopy height, but average for everything else
reprojected = acc_da.rio.reproject_match(ds, resampling=Resampling.max)
ds['acc'] = reprojected

# +
# Aspect
da = xr.DataArray(
    fdir, 
    dims=["y", "x"], 
    attrs={
        "transform": grid.affine,
        "crs": "EPSG:3857"
    }
)
da.rio.write_crs("EPSG:3857", inplace=True)

# Using Resampling.nearest for aspect
reprojected = da.rio.reproject_match(ds, resampling=Resampling.nearest)
ds['aspect'] = reprojected
# -

# Slope
da = xr.DataArray(
    slope, 
    dims=["y", "x"], 
    attrs={
        "transform": grid.affine,
        "crs": "EPSG:3857"
    }
)
da.rio.write_crs("EPSG:3857", inplace=True)
reprojected = da.rio.reproject_match(ds, resampling=Resampling.average) # Using Resampling.average for slope
ds['slope'] = reprojected

# Clay
filename = os.path.join(outdir, f"{stub}_Clay.tif")
array = rxr.open_rasterio(filename)
reprojected = array.rio.reproject_match(ds, resampling=Resampling.average)
ds['Clay'] = reprojected

# Silt
filename = os.path.join(outdir, f"{stub}_Silt.tif")
array = rxr.open_rasterio(filename)
reprojected = array.rio.reproject_match(ds, resampling=Resampling.average)
ds['Silt'] = reprojected

ds['Silt'].plot()

# Sand
variable = "Sand"
filename = os.path.join(outdir, f"{stub}_{variable}.tif")
array = rxr.open_rasterio(filename)
reprojected = array.rio.reproject_match(ds, resampling=Resampling.average)
ds[variable] = reprojected

ds[variable].plot()

# pH
variable = "Sand"
filename = os.path.join(outdir, f"{stub}_{variable}.tif")
array = rxr.open_rasterio(filename)
reprojected = array.rio.reproject_match(ds, resampling=Resampling.average)
ds[variable] = reprojected
