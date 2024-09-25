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
import geopandas as gpd
from shapely.geometry import box, Polygon
from rasterio.enums import Resampling
from rasterio import features
import scipy.ndimage
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, transform_bbox
from DAESIM_preprocess.topography import pysheds_accumulation, calculate_slope
from DAESIM_preprocess.silo_daily import merge_ozwald_silo, resample_weekly, visualise_water, visualise_temp
# -

stubs = {
    # "MULL": "Mulloon",
    # "CRGM": "Craig Moritz Farm",
    # "MILG": "Milgadara",
    # "ARBO": "Arboreturm",
    # "KOWN": "Kowen Forest",
    "ADAM": "Canowindra",
    "LCHV": "Lachlan Valley"
}
outdir = os.path.join(gdata_dir, "Data/PadSeg/")


def add_tiff_band(ds, variable, resampling_method, outdir, stub):
    """Add a new band to the xarray from a tiff file using the given resampling method"""
    filename = os.path.join(outdir, f"{stub}_{variable}.tif")
    array = rxr.open_rasterio(filename)
    reprojected = array.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected.isel(band=0).drop_vars('band')
    return ds


# +
# %%time

# Load the satellite imagery
# stub = "LCHV"
for i, stub in enumerate(stubs):
    print(f"Working on stub {i}/{len(stubs)}: {stub}")
    
    filename = os.path.join(outdir, f"{stub}_ds2.pkl")
    with open(filename, 'rb') as file:
        ds = pickle.load(file)
    
    # Add the canopy height
    ds = add_tiff_band(ds, "canopy_height", Resampling.max, outdir, stub)
    ds_trimmed = ds.isel(
        y=slice(1, -1),
        x=slice(1, -1) 
    )
    ds_original = ds.copy()
    ds = ds_trimmed

    # Save the resampled canopy height to file
    filename = os.path.join(scratch_dir, f'{stub}_canopy_height_resampled.tif')
    ds["canopy_height"].rio.to_raster(filename)
    print(filename)
    
    # # Better shelterscore showing the number of trees 
    # pixel_size = 10  # metres
    # distance = 20  # This corresponds to a 200m radius if the pixel size is 10m
    
    # # Classify anything with a height greater than 1 as a tree
    # tree_threshold = 1
    # tree_mask = ds['canopy_height'] >= tree_threshold
    
    # # # Find the pixels adjacent to trees
    # structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
    # adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)
    
    # # Create a circular kernel to determine the distance from a tree for each pixel
    # y, x = np.ogrid[-distance:distance+1, -distance:distance+1]
    # kernel = x**2 + y**2 <= distance**2
    # kernel = kernel.astype(float)
    # shelter_score = scipy.ndimage.convolve(tree_mask.astype(float), kernel, mode='constant', cval=0.0)
    
    # # Mask out trees and adjacent pixels
    # shelter_score[np.where(adjacent_mask)] = np.nan
    
    # # Add the shelter_score to the xarray
    # shelter_score_da = xr.DataArray(
    #     shelter_score, 
    #     dims=("y", "x"),  
    #     coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
    #     name="shelter_score" 
    # )
    # ds["num_trees_200m"] = shelter_score_da
    
    # # Save shelterscore tiff file
    # filename = os.path.join(scratch_dir, f'{stub}_num_trees_200m.tif')
    # ds["num_trees_200m"].rio.to_raster(filename)
    # print(filename)
    
    # # Calculate the productivity score
    # time = '2020-01-01'
    # ndvi = ds.sel(time=time, method='nearest')['NDVI']
    # productivity_score1 = ndvi.where(~adjacent_mask)
    # s = ds['num_trees_200m'].values
    
    # # Flatten the arrays for plotting
    # y = productivity_score1.values.flatten()
    # y_values = y[~np.isnan(y)]   # Remove all pixels that are trees or adjacent to trees
    # x = s.flatten()
    # x_values = x[~np.isnan(y)]   # Match the shape of the x_values
    
    # # Visualise the productivity of just the sheltered pixels
    # sheltered_productivities = productivity_score1.values
    # sheltered_productivities_da = xr.DataArray(
    #     sheltered_productivities, 
    #     dims=("y", "x"),  
    #     coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
    #     name="shelter_score" 
    # )
    # ds["sheltered_productivities"] = sheltered_productivities_da
    
    # # Save productivity score tiff file
    # filename = os.path.join(scratch_dir, f'{stub}_sheltered_productivities_2020-01-01.tif')
    # ds["sheltered_productivities"].rio.to_raster(filename)
    # print(filename)
# -






