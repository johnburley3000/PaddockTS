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
    "ADAM": "Canowindra",
    "LCHV": "Lachlan Valley"
}

# Filepaths
outdir = os.path.join(gdata_dir, "Data/PadSeg/")
stub = "MILG"

# %%time
# Sentinel imagery
filename = os.path.join(outdir, f"{stub}_ds2.pkl")
with open(filename, 'rb') as file:
    ds = pickle.load(file)

# %%time
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

ds_trimmed['NDVI'].isel(time=0).plot()

ds_original = ds.copy()
ds = ds_trimmed

# +
# Better shelterscore showing the number of trees 
pixel_size = 10  # metres
distance = 20  # This corresponds to a 200m radius if the pixel size is 10m

# Classify anything with a height greater than 1 as a tree
tree_threshold = 1
tree_mask = ds['canopy_height'] >= tree_threshold

# # Find the pixels adjacent to trees
structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

# Create a circular kernel to determine the distance from a tree for each pixel
y, x = np.ogrid[-distance:distance+1, -distance:distance+1]
kernel = x**2 + y**2 <= distance**2
kernel = kernel.astype(float)
shelter_score = scipy.ndimage.convolve(tree_mask.astype(float), kernel, mode='constant', cval=0.0)

# Mask out trees and adjacent pixels
shelter_score[np.where(adjacent_mask)] = np.nan

# Add the shelter_score to the xarray
shelter_score_da = xr.DataArray(
    shelter_score, 
    dims=("y", "x"),  
    coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
    name="shelter_score" 
)
ds["num_trees_200m"] = shelter_score_da

# +
# # Sanity check of linear regression on the manually created paddocks with absolutely no trees
# filename = os.path.join(gdata_dir,'lachlan_valley_notrees.geojson')
# pol = gpd.read_file(filename)

# # Change from multipolygon to polygon, because I created the layer with the wrong type in QGIS
# def convert_multipolygon_to_polygon(geometry):
#     return geometry.union(geometry)
# pol['geometry'] = pol['geometry'].apply(convert_multipolygon_to_polygon)

# # Create a mask from the geometries
# xarray_dataset = ds
# gdf = pol.to_crs(xarray_dataset.crs) 
# transform = xarray_dataset.rio.transform()
# out_shape = (xarray_dataset.dims['y'], xarray_dataset.dims['x'])
# geometries = gdf.geometry
# paddock_mask = features.geometry_mask(
#     [geom for geom in geometries],
#     transform=transform,
#     invert=True,
#     out_shape=out_shape
# )

# # Some of the pixels in my manually selected area are in the mask of tree + buffer pixels
# paddock_mask = paddock_mask & ~adjacent_mask
# plt.imshow(paddock_mask)

# # Calculate the productivity score
# # time = '2020-07-23'
# # ndvi = ds.sel(time=time, method='nearest')['NDVI']
# time = 'all years sum'
# ndvi = ds['NDVI'].sum(dim='time')
# productivity_score1 = ndvi.where(paddock_mask)
# s = ds['num_trees_200m'].where(paddock_mask)

# # Flatten the arrays for plotting
# x = productivity_score1.values.flatten()
# x_values = x[~np.isnan(x)]   # Remove all pixels that are trees or adjacent to trees
# y = s.values.flatten()
# y_values = y[~np.isnan(y)]   # Match the shape of the x_values

# # Plot
# plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
# plt.ylabel('NDVI', fontsize=12)
# pixel_size = 10
# plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
# plt.title("Manual Polygons with no trees" + ": " + str(time)[:10], fontsize=14)
# plt.show()

# res = stats.linregress(y_values, x_values)
# print(f"R-squared: {res.rvalue**2:.6f}")
# print(f"Slope: {res.slope**2:.12f}")
# plt.plot(y_values, x_values, 'o', label='original data')
# plt.plot(y_values, res.intercept + res.slope*y_values, 'r', label='fitted line')
# plt.legend()
# plt.show()

# +
time = 'all years sum'
ndvi = ds['NDVI'].sum(dim='time')
productivity_score1 = ndvi.where(~adjacent_mask)
s = ds['num_trees_200m'].where(~adjacent_mask)

# Flatten the arrays for plotting
x = productivity_score1.values.flatten()
x_values = x[~np.isnan(x)]   # Remove all pixels that are trees or adjacent to trees
y = s.values.flatten()
y_values = y[~np.isnan(y)]   # Match the shape of the x_values

# Plot
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel('NDVI', fontsize=12)
pixel_size = 10
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(f"{stub}" + ": " + str(time)[:10], fontsize=14)
plt.show()

res = stats.linregress(y_values, x_values)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope**2:.12f}")
plt.plot(y_values, x_values, 'o', label='original data')
plt.plot(y_values, res.intercept + res.slope*y_values, 'r', label='fitted line')
plt.legend()
plt.show()

# +
# Calculate the productivity score
time = '2020-01-03'
ndvi = ds.sel(time=time, method='nearest')['NDVI']
productivity_score1 = ndvi.where(~adjacent_mask)
s = ds['num_trees_200m'].values

# Flatten the arrays for plotting
x = productivity_score1.values.flatten()
x_values = x[~np.isnan(x)]   # Remove all pixels that are trees or adjacent to trees
y = s.flatten()
y_values = y[~np.isnan(x)]   # Match the shape of the x_values

# Plot
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel('NDVI', fontsize=12)
pixel_size = 10
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)
plt.show()
# -

res = stats.linregress(y_values, x_values)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope**2:.12f}")
plt.plot(y_values, x_values, 'o', label='original data')
plt.plot(y_values, res.intercept + res.slope*y_values, 'r', label='fitted line')
plt.legend()
plt.show()

# Winter
# time = '2020-07-23'
# time = '2020-01-03'
time = 'all years sum'
productivity_score = ds['NDVI'].sum(dim='time').where(~adjacent_mask)
# productivity_score = ds.sel(time=time, method='nearest')['NDVI'].where(~adjacent_mask)
shelter_score = ds['num_trees_200m']

# +
# Create a mask with the lowest n productivities for each shelterscore
# percentiles = [0,1,2,3,4,5,6,7,8,9,10, 25, 50, 75, 100]
# percentiles = [0,1,2,3,4,5,6,7,8,9,10]
# percentiles = [0,1,2,3,4,5]
percentiles = [0]
bins = 100

# Calculate the bin edges
flattened = shelter_score.values.flatten()
max_shelter_score = max(flattened[~np.isnan(flattened)])
bin_edges = np.linspace(0,max_shelter_score,bins)
shelter_bins = np.digitize(shelter_score, bin_edges)

mask = np.zeros_like(shelter_bins, dtype=bool)
for bin in range(1,bins):

    # Create a list of nan values in this bin
    bin_mask = (shelter_bins == bin)
    productivity_in_bin = productivity_score.where(bin_mask)
    flattened_productivities = productivity_in_bin.values.flatten()
    notnan_productivities = flattened_productivities[~np.isnan(flattened_productivities)]

    # Find the coordinate of the nearest value to each percentile, and add to the mask
    if len(notnan_productivities) > 0:
        for percentile in percentiles:
            value_at_percentile = np.percentile(notnan_productivities, percentile)
            closest_value = notnan_productivities[np.abs(notnan_productivities - value_at_percentile).argmin()]
            mask |= (productivity_in_bin == closest_value) & bin_mask
        
np.sum(mask.values)

# +
# Flatten the arrays for plotting
# ps = ds.sel(time=time, method='nearest')['NDVI'].where(mask)
ps = ds['NDVI'].sum(dim='time').where(mask)

x = ps.values.flatten()
x_values = x[~np.isnan(x)]   # Remove all pixels that are trees or adjacent to trees
y = s.flatten()
y_values = y[~np.isnan(x)]   # Match the shape of the x_values

# 2d histogram
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel(f'NDVI of lowest {n} pixels in {bins} bins', fontsize=12)
pixel_size = 10
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)
plt.show()

# Linear regression
res = stats.linregress(y_values, x_values)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope**2:.12f}")
plt.plot(y_values, x_values, 'o', label='original data')
plt.plot(y_values, res.intercept + res.slope*y_values, 'r', label='fitted line')
plt.legend()
plt.show()
# -
# Verifying that a random distribution wouldn't give the same results
p = np.random.normal(0.5, 0.1, 100000)
plt.hist(p, bins=30)
plt.title("Random distribution of productivity scores")
plt.show()

s = np.random.lognormal(0, 1, 100000)
s2 = s[s < np.percentile(s, 95)]
plt.hist(s2, bins=10)
plt.title("Log random distribution of shelter scores")
plt.show()

# +
x_values = p
y_values = s

# 2d histogram
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel(f'Productivity Score', fontsize=12)
pixel_size = 10
plt.xlabel(f'Shelter Score', fontsize=12)
plt.title("Random distribution comparison", fontsize=14)
plt.show()

# Linear regression
res = stats.linregress(y_values, x_values)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope**2:.12f}")
plt.plot(y_values, x_values, 'o', label='original data')
plt.plot(y_values, res.intercept + res.slope*y_values, 'r', label='fitted line')
plt.legend()
plt.show()

# +
# Create a mask with the lowest n productivities for each shelterscore
percentiles = [0]
bins = 100

# Calculate the bin edges
flattened = s
max_shelter_score = max(flattened)
bin_edges = np.linspace(0,max_shelter_score,bins)
shelter_bins = np.digitize(s, bin_edges)

s_mins = []
p_mins = []

# Find the minimum value in each bin
for bin in range(1,bins):
    # Mask out any values outside of this bin
    bin_mask = (shelter_bins == bin)
    productivity_in_bin = p.copy()
    productivity_in_bin[np.where(~bin_mask)] = np.nan
    
    # Find the minimum value in this bin
    if np.sum(~np.isnan(productivity_in_bin)) > 0:
        min_index = np.nanargmin(productivity_in_bin)
        s_min = s[min_index]
        p_min = p[min_index]
        s_mins.append(s_min)
        p_mins.append(p_min)

# +
x_values = p_mins
y_values = s_mins

# 2d histogram
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel(f'Productivity Score', fontsize=12)
pixel_size = 10
plt.xlabel(f'Shelter Score', fontsize=12)
plt.title("Lowest productivity in each shelter bin", fontsize=14)
plt.show()

# Linear regression
res = stats.linregress(y_values, x_values)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope**2:.12f}")
plt.plot(y_values, x_values, 'o', label='original data')
plt.plot(y_values, res.intercept + res.slope*np.array(y_values), 'r', label='fitted line')
plt.legend()
plt.show()
# -


