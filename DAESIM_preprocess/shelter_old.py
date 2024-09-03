# +
# Idea of this notebook is assign productivity and shelter scores and plot them against each other

# +
# Standard Libraries
import pickle
import os

# Dependencies
import numpy as np
import xarray as xr

import rioxarray as rxr
from rasterio import features
from rasterio.enums import Resampling
from shapely.geometry import box, Polygon
import scipy.ndimage
from scipy.spatial import cKDTree

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from dea_tools.plotting import rgb


# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, transform_bbox
# +
# 2023 example
# filename_sentinel = "/g/data/xe2/John/Data/PadSeg/MILG_b033_2023_ds2.pkl"
# filename_canopy_height = os.path.join(gdata_dir, "MILG14km_canopy_height.tif")

outdir = os.path.join(gdata_dir, "Data/PadSeg")
stub = "MILG"
filename_sentinel = os.path.join(outdir, f"{stub}_ds2.pkl")
# -

with open(filename_sentinel, 'rb') as file:
    ds = pickle.load(file)


ds.isel(time=0)['NDVI'].plot()

filename_canopy_height = "/g/data/xe2/cb8590/MILG14km_canopy_height_max.tif"
canopy_height_reprojected = rxr.open_rasterio(filename_canopy_height)
canopy_height_reprojected.isel(band=0).plot()

# +
# # Reprojecting the canopy height to the same dimensions as sentinel using Max Resampling
# canopy_height_reprojected = cropped_canopy_height.rio.reproject_match(ds, resampling=Resampling.max)

# filename = os.path.join(scratch_dir, f"{stub}_canopy_height_max.tif")
# canopy_height_reprojected.rio.to_raster(filename)

# Assigning the reprojected canopy height to a new band in the sentinel xarray
canopy_height_band = canopy_height_reprojected.isel(band=0)
ds['canopy_height'] = canopy_height_band
# +
# Get a mask of paddock polygons
filename = os.path.join(outdir,stub + '_filt.gpkg')
pol_filt = gpd.read_file(filename)

# filename = os.path.join(gdata_dir,'MILG_paddocks_notrees.geojson')
filename = os.path.join(gdata_dir,'MILG_paddocks_notrees.gpkg')
pol = gpd.read_file(filename)

# Change from multipolygon to polygon, because I created the layer with the wrong type in QGIS
def convert_multipolygon_to_polygon(geometry):
    return geometry.union(geometry)
pol['geometry'] = pol['geometry'].apply(convert_multipolygon_to_polygon)

# +
xarray_dataset = ds
xarray_crs = xarray_dataset.crs
xarray_bounds = xarray_dataset.rio.bounds()
pol_bounds = pol.total_bounds

print("Xarray bounds:", xarray_bounds)
print("Polygon bounds:", pol_bounds)

# +
gdf = pol.to_crs(xarray_crs) 
transform = xarray_dataset.rio.transform()
out_shape = (xarray_dataset.dims['y'], xarray_dataset.dims['x'])
geometries = gdf.geometry

# Create a mask
paddock_mask = features.geometry_mask(
    [geom for geom in geometries],
    transform=transform,
    invert=True,
    out_shape=out_shape
)
plt.imshow(paddock_mask)

# +
# Creating a mask for tree cells
tree_threshold = 1
tree_mask = ds['canopy_height'] >= tree_threshold

# Find the pixels adjacent to trees, which we ignore
# structuring_element = np.ones((3, 3))
structuring_element = np.ones((5, 5))
tree_buffer = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

# adjacent_mask = ~tree_buffer
# adjacent_mask = ~tree_buffer & paddock_mask
# adjacent_mask = np.ones(tree_mask.shape, dtype=bool)
# adjacent_mask = tree_mask
adjacent_mask = paddock_mask

# plt.imshow(adjacent_mask)
# -

# Using summed NDVI as a proxy for productivity
summed_ndvi = ds['NDVI'].sum(dim='time')
productivity_score1 = summed_ndvi.where(adjacent_mask)

# +
# Parameters
distance = 20  # This corresponds to a 200m radius if the pixel size is 10m
pixel_size = 10  # Assume each pixel represents a 10m x 10m area

# Create a circular kernel with a radius of 20 pixels (equivalent to 200m)
radius_in_pixels = distance
y, x = np.ogrid[-radius_in_pixels:radius_in_pixels+1, -radius_in_pixels:radius_in_pixels+1]
kernel = x**2 + y**2 <= radius_in_pixels**2
kernel = kernel.astype(float)
shelter_score1 = scipy.ndimage.convolve(tree_mask.astype(float), kernel, mode='constant', cval=0.0)

shelter_score_trees = shelter_score1.copy()

# Mask out the areas outside the desired buffer
shelter_score1[np.where(~adjacent_mask)] = np.nan
shelter_score_trees[np.where(adjacent_mask)] = np.nan

plt.imshow(shelter_score1)
plt.colorbar()

# +
# Add the shelter_score to the xarray
shelter_score_da = xr.DataArray(
    shelter_score1, 
    dims=("y", "x"),  
    coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
    name="shelter_score" 
)
ds["shelter_score"] = shelter_score_da

filename = os.path.join(scratch_dir, "MILG_num_trees_200m.tif")
ds["shelter_score"].rio.to_raster(filename)
print("Saved:", filename)
# -
time = '2020-01-03'
ndvi = ds.sel(time=time, method='nearest')['NDVI']

# +
# Using NDVI as a proxy for productivity
# ndvi = ds['NDVI'].sum(dim='time')
# for i in range(200,len(ds.time.values)):
productivity_score1 = ndvi.where(adjacent_mask)
s = shelter_score1[~np.isnan(productivity_score1)]

# Flatten the arrays for plotting
x = productivity_score1.values.flatten()
x_values = x[~np.isnan(x)]
y = s.flatten()
y_values = y[~np.isnan(y)]

# Plot
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
# plt.ylabel('Sum of NDVI over all years', fontsize=12)
plt.ylabel('NDVI', fontsize=12)
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)
plt.show()

# +
time = '2020-07-23'
ndvi = ds.sel(time=time, method='nearest')['NDVI']

productivity_score1 = ndvi.where(adjacent_mask)
s = shelter_score1[~np.isnan(productivity_score1)]

# Flatten the arrays for plotting
x = productivity_score1.values.flatten()
x_values = x[~np.isnan(x)]
y = s.flatten()
y_values = y[~np.isnan(y)]

# Plot
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
# plt.ylabel('Sum of NDVI over all years', fontsize=12)
plt.ylabel('NDVI', fontsize=12)
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)
plt.show()

# +
# Using NDVI as a proxy for productivity
ndvi = ds['NDVI'].sum(dim='time')
# for i in range(200,len(ds.time.values)):
productivity_score1 = ndvi.where(adjacent_mask)
s = shelter_score1[~np.isnan(productivity_score1)]

# Flatten the arrays for plotting
x = productivity_score1.values.flatten()
x_values = x[~np.isnan(x)]
y = s.flatten()
y_values = y[~np.isnan(y)]

# Plot
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
# plt.ylabel('Sum of NDVI over all years', fontsize=12)
plt.ylabel('NDVI summed over all years', fontsize=12)
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": overall", fontsize=14)
plt.show()

# +
# Using NDVI as a proxy for productivity
ndvi = ds['NDVI'].sum(dim='time')
productivity_score1 = ndvi.where(~adjacent_mask)
s = shelter_score_trees[~np.isnan(productivity_score1)]

# Flatten the arrays for plotting
x = productivity_score1.values.flatten()
x_values = x[~np.isnan(x)]
y = s.flatten()
y_values = y[~np.isnan(y)]

# Plot
plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
# plt.ylabel('Sum of NDVI over all years', fontsize=12)
plt.ylabel('NDVI summed over all years', fontsize=12)
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": trees", fontsize=14)
plt.show()

# +
# # Distance to the nearest group of n trees
# from scipy.ndimage import distance_transform_edt, label

# num_trees = 50

# # Assuming 'ds' is your xarray.Dataset and 'canopy_height' is your band
# canopy_height = ds['canopy_height']

# # 1. Identify tree pixels
# is_tree = canopy_height >= 1

# # 2. Assign NaN to shelter_score where there are trees
# shelter_score = xr.where(is_tree, np.nan, 0)

# # 3. Label connected tree pixels
# labeled_trees, num_features = label(is_tree)

# # 4. Compute distance to nearest tree pixel for non-tree areas
# distances = distance_transform_edt(~is_tree)

# # 5. Calculate the distance to the nearest group of n trees
# # Create a mask where groups of n trees or more exist
# tree_groups = np.zeros_like(labeled_trees, dtype=bool)
# for i in range(1, num_features + 1):
#     if np.sum(labeled_trees == i) >= num_trees:  # 'n' is 10
#         tree_groups |= labeled_trees == i

# # Compute distances to these groups
# shelter_score3_xr = xr.where(is_tree, np.nan, distance_transform_edt(~tree_groups))
# shelter_score3 = shelter_score3_xr.values * pixel_size

# # Remove the shelterscore for tree cells and adjacent pixels
# shelter_score3[np.where(~adjacent_mask)] = np.nan

# plt.imshow(shelter_score3)
# plt.show()

# +
# # # Comparing productivity with distance from a group of n trees
# y = shelter_score3.flatten()
# y_values = y[~np.isnan(y)]

# plt.hist2d(y_values, x_values, bins=100, norm=mcolors.PowerNorm(0.1))
# plt.xlabel(f'Distance from a group of {num_trees} trees (m)', fontsize=12)
# plt.ylabel('Sum of NDVI over all years', fontsize=12)
# plt.show()
# +
# # Save the tree groups as a tiff for viewing in QGIS
# layername = "tree_groups_50"
# layer = tree_groups.astype(float)
# da = xr.DataArray(
#     layer, 
#     dims=("y", "x"),  
#     coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
#     name=layername
# )
# ds[layername] = da

# filename = os.path.join(scratch_dir, f"MILG_{layername}.tif")
# ds[layername].rio.to_raster(filename)
# print("Saved:", filename)
# -



