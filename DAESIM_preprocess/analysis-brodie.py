# Standard library
import os
import pickle
import datetime
import math

# Dependencies
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import box, Polygon
from rasterio.enums import Resampling
from rasterio import features
import scipy.ndimage
from scipy import stats
from scipy.signal import fftconvolve
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns
import contextily as ctx

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, transform_bbox
from DAESIM_preprocess.topography import pysheds_accumulation, calculate_slope
from DAESIM_preprocess.silo_daily import merge_ozwald_silo, resample_weekly, visualise_water, visualise_temp
from DAESIM_preprocess.ozwald_daily import ozwald_daily, ozwald_daily_abbreviations
from DAESIM_preprocess.topography import show_acc, show_aspect, show_slope, show_ridge_gullies, pysheds_accumulation, catchment_gullies, catchment_ridges, calculate_slope

# Checking memory usage
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# region
# Filepaths
outdir = os.path.join(gdata_dir, "Data/shelter/")
stub = "BRODIE"

# outdir = os.path.join(gdata_dir, "Data/shelter/")
# # stub = "34_5_148_2"
# stub = "33_8_148_1"

# Global variables
tree_cover_threshold = 5
pixel_size = 10  # metres
distances = (0, 30) # pixels. So all pixels within 300m. Might be more robust to look at pixels in a donut, e.g. 50m-300m.
min_distance = distances[0]
max_distance = distances[1]
# endregion

# %%time
# Load the sentinel imagery xarray 
# filename = os.path.join(outdir, f"{stub}_ds2.pkl")
filename = os.path.join(outdir, f"{stub}_ds2_frac.pkl")
with open(filename, 'rb') as file:
    ds_original = pickle.load(file)

ds = ds_original

print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# %%time
# Calculate the percentage of tree cover in each sentinel pixel, based on the global canopy height map
variable = "canopy_height"
filename = os.path.join(outdir, f"{stub}_{variable}.tif")
array = rxr.open_rasterio(filename)
ds['max_tree_height'] = array.rio.reproject_match(ds, resampling=Resampling.max)
binary_mask = (array >= 1).astype(float)
ds['tree_percent'] = binary_mask.rio.reproject_match(ds, resampling=Resampling.average)

# The reproject_match often messes up the boundary, so we trim the outside pixels after adding all the resampled bounds
ds = ds.isel(
    y=slice(1, -1),
    x=slice(1, -1) 
)

# region
# # Crop the region to just Brodie's transects

# # Define the coordinate transformation from EPSG:4326 to EPSG:6933
# transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

# # Convert given lat/lon bounds to EPSG:6933
# lon_min, lat_min = 147 + 9/60 + 35.17/3600, -(25 + 22/60 + 51.54/3600)
# lon_max, lat_max = 147 + 9/60 + 41.09/3600, -(25 + 22/60 + 47.40/3600)

# x_min, y_min = transformer.transform(lon_min, lat_min)
# x_max, y_max = transformer.transform(lon_max, lat_max)

# cropped_bounds = x_min, y_min, x_max, y_max
# endregion

cropped_bounds = (14198885, -3135710, 14199267, -3135470)
# cropped_bounds =   (14198898, -3135643, 14199057, -3135510)
x_min, y_min, x_max, y_max = cropped_bounds
cropped_bounds

# region
# ds = ds_uncropped
# endregion

# Crop the dataset
ds_cropped = ds.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
ds_uncropped = ds
ds = ds_cropped

# region
# Global canopy height tree_mask
tree_percent = ds['tree_percent'].values[0]
tree_mask = tree_percent > 0

# Shelterscore showing the number of trees within a donut at a given distance away from the crop/pasture pixel
structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)
# adjacent_mask = tree_mask

# for i in range(len(distances) - 1):

#     min_distance = distances[i]
#     max_distance = distances[i+1]
    
#     # Calculate the number of trees in a donut between the inner and outer circle
#     y, x = np.ogrid[-max_distance:max_distance+1, -max_distance:max_distance+1]
#     kernel = (x**2 + y**2 <= max_distance**2) & (x**2 + y**2 >= min_distance**2)
#     kernel = kernel.astype(float)
    
#     total_tree_cover = fftconvolve(tree_percent, kernel, mode='same')
#     shelter_score = (total_tree_cover / kernel.sum()) * 100
    
#     # Mask out trees and adjacent pixels
#     shelter_score[np.where(adjacent_mask)] = np.nan
#     shelter_score[shelter_score < 1] = 0
    
#     # Add the shelter_score to the xarray
#     shelter_score_da = xr.DataArray(
#         shelter_score, 
#         dims=("y", "x"),  
#         coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
#         name="shelter_score" 
#     )

#     layer_name = f"percent_trees_{pixel_size * min_distance}m-{pixel_size * max_distance}m"
#     ds[layer_name] = shelter_score_da
#     print(f"Added layer: {layer_name}")
# endregion

# Enhanced Vegetation Index
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
productivity_variable = 'EVI'

time = '2022-08-04'
ds_timepoint = ds.sel(time=time, method='nearest')

plt.imshow(ds_timepoint['EVI'])

# region
# Compute Euclidean distance from each non-tree pixel to the nearest tree
distance_to_tree = distance_transform_edt(~tree_mask) * pixel_size

# Mask out trees and adjacent pixels
distance_to_tree[np.where(adjacent_mask)] = np.nan

# Create an xarray DataArray for the distance layer
distance_da = xr.DataArray(
    distance_to_tree,
    dims=("y", "x"),
    coords={"y": ds.coords["y"], "x": ds.coords["x"]},
    name="distance_to_tree"
)

# Add to dataset
ds["distance_to_tree"] = distance_da
ds["distance_to_tree"].plot()
# endregion

# region
# Save the shelterscore, productivity score, and canopy height to tiff files for double checking in QGIS
filename = os.path.join(scratch_dir, f"{stub}_10m_canopy_height.tiff")
ds['max_tree_height'].rio.to_raster(filename)
print(filename)

filename = os.path.join(scratch_dir, f"{stub}_EVI_2022_08_03.tiff")
ds_timepoint['EVI'].rio.to_raster(filename)
print(filename)

filename = os.path.join(scratch_dir, f"{stub}_distance_to_tree.tiff")
ds['distance_to_tree'].rio.to_raster(filename)
print(filename)
# endregion

bg = ds_timepoint['bg'].values
bg[np.where(adjacent_mask)] = np.nan
ds_timepoint['bg'].plot()

filename = os.path.join(scratch_dir, f"{stub}_bg_2022_08_03.tiff")
ds_timepoint['bg'].rio.to_raster(filename)
print(filename)

ds_timepoint = ds.sel(time=time, method='nearest')
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
red = ds_timepoint['nbart_red']
green = ds_timepoint['nbart_green']
blue = ds_timepoint['nbart_blue']
rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)

# region
# RGB Image
label_size = 14
bounds = ds[productivity_variable].rio.bounds()
left, bottom, right, top = bounds
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(rgb, extent=(left, right, bottom, top))

scalebar = AnchoredSizeBar(
    ax.transData, 100, '100m', loc='lower center', pad=0.1, 
    color='black', frameon=False, size_vertical=10,
    fontproperties=fm.FontProperties(size=label_size),
    bbox_to_anchor=(0.3, -0.15),  # Position below the plot
    bbox_transform=ax.transAxes,
)
ax.add_artist(scalebar)
plt.axis('off')

filename = os.path.join(scratch_dir, stub+'_Paddocks_map_auto.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
print(filename)
# endregion

# Remove unnecessary variables from ds
useful_variables = ['nbart_red', 'nbart_green', 'nbart_blue', 'EVI', 'tree_percent', 'max_tree_height', 'distance_to_tree']
ds_small = ds.isel(band=0)[useful_variables]

# region
# Prep arrays for histogram

productivity_variable = 'bg'
ds_timepoint = ds.sel(time=time, method='nearest')
    
# Calculate shelter score and productivity index for this timepoint
ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
p = ndvi.where(~adjacent_mask) #  & (grassland | cropland))
# layer_name = f"percent_trees_0m-300m"
layer_name = f"distance_to_tree"
s = ds[layer_name]
x = s.values.flatten()

# Make sure that any nan values in the shelter score are also nan in the productivity_score
p = p.where(~s.isnull())

# Remove all pixels that are trees or adjacent to trees
y = p.values.flatten()
y_values_outliers = y[~np.isnan(y)]  

# Outlier boundary
lower_bound = 0
upper_bound = max(np.percentile(y_values_outliers, 99.9), 1)

# Find the shelter scores not obstructed by cloud cover or outliers
y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
x_values_outliers = x[~np.isnan(y)]
x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]

# endregion

# region

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Scatter plot
sc = ax1.scatter(
    x_values, y_values, 
    s=10,  
    c='blue', 
    alpha=0.5 
)

# Titles and labels
ax1.set_title(f"Bare Ground vs Shelter Score, 3 Aug 2022", fontsize=18)
ax1.set_xlabel("Distance from nearest tree (m)", fontsize=18)
ax1.set_ylabel(f"bare ground (unitless)", fontsize=18)
ax1.tick_params(axis='both', labelsize=14)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# ax1.set_ylim(0.15, 0.4)
# ax1.set_xlim(0, 350)
# endregion



