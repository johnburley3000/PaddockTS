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
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, transform_bbox
from DAESIM_preprocess.topography import pysheds_accumulation, calculate_slope
from DAESIM_preprocess.silo_daily import merge_ozwald_silo, resample_weekly, visualise_water, visualise_temp
from DAESIM_preprocess.ozwald_daily import ozwald_daily, ozwald_daily_abbreviations
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

# +
# Weather data
filename_silo = os.path.join(outdir, f"{stub}_silo_daily.nc")
ds_silo = xr.open_dataset(filename_silo)

variable = 'max_temp'
df_drought = pd.DataFrame(ds_silo.isel(lat=0, lon=0)[variable], index=ds_silo.isel(lat=0, lon=0).time, columns=[variable])


# -

def add_tiff_band(ds, variable, resampling_method, outdir, stub):
    """Add a new band to the xarray from a tiff file using the given resampling method"""
    filename = os.path.join(outdir, f"{stub}_{variable}.tif")
    array = rxr.open_rasterio(filename)
    reprojected = array.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected.isel(band=0).drop_vars('band')
    return ds


world_cover_layers = {
    "Tree cover": 10, # Green
    "Shrubland": 20, # Orange
    "Grassland": 30, # Yellow
    "Cropland": 40, # pink
    "Built-up": 50, # red
    "Permanent water bodies": 80, # blue
}

# Calculate the percentage of tree cover in each sentinel pixel, based on the global canopy height map
variable = "canopy_height"
filename = os.path.join(outdir, f"{stub}_{variable}.tif")
array = rxr.open_rasterio(filename)
binary_mask = (array >= 1).astype(float)
tree_percent = binary_mask.rio.reproject_match(ds, resampling=Resampling.average)
ds['tree_percent'] = tree_percent

ds = add_tiff_band(ds, "canopy_height", Resampling.max, outdir, stub)   # Maximum canopy height in each sentinel pixel

# The resampling often messes up the boundary, so we trim the outside pixels after adding all the resampled bounds
ds_trimmed = ds.isel(
    y=slice(1, -1),
    x=slice(1, -1) 
)

# +
# ds_trimmed['canopy_height'].plot()

# +
# ds_trimmed['NDVI'].isel(time=0).plot()
# -

ds_original = ds.copy()
ds = ds_trimmed

# +
# %%time
# Add worldcover classes to the xarray
worldcover_path = os.path.join("/g/data/xe2/cb8590/WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/")
MILG_id = "S36E147"
filename = os.path.join(worldcover_path, f"ESA_WorldCover_10m_2021_v200_{MILG_id}_Map", f"ESA_WorldCover_10m_2021_v200_{MILG_id}_Map.tif")
array = rxr.open_rasterio(filename)
reprojected = array.rio.reproject_match(ds)
ds["worldcover"] = reprojected.isel(band=0).drop_vars('band')

cropland = ds["worldcover"].values == world_cover_layers["Cropland"]
grassland = ds["worldcover"].values == world_cover_layers["Grassland"]
tree_cover = ds["worldcover"].values == world_cover_layers["Tree cover"]

crop_or_grass = cropland | grassland

tree_percent = ds['tree_percent'].isel(band=0).values

# +
# %%time
# Shelterscore showing the number of trees within a donut at a given distance away from the crop/pasture pixel

# distances = 0, 4, 6, 8, 10, 12   # A distance of 20 would correspond to a 200m radius if the pixel size is 10m
# distances = 0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40    # A distance of 20 would correspond to a 200m radius if the pixel size is 10m
# distances = 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
# distances = 1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 # , 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
# distances = 1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
# distances=0,10
distances = 0, 30
# distances = 1,2,3,4


# Classify anything with a height greater than 1 as a tree
tree_threshold = 1
tree_mask = ds['canopy_height'] >= tree_threshold

# Use the sentinel tree cover instead of global canopy height model
tree_mask = tree_cover
tree_percent = tree_cover

distance = 6
min_distance = 4
max_distance = 6
pixel_size = 10  # metres

# Find all the pixels directly adjacent to trees
structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

for i in range(len(distances) - 1):

    min_distance = distances[i]
    max_distance = distances[i+1]
    
    # Calculate the number of trees in a donut between the inner and outer circle
    y, x = np.ogrid[-max_distance:max_distance+1, -max_distance:max_distance+1]
    kernel = (x**2 + y**2 <= max_distance**2) & (x**2 + y**2 >= min_distance**2)
    kernel = kernel.astype(float)
    
    total_tree_cover = fftconvolve(tree_percent, kernel, mode='same')
    shelter_score = (total_tree_cover / kernel.sum()) * 100
    
    # Mask out trees and adjacent pixels
    shelter_score[np.where(adjacent_mask)] = np.nan
    shelter_score[shelter_score < 1] = 0
    
    # Add the shelter_score to the xarray
    shelter_score_da = xr.DataArray(
        shelter_score, 
        dims=("y", "x"),  
        coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
        name="shelter_score" 
    )

    layer_name = f"percent_trees_{pixel_size * min_distance}m-{pixel_size * max_distance}m"
    ds[layer_name] = shelter_score_da
    print(f"Added layer: {layer_name}")
# +
# Additional vegetation indices
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))


# ds['NIRV'] = ds['NDVI'] * ds['nbart_nir_1']
# ds['kNDVI'] = np.tanh(ds['NDVI'] * ds['NDVI'])
# ds['EVI2'] = 2.5 * ((B8 - B4)/(1 + B8 + 2.4 * B4))

# +
# # Fractional Cover
# import tensorflow as tf
# from fractionalcover3 import unmix_fractional_cover
# from fractionalcover3 import data 

# def calculate_fractional_cover(ds, band_names, i=3):
#     # Check if the number of band names is exactly 6
#     if len(band_names) != 6:
#         raise ValueError("Exactly 6 band names must be provided")
    
#     # Extract the specified bands and stack them into a numpy array with shape (time, bands, x, y)
#     inref = np.stack([ds[band].values for band in band_names], axis=1)
#     print(inref.shape)  # This should now be (time, bands, x, y)

#     #inref = inref * 0.0001 # if not applying the correcion factors below

#     # Array for correction factors 
#     # This is taken from here: https://github.com/petescarth/fractionalcover/blob/main/notebooks/ApplyModel.ipynb
#     # and described in a paper by Neil Floodfor taking Landsat to Sentinel 2 reflectance (and visa versa).
#     correction_factors = np.array([0.9551, 1.0582, 0.9871, 1.0187, 0.9528, 0.9688]) + \
#                          np.array([-0.0022, 0.0031, 0.0064, 0.012, 0.0079, -0.0042])

#     # Apply correction factors using broadcasting
#     inref = inref * correction_factors[:, np.newaxis, np.newaxis]

#     # Initialize an array to store the fractional cover results
#     fractions = np.empty((inref.shape[0], 3, inref.shape[2], inref.shape[3]))

#     # Loop over each time slice and apply the unmix_fractional_cover function
#     for t in range(inref.shape[0]):
#         fractions[t] = unmix_fractional_cover(inref[t], fc_model=data.get_model(n=i))
    
#     return fractions

# def add_fractional_cover_to_ds(ds, fractions):
#     """
#     Add the fractional cover bands to the original xarray.Dataset.

#     Parameters:
#     ds (xarray.Dataset): The original xarray Dataset containing the satellite data.
#     fractions (numpy.ndarray): The output array with fractional cover (time, bands, x, y).

#     Returns:
#     xarray.Dataset: The updated xarray Dataset with the new fractional cover bands.
#     """
#     # Create DataArray for each vegetation fraction
#     bg = xr.DataArray(fractions[:, 0, :, :], coords=[ds.coords['time'], ds.coords['y'], ds.coords['x']], dims=['time', 'y', 'x'])
#     pv = xr.DataArray(fractions[:, 1, :, :], coords=[ds.coords['time'], ds.coords['y'], ds.coords['x']], dims=['time', 'y', 'x'])
#     npv = xr.DataArray(fractions[:, 2, :, :], coords=[ds.coords['time'], ds.coords['y'], ds.coords['x']], dims=['time', 'y', 'x'])
    
#     # Assign new DataArrays to the original Dataset
#     ds_updated = ds.assign(bg=bg, pv=pv, npv=npv)
    
#     return ds_updated

# # band_names = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3']
# # # fractions = calculate_fractional_cover(ds, band_names)
# # ds = add_fractional_cover_to_ds(ds, fractions)
# -

# # Linear Regression and 2D Histogram

# +
# time = '2019-12-31'
time = '2020-01-08'
# time = '2024-03-07'
# time = '2021-02-06'
# time = '2022-06-01'
# time = '2020-03-20'


productivity_variable = 'EVI'
ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
productivity_score1 = ndvi.where(~adjacent_mask) #  & (grassland | cropland))

# Visualise a linear regression for this timepoint
# layer_name = f"percent_trees_0m-100m"
# layer_name = f"percent_trees_490m-500m"
# layer_name = f"percent_trees_100m-110m"
# layer_name = f"percent_trees_50m-300m"
# layer_name = f"percent_trees_750m-800m"
# layer_name = "percent_trees_950m-1000m"
layer_name = f"percent_trees_100m-300m"

s = ds[layer_name].values

# Remove all pixels that are trees or adjacent to trees
y = productivity_score1.values.flatten()
y_values_outliers = y[~np.isnan(y)]   

# Remove outliers
q1 = np.percentile(y_values_outliers, 25)
q3 = np.percentile(y_values_outliers, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Find the shelter scores not obstructed by cloud cover or outliers
y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]

x = s.flatten()
x_values_outliers = x[~np.isnan(y)]
x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]

# Keeping outliers
# x_values = x_values_outliers
# y_values = y_values_outliers

# Normalise
x_values_normalised = (x_values - min(x_values)) / (max(x_values) - min(x_values))
y_values_normalised = (y_values - min(y_values)) / (max(y_values) - min(y_values))

lower_bound, upper_bound

# +
# 2D histogram with logarithmic normalization
plt.figure(figsize=(14, 8))  # Width = 12, Height = 8
plt.hist2d(
    x_values, y_values, 
    bins=100, 
    norm=mcolors.LogNorm(),  # Logarithmic color scale
    cmap='viridis'
)
pixel_size = 10
plt.title("Productivity Index vs Shelter Score", fontsize=30)
plt.xlabel(layer_name, fontsize=18)
plt.ylabel(f'Enhanced Vegetation Index ({productivity_variable})', fontsize=18)

# Add color bar with custom ticks
cbar = plt.colorbar(label='Number of pixels')
cbar.set_label('Number of pixels', fontsize=18)

cbar.set_ticks([1, 10, 100, 1000, 10000])  # Set the desired tick marks
cbar.set_ticklabels(['1', '10', '100', '1000', '10000'])  # Ensure labels match the ticks

# Linear regression line
res = stats.linregress(x_values, y_values)
x_fit = np.linspace(min(x_values), max(x_values), 500)
y_fit = res.intercept + res.slope * x_fit
line_handle, = plt.plot(x_fit, y_fit, 'r-', label=f"$R^2$ = {res.rvalue**2:.2f}")
plt.legend(fontsize=14)

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_histregression_{time}.png")
plt.savefig(filename, bbox_inches='tight')
print(filename)
plt.show()


# -

# # Box Plot

# +
# Example box plot
# y_values = (y_values - min(y_values)) / (max(y_values) - min(y_values)) # Normalisation for the fractional cover

plt.figure(figsize=(8, 8))  # Width = 12, Height = 8
percent_tree_threshold = 10

sheltered = y_values[np.where(x_values >= percent_tree_threshold)]
unsheltered = y_values[np.where(x_values < percent_tree_threshold)]

box_data = [unsheltered, sheltered]
plt.boxplot(box_data, labels=['Unsheltered', 'Sheltered'], showfliers=False)
plt.xticks(fontsize=18)

plt.title("Unsheltered vs Sheltered Pixels", fontsize=30)
plt.ylabel('Enhanced Vegetation Index (EVI)', fontsize=18)
plt.xlabel('Shelter threshold of 10% tree cover within 100m', fontsize=18, labelpad=18)


# Add median values next to each box plot
medians = [np.median(data) for data in box_data]
# for i, median in enumerate(medians, start=1):  # `start=1` because boxplot positions start at 1
#     plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontsize=14, color='blue')
for i, median in enumerate(medians, start=1):  # `start=1` because boxplot positions start at 1
    plt.text(i + 0.09, median, f'{median:.2f}', ha='left', va='center', fontsize=14)


print(f"Shelter threshold = {int(percent_tree_threshold)}% tree cover within {distance * pixel_size}m")
print("Number of sheltered pixels: ", len(sheltered))
print("Number of unsheltered pixels: ", len(unsheltered))

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_boxplot_{time}.png")
plt.savefig(filename, bbox_inches='tight')
print(filename)
# -
# # Matrix of benefits at specific distances at a single timepoint

# +
# %%time
# Shelterscore showing the number of trees within a donut at a given distance away from the crop/pasture pixel

# distances = 0, 4, 6, 8, 10, 12   # A distance of 20 would correspond to a 200m radius if the pixel size is 10m
# distances = 0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40    # A distance of 20 would correspond to a 200m radius if the pixel size is 10m
# distances = 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
# distances = 1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50

# distances = list(range(10, 32, 4))
distances = list(range(4, 30, 1))

# Classify anything with a height greater than 1 as a tree
# tree_threshold = 1
# tree_mask = ds['canopy_height'] >= tree_threshold

distance = 6
min_distance = 4
max_distance = 6
pixel_size = 10  # metres

# Find all the pixels directly adjacent to trees
structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

for i in range(len(distances) - 1):

    min_distance = distances[i]
    max_distance = distances[i+1]
    
    # Calculate the number of trees in a donut between the inner and outer circle
    y, x = np.ogrid[-max_distance:max_distance+1, -max_distance:max_distance+1]
    kernel = (x**2 + y**2 <= max_distance**2) & (x**2 + y**2 >= min_distance**2)
    kernel = kernel.astype(float)
    
    total_tree_cover = fftconvolve(tree_percent, kernel, mode='same')
    shelter_score = (total_tree_cover / kernel.sum()) * 100
    
    # Mask out trees and adjacent pixels
    shelter_score[np.where(adjacent_mask)] = np.nan
    shelter_score[shelter_score < 1] = 0
    
    # Add the shelter_score to the xarray
    shelter_score_da = xr.DataArray(
        shelter_score, 
        dims=("y", "x"),  
        coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
        name="shelter_score" 
    )

    layer_name = f"percent_trees_{pixel_size * min_distance}m-{pixel_size * max_distance}m"
    ds[layer_name] = shelter_score_da
    print(f"Added layer: {layer_name}")

# +
# %%time
benefits = []
percentage_tree_thresholds = list(range(1, 30, 1))


for i in range(len(distances) - 1):

    # Shelter score 
    min_distance = distances[i]
    max_distance = distances[i+1]
    layer_name = f"percent_trees_{pixel_size * min_distance}m-{pixel_size * max_distance}m"
    s = ds[layer_name].values
    
    # Flatten the arrays for plotting
    y = productivity_score1.values.flatten()
    y_values = y[~np.isnan(y)]   # Remove all pixels that are trees or adjacent to trees
    x = s.flatten()
    x_values = x[~np.isnan(y)]   # Match the shape of the x_values

    # Sheltered vs unsheltered pixels
    for percentage_tree_threshold in percentage_tree_thresholds:
        sheltered = y_values[np.where(x_values >= percentage_tree_threshold)]
        unsheltered = y_values[np.where(x_values < percentage_tree_threshold)]
        median_diff = np.median(sheltered) - np.median(unsheltered)
        median_diff_percentage = median_diff/np.median(y_values)
        sample_size = min(len(sheltered), len(unsheltered))
        benefit = {
            "distance": max_distance * 10,
            "percentage_tree_threshold": percentage_tree_threshold,
            "median_diff_percentage": 100 * np.round(median_diff_percentage, 2),
            "sample_size": sample_size
        }
        benefits.append(benefit)

df = pd.DataFrame(benefits)
# +
# Visualise the benefits matrix
heatmap_data = df.pivot(index='percentage_tree_threshold', columns='distance', values='median_diff_percentage')

plt.figure(figsize=(20, 10))  # Width = 12, Height = 8

ax = sns.heatmap(heatmap_data, annot=True, cmap="YlGn", cbar=True)
ax.invert_yaxis()

label_size = 24
pad_size = 10
plt.title(f'Shelter benefits at Milgadara on {time}', fontsize = 30)
plt.ylabel('Tree Cover (%)', fontsize=label_size, labelpad=pad_size)
plt.xlabel('Distance (m)', fontsize=label_size, labelpad=pad_size)
cbar = ax.collections[0].colorbar
cbar.set_label(f'Sheltered vs Unsheltered difference in {productivity_variable} (%)', fontsize=label_size)

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_bigmatrix_{time}.png")
plt.savefig(filename, bbox_inches='tight')
print(filename)

plt.show()

# +
# Visualise the sample sizes
heatmap_data = df.pivot(index='percentage_tree_threshold', columns='distance', values='sample_size')

threshold = 10000
annotations = np.where(heatmap_data < threshold, heatmap_data.astype(int).astype(str), "")

plt.figure(figsize=(20, 10))  # Width = 12, Height = 8
ax = sns.heatmap(heatmap_data, annot=annotations, fmt='', cbar=True)
ax.invert_yaxis()

plt.title(f'{stub}: Sample Sizes')
plt.ylabel('Tree Cover (%)')
plt.xlabel('Distance (m)')
cbar = ax.collections[0].colorbar
cbar.set_label('Sample size')
plt.show()
# -

# # Temporal Variation

# +
# %%time
# Look at how the r2, slope and median difference between EVI and shelter threshold compares over time, using a donut of 50m-300m (and each different thresholds)
tree_cover_threshold = 10

benefits = []

# distances = 1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
# distances = 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
distances = 0, 30

for i in range(len(distances) - 1):
    min_distance = distances[i]
    max_distance = distances[i+1]
    layer_name = f"percent_trees_{pixel_size * min_distance}m-{pixel_size * max_distance}m"
    shelter_score = ds[layer_name]
    x = shelter_score.values.flatten()

    for i, time in enumerate(ds.time.values):
        ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
        productivity_score = ndvi.where(~adjacent_mask)

        # Remove all pixels that are trees, adjacent to trees, or masked by cloud cover
        y = productivity_score.values.flatten()
        y_values_outliers = y[~np.isnan(y)]   

        # Remove outliers
        q1 = np.percentile(y_values_outliers, 25)
        q3 = np.percentile(y_values_outliers, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
        x_values_outliers = x[~np.isnan(y)]
        x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]

        # Uncomment to keep outliers
        # x_values = x_values_outliers
        # y_values = y_values_outliers
        
        sheltered = y_values[np.where(x_values >= tree_cover_threshold)]
        unsheltered = y_values[np.where(x_values < tree_cover_threshold)]
        percentage_benefit = (np.median(sheltered) - np.median(unsheltered))/np.median(y_values)
        sample_size = min(len(sheltered), len(unsheltered))
        
        res = stats.linregress(x_values, y_values)
    
        benefit = {
            "distance":max_distance,
            "time": time,
            "r2": res.rvalue**2,
            "slope": res.slope,
            "percentage_benefit": percentage_benefit,
            "sample_size": sample_size,
            "median": np.median(y_values)
        }
        benefits.append(benefit)

len(benefits)
# -

pd.DataFrame(benefits)

# Load weather data
filename_ozwald = os.path.join(outdir, f"{stub}_ozwald_8day.nc")
filename_silo = os.path.join(outdir, f"{stub}_silo_daily.nc")
ds_ozwald = xr.open_dataset(filename_ozwald)
ds_silo = xr.open_dataset(filename_silo)
df_daily = merge_ozwald_silo(ds_ozwald, ds_silo)
df_weekly = resample_weekly(df_daily)

# Merge shelter benefits
df_benefits = pd.DataFrame(benefits)
df_benefits['date'] = df_benefits['time'].dt.date
df_benefits = df_benefits.set_index('date')
df_benefits.index = pd.to_datetime(df_benefits.index)
df_merged = pd.merge_asof(df_weekly, df_benefits, left_index=True, right_index=True, direction='nearest')

# +
# Visualise the shelter benefits
plt.figure(figsize=(50, 10))
r2_plot = plt.plot(df.index, df["r2"], color='black')
benefits_plot = plt.plot(df.index, df["percentage_benefit"], color='yellowgreen')
opacity = 0.2
plt.fill_between(
    df.index, 
    0, 
    df["percentage_benefit"], 
    where=(df["percentage_benefit"] > 0), 
    color='green', 
    alpha=opacity, 
    interpolate=True,  # Smooth transitions
    label='Positive Benefit'
)
plt.fill_between(
    df.index, 
    0, 
    df["percentage_benefit"], 
    where=(df["percentage_benefit"] < 0), 
    color='red', 
    alpha=opacity, 
    interpolate=True, 
    label='Negative Benefit'
)
handles = [r2_plot[0], benefits_plot[0]]
labels = ['Shelter score vs productivity index (r2)', 'Sheltered median vs unsheltered median (%)']

plt.title("Shelter Benefits", fontsize=50)
plt.legend(handles=handles, labels=labels, fontsize=30, loc='upper left')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

filename_benefits = os.path.join(outdir, f"{stub}_shelter_benefits_time_series.png")
plt.savefig(filename_benefits)
plt.show()


# Visualise the weather data
plt.figure(figsize=(50, 20))
df = df_merged

rainfall_plot = plt.bar(df.index, df['Rainfall'], color='skyblue', width=5)
et_potential_plot = plt.bar(df.index, df['Potential Evapotranspiration'], color='orange')
moisture_min_plot = plt.plot(df.index, df['Minimum Soil Moisture'], color='blue')
EVI_plot = plt.plot(df.index, df["median"] * 100, color='green')

handles = [rainfall_plot, et_potential_plot[0], moisture_min_plot[0], EVI_plot[0]]
labels = ['Rainfall (mm)', "Evapotranspiration (mm)", "Soil moisture (mm)", "Overall productivity x100"]

plt.legend(handles=handles, labels=labels, fontsize=30, loc='upper left')
plt.title("Environmental Variables", fontsize=50)
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)  
plt.tight_layout()

filename_weather = os.path.join(outdir, f"{stub}_environmental_variables.png")
plt.savefig(filename)
plt.show()

print("Saved", filename_benefits)
print("Saved", filename_weather)
# -

# # Spatial Variation

df_top10 = df_benefits.nlargest(10, 'r2')
df_top10[['r2', 'slope', 'percentage_benefit', 'sample_size']]

# +
# Visualise the spatial variation in EVI
time = df_top10.index[0].date()

ds_productivity = ds.sel(time=time, method='nearest')[productivity_variable]
ds_masked = ds_productivity.where(~adjacent_mask)

# Calculate the productivity and shelter scores
layer_name = f"percent_trees_0m-300m"
s = ds[layer_name].values
y = ds_masked.values.flatten()
y_values_outliers = y[~np.isnan(y)]  
x = s.flatten()
x_values_outliers = x[~np.isnan(y)]  

# Remove outliers from list
q1 = np.percentile(y_values_outliers, 25)
q3 = np.percentile(y_values_outliers, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]    
x = s.flatten()
x_values_outliers = x[~np.isnan(y)]
x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]

unsheltered = y_values[np.where(x_values < percent_tree_threshold)]
median_value = np.median(unsheltered)

# Define color map
plt.figure(figsize=(8, 8))  # Width=10, Height=8 (adjust as needed)

cmap = plt.cm.coolwarm  
cmap.set_bad(color='green')  # Set NaN pixels to green
ax = ds_masked.plot(
    cmap=cmap,
    vmin=median_value - (upper_bound - lower_bound) / 2,
    vmax=median_value + (upper_bound - lower_bound) / 2,
)

# Plot the map
ax = plt.gca() 
ax.set_title(f'Productivity at {stubs[stub]} on {time}', fontsize=22) 
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([]) 
ax.set_yticks([])
cbar = ax.collections[0].colorbar
cbar.set_label(f"Enhanced Vegetation Index ({productivity_variable})", fontsize=18)  # Set font size

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_spatial_variation_{time}.png")
plt.savefig(filename, bbox_inches='tight')
print(filename)
plt.show()
# -
ds['worldcover'].plot(figsize=(8,8))
plt.show()











