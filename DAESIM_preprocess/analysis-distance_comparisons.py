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
stub = "ADAM"

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
crop_or_grass = cropland | grassland

tree_percent = ds['tree_percent'].isel(band=0).values

# +
# %%time
# Shelterscore showing the number of trees 
pixel_size = 10  # metres
# distances = 5, 10, 20, 50, 100, 200 # A distance of 20 would correspond to a 200m radius if the pixel size is 10m
distances = 4, 6, 8, 10, 12   # A distance of 20 would correspond to a 200m radius if the pixel size is 10m
# distances = 5, 10, 15, 20, 25   # A distance of 20 would correspond to a 200m radius if the pixel size is 10m

# Classify anything with a height greater than 1 as a tree
tree_threshold = 1
tree_mask = ds['canopy_height'] >= tree_threshold

# # Find the pixels adjacent to trees
structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

for distance in distances:
    
    # Calculate the number of trees within a given distance for each pixel
    y, x = np.ogrid[-distance:distance+1, -distance:distance+1]
    kernel = x**2 + y**2 <= distance**2
    kernel = kernel.astype(float)

    # This method will overeestimate the tree cover percent
    # shelter_score = fftconvolve(tree_mask.astype(float), kernel, mode='same')

    # More accurate tree cover percent calculation
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

    layer_name = f"percent_trees_{pixel_size * distance}m"
    ds[layer_name] = shelter_score_da
    print(f"Added layer: {layer_name}")


# +
# Example shelter score
layer_name = "percent_trees_100m"
filename = os.path.join(scratch_dir, f'{stub}_{layer_name}.tif')
ds[layer_name].rio.to_raster(filename)
print(filename)

ds[layer_name].plot(cmap='viridis')
plt.title(f"{stub} Shelter Score")
filename = filename = os.path.join(scratch_dir, f'{stub}_{layer_name}.png')
plt.savefig(filename)
print(filename)

# +
# Additional vegetation indices
ds['NIRV'] = ds['NDVI'] * ds['nbart_nir_1']

B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))

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

plt.imshow(~adjacent_mask & cropland)

# +
# Example shelter vs productivity score
time = '2020-05-17'
productivity_variable = 'EVI'
ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
productivity_score1 = ndvi.where(~adjacent_mask) #  & (grassland | cropland))
distance = 10
layer_name = f"percent_trees_{pixel_size * distance}m"
s = ds[layer_name].values

# Flatten the arrays for plotting
y = productivity_score1.values.flatten()
y_values = y[~np.isnan(y)]   # Remove all pixels that are trees or adjacent to trees
x = s.flatten()
x_values = x[~np.isnan(y)]   # Match the shape of the x_values

# # Remove infinity values from the fractional cover (I haven't looked into what these mean)
# not_infinity = np.where(y_values != np.inf)[0]
# y_values = y_values[not_infinity]
# x_values = x_values[not_infinity]


# +
# Example 2d histogram
plt.hist2d(x_values, y_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel(productivity_variable, fontsize=12)
pixel_size = 10
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_2dhist_{time}.png")
plt.savefig(filename)
print(filename)

# +
# Example linear regression

# Min max normalisation for the shelter and productivity scores to make the slope more meaningful
x_values_normalised = (x_values - min(x_values)) / (max(x_values) - min(x_values))
y_values_normalised = (y_values - min(y_values)) / (max(y_values) - min(y_values))

res = stats.linregress(x_values_normalised, y_values_normalised)
print(f"Sample size: {len(x_values)}")
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope:.6f}")
plt.plot(x_values_normalised, y_values_normalised, 'o', label='original data')
plt.plot(x_values_normalised, res.intercept + res.slope*x_values_normalised, 'r', label='fitted line')

plt.ylabel('Productivity Score', fontsize=12)
pixel_size = 10
plt.xlabel(f'Shelter Score', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)

plt.legend()

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_lineplot_{time}.png")
plt.savefig(filename)
print(filename)

# +
# # Example sheltered vs unsheltered threshold
# distance = 20
# shelter_threshold = 0.1  # Percentage tree cover
# num_trees_threshold = ((distance * 2) ** 2) * shelter_threshold # Number of tree pixels
# num_trees_threshold

# +
# Example box plot
# y_values = (y_values - min(y_values)) / (max(y_values) - min(y_values)) # Normalisation for the fractional cover
percent_tree_threshold = 10

sheltered = y_values[np.where(x_values >= percent_tree_threshold)]
unsheltered = y_values[np.where(x_values < percent_tree_threshold)]

plt.boxplot([unsheltered, sheltered], labels=['Unsheltered', 'Sheltered'], showfliers=False)

plt.title(stub + ": " + str(time)[:10], fontsize=14)
plt.ylabel(productivity_variable, fontsize=12)

print(f"Shelter threshold = {int(percent_tree_threshold)}% tree cover within {distance * pixel_size}m")
print("Number of sheltered pixels: ", len(sheltered))
print("Number of unsheltered pixels: ", len(unsheltered))

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_boxplot_{time}.png")
plt.savefig(filename)
print(filename)
# -
# 11% increase in EVI at this timepoint in sheltered pixels compared to unsheltered
(np.median(sheltered) - np.median(unsheltered))/np.median(y_values)

# An actual effect size
standardized_mean_difference = (np.mean(sheltered) - np.mean(unsheltered))/np.std(y_values)
standardized_mean_difference

# I made up this effect size 
standardized_median_difference = (np.median(sheltered) - np.median(unsheltered)) / stats.median_abs_deviation(y_values)
standardized_median_difference

# +
# Find the max_temp for each date with satellite imagery 
xarray_times = ds['time'].values
nearest_times_indices = df_drought.index.get_indexer(xarray_times, method='nearest')
nearest_df_times = df_drought.index[nearest_times_indices]

# Find the timepoints where the temp is greater than 25 degrees
temp_threshold = 25
selected_times = nearest_df_times[df_drought['max_temp'].iloc[nearest_times_indices] > temp_threshold]
ds_drought = ds.sel(time=selected_times, method='nearest')

# +
# %%time
# Calculate the productivity score

# shelter_thresholds = 0.01, 0.02, 0.05, 0.1, 0.2, 0.3   # Percentage tree cover
shelter_thresholds = 0.005, 0.01, 0.03, 0.05, 0.1   # Percentage tree cover
# shelter_thresholds = 0.02, 0.04, 0.06, 0.08, 0.1   # Percentage tree cover

benefit_scores_dict = {}
sample_sizes_dict = {}

for i, distance in enumerate(distances):
    print(f"\nDistance threshold {i}/{len(distances)}", distance)
    layer_name = f"percent_trees_{distance * pixel_size}m"

    for i, shelter_threshold in enumerate(shelter_thresholds):
        print(f"Shelter threshold {i}/{len(shelter_thresholds)}", shelter_threshold)
        num_trees_threshold = ((distance * 2) ** 2) * shelter_threshold # Number of tree pixels
    
        benefit_scores = []
    
        for i, time in enumerate(ds.time.values):
            ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
            # productivity_score1 = ndvi.where(~adjacent_mask & (grassland | cropland))
            productivity_score1 = ndvi.where(~adjacent_mask) #  & (grassland | cropland))
            s = ds[layer_name].values
            
            # Flatten the arrays for plotting
            y = productivity_score1.values.flatten()
            y_values = y[~np.isnan(y)]   # Remove all pixels that are trees or adjacent to trees
            x = s.flatten()
            x_values = x[~np.isnan(y)]   # Match the shape of the x_values

            # Normalise the y values (for bare ground and non-photosynethetic vegetation comparison)
            # y_values = (y_values - min(y_values)) / (max(y_values) - min(y_values))
        
            # Select sheltered pixels and calculate z scores for NDVI at each pixel
            sheltered = y_values[np.where(x_values >= num_trees_threshold)]
            unsheltered = y_values[np.where(x_values < num_trees_threshold)]

            # Take a random sample to keep the sample sizes consistent across experiments
            # random_sample_size = 1000
            # sample_size = min(len(unsheltered), len(sheltered))
            # if sample_size < random_sample_size:
            #     sheltered = []
            #     unsheltered = []
            # else:
            #     random_values = np.random.choice(np.arange(0, len(sheltered)), size=random_sample_size, replace=False)
            #     sheltered = sheltered[random_values]
                
            #     random_values = np.random.choice(np.arange(0, len(unsheltered)), size=random_sample_size, replace=False)
            #     unsheltered = unsheltered[random_values]

            # Calculate the effect sizes
            median_diff = np.median(sheltered) - np.median(unsheltered)
            median_diff_standard = (np.median(sheltered) - np.median(unsheltered)) / stats.median_abs_deviation(y_values)
            mean_diff_standard = (np.mean(sheltered) - np.mean(unsheltered))/np.std(y_values)
    
            # Store the results
            benefit_score = {
                "time":time,
                f"median_{productivity_variable}":np.median(y_values),
                "median_sheltered": np.median(sheltered), 
                "median_unsheltered": np.median(unsheltered),
                "median_diff":median_diff,
                "median_diff_standard": median_diff_standard,
                "mean_diff_standard": mean_diff_standard,
            }
            benefit_score = benefit_score
            benefit_scores.append(benefit_score)
    
        # Save the results in a dictionary
        key = f"d:{distance}, s:{shelter_threshold}"
        benefit_scores_dict[key] = benefit_scores
        sample_sizes_dict[key] = {"sheltered": len(sheltered), "unsheltered": len(unsheltered)}
    
print(len(total_benefits))
# pd.DataFrame(total_benefits).head()

# +
# Comparing the median sheltered and unsheltered EVI
df = pd.DataFrame(benefit_scores_dict['d:10, s:0.01'])
df = df.set_index('time')
df = df.astype(float)
df.index = pd.to_datetime(df.index)
df = df[['median_sheltered', 'median_unsheltered']]
df_merged = pd.merge_asof(df, df_drought, left_index=True, right_index=True, direction='nearest')

# Time series plot
df_merged[['median_sheltered', 'median_unsheltered']].plot(figsize=(20,10))

# Add horizontal line at y=0
ax = plt.gca()
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Colour in dates where max temp > temp_threshold
temperature_threshold = 25
above_25 = df_merged['max_temp'] > temperature_threshold
start = None
for i, (date, above) in enumerate(above_25.items()):
    if above and start is None:
        start = date
    elif not above and start is not None:
        ax.axvspan(start, date, color='red', alpha=0.1)
        start = None

# Key
patch = mpatches.Patch(color='red', alpha=0.1, label=f'max_temp > {temperature_threshold}°C')
ax.legend(handles=[patch] + ax.get_legend().legendHandles, loc='upper left')

plt.ylabel(f"median EVI")
plt.xlabel("")
plt.title(f'{stub}: Shelter Benefit Time Series')

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_time_series.png")
plt.savefig(filename)
print(filename)

# +
# Creating time series dataframe of shelter_benefits alongsde max_temp
# variable = 'median_diff_standard' # The median diff standard shows the strongest benefits in 2020 because of the normalisation
variable = 'percentage_benefit'     # The percentage benefit is more interpretable like a productivity boost

df = pd.DataFrame(benefit_scores_dict['d:10, s:0.01'])
# df = pd.DataFrame(benefit_scores_dict['d:100, s:0.02'])
df = df.set_index('time')
df = df.astype(float)
df.index = pd.to_datetime(df.index)
df['percentage_benefit'] = 100 * df['median_diff']/df[f'median_{productivity_variable}']
df = df[variable]
df_merged = pd.merge_asof(df, df_drought, left_index=True, right_index=True, direction='nearest')

# Time series plot
df_merged[variable].plot(figsize=(20,10))

# Add horizontal line at y=0
ax = plt.gca()
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Colour in dates where max temp > temp_threshold
temperature_threshold = 25
above_25 = df_merged['max_temp'] > temperature_threshold
start = None
for i, (date, above) in enumerate(above_25.items()):
    if above and start is None:
        start = date
    elif not above and start is not None:
        ax.axvspan(start, date, color='red', alpha=0.1)
        start = None

# Key
patch = mpatches.Patch(color='red', alpha=0.1, label=f'max_temp > {temperature_threshold}°C')
ax.legend(handles=[patch], loc='upper left')

plt.ylabel(f"median {productivity_variable} increase (%)")
plt.xlabel("")
plt.title(f'{stub}: Shelter Benefit Time Series')

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_time_series.png")
plt.savefig(filename)
print(filename)

# +
# Creating time series dataframe of shelter_benefits alongsde max_temp

df = pd.DataFrame(benefit_scores_dict['d:10, s:0.01'])
# df = pd.DataFrame(benefit_scores_dict['d:100, s:0.02'])
df = df.set_index('time')
df = df.astype(float)
df.index = pd.to_datetime(df.index)

df['cumulative_benefit'] = df['median_diff'].cumsum()
df = df['cumulative_benefit']
df_merged = pd.merge_asof(df, df_drought, left_index=True, right_index=True, direction='nearest')

# Time series plot
df_merged['cumulative_benefit'].plot(figsize=(20,10))

# Add horizontal line at y=0
ax = plt.gca()
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Colour in dates where max temp > temp_threshold
temperature_threshold = 25
above_25 = df_merged['max_temp'] > temperature_threshold
start = None
for i, (date, above) in enumerate(above_25.items()):
    if above and start is None:
        start = date
    elif not above and start is not None:
        ax.axvspan(start, date, color='red', alpha=0.1)
        start = None

# Key
patch = mpatches.Patch(color='red', alpha=0.1, label=f'max_temp > {temperature_threshold}°C')
ax.legend(handles=[patch], loc='upper left')

plt.ylabel(f"cumulative {productivity_variable} increase")
plt.xlabel("")
plt.title(f'{stub}: Shelter Benefit Time Series')

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_time_series.png")
plt.savefig(filename)
print(filename)

# +
total_benefits = []

for key in benefit_scores_dict.keys():
    
    # Parse the distance and shelter thresholds
    distance_shelter = key.split(',')
    distance = int(distance_shelter[0].split(':')[1])
    shelter_threshold = float(distance_shelter[1].split(':')[1])
    layer_name = f"percent_trees_{distance * pixel_size}m"
    num_trees_threshold = ((distance * 2) ** 2) * shelter_threshold

    # Create a dataframe
    benefit_scores = benefit_scores_dict[key]
    df_shelter = pd.DataFrame(benefit_scores)
    df_shelter = df_shelter.set_index('time')
    
    # Join the weather data onto the shelter scores
    df_merged = pd.merge_asof(df_shelter, df_drought, left_index=True, right_index=True, direction='nearest')
    temperature_threshold = 25
    hot_days = np.where(df_merged['max_temp'] > temperature_threshold)
    
    # Aggregate results over all hot timepoints
    overall_median_ndvi = df_merged[f'median_{productivity_variable}'].iloc[hot_days].median()
    overall_median_diff = df_merged['median_diff'].iloc[hot_days].median()
    overall_median_diff_standard = df_merged['median_diff_standard'].iloc[hot_days].median()
    overall_mean_diff_standard = df_merged['mean_diff_standard'].iloc[hot_days].median()
    
    # Store the aggregated results
    total_benefit = {
            "distance_threshold":distance,
            "shelter_threshold":shelter_threshold,
            "sheltered_pixels":sample_sizes_dict[key]['sheltered'],
            "unsheltered_pixels":sample_sizes_dict[key]['unsheltered'],
            f"median_{productivity_variable}":overall_median_ndvi,
            "median_difference": overall_median_diff,
            "standardized_median_difference": overall_median_diff_standard,
            "standardized_mean_difference": overall_mean_diff_standard,
        }
    total_benefits.append(total_benefit)

len(total_benefits)
# -

df = pd.DataFrame(total_benefits)
df['distance'] = df['distance_threshold'] * pixel_size
df['percentage_trees'] = df['shelter_threshold'] * 100
df['percentage_benefit'] = 100 * df['median_difference']/df[f'median_{productivity_variable}']
df['min_sample_size'] = df['median_difference']/df[f'median_{productivity_variable}']
df['min_sample_size'] = df[['sheltered_pixels', 'unsheltered_pixels']].min(axis=1)
df

# +
# Visualise the sample sizes
heatmap_data = df.pivot(index='distance', columns='percentage_trees', values='min_sample_size')

threshold = 30000
annotations = np.where(heatmap_data < threshold, heatmap_data.astype(int).astype(str), "")
ax = sns.heatmap(heatmap_data, annot=annotations, fmt='', cbar=True)
ax.invert_yaxis()

plt.title(f'{stub}: Sheltered vs Unsheltered')
plt.xlabel('Tree Cover (%)')
plt.ylabel('Distance (m)')
cbar = ax.collections[0].colorbar
cbar.set_label('Sample size')

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_sample_sizes.png")
plt.savefig(filename)
print(filename)
# +
# Heatmap comparison
heatmap_data = df.pivot(index='distance', columns='percentage_trees', values='percentage_benefit')
ax = sns.heatmap(heatmap_data, annot=True, cmap="YlGn", cbar=True)
ax.invert_yaxis()

plt.title(f'{stub}: Sheltered vs Unsheltered')
plt.xlabel('Tree Cover (%)')
plt.ylabel('Distance (m)')
cbar = ax.collections[0].colorbar
cbar.set_label(f'median {productivity_variable} increase (%)')

filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_shelter_benefits.png")
plt.savefig(filename)
print(filename)

# +
# # Visualise the summer productivity
ds_drought_median = ds_drought[productivity_variable].median(dim='time')
ds_drought_masked = ds_drought_median.where(~adjacent_mask)

filename = os.path.join(scratch_dir, f'{stub}_summer_{productivity_variable}.tif')
ds_drought_masked.rio.to_raster(filename)
print(filename)

ds_drought_masked.plot(cbar_kwargs={'label': f'Median {productivity_variable} when max_temp > {temperature_threshold}°C'})
plt.title(f"{stub} Productivity Score")
filename = os.path.join(scratch_dir, f'{stub}_summer_{productivity_variable}.png')
plt.savefig(filename)
print(filename)
# -






