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
from pyproj import Transformer

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS (degrees, minutes, seconds) to decimal degrees."""
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:  # South and West are negative
        decimal *= -1
    return decimal

# Convert the given coordinates
lat_min = dms_to_decimal(25, 22, 51.54, 'S')
lon_min = dms_to_decimal(147, 9, 35.17, 'E')

lat_max = dms_to_decimal(25, 22, 47.40, 'S')
lon_max = dms_to_decimal(147, 9, 41.09, 'E')

# Print results
cropped_bounds = (lat_min, lon_min, lat_max, lon_max)

# endregion

# region
# Crop the region to just Brodie's transects

# Define the coordinate transformation from EPSG:4326 to EPSG:6933
transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

# Convert given lat/lon bounds to EPSG:6933
lon_min, lat_min = 147 + 9/60 + 35.17/3600, -(25 + 22/60 + 51.54/3600)
lon_max, lat_max = 147 + 9/60 + 41.09/3600, -(25 + 22/60 + 47.40/3600)

x_min, y_min = transformer.transform(lon_min, lat_min)
x_max, y_max = transformer.transform(lon_max, lat_max)

cropped_bounds = x_min, y_min, x_max, y_max

# Crop the dataset
ds_cropped = ds.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)

# Check the new bounds
print(ds_cropped.rio.bounds())

ds_uncropped = ds
ds = ds_cropped
# endregion

# region
# Global canopy height tree_mask
tree_percent = ds['tree_percent'].values[0]
tree_mask = tree_percent > 0

# Shelterscore showing the number of trees within a donut at a given distance away from the crop/pasture pixel
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
# endregion

# Enhanced Vegetation Index
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
productivity_variable = 'EVI'

time = "2022-08-04"

ds_timepoint = ds.sel(time=time, method='nearest')
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
red = ds_timepoint['nbart_red']
green = ds_timepoint['nbart_green']
blue = ds_timepoint['nbart_blue']
rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)

# region
# RGB Image
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
useful_variables = ['nbart_red', 'nbart_green', 'nbart_blue', 'EVI', 'tree_percent', 'max_tree_height', 'percent_trees_0m-300m']
ds_small = ds.isel(band=0)[useful_variables]

# region
from matplotlib import colors

# Prep formatting functions
def remove_axis_labels(ax):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

# Create a dummy white colorbar to align the plots nicely
white_cmap = LinearSegmentedColormap.from_list("white_cmap", ["white", "white"])
white_norm = Normalize(vmin=0, vmax=1)
sm_white = ScalarMappable(norm=white_norm, cmap=white_cmap)

# Add a colour bar with a specified title and label
def add_cbar(im, title="", label_size=16):
    cbar = im.colorbar
    cbar.set_label(title, fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size)
    return cbar

# Prep cmaps
cmap_EVI = plt.cm.coolwarm
cmap_EVI.set_bad(color='green')  # Set NaN pixels to green
cmap_tree_height = plt.cm.viridis
cmap_tree_height.set_bad(color='white')
# endregion

# region
# Compare the different productivity scores across the whole region

# Extracting a single timepoint for RGB and productivity plots
ds_timepoint = ds.sel(time=time, method='nearest')
layer_name = f"percent_trees_0m-300m"
s = ds[layer_name].values
x = s.flatten()

productivity_variable = 'EVI'

# Calculate the productivity and shelter scores
ds_productivity = ds.sel(time=time, method='nearest')[productivity_variable]
ds_masked = ds_productivity.where(~adjacent_mask)
y = ds_masked.values.flatten()
y_values_outliers = y[~np.isnan(y)]  
x_values_outliers = x[~np.isnan(y)]  
lower_bound = np.percentile(y_values_outliers, 1)
upper_bound = np.percentile(y_values_outliers, 99)
y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]    
x_values_outliers = x[~np.isnan(y)]
x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
unsheltered = y_values[np.where(x_values < tree_cover_threshold)]
median_value = np.median(unsheltered)

vmin_EVI = median_value - (upper_bound - lower_bound) / 2
vmax_EVI = median_value + (upper_bound - lower_bound) / 2
ds_trees = ds_productivity.where(~tree_mask)

productivity_stats[productivity_variable] = {
    "vmin_EVI": vmin_EVI,
    "vmax_EVI": vmax_EVI,
    "ds_trees": ds_trees
}

# Plotting the maps in subplots
fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle(f"Paddock {paddock_id} on {time}", fontsize=26)

# Fontsizes
title_size = 20
label_size = 16
annotation_size = 12

# Axes
productivity_variable = "EVI"

vmin_EVI = productivity_stats[productivity_variable]['vmin_EVI']
vmax_EVI = productivity_stats[productivity_variable]['vmax_EVI']
ds_trees = productivity_stats[productivity_variable]['ds_trees']
im = ds_trees.plot(cmap=cmap_EVI, vmin=vmin_EVI, vmax=vmax_EVI, add_colorbar=True)
ax.set_title(f"Productivity Proxy", fontsize=title_size)
add_cbar(im, productivity_variable, label_size)

plt.tight_layout()
filename = os.path.join(scratch_dir, f"{stub}_Paddock_productivities_{time}.png")
plt.savefig(filename)
plt.show()
print("Saved", filename)

# endregion

ds_buffered = ds


# region
def calculate_shelter_effects(ds, adjacent_mask, tree_cover_threshold=1):
    benefits = []
    layer_name = f"percent_trees_{pixel_size * min_distance}m-{pixel_size * max_distance}m"
    shelter_score = ds[layer_name]
    x = shelter_score.values.flatten()
    for i, time in enumerate(ds.time.values):
        ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
        productivity_score = ndvi.where(~adjacent_mask)
    
        # Remove all pixels that are trees, adjacent to trees, or masked by cloud cover
        y = productivity_score.values.flatten()
        y_values_outliers = y[~np.isnan(y)]   
    
        if len(y_values_outliers) == 0:
            continue
    
        # Remove outliers
        lower_bound = 0
        upper_bound = max(np.percentile(y_values_outliers, 99.9), 1)
        y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
        x_values_outliers = x[~np.isnan(y)]
        x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
        
        sheltered = y_values[np.where(x_values >= tree_cover_threshold)]
        unsheltered = y_values[np.where(x_values < tree_cover_threshold)]
        
        if len(sheltered) == 0 or len(unsheltered) == 0:
            continue
            
        percentage_benefit = (np.median(sheltered) - np.median(unsheltered))/np.median(unsheltered)
        sample_size = min(len(sheltered), len(unsheltered))
        
        res = stats.linregress(x_values, y_values)
    
        benefit = {
            "distance":max_distance,
            "time": time,
            "r2": res.rvalue**2,
            "slope": res.slope,
            "percentage_benefit": percentage_benefit,
            "sample_size": sample_size,
            "median": np.median(y_values),
            "q1": np.percentile(y_values, 25),
            "q3": np.percentile(y_values, 75)
        }
        benefits.append(benefit)

    # Create a dataframe of the shelter benefits
    df_benefits = pd.DataFrame(benefits)
    if len(df_benefits) == 0:
        df_benefits.index = pd.to_datetime(df_benefits.index)
        return df_benefits
    
    df_benefits['date'] = df_benefits['time'].dt.date
    df_benefits = df_benefits.set_index('date')
    df_benefits.index = pd.to_datetime(df_benefits.index)
    
    filename = os.path.join(scratch_dir, f"{stub}_benefits.csv")
    df_benefits.to_csv(filename)
    print("Saved: ", filename)

    return df_benefits

df_benefits = calculate_shelter_effects(ds_buffered, adjacent_mask)
# endregion

# region
def plot_histogram(ds, time, tree_cover_threshold=1):
    
    ds_timepoint = ds.sel(time=time, method='nearest')
        
    # Calculate shelter score and productivity index for this timepoint
    ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
    p = ndvi.where(~adjacent_mask) #  & (grassland | cropland))
    layer_name = f"percent_trees_0m-300m"
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
    
    # Plot 1: 2D histogram 
    fig, axes = plt.subplots(2, 1, figsize=(14, 16)) 
    title_size = 30
    label_size = 26
    annotations_size = label_size
    ax1 = axes[0]
    hist = ax1.hist2d(
        x_values, y_values, 
        bins=100, 
        norm=mcolors.LogNorm(),
        cmap='viridis',
    )
    ax1.set_title(f"Vegetation Index vs Shelter Score on {time}", fontsize=title_size)
    ax1.set_xlabel(f"Tree cover within {max_distance * pixel_size}m (%)", fontsize=label_size)
    ax1.set_ylabel(f'{productivity_variable}', fontsize=label_size)
    ax1.tick_params(axis='both', labelsize=annotations_size)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    cbar = plt.colorbar(hist[3], ax=ax1)  # hb[3] contains the QuadMesh, which is used for colorbar
    cbar.set_label(f"Number of pixels", fontsize=label_size)
    cbar.ax.tick_params(labelsize=annotations_size)
    
    # Linear regression line
    if len(np.unique(x_values)) > 1:
        res = stats.linregress(x_values, y_values)
        x_fit = np.linspace(min(x_values), max(x_values), 500)
        y_fit = res.intercept + res.slope * x_fit
        ax1.plot(x_fit, y_fit, 'r-', label=f"$R^2$ = {res.rvalue**2:.2f}")
        ax1.legend(fontsize=label_size)
    
    # Add vertical black dotted line at the tree cover threshold
    ax1.axvline(
        tree_cover_threshold, 
        color='black', 
        linestyle='dotted', 
        linewidth=2, 
        label=f"Tree cover = {tree_cover_threshold}%"
    )
    
    # Plot 2: Box plot
    ax2 = axes[1]

    # Calculate sheltered and unsheltered pixels
    sheltered = y_values[np.where(x_values >= tree_cover_threshold)]
    unsheltered = y_values[np.where(x_values < tree_cover_threshold)]

    # Don't draw the boxplot if all the pixels are in the same category
    all_same_category = (len(sheltered) == 0) or (len(unsheltered) == 0)
    if all_same_category:
        ax2.set_title(f"Pixels all same category of unsheltered or sheltered", fontsize=title_size)

    elif not(all_same_category):    
        box_data = [unsheltered, sheltered]
        im = ax2.boxplot(box_data, labels=['Unsheltered', 'Sheltered'], showfliers=False)
        ax2.set_title(f'Shelter threshold of {tree_cover_threshold}% tree cover within {max_distance * pixel_size}m', fontsize=title_size)
        ax2.set_ylabel(productivity_variable, fontsize=label_size)
        ax2.tick_params(axis='both', labelsize=annotations_size)
        
        # Add medians and sample size next to each box plot
        medians = [np.median(data) for data in box_data]
        number_of_pixels = [len(unsheltered), len(sheltered)]
        
        placement_unsheltered = np.percentile(unsheltered, 75) + (1.5 * (np.percentile(unsheltered, 75) - np.percentile(unsheltered, 25)))
        placement_sheltered = np.percentile(sheltered, 75) + (1.5 * (np.percentile(sheltered, 75) - np.percentile(sheltered, 25)))
        n_placements = [placement_unsheltered, placement_sheltered]
        
        for i, median in enumerate(medians):
            ax2.text(i + 1 + 0.09, median, f'{median:.2f}', ha='left', va='center', fontsize=label_size)
            ax2.text(i + 1 - 0.09, n_placements[i] + 0.015, f'n={number_of_pixels[i]}', ha='left', va='center', fontsize=label_size)
        
        # Add some space above the sample size text
        y_max = max(placement_unsheltered, placement_sheltered) + 0.1 * max(placement_unsheltered, placement_sheltered)
        ax2.set_ylim(None, y_max)
        
        # Create a dummy white colorbar to align the plots nicely
        white_cmap = LinearSegmentedColormap.from_list("white_cmap", ["white", "white"])
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=white_cmap)
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
        cbar.set_ticks([])  
        cbar.set_label('')  
        cbar.outline.set_visible(False)
        
    # Save the plots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3) 
    filename = os.path.join(scratch_dir, f"{stub}_{time}_regression.png")
    plt.savefig(filename)
    plt.show()
    print("Saved", filename)
    
time = df_benefits.index[0].date()
# time = "2020-01-08"  
time = "2022-08-04"

plot_histogram(ds_buffered, time)
# endregion
# region
def plot_timeseries(ds, df_benefits, stub):
    filename_ozwald = os.path.join(outdir, f"{stub}_ozwald_8day.nc")
    filename_silo = os.path.join(outdir, f"{stub}_silo_daily.nc")
    ds_ozwald = xr.open_dataset(filename_ozwald)
    ds_silo = xr.open_dataset(filename_silo)
    df_daily = merge_ozwald_silo(ds_ozwald, ds_silo)
    df_weekly = resample_weekly(df_daily)
    
    # Merge shelter benefits
    df_merged = pd.merge_asof(df_weekly, df_benefits, left_index=True, right_index=True, direction='nearest')
    df = df_merged
    
    # Plot 1: shelter benefits
    fig, axes = plt.subplots(2, 1, figsize=(50, 30))  # Create two vertically stacked subplots
    title_fontsize = 70
    tick_size = 42
    
    ax = axes[0]
    ax.plot(df.index, df["r2"] * 100, color='black', label='Shelter score vs productivity index ($r^2 \\times 100$)')
    ax.plot(df.index, df["percentage_benefit"] * 100, color='grey')
    opacity = 0.3
    ax.fill_between(
        df.index, 
        0, 
        df["percentage_benefit"] * 100, 
        where=(df["percentage_benefit"] > 0), 
        color='limegreen', 
        alpha=opacity, 
        interpolate=True,
        label='Sheltered > unsheltered (%)'
    )
    ax.fill_between(
        df.index, 
        0, 
        df["percentage_benefit"] * 100, 
        where=(df["percentage_benefit"] < 0), 
        color='red', 
        alpha=opacity, 
        interpolate=True,
        label='Sheltered < unsheltered (%)'
    )
    ax.set_title(f"Time Series of Shelter Benefits", fontsize=title_fontsize)
    ax.legend(fontsize=tick_size, loc='upper left')
    ax.tick_params(axis='both', labelsize=tick_size)
    
    # Plot 2: Weather data
    ax = axes[1]
    ax.set_title(f"Environmental Variables", fontsize=title_fontsize)
    EVI_scale_factor = 100
    
    rainfall_plot = ax.bar(df.index, df['Rainfall']/EVI_scale_factor, color='skyblue', width=5, label=r'Weekly Rainfall (mm $\times 10^2$)')
    ax.bar(df.index, df['Potential Evapotranspiration']/EVI_scale_factor, color='orange', label=r"Potential Evapotranspiration (mm $\times 10^2$)")
    ax.plot(df.index, df['Minimum Soil Moisture']/EVI_scale_factor, color='blue', label="Soil moisture (mm)")
    ax.plot(df.index, df["q1"], color='grey')
    ax.plot(df.index, df["q3"], color='grey')
    
    # Plot the interquartile range
    q1 = df["q1"] 
    q3 = df["q3"]
    ax.fill_between(df.index, q1, q3, color='green', alpha=opacity, label="Enhanced Vegetation Index (IQR)")
    
    ax.legend(fontsize=tick_size, loc='upper left')
    ax.tick_params(axis='both', labelsize=tick_size)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    
    # Save as a single image
    filename_combined = os.path.join(scratch_dir, f"{stub}_time_series.png")
    plt.savefig(filename_combined)
    print("Saved", filename_combined)

ds_buffered = ds
plot_timeseries(ds_buffered, df_benefits, stub)
# endregion

# region
def plot_maps(ds, tree_mask, stub):

    # Extracting a single timepoint for RGB and productivity plots
    ds_timepoint = ds.sel(time=time, method='nearest')

    # Setting up the RGB layers
    red = ds_timepoint['nbart_red']
    green = ds_timepoint['nbart_green']
    blue = ds_timepoint['nbart_blue']
    rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
    bounds = ds_buffered[productivity_variable].rio.bounds()
    left, bottom, right, top = bounds

    # Calculate the productivity and shelter scores
    ds_productivity = ds.sel(time=time, method='nearest')[productivity_variable]
    ds_masked = ds_productivity.where(~adjacent_mask)
    layer_name = f"percent_trees_0m-300m"
    s = ds[layer_name].values
    y = ds_masked.values.flatten()
    y_values_outliers = y[~np.isnan(y)]  
    x = s.flatten()
    x_values_outliers = x[~np.isnan(y)]  
    
    # Remove outliers
    lower_bound = np.percentile(y_values_outliers, 1)
    upper_bound = np.percentile(y_values_outliers, 99)
    y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]    
    x = s.flatten()
    x_values_outliers = x[~np.isnan(y)]
    x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
    unsheltered = y_values[np.where(x_values < tree_cover_threshold)]
    median_value = np.median(unsheltered)

    # Calculate colour bar boundaries
    vmin_EVI = median_value - (upper_bound - lower_bound) / 2
    vmax_EVI = median_value + (upper_bound - lower_bound) / 2
    ds_trees = ds_productivity.where(~tree_mask)

    ###############################
    # Plotting the maps in subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"{time}", fontsize=26)
    
    # Fontsizes
    title_size = 20
    label_size = 16
    annotation_size = 12
    
    # EVI
    ax = axes[0,0]
    im = ds_trees.plot(ax=ax, cmap=cmap_EVI, vmin=vmin_EVI, vmax=vmax_EVI, add_colorbar=True)
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Productivity Proxy", fontsize=title_size)
    add_cbar(im, productivity_variable, label_size)
    
    # RGB
    ax = axes[0,1]
    ax.imshow(rgb, extent=(left, right, bottom, top))
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Sentinel-2 Imagery", fontsize=title_size)
    
    scalebar = AnchoredSizeBar(
        ax.transData, 1000, '1km', loc='lower center', pad=0.1, 
        color='white', frameon=False, size_vertical=10, 
        fontproperties=fm.FontProperties(size=label_size)
    )
    ax.add_artist(scalebar)
    
    cbar = plt.colorbar(sm_white, ax=ax, orientation='vertical')
    cbar.set_ticks([])  
    cbar.set_label('')  
    cbar.outline.set_visible(False)
    
    # Shelter score
    ax = axes[1,0] 
    im = ds_buffered['percent_trees_0m-300m'].plot(ax=ax, cmap=cmap_EVI, vmin=0, vmax=30) 
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Shelter Score", fontsize=title_size)
    add_cbar(im, "Tree cover within 300m (%)", annotation_size)
    
    # Canopy Height
    ax = axes[1,1] 
    data = ds_buffered['max_tree_height'].where(ds_buffered['max_tree_height'] != 0, np.nan) 
    im = data.plot(ax=ax, cmap=cmap_tree_height, vmin=0, vmax=20) 
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Canopy Height", fontsize=title_size)
    add_cbar(im, "metres", label_size)
    
    # Remove axes
    for row in axes:
        for ax in row:
            remove_axis_labels(ax)
    
    plt.tight_layout()
    filename = os.path.join(scratch_dir, f"{stub}_maps_{time}.png")
    plt.savefig(filename)
    plt.show()
    print("Saved", filename)

    # True colour tiff
    filename = os.path.join(scratch_dir, f"{stub}_RGB_{time}.tif")
    ds_timepoint.attrs = {}
    rgb_stack = ds_timepoint[['nbart_red', 'nbart_green', 'nbart_blue']]
    rgb_stack.rio.to_raster(filename)
    print("Saved", filename)
    
    # Productivity tiff
    filename = os.path.join(scratch_dir, f"{stub}_{productivity_variable}_{time}.tif")
    clipped = ds_masked.fillna(upper_bound + 0.1)
    clipped = clipped.clip(min=lower_bound, max=upper_bound)
    clipped.attrs = {}
    clipped.rio.to_raster(filename)
    print("Saved", filename)
    
plot_maps(ds_buffered, tree_mask, stub)

# endregion

# region
def plot_productivities(ds, tree_mask, stub):
# ds = ds_buffered

    # paddock_row = pol[pol['paddock'] == paddock_id]
    
    # Extracting a single timepoint for RGB and productivity plots
    ds_timepoint = ds.sel(time=time, method='nearest')
    layer_name = f"percent_trees_0m-300m"
    s = ds[layer_name].values
    x = s.flatten()
    
    ###############################
    
    productivity_variables = ['EVI', 'bg', 'pv', 'npv']
    productivity_stats = dict()
    
    for productivity_variable in productivity_variables:
    
        # Calculate the productivity and shelter scores
        ds_productivity = ds.sel(time=time, method='nearest')[productivity_variable]
        ds_masked = ds_productivity.where(~adjacent_mask)
        y = ds_masked.values.flatten()
        y_values_outliers = y[~np.isnan(y)]  
        x_values_outliers = x[~np.isnan(y)]  
        lower_bound = np.percentile(y_values_outliers, 1)
        upper_bound = np.percentile(y_values_outliers, 99)
        y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]    
        x_values_outliers = x[~np.isnan(y)]
        x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
        unsheltered = y_values[np.where(x_values < tree_cover_threshold)]
        median_value = np.median(unsheltered)
        
        vmin_EVI = median_value - (upper_bound - lower_bound) / 2
        vmax_EVI = median_value + (upper_bound - lower_bound) / 2
        ds_trees = ds_productivity.where(~tree_mask)
    
        productivity_stats[productivity_variable] = {
            "vmin_EVI": vmin_EVI,
            "vmax_EVI": vmax_EVI,
            "ds_trees": ds_trees
        }
    
    
    
    # Plotting the maps in subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"{time}", fontsize=26)
    
    # Fontsizes
    title_size = 20
    label_size = 16
    annotation_size = 12
    
    # Axes
    ax = axes[0,0]
    productivity_variable = "EVI"
    
    vmin_EVI = productivity_stats[productivity_variable]['vmin_EVI']
    vmax_EVI = productivity_stats[productivity_variable]['vmax_EVI']
    ds_trees = productivity_stats[productivity_variable]['ds_trees']
    im = ds_trees.plot(ax=ax, cmap=cmap_EVI, vmin=vmin_EVI, vmax=vmax_EVI, add_colorbar=True)
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Productivity Proxy", fontsize=title_size)
    add_cbar(im, productivity_variable, label_size)
    
    ax = axes[0,1]
    productivity_variable = "bg"
    
    vmin_EVI = productivity_stats[productivity_variable]['vmin_EVI']
    vmax_EVI = productivity_stats[productivity_variable]['vmax_EVI']
    ds_trees = productivity_stats[productivity_variable]['ds_trees']
    im = ds_trees.plot(ax=ax, cmap=cmap_EVI, vmin=vmin_EVI, vmax=vmax_EVI, add_colorbar=True)
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Productivity Proxy", fontsize=title_size)
    add_cbar(im, productivity_variable, label_size)
    
    ax = axes[1,0]
    productivity_variable = "pv"
    
    vmin_EVI = productivity_stats[productivity_variable]['vmin_EVI']
    vmax_EVI = productivity_stats[productivity_variable]['vmax_EVI']
    ds_trees = productivity_stats[productivity_variable]['ds_trees']
    im = ds_trees.plot(ax=ax, cmap=cmap_EVI, vmin=vmin_EVI, vmax=vmax_EVI, add_colorbar=True)
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Productivity Proxy", fontsize=title_size)
    add_cbar(im, productivity_variable, label_size)
    
    ax = axes[1,1]
    productivity_variable = "npv"
    
    vmin_EVI = productivity_stats[productivity_variable]['vmin_EVI']
    vmax_EVI = productivity_stats[productivity_variable]['vmax_EVI']
    ds_trees = productivity_stats[productivity_variable]['ds_trees']
    im = ds_trees.plot(ax=ax, cmap=cmap_EVI, vmin=vmin_EVI, vmax=vmax_EVI, add_colorbar=True)
    # paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)
    ax.set_title(f"Productivity Proxy", fontsize=title_size)
    add_cbar(im, productivity_variable, label_size)
    
    # Remove axes
    for row in axes:
        for ax in row:
            remove_axis_labels(ax)
    
    plt.tight_layout()
    filename = os.path.join(scratch_dir, f"{stub}_productivities_{time}.png")
    plt.savefig(filename)
    plt.show()
    print("Saved", filename)

plot_productivities(ds_buffered, tree_mask, stub)

# endregion

# region
# %%time
# adjacent_mask, tree_mask, ds_buffered = calculate_adjacency_mask(pol, ds_small, paddock_id)

df_benefits = calculate_shelter_effects(ds_buffered, adjacent_mask)
# time = "2020-01-08"   
plot_histogram(ds_buffered, time)
if len(df_benefits) > 0:
    plot_timeseries(ds_buffered, df_benefits, stub)
plot_maps(ds_buffered, tree_mask, stub)
# endregion



