# +
# Aim of this notebook is to combine all the datasources into a single xarray

# +
# # !pip install contextily

# +
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
from matplotlib.font_manager import FontProperties
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

world_cover_layers = {
    "Tree cover": 10, # Green
    "Shrubland": 20, # Orange
    "Grassland": 30, # Yellow
    "Cropland": 40, # pink
    "Built-up": 50, # red
    "Permanent water bodies": 80, # blue
}

# The resampling often messes up the boundary, so we trim the outside pixels after adding all the resampled bounds
ds = ds.isel(
    y=slice(1, -1),
    x=slice(1, -1) 
)

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

# +
# %%time
# Shelterscore showing the number of trees within a donut at a given distance away from the crop/pasture pixel
distances = 0, 30

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
# Enhanced Vegetation Index
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))

productivity_variable = 'EVI'
# -

# # All Timepoints

# +
# %%time
tree_cover_threshold = 10

benefits = []

min_distance = 0
max_distance = 30
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
    lower_bound = 0
    upper_bound = max(np.percentile(y_values_outliers, 99.9), 1)
    y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
    x_values_outliers = x[~np.isnan(y)]
    x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
    
    sheltered = y_values[np.where(x_values >= tree_cover_threshold)]
    unsheltered = y_values[np.where(x_values < tree_cover_threshold)]
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

len(benefits)
# -

df_benefits = pd.DataFrame(benefits)
df_benefits['date'] = df_benefits['time'].dt.date
df_benefits = df_benefits.set_index('date')
df_benefits.index = pd.to_datetime(df_benefits.index)
df_top10 = df_benefits.nlargest(10, 'r2')
df_top10

# # Max r2 timepoint

time = df_top10.index[0].date()
ds_timepoint = ds.sel(time=time, method='nearest')

# +
# Calculate shelter score and productivity index for this timepoint
ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
productivity_score1 = ndvi.where(~adjacent_mask) #  & (grassland | cropland))
layer_name = f"percent_trees_0m-300m"
s = ds[layer_name].values

# Remove all pixels that are trees or adjacent to trees
y = productivity_score1.values.flatten()
y_values_outliers = y[~np.isnan(y)]   

# Outlier boundary
lower_bound = 0
upper_bound = max(np.percentile(y_values_outliers, 99.9), 1)

# Find the shelter scores not obstructed by cloud cover or outliers
y_values = y_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]
x = s.flatten()
x_values_outliers = x[~np.isnan(y)]
x_values = x_values_outliers[(y_values_outliers > lower_bound) & (y_values_outliers < upper_bound)]

# Calculate sheltered and unsheltered pixels
percent_tree_threshold = 10
sheltered = y_values[np.where(x_values >= percent_tree_threshold)]
unsheltered = y_values[np.where(x_values < percent_tree_threshold)]
# +
fig, axes = plt.subplots(2, 1, figsize=(14, 16)) 
title_size = 22
label_size = 18
annotations_size = 14

# Plot 1: 2D histogram 
ax1 = axes[0]
hist = ax1.hist2d(
    x_values, y_values, 
    bins=100, 
    norm=mcolors.LogNorm(),
    cmap='viridis',
)
ax1.set_title(f"{stubs[stub]} Productivity Index vs Shelter Score at {time}", fontsize=title_size)
ax1.set_xlabel(f"Tree cover within {max_distance * pixel_size}m (%)", fontsize=label_size)
ax1.set_ylabel(f'Enhanced Vegetation Index ({productivity_variable})', fontsize=label_size)
ax1.tick_params(axis='both', labelsize=annotations_size)

cbar = plt.colorbar(hist[3], ax=ax1)  # hb[3] contains the QuadMesh, which is used for colorbar
cbar.set_label(f"Number of pixels ({productivity_variable})", fontsize=label_size)
cbar.ax.tick_params(labelsize=annotations_size)

# Linear regression line
res = stats.linregress(x_values, y_values)
x_fit = np.linspace(min(x_values), max(x_values), 500)
y_fit = res.intercept + res.slope * x_fit
ax1.plot(x_fit, y_fit, 'r-', label=f"$R^2$ = {res.rvalue**2:.2f}")
ax1.legend(fontsize=14)

# Add vertical black dotted line at the tree cover threshold
ax1.axvline(
    percent_tree_threshold, 
    color='black', 
    linestyle='dotted', 
    linewidth=2, 
    label=f"Tree cover = {tree_cover_threshold}%"
)


# Plot 2: Box plot
ax2 = axes[1]
box_data = [unsheltered, sheltered]
im = ax2.boxplot(box_data, labels=['Unsheltered', 'Sheltered'], showfliers=False)
ax2.set_title(f'Shelter threshold of {percent_tree_threshold}% tree cover within {max_distance * pixel_size}m', fontsize=title_size)
ax2.set_ylabel('Enhanced Vegetation Index (EVI)', fontsize=label_size)
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

# Explanatory text for calculating percentage benefit
shelter_vs_unsheltered = (np.median(sheltered) - np.median(unsheltered)) / np.median(unsheltered) * 100
ax2.text(
    0.53, y_max - 0.02,  # Position text in top left
    f'Sheltered vs unsheltered (%) = ({medians[1]:.2f} - {medians[0]:.2f})/{medians[0]:.2f} = {shelter_vs_unsheltered:.2f}%',
    fontsize=annotations_size, ha='left', va='top'
)

# Create a dummy white colorbar to align the plots nicely
white_cmap = LinearSegmentedColormap.from_list("white_cmap", ["white", "white"])
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap=white_cmap)
cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
cbar.set_ticks([])  
cbar.set_label('')  
cbar.outline.set_visible(False)

# Save the plots
fig.tight_layout()
plt.subplots_adjust(hspace=0.2) 

filename = os.path.join(scratch_dir, f"{stub}_hist_and_boxplot.png")
plt.savefig(filename)
plt.show()
print("Saved", filename)


# -
# # Temporal Variation

# Load weather data
filename_ozwald = os.path.join(outdir, f"{stub}_ozwald_8day.nc")
filename_silo = os.path.join(outdir, f"{stub}_silo_daily.nc")
ds_ozwald = xr.open_dataset(filename_ozwald)
ds_silo = xr.open_dataset(filename_silo)
df_daily = merge_ozwald_silo(ds_ozwald, ds_silo)
df_weekly = resample_weekly(df_daily)

# Merge shelter benefits
df_merged = pd.merge_asof(df_weekly, df_benefits, left_index=True, right_index=True, direction='nearest')
df = df_merged

# +
fig, axes = plt.subplots(2, 1, figsize=(50, 30))  # Create two vertically stacked subplots
title_fontsize = 70
tick_size = 30

# Visualise the shelter benefits
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
ax.set_title(f"{stubs[stub]} Time Series of Shelter Benefits", fontsize=title_fontsize)
ax.legend(fontsize=tick_size, loc='upper left')
ax.tick_params(axis='both', labelsize=tick_size)

# Visualise the weather data
ax = axes[1]
rainfall_plot = ax.bar(df.index, df['Rainfall'], color='skyblue', width=5, label='Rainfall (mm)')
ax.bar(df.index, df['Potential Evapotranspiration'], color='orange', label="Evapotranspiration (mm)")
ax.plot(df.index, df['Minimum Soil Moisture'], color='blue', label="Soil moisture (mm)")
ax.plot(df.index, df["q1"] * 100, color='grey')
ax.plot(df.index, df["q3"] * 100, color='grey')

# Plot the interquartile range
q1 = df["q1"] * 100 
q3 = df["q3"] * 100
ax.fill_between(df.index, q1, q3, color='green', alpha=opacity, label="Overall productivity x100")

ax.set_title(f"{stubs[stub]} Weather", fontsize=title_fontsize)
ax.legend(fontsize=tick_size, loc='upper left')
ax.tick_params(axis='both', labelsize=tick_size)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Save as a single image
filename_combined = os.path.join(scratch_dir, f"{stub}_shelter_weather.png")
plt.savefig(filename_combined)
plt.show()
print("Saved", filename_combined)
# -

# # Spatial Variation

# +
# Calculate aspect ratio of this region
earth_radius_km = 6371

# Lat and lon parameters used to generate the imagery
filename = os.path.join(outdir, f"{stub}_ds2_query.pkl")
with open(filename, 'rb') as file:
    query = pickle.load(file)
latitude_deg = query['y'][0]
lat_diff_deg = query['y'][1] - query['y'][0]
lon_diff_deg = query['x'][1] - query['x'][0]

# Conversion to km
latitude_rad = math.radians(latitude_deg)
lat_distance_km = lat_diff_deg * (math.pi * earth_radius_km / 180)
lon_distance_km = lon_diff_deg * (math.cos(latitude_rad) * (math.pi * earth_radius_km / 180))

# Conversion to aspect for matplotlib plotting
lon_distance_km, lat_distance_km
# -

# Load and calculate topography layers
filename = os.path.join(outdir, f"{stub}_terrain.tif")
grid, dem, fdir, acc = pysheds_accumulation(filename)
num_catchments = 20
gullies, full_branches = catchment_gullies(grid, fdir, acc, num_catchments)
ridges = catchment_ridges(grid, fdir, acc, full_branches)
slope = calculate_slope(filename)


# +
def add_tiff_band(ds, variable, resampling_method, outdir, stub):
    """Add a new band to the xarray from a tiff file using the given resampling method"""
    filename = os.path.join(outdir, f"{stub}_{variable}.tif")
    array = rxr.open_rasterio(filename)
    reprojected = array.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected.isel(band=0).drop_vars('band')
    return ds

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

# Add the topography bands
ds = add_tiff_band(ds, "terrain", Resampling.average, outdir, stub)
ds = add_numpy_band(ds, "ridges", ridges.astype(int), grid.affine, Resampling.max)
ds = add_numpy_band(ds, "gullies", gullies.astype(int), grid.affine, Resampling.max)

dem = ds['terrain']
ridges = ds['ridges']
gullies = ds['gullies']

# +
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

unsheltered = y_values[np.where(x_values < percent_tree_threshold)]
median_value = np.median(unsheltered)

# +
# Calculate bounding box of a larger region to get an idea of the location
image_bbox = {
    'y': query['y'],
    'x': query['x'],
}
buffer = 1  
region_bbox = {
    'y': (image_bbox['y'][0] - buffer, image_bbox['y'][1] + buffer),
    'x': (image_bbox['x'][0] - buffer, image_bbox['x'][1] + buffer),
}

# Create GeoDataFrames for the image and region
image_gdf = gpd.GeoDataFrame(
    {'geometry': [box(image_bbox['x'][0], image_bbox['y'][0], image_bbox['x'][1], image_bbox['y'][1])]},
    crs='EPSG:4326', 
)
region_gdf = gpd.GeoDataFrame(
    {'geometry': [box(region_bbox['x'][0], region_bbox['y'][0], region_bbox['x'][1], region_bbox['y'][1])]},
    crs='EPSG:4326', 
)

# +
# Set up 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
title_size = 22
label_size = 18
annotations_size = 14

white_cmap = LinearSegmentedColormap.from_list("white_cmap", ["white", "white"])
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap=white_cmap)

# Plot 1: Productivity Map
ax = axes[0, 0]
cmap = plt.cm.coolwarm
cmap.set_bad(color='green')  # Set NaN pixels to green
im = ds_masked.plot(
    cmap=cmap,
    vmin=median_value - (upper_bound - lower_bound) / 2,
    vmax=median_value + (upper_bound - lower_bound) / 2,
    ax=ax
)
ax.set_title(f'{stubs[stub]} Productivity on {time}', fontsize=title_size)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
xlim, ylim = ax.get_xlim(), ax.get_ylim()
lat_lon_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
ax.set_aspect(lat_lon_ratio)
cbar = ax.collections[0].colorbar
cbar.set_label(f"Enhanced Vegetation Index ({productivity_variable})", fontsize=label_size)
cbar.ax.tick_params(labelsize=annotations_size)

# Plot 2: Topography Map
ax = axes[0, 1]
im = ax.imshow(dem, cmap='terrain', zorder=1, interpolation='bilinear')
cbar = plt.colorbar(im, ax=ax, label='Elevation (m)')
cbar.set_label('Elevation (m)', fontsize=label_size)
cbar.ax.tick_params(labelsize=annotations_size)
ax.contour(ridges, levels=[0.5], colors='red', linewidths=1.5, zorder=2)
ax.contour(gullies, levels=[0.5], colors='blue', linewidths=1.5, zorder=3)
ax.contour(dem, colors='black', linewidths=0.5, zorder=4, alpha=0.5)
ax.set_title('Ridges and Gullies', fontsize=title_size)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect(lat_lon_ratio)

# Plot 3: Larger region and bounding box
ax = axes[1, 1]
region_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='200km Region')
image_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label='10km Bounding Box')
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=image_gdf.crs)
ax.set_xlim(region_bbox['x'][0], region_bbox['x'][1])
ax.set_ylim(region_bbox['y'][0], region_bbox['y'][1])
ax.legend()
ax.set_title('Location', fontsize=title_size)

# Create a dummy white colorbar to align the plots nicely
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_ticks([])  
cbar.set_label('')  
cbar.outline.set_visible(False)

# Plot 4: True Colour Image
ax = axes[1, 0]
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
red = ds_timepoint['nbart_red']
green = ds_timepoint['nbart_green']
blue = ds_timepoint['nbart_blue']
rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
ax.imshow(rgb)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect(lat_lon_ratio)
width_km = 10  # Scale bar settings
pixels_per_km = ds.sizes['x'] / width_km
fontprops = FontProperties(size=12)
scalebar = AnchoredSizeBar(
    ax.transData,
    pixels_per_km,
    '1 km',
    'lower left',
    pad=0.5,
    color='white',
    frameon=False,
    size_vertical=2,
    fontproperties=fontprops,
)
ax.add_artist(scalebar)
ax.set_title('True Colour', fontsize=title_size)

# Create a dummy white colorbar to align the plots nicely
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_ticks([])  
cbar.set_label('')  
cbar.outline.set_visible(False)

# Adjust layout and save
plt.tight_layout()
filename = os.path.join(scratch_dir, f"{stub}_spatial_variation.png")
plt.savefig(filename)
plt.show()
print("Saved:", filename)
