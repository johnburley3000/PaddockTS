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

# region
# Filepaths
outdir = os.path.join(gdata_dir, "Data/PadSeg/")
stub = "MILG"

# Global variables
tree_cover_threshold = 5
pixel_size = 10  # metres
distances = (0, 30) # pixels. So all pixels within 300m. Might be more robust to look at pixels in a donut, e.g. 50m-300m.
min_distance = distances[0]
max_distance = distances[1]
# endregion

# region
# outdir = '/g/data/xe2/cb8590/Data/shelter/'
# stub = '34_0_148_5'
# -
# endregion

# %%time
# Load the sentinel imagery xarray 
filename = os.path.join(outdir, f"{stub}_ds2.pkl")
with open(filename, 'rb') as file:
    ds_original = pickle.load(file)

ds = ds_original

# %%time
# Calculate the percentage of tree cover in each sentinel pixel, based on the global canopy height map
variable = "canopy_height"
filename = os.path.join(outdir, f"{stub}_{variable}.tif")
array = rxr.open_rasterio(filename)
binary_mask = (array >= 1).astype(float)
ds['tree_percent'] = binary_mask.rio.reproject_match(ds, resampling=Resampling.average)

# Add worldcover classes to the xarray
world_cover_layers = {
    "Tree cover": 10, # Green
    # "Shrubland": 20, # Orange
    "Grassland": 30, # Yellow
    "Cropland": 40, # pink
    "Built-up": 50, # red
    "Permanent water bodies": 80, # blue
}

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

# The resampling often messes up the boundary, so we trim the outside pixels after adding all the resampled bounds
ds = ds.isel(
    y=slice(1, -1),
    x=slice(1, -1) 
)

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

# region
# Selecting an individual paddock
# endregion

time = "2020-01-08"
ds_timepoint = ds.sel(time=time, method='nearest')
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
red = ds_timepoint['nbart_red']
green = ds_timepoint['nbart_green']
blue = ds_timepoint['nbart_blue']
rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)

# region
# Read in the polygons from SAMGeo (these will not neccesarily match user-provided paddocks)
pol = gpd.read_file(outdir+stub+'_filt.gpkg')

# have to set a paddock id. Preferably do this in earlier step in future... 
pol['paddock'] = range(1,len(pol)+1)
pol['paddock'] = pol.paddock.astype('category')
# endregion

# region
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

lat_lon_ratio = lat_distance_km/lon_distance_km
lat_lon_ratio

# endregion

# region
# Plotting a larger region and bounding box with contextily
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

region_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='200km Region')
image_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label='10km Bounding Box')
ax.set_xlim(region_bbox['x'][0], region_bbox['x'][1])
ax.set_ylim(region_bbox['y'][0], region_bbox['y'][1])
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=image_gdf.crs)
ax.legend()
ax.set_title('Location', fontsize=title_size)

# Add the dummy white colorbar
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
# endregion

# region
# Generate a map of the paddocks 
bounds = ds[productivity_variable].rio.bounds()
left, bottom, right, top = bounds
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(rgb, extent=(left, right, bottom, top))

pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='yellow')

scalebar = AnchoredSizeBar(
    ax.transData, 1000, '1km', loc='lower center', pad=0.1, 
    color='black', frameon=False, size_vertical=10,
    fontproperties=fm.FontProperties(size=label_size),
    bbox_to_anchor=(0.05, -0.05),  # Position below the plot
    bbox_transform=ax.transAxes,
)
ax.add_artist(scalebar)

ax.set_aspect(lat_lon_ratio)

filename = os.path.join(scratch_dir, stub+'_paddock_map_auto.tif')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.axis('off')
plt.show()
print(filename)
# endregion

# region
# WorldCover Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Abbreviate the WorldCover names to take less space on the plot
world_cover_layers = {
    "Tree": 10, # Green
    "Grass": 30, # Yellow
    "Crop": 40, # pink
    "Urban": 50, # red
    "Water": 80, # blue
}
colors = ['green', 'yellow', 'violet', 'red', 'blue']
values = list(world_cover_layers.values())
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(values + [max(values) + 10], cmap.N)  # Add an extra upper bound

# Select the worldcover layer
ds_worldcover = ds.sel(time=time, method='nearest')['worldcover']
im = ds_worldcover.plot(
    cmap=cmap,
    norm=norm,
    ax=ax,
    add_colorbar=False
)
ax.set_title(f'WorldCover 2021', fontsize=title_size)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])

pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='purple')

scalebar = AnchoredSizeBar(
    ax.transData, 1000, '1km', loc='lower center', pad=0.1, 
    color='black', frameon=False, size_vertical=10,
    fontproperties=fm.FontProperties(size=label_size),
    bbox_to_anchor=(0.05, -0.05),  # Position below the plot
    bbox_transform=ax.transAxes,
)
ax.add_artist(scalebar)

ax.set_aspect(lat_lon_ratio)
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
labels = list(world_cover_layers.keys())
ax.legend(handles, labels, loc='lower right', fontsize=annotations_size)

plt.show()
# endregion

# Remove unnecessary variables from ds
useful_variables = ['nbart_red', 'nbart_green', 'nbart_blue', 'EVI', 'worldcover', 'tree_percent', 'percent_trees_0m-300m']
ds_small = ds.isel(band=0)[useful_variables]

# Select a paddock
paddock_id = 66
paddock_row = pol[pol['paddock'] == 66]
paddock_geometry = paddock_row['geometry'].iloc[0]

# Create a rectangular buffer
minx, miny, maxx, maxy = paddock_geometry.bounds
buffer_distance = max_distance * pixel_size
expanded_minx = minx - buffer_distance
expanded_miny = miny - buffer_distance
expanded_maxx = maxx + buffer_distance
expanded_maxy = maxy + buffer_distance
rectangular_buffer = box(expanded_minx, expanded_miny, expanded_maxx, expanded_maxy)
buffered_gdf = gpd.GeoDataFrame(geometry=[rectangular_buffer])
# buffered_gdf.iloc[0,0]

# %%time
# Clip the xarray to this paddock
ds_buffered = ds_small.rio.clip(buffered_gdf.geometry, drop=True, invert=False)

# region
# Recreate the adjacency mask for just this paddock
paddock_geometry = paddock_row['geometry'].iloc[0]
ds = ds_buffered
tree_percent = ds_buffered['tree_percent'].values
tree_mask = tree_percent > 0
structuring_element = np.ones((3, 3))
adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

# Exclude pixels outside the paddock for the rest of this analysis
paddock_mask = geometry_mask(
    [paddock_geometry],
    out_shape=(ds.sizes["y"], ds.sizes["x"]),
    transform=ds.rio.transform())
adjacent_mask |= paddock_mask

plt.imshow(adjacent_mask)
# endregion

# # All Timepoints

# %%time
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

# region
df_benefits = pd.DataFrame(benefits)
df_benefits['date'] = df_benefits['time'].dt.date
df_benefits = df_benefits.set_index('date')
df_benefits.index = pd.to_datetime(df_benefits.index)

filename = os.path.join(scratch_dir, f"{stub}_df_benefits.csv")
df_benefits.to_csv(filename)
print(filename)
# endregion

df_top10 = df_benefits.nlargest(10, 'r2')
df_top10

# # Max r2 timepoint

# time = df_top10.index[0].date()
time = "2020-01-08"
ds_timepoint = ds.sel(time=time, method='nearest')

# region
# Calculate shelter score and productivity index for this timepoint
ndvi = ds.sel(time=time, method='nearest')[productivity_variable]
p = ndvi.where(~adjacent_mask) #  & (grassland | cropland))
layer_name = f"percent_trees_0m-300m"
s = ds[layer_name]
x = s.values.flatten()

# Make sure that any nan values in the shelter score are also nan in the productivity_score
p = p.where(~s.isnull())
# endregion

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

# Calculate sheltered and unsheltered pixels
percent_tree_threshold = 10
sheltered = y_values[np.where(x_values >= percent_tree_threshold)]
unsheltered = y_values[np.where(x_values < percent_tree_threshold)]

# region
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
res = stats.linregress(x_values, y_values)
x_fit = np.linspace(min(x_values), max(x_values), 500)
y_fit = res.intercept + res.slope * x_fit
ax1.plot(x_fit, y_fit, 'r-', label=f"$R^2$ = {res.rvalue**2:.2f}")
ax1.legend(fontsize=label_size)

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
ax2.set_ylabel('EVI', fontsize=label_size)
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
# shelter_vs_unsheltered = (np.median(sheltered) - np.median(unsheltered)) / np.median(unsheltered) * 100
# ax2.text(
#     0.53, y_max - 0.02,  # Position text in top left
#     f'    Sheltered vs unsheltered \n = ({medians[1]:.2f} - {medians[0]:.2f})/{medians[0]:.2f} = {shelter_vs_unsheltered:.2f}%',
#     fontsize=annotations_size, ha='left', va='top'
# )

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
filename = os.path.join(scratch_dir, f"{stub}_hist_and_boxplot.png")
plt.savefig(filename)
plt.show()
print("Saved", filename)
# endregion


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

# region
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
plt.show()
print("Saved", filename_combined)
# endregion

# # Spatial Variation

# Load and calculate topography layers
filename = os.path.join(outdir, f"{stub}_terrain.tif")
grid, dem, fdir, acc = pysheds_accumulation(filename)
num_catchments = 20
gullies, full_branches = catchment_gullies(grid, fdir, acc, num_catchments)
ridges = catchment_ridges(grid, fdir, acc, full_branches)
slope = calculate_slope(filename)


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

# region
# Calculate aspect ratio of this paddock buffer
paddock_row_4326 = paddock_row.to_crs(epsg=4326)

minx, miny, maxx, maxy = paddock_row_4326['geometry'].iloc[0].bounds
buffer_distance = max_distance * pixel_size
expanded_minx = minx - buffer_distance
expanded_miny = miny - buffer_distance
expanded_maxx = maxx + buffer_distance
expanded_maxy = maxy + buffer_distance

lat_diff_deg = expanded_maxy - expanded_miny
lon_diff_deg = expanded_maxx - expanded_minx
latitude_rad = math.radians(latitude_deg)
lat_distance_km = lat_diff_deg * (math.pi * earth_radius_km / 180)
lon_distance_km = lon_diff_deg * (math.cos(latitude_rad) * (math.pi * earth_radius_km / 180))
lat_lon_ratio = lat_distance_km/lon_distance_km
lat_lon_ratio
# endregion

lat_distance_km

# region
# Productivity Map
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
title_size = 22
label_size = 18
annotations_size = 14

vmin = median_value - (upper_bound - lower_bound) / 2
vmax = median_value + (upper_bound - lower_bound) / 2
cmap = plt.cm.coolwarm
cmap.set_bad(color='green')  # Set NaN pixels to green

ds_trees = ds_productivity.where(~tree_mask)

im = ds_trees.plot(
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    ax=ax,
    add_colorbar=False  # Suppress the automatic color bar
)
paddock_row.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=5)

ax.set_title(f'Paddock {paddock_id} on {time}', fontsize=title_size)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
xlim, ylim = ax.get_xlim(), ax.get_ylim()
lat_lon_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

# Add a color bar axis manually
colorbar_ax = fig.add_axes([1, 0.125, 0.03, 0.75])  # [x-position, y-position, width, height]
cbar = fig.colorbar(im, cax=colorbar_ax)

# Add labels and ticks
cbar.set_label(f"Enhanced Vegetation Index ({productivity_variable})", fontsize=label_size)
cbar.ax.tick_params(labelsize=annotations_size)

# Add legend for tree pixels
tree_patch = mpatches.Patch(color='green', label='Tree')  # Custom legend entry
ax.legend(handles=[tree_patch], loc='upper left', fontsize=label_size)

scalebar = AnchoredSizeBar(
    ax.transData, 1000, '1km', loc='lower center', pad=0.1, 
    color='dimgrey', frameon=False, size_vertical=10, 
    fontproperties=fm.FontProperties(size=label_size)
)
ax.add_artist(scalebar)

ax.set_aspect(lat_lon_ratio)
plt.tight_layout()
plt.show()
# endregion

# region
# Visualise a panchromatic image of this paddock
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

time = "2020-01-08"
ds_timepoint = ds_buffered.sel(time=time, method='nearest')
red = ds_timepoint['nbart_red']
green = ds_timepoint['nbart_green']
blue = ds_timepoint['nbart_blue']
rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
bounds = ds_buffered[productivity_variable].rio.bounds()
left, bottom, right, top = bounds

ax.set_aspect(lat_lon_ratio)
ax.set_title(f'Paddock {paddock_id} on {time}', fontsize=title_size)

ax.imshow(rgb, extent=(left, right, bottom, top))
paddock_row.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=5)

scalebar = AnchoredSizeBar(
    ax.transData, 1000, '1km', loc='lower center', pad=0.1, 
    color='white', frameon=False, size_vertical=10, 
    fontproperties=fm.FontProperties(size=label_size)
)
ax.add_artist(scalebar)

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])

ax.set_aspect(lat_lon_ratio)
plt.show()
# endregion
# region
# True colour tiff
filename = os.path.join(scratch_dir, f"RGB_{stub}_{time}.tif")
ds_timepoint.attrs = {}
rgb_stack = ds_timepoint[['nbart_red', 'nbart_green', 'nbart_blue']]
rgb_stack.rio.to_raster(filename)
print(filename)

# Productivity Tiff
filename = os.path.join(scratch_dir, f"{productivity_variable}_{stub}_{time}.tif")
clipped = ds_masked.fillna(upper_bound + 0.1)
clipped = clipped.clip(min=lower_bound, max=upper_bound)
clipped.attrs = {}
clipped.rio.to_raster(filename)
print(filename)
# endregion
