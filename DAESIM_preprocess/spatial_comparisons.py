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
# Weather data
filename_silo = os.path.join(outdir, f"{stub}_silo_daily.nc")
ds_silo = xr.open_dataset(filename_silo)

variable = 'max_temp'
df_drought = pd.DataFrame(ds_silo.isel(lat=0, lon=0)[variable], index=ds_silo.isel(lat=0, lon=0).time, columns=[variable])


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

ds_trimmed['canopy_height'].plot()

ds_trimmed['NDVI'].isel(time=0).plot()

ds_original = ds.copy()
ds = ds_trimmed

# +
# Visualise the distribution of all summed (or mean) NDVI's (both trees and grasses)
total_ndvi_summed = ds['NDVI'].mean(dim='time')
plt.imshow(total_ndvi_summed)
plt.show()

total_ndvi_flattened = total_ndvi_summed.values.flatten()
plt.hist(total_ndvi_flattened, bins=1000)
plt.show()
# -
# Find the max_temp for each date with satellite imagery 
xarray_times = ds['time'].values
nearest_times_indices = df_drought.index.get_indexer(xarray_times, method='nearest')
nearest_df_times = df_drought.index[nearest_times_indices]

# Find the timepoints where the temp is greater than 25 degrees
temp_threshold = 25
selected_times = nearest_df_times[df_drought['max_temp'].iloc[nearest_times_indices] > temp_threshold]
ds_drought = ds.sel(time=selected_times, method='nearest')

# +
total_ndvi_summed = ds_drought['NDVI'].median(dim='time')
plt.imshow(total_ndvi_summed)
plt.show()

total_ndvi_flattened = total_ndvi_summed.values.flatten()
plt.hist(total_ndvi_flattened, bins=1000)
plt.show()
# -





# +
# Visualise the distribution of the mean NDVI in summer (when max temp > 25 degrees)
total_ndvi_summed = ds['NDVI'].mean(dim='time')
plt.imshow(total_ndvi_summed)
plt.show()

total_ndvi_flattened = total_ndvi_summed.values.flatten()
plt.hist(total_ndvi_flattened, bins=1000)
plt.show()

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
ds["num_trees_200m"].plot()

# +
# Repeating the linear regression on the manually created paddocks with absolutely no trees
filename = os.path.join(gdata_dir,'MILG_paddocks_notrees.gpkg')
pol = gpd.read_file(filename)

# Change from multipolygon to polygon, because I created the layer with the wrong type in QGIS
def convert_multipolygon_to_polygon(geometry):
    return geometry.union(geometry)
pol['geometry'] = pol['geometry'].apply(convert_multipolygon_to_polygon)

# Create a mask from the geometries
xarray_dataset = ds
gdf = pol.to_crs(xarray_dataset.crs) 
transform = xarray_dataset.rio.transform()
out_shape = (xarray_dataset.dims['y'], xarray_dataset.dims['x'])
geometries = gdf.geometry
paddock_mask = features.geometry_mask(
    [geom for geom in geometries],
    transform=transform,
    invert=True,
    out_shape=out_shape
)
# Some of the pixels in my manually selected area are in the mask of tree + buffer pixels
paddock_mask = paddock_mask & ~adjacent_mask

# +
# plt.imshow(paddock_mask)

# +
# Calculate the productivity score
time = '2020-01-01'
ndvi = ds.sel(time=time, method='nearest')['NDVI']
productivity_score1 = ndvi.where(~adjacent_mask)
# productivity_score1 = ndvi.where(paddock_mask)

s = ds['num_trees_200m'].values

## Filtering by soils or topography
# new_mask = ~adjacent_mask & (ds['terrain'] > 560) # & (ds['Silt'] < 15)
# new_mask = ~adjacent_mask & (ds['aspect'] == 4) & (ds['slope'] > 10)
# productivity_score1 = ndvi.where(new_mask)

# Flatten the arrays for plotting
y = productivity_score1.values.flatten()
y_values = y[~np.isnan(y)]   # Remove all pixels that are trees or adjacent to trees
x = s.flatten()
x_values = x[~np.isnan(y)]   # Match the shape of the x_values
# -


# Visualise the productivity of just the sheltered pixels
sheltered_productivities = productivity_score1.values
sheltered_productivities_da = xr.DataArray(
    sheltered_productivities, 
    dims=("y", "x"),  
    coords={"y": ds.coords["y"], "x": ds.coords["x"]}, 
    name="shelter_score" 
)
ds["sheltered_productivities"] = sheltered_productivities_da
ds["sheltered_productivities"].plot()

filename = os.path.join(scratch_dir, f'{stub}_sheltered_productivities_2020-01-01.tif')
ds["sheltered_productivities"].rio.to_raster(filename)
print(filename)

filename = os.path.join(scratch_dir, f'{stub}_num_trees_200m.tif')
ds["num_trees_200m"].rio.to_raster(filename)
print(filename)

# Plot
plt.hist2d(x_values, y_values, bins=100, norm=mcolors.PowerNorm(0.1))
plt.ylabel('NDVI', fontsize=12)
pixel_size = 10
plt.xlabel(f'Number of tree pixels within {distance * pixel_size}m', fontsize=12)
plt.title(stub + ": " + str(time)[:10], fontsize=14)
plt.show()

res = stats.linregress(x_values, y_values)
print(f"Sample size: {len(x_values)}")
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Slope: {res.slope:.6f}")
plt.plot(x_values, y_values, 'o', label='original data')
plt.plot(x_values, res.intercept + res.slope*x_values, 'r', label='fitted line')
plt.legend()
plt.show()

shelter_threshold = 0.1  # Percentage tree cover
num_trees_threshold = ((distance * 2) ** 2) * shelter_threshold # Number of tree pixels
num_trees_threshold

sheltered = y_values[np.where(x_values >= num_trees_threshold)]
unsheltered = y_values[np.where(x_values < num_trees_threshold)]
F_statistic, p_value = stats.f_oneway(sheltered, unsheltered)
print(f"Number of sheltered pixels: {len(sheltered)}/{len(y_values)}")
print("F_statistic:", F_statistic) 
print("p_value:", p_value)
plt.boxplot([sheltered, unsheltered], labels=['Sheltered', 'Unsheltered'])
plt.show()

# +
# %%time
# Calculate the productivity score

shelter_thresholds = 0, 0.05, 0.1, 0.2, 0.3   # Percentage tree cover
total_benefits = []
benefit_scores_dict = {}

for i, shelter_threshold in enumerate(shelter_thresholds):
    print(f"Working on {i}/{len(shelter_thresholds)}", shelter_threshold)
    num_trees_threshold = ((distance * 2) ** 2) * shelter_threshold # Number of tree pixels

    benefit_scores = []
    for i, time in enumerate(ds.time.values):
        ndvi = ds.sel(time=time, method='nearest')['NDVI']
        productivity_score1 = ndvi.where(~adjacent_mask)
        # productivity_score1 = ndvi.where(paddock_mask)
        # productivity_score1 = ndvi.where(new_mask)
        s = ds['num_trees_200m'].values
        
        # Flatten the arrays for plotting
        y = productivity_score1.values.flatten()
        y_values = y[~np.isnan(y)]   # Remove all pixels that are trees or adjacent to trees
        x = s.flatten()
        x_values = x[~np.isnan(y)]   # Match the shape of the x_values
    
        # Select sheltered pixels and calculate z scores for NDVI at each pixel
        sheltered = y_values[np.where(x_values >= num_trees_threshold)]
        if len(sheltered) == 0:
            print(f"No pixels with a shelterscore > {num_trees_threshold}")
            continue
        sheltered_z = (sheltered - np.mean(y_values))/np.std(y_values)
        all_z = (y_values - np.mean(y_values))/np.std(y_values)
        
        percentiles = 10, 25, 50, 75, 90
        percentile_benefits = {}
        for percentile in percentiles:
            percentile_benefits[f"p{percentile}"] = np.percentile(sheltered_z, percentile) - np.percentile(all_z, percentile)

        # # Min max normalisation
        # x_values = (x_values - min(x_values)) / (max(x_values) - min(x_values))
        # y_values = (y_values - min(y_values)) / (max(y_values) - min(y_values))
        # res = stats.linregress(x_values, y_values)
        # F_statistic, p_value = stats.f_oneway(sheltered, unsheltered)    

        benefit_score = {
            "time":time,
            # "r2":res.rvalue**2,
            # "slope":res.slope,
            # "f":F_statistic,
            # "p":p_value
        }
        benefit_score = benefit_score | percentile_benefits
        benefit_scores.append(benefit_score)

    if len(sheltered) == 0:
        continue

    benefit_scores_dict[shelter_threshold] = benefit_scores
    df_shelter = pd.DataFrame(benefit_scores)
    df_shelter = df_shelter.set_index('time')
    
    max_z = max(df_shelter['p50'].values)

    # Join the weather data onto the shelter scores
    df_merged = pd.merge_asof(df_shelter, df_drought, left_index=True, right_index=True, direction='nearest')

    temperature_threshold = 25
    hot_days = np.where(df_merged['max_temp'] > temperature_threshold)

    median_z = df_merged['p50'].median()
    median_z_summer = df_merged['p50'].iloc[hot_days].median()
        
    total_benefit = {
            "shelter_threshold":shelter_threshold,
            "sheltered_pixels":len(sheltered),
            "unsheltered_pixels":len(y_values) - len(sheltered),
            "max_z":max_z,
            "median_z_summer": median_z_summer,
        }
    total_benefits.append(total_benefit)

print(len(total_benefits))
pd.DataFrame(total_benefits)
# -

# Plotting benefits over time at different productivity percentiles
df = pd.DataFrame(benefit_scores_dict[0.1])
df = df.set_index('time')
df = df.astype(float)
df.index = pd.to_datetime(df.index)
df = df[['p10', 'p25', 'p50', 'p75', 'p90']]
df.plot(figsize=(50,20))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(100))
ax.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xticks(rotation=45)
plt.show()

# +
# Plot a comparison of max temp and shelter benefit
df = df_merged
fig, ax1 = plt.subplots(figsize=(50, 20))  # Create the base figure and axis

# Maximum temperature
ax1.plot(df.index, df['max_temp'], color='tab:red', label='Maximum temperature')
ax1.set_xlabel('Date')
ax1.set_ylabel('Maximum temperature (°C)', color='tab:red')  # Set the y-axis label
ax1.tick_params(axis='y', labelcolor='tab:red')  # Change tick colors to match the line

# Shelter benefit
ax2 = ax1.twinx()  
ax2.plot(df.index, df['p50'], color='tab:green', label='Shelter benefit')
ax2.set_ylabel('Shelter benefit', color='tab:green')  # Set the second y-axis label
ax2.tick_params(axis='y', labelcolor='tab:green')  # Change tick colors to match the line

plt.xticks(rotation=45)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
# -


