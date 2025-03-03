# +
import os
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.topography import show_acc, show_aspect, show_slope, show_ridge_gullies, pysheds_accumulation, catchment_gullies, catchment_ridges, calculate_slope

import argparse
import logging
import pickle
import xarray as xr
import geopandas as gpd
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# -

import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib import colors
from scipy.ndimage import zoom
import rasterio
from scipy.ndimage import gaussian_filter


stub = "TEST6"
outdir = "/g/data/xe2/cb8590/Data/PadSeg/"

# Load the imagery stack
filename = os.path.join(outdir, f"{stub}_ds2.pkl")
with open(filename, 'rb') as file:
    ds_original = pickle.load(file)
ds = ds_original

# Load the paddocks
pol = gpd.read_file(outdir+stub+'_filt.gpkg')
pol['paddock'] = range(1,len(pol)+1)
pol['paddock'] = pol.paddock.astype('category')

# +
# Gaussian smooth the dem before processing with pysheds, because values are stored as ints in terrain tiles
filename = os.path.join(outdir, f"{stub}_terrain.tif")
with rasterio.open(filename) as src:
    dem = src.read(1)  # Read DEM data
    transform = src.transform  # Get geospatial transformation
    crs = src.crs  # Get the coordinate reference system
    nodata = src.nodata  # Get the NoData value, if any
    width = src.width  # Get the width (number of columns)
    height = src.height  # Get the height (number of rows)

# Apply Gaussian smoothing to the DEM
sigma = 10  # Standard deviation for Gaussian filter, adjust to control smoothing
dem_smooth = gaussian_filter(dem.astype(float), sigma=sigma)

filename = os.path.join(outdir, f"{stub}_terrain_smoothed.tif")
with rasterio.open(filename, 'w', driver='GTiff', height=height, width=width,
                   count=1, dtype=dem_smooth.dtype, crs=crs, transform=transform,
                   nodata=nodata) as dst:
    dst.write(dem_smooth, 1)  # Write the smoothed DEM to band 1
print(f"Smoothed DEM saved to {filename}")
# -

# Load the terrain and calculate topographic variables
grid, dem, fdir, acc = pysheds_accumulation(filename)
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


# Align & resample & reproject the topographic variables to match the imagery stack
ds = add_numpy_band(ds, "terrain", dem, grid.affine, Resampling.average)
ds = add_numpy_band(ds, "slope", slope, grid.affine, Resampling.average)
ds = add_numpy_band(ds, "topographic_index", acc, grid.affine, Resampling.max)
ds = add_numpy_band(ds, "aspect", fdir, grid.affine, Resampling.nearest)

# +
# Elevation Plot
fig, ax = plt.subplots(figsize=(8, 6))
left, bottom, right, top = ds.rio.bounds()

# Display the DEM with origin='lower' to match contour behavior
im = ax.imshow(dem, cmap='terrain', zorder=1, interpolation='bilinear', 
               extent=(left, right, bottom, top))

# Plot polygon
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)

# Add polygon labels
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=10, ha='center', va='center', color='black')

# Add colorbar
plt.colorbar(im, ax=ax, label='Elevation (m)')

# Specify the contour intervals
interval = 10
contour_levels = np.arange(np.floor(np.min(dem)), np.ceil(np.max(dem)), interval)

# Create the contour plot (correctly aligned with imshow)
contours = ax.contour(dem, levels=contour_levels, colors='black', 
                      linewidths=0.5, zorder=4, alpha=0.5, 
                      extent=(left, right, bottom, top), origin='upper')

# Add contour labels
ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')

# plt.axis('off')
plt.tight_layout()

# +
# Water Accumulation
fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(acc,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear', 
               extent=(left, right, bottom, top))

# Plot polygon
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)

# Add polygon labels
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=10, ha='center', va='center', color='yellow')


plt.colorbar(im, ax=ax, label='Upstream Cells')
# -

# Make the flow directions sequential for easier plotting
arcgis_dirs = np.array([1, 2, 4, 8, 16, 32, 64, 128]) 
sequential_dirs = np.array([1, 2, 3, 4, 5, 6, 7, 8])  
fdir_equal_spacing = np.zeros_like(fdir)  
for arcgis_dir, sequential_dir in zip(arcgis_dirs, sequential_dirs):
    fdir_equal_spacing[fdir == arcgis_dir] = sequential_dir 

# +
# Plot Aspect with compass direction labels
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(fdir_equal_spacing, cmap="hsv", origin="upper", extent=(left, right, bottom, top))

# Plot polygon
pol.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

# Add polygon labels
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='black')

cbar = fig.colorbar(im, ax=ax)

cbar.set_ticks(sequential_dirs)  
cbar.set_ticklabels(["E", "SE", "S", "SW", "W", 'NW', "N", "NE"])  

ax.set_title("Aspect with Compass Directions")
plt.tight_layout()
plt.show()
# +
# Slope
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(slope, cmap="Greys", origin="upper", extent=(left, right, bottom, top))

# Plot polygon
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)

# Add polygon labels
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='black')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Slope (degrees)")

plt.tight_layout()
plt.show()

