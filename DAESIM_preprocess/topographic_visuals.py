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




# !ls /g/data/xe2/cb8590/Data/PadSeg/TEST6*

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

# Load the terrain and calculate topographic variables
filename = os.path.join(outdir, f"{stub}_terrain.tif")
grid, dem, fdir, acc = pysheds_accumulation(filename)
slope = calculate_slope(filename)

# Calculate a continuous aspect (the 'fdir' is categorical)
dzdx, dzdy = np.gradient(dem)
aspect = np.arctan2(-dzdy, -dzdx) * (180 / np.pi)
aspect[aspect < 0] += 360  


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
sequential_dirs = np.arange(1, 9)  
fdir_equal_spacing = np.zeros_like(fdir)  
for arcgis_dir, sequential_dir in zip(arcgis_dirs, sequential_dirs):
    fdir_equal_spacing[fdir == arcgis_dir] = sequential_dir 

# +
# Plot Aspect with compass direction labels
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(fdir_equal_spacing, cmap="twilight", origin="upper", extent=(left, right, bottom, top))

# Plot polygon
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)

# Add polygon labels
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='yellow')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Aspect")

cbar.set_ticks(sequential_dirs)  
cbar.set_ticklabels(["E", "SE", "S", "SW", "W", 'NW', "N", "NE"])  

ax.set_title("Aspect with Compass Directions")
plt.tight_layout()
plt.show()
# -


