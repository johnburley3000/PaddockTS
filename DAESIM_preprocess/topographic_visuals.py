import os
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.topography import pysheds_accumulation, calculate_slope

import numpy as np
import pickle
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from scipy.ndimage import gaussian_filter

import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


stub = "TEST6"
outdir = "/g/data/xe2/cb8590/Data/PadSeg/"

from DAESIM_preprocess.util import scratch_dir
tmpdir = scratch_dir

# Load the imagery stack
filename = os.path.join(outdir, f"{stub}_ds2.pkl")
with open(filename, 'rb') as file:
    ds_original = pickle.load(file)
ds = ds_original

# Load the paddocks
pol = gpd.read_file(outdir+stub+'_filt.gpkg')
pol['paddock'] = range(1,len(pol)+1)
pol['paddock'] = pol.paddock.astype('category')

# Gaussian smooth the dem before processing with pysheds (because values are stored as ints in terrain tiles)
filename = os.path.join(outdir, f"{stub}_terrain.tif")
with rasterio.open(filename) as src:
    dem = src.read(1)  
    transform = src.transform  
    crs = src.crs
    nodata = src.nodata 
    width = src.width 
    height = src.height 

sigma = 10  # Adjust this value to control how much smoothing gets applied

dem_smooth = gaussian_filter(dem.astype(float), sigma=sigma)
filename = os.path.join(outdir, f"{stub}_terrain_smoothed.tif")
with rasterio.open(filename, 'w', driver='GTiff', height=height, width=width,
                   count=1, dtype=dem_smooth.dtype, crs=crs, transform=transform,
                   nodata=nodata) as dst:
    dst.write(dem_smooth, 1) 
print(f"Smoothed DEM saved to {filename}")

# Load the terrain and calculate topographic variables
grid, dem, fdir, acc = pysheds_accumulation(filename)
slope = calculate_slope(filename)

# Make the flow directions sequential for easier plotting later
arcgis_dirs = np.array([1, 2, 4, 8, 16, 32, 64, 128]) 
sequential_dirs = np.array([1, 2, 3, 4, 5, 6, 7, 8])  
fdir_equal_spacing = np.zeros_like(fdir)  
for arcgis_dir, sequential_dir in zip(arcgis_dirs, sequential_dirs):
    fdir_equal_spacing[fdir == arcgis_dir] = sequential_dir 


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
ds = add_numpy_band(ds, "topographic_index", acc, grid.affine, Resampling.max)
ds = add_numpy_band(ds, "aspect", fdir, grid.affine, Resampling.nearest)
ds = add_numpy_band(ds, "slope", slope, grid.affine, Resampling.average)

# Clip everything by 1 cell because these algorithms can mess up at the boundary
ds = ds.isel(
    y=slice(1, -1),
    x=slice(1, -1) 
)


# Save the layers as tiff files for viewing in QGIS
filepath = os.path.join(tmpdir, stub + "_elevation_QGIS.tif")
ds['terrain'].rio.to_raster(filepath)
print(filepath)

filepath = os.path.join(tmpdir, stub + "_topographic_index_QGIS.tif")
ds['topographic_index'].rio.to_raster(filepath)
print(filepath)

# Need to specify the datatype for the aspect to save correctly
filepath = os.path.join(tmpdir, stub + "_aspect_QGIS.tif")
ds['aspect'].rio.to_raster(
    filepath,
    dtype="int8", 
    nodata=-1, 
)
print(filepath)

filepath = os.path.join(tmpdir, stub + "_slope_QGIS.tif")
ds['slope'].rio.to_raster(filepath)
print(filepath)

# Prep the plotting boundaries to align rasters with polygons
left, bottom, right, top = ds.rio.bounds()

# +
###### Elevation Plot #########
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(dem, cmap='terrain', zorder=1, interpolation='bilinear', 
               extent=(left, right, bottom, top))
plt.title("Elevation")
plt.colorbar(im, ax=ax, label='height above sea level (m)')# +

# Create the contour plot (correctly aligned with imshow)
interval = 10  # Contour interval (m)
contour_levels = np.arange(np.floor(np.min(dem)), np.ceil(np.max(dem)), interval)
contours = ax.contour(dem, levels=contour_levels, colors='black', 
                      linewidths=0.5, zorder=4, alpha=0.5, 
                      extent=(left, right, bottom, top), origin='upper')
ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')   # Add contour labels

# Scale bar
scalebar = AnchoredSizeBar(
    ax.transData, 1000, '1km', loc='lower left', pad=0.1, 
    color='white', frameon=False, size_vertical=10, 
    fontproperties=fm.FontProperties(size=14)
)
ax.add_artist(scalebar)

# Add the polygons
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=10, ha='center', va='center', color='black')

# Save the preview
plt.tight_layout()
filepath = os.path.join(tmpdir, stub + "_elevation_preview.png")
plt.savefig(filepath)
print(filepath)
# -

# +
########## Topographic Index Plot ##############
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(acc,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear', 
               extent=(left, right, bottom, top))
plt.title("Topographic Index")
plt.colorbar(im, ax=ax, label='upstream cells')

# Add the polygons
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=10, ha='center', va='center', color='yellow')

# Save the preview
filepath = os.path.join(tmpdir, stub + "_topographic_index_preview.png")
plt.savefig(filepath)
print(filepath)
# -

# +
########### Aspect Plot ###############
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(fdir_equal_spacing, cmap="hsv", origin="upper", extent=(left, right, bottom, top))
ax.set_title("Aspect")
plt.tight_layout()

# Colour bar with compass directions
cbar = fig.colorbar(im, ax=ax)
cbar.set_ticks(sequential_dirs)  
cbar.set_ticklabels(["E", "SE", "S", "SW", "W", 'NW', "N", "NE"])  

# Add polygons
pol.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='black')

filepath = os.path.join(tmpdir, stub + "_aspect_preview.png")
plt.savefig(filepath)
print(filepath)
# -

# +
####### Slope Plot ##########
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(slope, cmap="Purples", origin="upper", extent=(left, right, bottom, top))
plt.title("Slope")
plt.tight_layout()
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("degrees")

# Add Polygons
pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
for x, y, label in zip(pol.geometry.centroid.x, pol.geometry.centroid.y, pol['paddock']):
    ax.text(x, y, label, fontsize=12, ha='center', va='center', color='black')

# Save the preview
filepath = os.path.join(tmpdir, stub + "_slope_preview.png")
plt.savefig(filepath)
print(filepath)
# -
