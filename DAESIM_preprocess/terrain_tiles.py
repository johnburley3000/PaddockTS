# # NCI ARE Setup
# Modules: gdal/3.6.4  
# Environment base: /g/data/xe2/John/geospatenv

# Standard library
import subprocess
import os

# Dependencies
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import rioxarray as rxr

# Local imports
paddockTS_dir = os.path.join(os.path.expanduser('~'), "Projects/PaddockTS")
os.chdir(paddockTS_dir)
from DAESIM_preprocess.util import gdata_dir, scratch_dir, create_bbox, transform_bbox

def run_gdalwarp(bbox=[148.464499, -34.394042, 148.474499, -34.3840426], filename="output.tif"):
    """Use gdalwarp to download a tif from terrain tiles"""

    if os.path.exists(filename):
        os.remove(filename)
    
    xml=os.path.join("DAESIM_preprocess/terrain_tiles.xml")
        
    bbox_3857 = transform_bbox(bbox)
    min_x, min_y, max_x, max_y = bbox_3857
    command = [
        "gdalwarp",
        "-of", "GTiff",
        "-te", str(min_x), str(min_y), str(max_x), str(max_y),
        xml, filename
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # print("Terrain Tiles STDOUT:", result.stdout, flush=True)
    # print("Terrain Tiles STDERR:", result.stderr, flush=True)
    print(f"Downloaded {filename}")

def interpolate_nan(filename="output.tif"):
    """Fix bad measurements in terrain tiles dem"""

    # Load the tiff into a numpy array rasterio
    with rasterio.open(filename) as dataset:
        dem = dataset.read(1) 
        meta = dataset.meta.copy()
    
    # There are some clearly bad measurements in terrain tiles and this attempts to assign them np.nan.
    threshold = 10
    heights = sorted(set(dem.flatten()))
    lowest_correct_height = min(heights)
    for i in range(len(heights)//2, -1, -1):
        if heights[i + 1] - heights[i] > threshold:
            lowest_correct_height = heights[i + 1] 
            break
    Z = np.where(dem < lowest_correct_height, np.nan, dem)
    
    # Extract into lists for interpolating
    x_coords, y_coords = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = Z.flatten()
    
    # Remove NaN values before interpolating
    mask = ~np.isnan(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]
    xy_coords = np.vstack((x_flat, y_flat), dtype=float).T
    
    # Replace bad/nan/missing values with the nearest neighbour
    X, Y = np.meshgrid(np.linspace(0, Z.shape[1] - 1, Z.shape[1]),
                np.linspace(0, Z.shape[0] - 1, Z.shape[0]))
    nearest = griddata(xy_coords, z_flat, (X, Y), method='nearest')

    return dem, meta

def downsample(dem):
    """Downsample from 10m dem to 30m dem"""
    zoomed = zoom(dem, 1/3, order=0) 
    return zoomed

def download_dem(dem, meta, filename="terrain_tiles.tif"):
    meta.update({
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,  # Number of bands
        "dtype": dem.dtype
    })
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(dem, 1)
    print(f"Saved {filename}")

def visualise_tif(filename="terrain_tiles.tif", title="Terrain Tiles"):
    ds = rxr.open_rasterio(filename)
    band = ds.sel(band=1)
    band.plot()
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="", stub="test", tmp_dir=""):
    """Download 10m resolution elevation from terrain_tiles"""
    
    # Load the raw data
    bbox = create_bbox(lat, lon, buffer)
    filename = os.path.join(tmp_dir, f"{stub}_terrain_original.tif")
    run_gdalwarp(bbox, filename)

    # Fix bad measurements
    dem, meta = interpolate_nan(filename)        
    filename = os.path.join(outdir, f"{stub}_terrain.tif")
    download_dem(dem, meta, filename)

if __name__ == '__main__':
    # Choosing location
    lat, lon = -34.3890427, 148.469499
    buffer = 0.005  # 0.01 degrees is about 1km in each direction, so 2km total
    stub = "MILG_1km"

    # Specify output destinations
    outdir = os.path.join(gdata_dir, "Data/PadSeg/")
    tmp_dir = os.path.join(scratch_dir, "tmp")

    # Download elevation from terrain tiles
    terrain_tiles(lat, lon, buffer, outdir, stub, tmp_dir)

    # Visualise the downloaded data
    filename = os.path.join(outdir, f"{stub}_terrain.tif")
    visualise_tif(filename, "Terrain Tiles")
