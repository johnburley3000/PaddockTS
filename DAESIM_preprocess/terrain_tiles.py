# +
# # NCI ARE Setup
# Modules: gdal/3.6.4  
# Environment base: /g/data/xe2/John/geospatenv

# Standard library
import subprocess
import os

# Dependencies
import numpy as np
import rasterio
from scipy.interpolate import griddata
from pyproj import Transformer

# +
def transform_bbox(bbox=[148.464499, -34.394042, 148.474499, -34.384042], inputEPSG="EPSG:4326", outputEPSG="EPSG:3857"):
    transformer = Transformer.from_crs(inputEPSG, outputEPSG)
    x1,y1 = transformer.transform(bbox[1], bbox[0])
    x2,y2 = transformer.transform(bbox[3], bbox[2])
    return (x1, y1, x2, y2)

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

    return nearest, meta

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

def terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=".", stub="Test", tmp_dir="."):
    """Download 10m resolution elevation from terrain_tiles
    
    Parameters
    ----------
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area
        outdir: The directory that the tiff file gets saved
        stub: The name to be prepended to each file download
        depths: See 'identifiers' at the top of this file for a complete list
    
    Downloads
    ---------
        A Tiff file of elevation with severe outlier pixels replaced by the nearest neighbour

    """
    
    # Load the raw data
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    filename = os.path.join(tmp_dir, f"{stub}_terrain_original.tif")
    run_gdalwarp(bbox, filename)

    # Fix bad measurements
    dem, meta = interpolate_nan(filename)        
    filename = os.path.join(outdir, f"{stub}_terrain.tif")
    download_dem(dem, meta, filename)


# -

if __name__ == '__main__':
    # Change directory to the PaddockTS repo so that the 'DAESIM_preprocess/terrain_tiles.xml' is in the pythonpath
    if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
        paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/PaddockTS")
    elif os.path.basename(os.getcwd()) != "PaddockTS":
        paddockTS_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
    else:  # Already running locally from PaddockTS root
        paddockTS_dir = os.getcwd()

    print("Changing directory to:",paddockTS_dir)
    os.chdir(paddockTS_dir)
    
    terrain_tiles()