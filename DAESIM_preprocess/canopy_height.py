# region
# The registry is here: https://registry.opendata.aws/dataforgood-fb-forests/
# endregion

# region
# Standard Libraries
import os
import subprocess

# Dependencies
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.transform import Affine
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib import colors

import boto3
# Note: To make boto3 work, I had to create a file named .aws/credentials in my /home/147/cb8590 with these contents:  
# [default]
# aws_access_key_id = ACCESS_KEY
# aws_secret_access_key = SECRET_KEY

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import create_bbox, transform_bbox, scratch_dir, gdata_dir
# endregion

canopy_height_dir ='/g/data/xe2/datasets/Global_Canopy_Height'

def identify_relevant_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005):
    """Find the tiles that overlap with the region of interest"""
    tiles_geojson_filename = os.path.join(canopy_height_dir, 'tiles_global.geojson')
    gdf = gpd.read_file(tiles_geojson_filename)
    bbox = create_bbox(lat, lon, buffer)
    roi_coords = box(*bbox)
    roi_polygon = Polygon(roi_coords)

    relevant_tiles = []
    for idx, row in gdf.iterrows():
        tile_polygon = row['geometry']
        if tile_polygon.intersects(roi_polygon):
            relevant_tiles.append(row['tile'])
    return relevant_tiles


def download_new_tiles(tiles):
    """Download any tiles that we haven't already downloaded"""
    # Find tiles we haven't downloaded yet
    to_download = []
    for tile in tiles:
        tile_path = os.path.join(canopy_height_dir, f"{tile}.tif")
        if not os.path.isfile(tile_path):
            to_download.append(tile)
            
    if len(to_download) == 0:
        return
    
    # Setup the AWS connection
    s3 = boto3.client('s3')
    
    # Download tiles if we don't have them in gdata already
    print(f"Downloading {tiles}")
    for tile in to_download:
        bucket_name = 'dataforgood-fb-data'
        file_key = f'forests/v1/alsgedi_global_v6_float/chm/{tile}.tif'
        local_file_path = os.path.join(canopy_height_dir, f'{tile}.tif')
        s3.download_file(bucket_name, file_key, local_file_path)
        print("Downloaded:", local_file_path)


def merge_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="/g/data/xe2/cb8590/", stub="Test", tmp_dir='/scratch/xe2/cb8590/tmp'):
    """Create a tiff file with just the region of interest. This may use just one tile, or merge multiple tiles"""
    
    # Convert the bounding box to EPSG:3857 (tiles.geojson uses EPSG:4326, but the tiff files use EPSG:3857')
    bbox = create_bbox(lat, lon, buffer)
    bbox_3857 = transform_bbox(bbox)
    roi_coords_3857 = box(*bbox_3857)
    roi_polygon_3857 = Polygon(roi_coords_3857)
    
    relevant_tiles = identify_relevant_tiles(lat, lon, buffer)
    
    # Crop the images and save a cropped tiff file for each one
    for tile in relevant_tiles:
        tiff_file = os.path.join(canopy_height_dir, f'{tile}.tif')
        with rasterio.open(tiff_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, [mapping(roi_polygon_3857)], crop=True)
            out_meta = src.meta.copy()
        cropped_tiff_filename = os.path.join(tmp_dir, f'{tile}_cropped.tif')
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(cropped_tiff_filename, "w", **out_meta) as dest:
            dest.write(out_image)
            
    # Merge the cropped tiffs
    src_files_to_mosaic = []
    for tile in relevant_tiles:
        tiff_file = os.path.join(tmp_dir, f'{tile}_cropped.tif')
        src = rasterio.open(tiff_file)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    # From visual inspection, it looks like the canopy height map is offset by about 10m south. This corrects that.
    original_transform = out_meta['transform']
    new_transform = original_transform * Affine.translation(0, -10)

    # Write the merged raster to a new tiff
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": new_transform
    })
    output_tiff_filename = os.path.join(outdir, f'{stub}_canopy_height.tif')
    with rasterio.open(output_tiff_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files_to_mosaic:
        src.close()
    print("Saved:", output_tiff_filename)


def visualise_canopy_height(filename, outpath=scratch_dir, stub="Test"):
    """Pretty visualisation of the canopy height"""

    with rasterio.open(filename) as src:
        image = src.read(1)  
        transform = src.transform 
    
    # Bin the slope into categories
    bin_edges = np.arange(0, 16, 1) 
    categories = np.digitize(image, bin_edges, right=True)
    
    # Define a color for each category
    colours = plt.cm.viridis(np.linspace(0, 1, len(bin_edges) - 2))
    cmap = colors.ListedColormap(['white'] + list(colours))
    
    # Plot the values
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(categories, cmap=cmap)
    
    # Assign the colours
    labels = [f'{bin_edges[i]}' for i in range(len(bin_edges))]
    labels[-1] = '>=15'
    
    # Place the tick label in the middle of each category
    num_categories = len(bin_edges)
    start_position = 0.5
    end_position = num_categories + 0.5
    step = (end_position - start_position)/(num_categories)
    tick_positions = np.arange(start_position, end_position, step)
    
    cbar = plt.colorbar(im, ticks=tick_positions)
    cbar.ax.set_yticklabels(labels)
    
    plt.title('Canopy Height (m)', size=14)
    plt.tight_layout()
    filename = os.path.join(outpath, f"{stub}_canopy_height.png")
    plt.savefig(filename)
    print("Saved", filename)
    plt.show()

def canopy_height(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=scratch_dir, stub="Test", tmp_dir='/scratch/xe2/cb8590/tmp'):
    """Create a merged canopy height raster, downloading new tiles if necessary"""
    tiles = identify_relevant_tiles(lat, lon, buffer)
    download_new_tiles(tiles)
    merge_tiles(lat, lon, buffer, outdir, stub, tmp_dir)


# %%time
if __name__ == '__main__':

    outdir = '/g/data/xe2/cb8590/Data/shelter/'
    stub = '34_0_148_5'
    lat = -34.0
    lon = 148.5
    buffer = 0.05
    canopy_height(lat, lon, buffer, outdir, stub)
    # visualise_canopy_height("/g/data/xe2/cb8590/Data/PadSeg/MILG_canopy_height.tif")


