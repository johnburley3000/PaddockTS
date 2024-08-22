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
import boto3
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.transform import Affine
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib import colors


# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import create_bbox, transform_bbox, scratch_dir
# endregion

# region
# Setup the AWS connection

# To make boto3 work, I had to create a file named .aws/credentials in my /home/147/cb8590 with these contents:  
# [default]
# aws_access_key_id = ACCESS_KEY
# aws_secret_access_key = SECRET_KEY

s3 = boto3.client('s3')
# endregion

# Specify the region of interest
lat, lon = -34.3890427, 148.469499
buffer = 0.033  # 0.01 degrees is about 1km in each direction, so 2km total

# Filenames
stub = "MILG_1km"
outdir = '/g/data/xe2/cb8590/Global_Canopy_Height'
tmp_dir = os.path.join(scratch_dir, "tmp")
tiles_geojson = os.path.join(outdir, 'tiles_global.geojson')

# %%time
# Read in the geometries
gdf = gpd.read_file(tiles_geojson)

# region
# Figure out which tiles intersect the region of interest
bbox = create_bbox(lat, lon, buffer)
roi_coords = box(*bbox)
roi_polygon = Polygon(roi_coords)

relevant_tiles = []
for idx, row in gdf.iterrows():
    tile_polygon = row['geometry']
    if tile_polygon.intersects(roi_polygon):
        relevant_tiles.append(row['tile'])
relevant_tiles
# endregion

# region
# %%time
# Find tiles we haven't downloaded yet
to_download = []
for tile in relevant_tiles:
    tile_path = os.path.join(outdir, f"{tile}.tif")
    if not os.path.isfile(tile_path):
        to_download.append(tile)

# Download tiles if we don't have them in gdata already
for tile in to_download:
    bucket_name = 'dataforgood-fb-data'
    file_key = f'forests/v1/alsgedi_global_v6_float/chm/{tile}.tif'
    local_file_path = os.path.join(outdir, f'{tile}.tif')
    s3.download_file(bucket_name, file_key, local_file_path)
    print("Downloaded:", local_file_path)
# endregion

# !ls /g/data/xe2/cb8590/Global_Canopy_Height

# Convert the bounding box to EPSG:3857 (tiles.geojson uses EPSG:4326, but the tiff files use EPSG:3857')
bbox = create_bbox(lat, lon, buffer)
bbox_3857 = transform_bbox(bbox)
roi_coords_3857 = box(*bbox_3857)
roi_polygon_3857 = Polygon(roi_coords_3857)

# region
# %%time
# Crop the images and save a cropped tiff file for each one
for tile in relevant_tiles:
    tiff_file = os.path.join(outdir, f'{tile}.tif')
    with rasterio.open(tiff_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, [mapping(roi_polygon_3857)], crop=True)
        out_meta = src.meta.copy()
    cropped_tiff_file = os.path.join(tmp_dir, f'{tile}_cropped.tif')
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    with rasterio.open(cropped_tiff_file, "w", **out_meta) as dest:
        dest.write(out_image)

# Merge the cropped tiffs
src_files_to_mosaic = []
for tile in relevant_tiles:
    tiff_file = os.path.join(tmp_dir, f'{tile}_cropped.tif')
    src = rasterio.open(tiff_file)
    src_files_to_mosaic.append(src)
mosaic, out_trans = merge(src_files_to_mosaic)
out_meta = src_files_to_mosaic[0].meta.copy()

# From visual inspection, it looks like the canopy height map is offset by about 10m south and 10m west. This corrects that.
original_transform = out_meta['transform']
# new_transform = original_transform * Affine.translation(10, -10)
new_transform = original_transform * Affine.translation(0, -10)

# Write the merged raster to a new tiff
out_meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": new_transform
})
output_tiff = os.path.join(tmp_dir, 'combined_image.tif')
with rasterio.open(output_tiff, "w", **out_meta) as dest:
    dest.write(mosaic)
for src in src_files_to_mosaic:
    src.close()
print("Saved:", output_tiff)
# endregion

# region
# %%time
tiff_file = os.path.join(tmp_dir, "combined_image.tif")
with rasterio.open(tiff_file) as src:
    image = src.read(1)  
    transform = src.transform 

show(image)
# endregion

# region
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
cbar = plt.colorbar(im, ticks=np.arange(len(bin_edges)))
cbar.ax.set_yticklabels(labels)

plt.title('Canopy Height', size=14)
plt.tight_layout()
plt.show()
# endregion
