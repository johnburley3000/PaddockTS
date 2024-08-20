# region
# The registry is here: https://registry.opendata.aws/dataforgood-fb-forests/
# endregion

# region
# Standard Libraries
import os
import subprocess

# Dependencies
import geopandas as gpd
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform
import boto3
import rasterio
from rasterio.plot import show
from pyproj import Transformer

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import create_bbox, transform_bbox
# endregion

# Setup the AWS connection
s3 = boto3.client('s3')

# region
# To make boto3 work, I had to create a file named .aws/credentials in my /home/147/cb8590 with these contents:  
# [default]
# aws_access_key_id = ACCESS_KEY
# aws_secret_access_key = SECRET_KEY
# region = ap-southeast-2
# endregion

# !ls .aws/credentials

# Specify the region of interest
lat, lon = -34.3890427, 148.469499
buffer = 0.1  # 0.01 degrees is about 1km in each direction, so 2km total

# Filenames
stub = "MILG_1km"
outdir = '/g/data/xe2/cb8590/Global_Canopy_Height'
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

# region
# Load in all the relevant files and crop them to the region of interest
bbox_3857 = transform_bbox(bbox)
roi_3857 = box(*bbox_3857)
roi_polygon_3857 = Polygon(roi_3857)

# Not sure if I'll need all these lists, but making them for now
images = []
transforms = []
metas = []
for tile in relevant_tiles:
    tiff_file = os.path.join(outdir, f'{tile}.tif')
    with rasterio.open(tiff_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, [mapping(roi_polygon_3857)], crop=True)
        out_meta = src.meta.copy()
        images.append(out_image)
        transforms.append(out_transform)
        metas.append(meta)
# endregion

# region
tile = '311230211'
tiff_file = os.path.join(outdir, f'{tile}.tif')

# Crop the image to the region of interest
with rasterio.open(tiff_file) as src:
    out_image, out_transform = rasterio.mask.mask(src, [mapping(roi_polygon_3857)], crop=True)
    out_meta = src.meta.copy()


show(out_image)
# endregion

# region
# Update the metadata to reflect the new dimensions
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

# Save the cropped image as a new GeoTIFF file
cropped_tiff_file = '311230211_cropped.tif'
with rasterio.open(cropped_tiff_file, "w", **out_meta) as dest:
    dest.write(out_image)

print(f'Cropped image saved to {cropped_tiff_file}')

# endregion

# !ls
