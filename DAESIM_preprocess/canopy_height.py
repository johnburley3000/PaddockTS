# +
import os
import subprocess

import geopandas as gpd
from shapely.geometry import Polygon
import boto3
# -

# Setup the AWS connection
s3 = boto3.client('s3')

# +
# To make boto3 work, I had to create a file named .aws/credentials in my /home/147/cb8590 with these contents:
# [default]
# aws_access_key_id = ACCESS_KEY
# aws_secret_access_key = SECRET_KEY
# region = ap-southeast-2

# !ls .aws/credentials
# -

# Filenames
outdir = '/g/data/xe2/cb8590/Global_Canopy_Height'
tiles_geojson = os.path.join(outdir, 'tiles_global.geojson')

# %%time
# Read in the geometries
gdf = gpd.read_file(tiles_geojson)

# +
# s
# -

# Filter tiles based on intersection with ROI
relevant_tiles = []
for idx, row in gdf.iterrows():
    tile_polygon = row['geometry']
    if tile_polygon.intersects(roi_polygon):
        relevant_tiles.append(row['tile'])

# +
# Specify the bucket name and prefix
bucket_name = 'dataforgood-fb-data'
prefix = 'forests/v1/alsgedi_global_v6_float/chm/311230213.tif'

# List the files in the specified path
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)


# +
# Downloading the file
# bucket_name = 'dataforgood-fb-data'
# file_key = 'forests/v1/alsgedi_global_v6_float/chm/your_filename.tif'
# local_file_path = '/path/to/your/local/directory/your_filename.tif'
# s3.download_file(bucket_name, file_key, local_file_path)
# print(f'File downloaded successfully to {local_file_path}')
# -

# Define the region of interest (ROI) polygon from provided coordinates
roi_coords = [
    [-120.7150726268212, 37.222377447414104],
    [-120.7150726268212, 37.10540660593651],
    [-120.58173775727298, 37.10540660593651],
    [-120.58173775727298, 37.222377447414104],
    [-120.7150726268212, 37.222377447414104]
]
roi_polygon = Polygon(roi_coords)

# Load the tiles.geojson file to identify relevant tiles
tiles_geojson = 'tiles.geojson'
gdf = gpd.read_file(tiles_geojson)

# !aws
