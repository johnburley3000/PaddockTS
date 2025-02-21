# !pip install sentinelhub

# +
# Need to create a Copernicus CLIENT_ID and CLIENT_SECRET by following these instructions
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html

# Then add these details to credentials.json 
# (this file is ignored by git, so you don't commit your credentials to github)

# +
# Useful links:
# Example notebook demo-ing Sentinel Hub: https://github.com/eu-cdse/notebook-samples/blob/c0e0ade601973c5d4e4bf66a13c0b76ebb099805/sentinelhub/migration_from_scihub_guide.ipynb
# Copernicus API reference is here: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/ApiReference.html
# SentinelHub ReadTheDocs is here: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html
# -

import json
import requests
import xarray as xr
import numpy as np
import rasterio

from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, Geometry)

# For using their plotting function
from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt

# +
# Load your Copernicus Client ID and Secret from a json file that doesn't get committed to the repository
with open("credentials.json", "r") as file:
    credentials = json.load(file)
    
CLIENT_ID = credentials["CLIENT_ID"]
CLIENT_SECRET = credentials["CLIENT_SECRET"]

print("Client ID:", CLIENT_ID)
print("Client Secret:", CLIENT_SECRET) 

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
# -

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"


# +
def define_query(lat=-34.389042, lon=148.469499, buffer=0.005 , start_year=2020, end_year=2021):
    """Just requesting the bands needed for RGB and fractional cover"""
    lat_range = (lat-buffer, lat+buffer)
    lon_range = (lon-buffer, lon+buffer)
    time_range = (f'{start_year}-01-01', f'{end_year}-12-31')
    query = {
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'measurements': ['nbart_red', 'nbart_green', 'nbart_blue', 'nbart_nir_1', 'nbart_swir_2','nbart_swir_3'],

    }
    return query
    
lat = -34.389042
lon = 148.469499
buffer = 0.005    # 0.01 is 1km in each direction to 2kmx2km total     
start_year = 2010  # This automatically gets the earlist timepoint (late 2015)
end_year = 2030    # This automatically gets the most recent timepoint

query = define_query(lat, lon, buffer, start_year, end_year)
query

# +
# aoi_coords_wgs84 = [4.20762, 50.764694, 4.487708, 50.916455]
aoi_coords_wgs84 = [148.46449900000002, -34.394042000000006, 148.474499, -34.384042]

time_interval = '2022-07-01', '2022-07-20'

resolution = 10
aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {aoi_size} pixels')
# +
catalog = SentinelHubCatalog(config=config)
search_iterator = catalog.search(
    DataCollection.SENTINEL2_L2A,
    bbox=aoi_bbox,
    time=time_interval,
    fields={"include": ["id", "properties.datetime"], "exclude": []},

)
results = list(search_iterator)
results
# -

# %%time
catalog.get_feature(DataCollection.SENTINEL2_L2A, 'S2A_MSIL2A_20220701T001121_N0510_R073_T55HFC_20240625T103645.SAFE')







# +
# Adding a 4th band
# evalscript_true_color = """
#     //VERSION=3

#     function setup() {
#         return {
#             input: [{
#                 bands: ["B02", "B03", "B04", "B08"]
#             }],
#             output: {
#                 bands: 4
#             }
#         };
#     }

#     function evaluatePixel(sample) {
#         return [sample.B04, sample.B03, sample.B02, sample.B08];
#     }
# """
# -

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

# +
# %%time
request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                name="s2", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=time_interval,
            other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}           )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=aoi_bbox,
    size=aoi_size,
    config=config,
)
true_color_imgs = request_true_color.get_data()

print(true_color_imgs[0].shape)


# +
def plot_image(
    image: np.ndarray,
    factor: float = 1.0,
    clip_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

plot_image(true_color_imgs[0], factor=3.5 / 255, clip_range=(0, 1))
# -






