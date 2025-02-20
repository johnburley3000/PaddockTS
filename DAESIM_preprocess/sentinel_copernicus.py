# !pip install sentinelhub

# +
# Need to create a Copernicus CLIENT_ID and CLIENT_SECRET by following these instructions
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html

# Then add these details to credentials.json 
# (this file is ignored by git, so you don't commit your credentials to github)

# +
# The Copernicus API reference is here: 
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/ApiReference.html
# -

import json
import requests
import xarray as xr
import numpy as np
import rasterio

from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, Geometry)

# +
# Load your Copernicus Client ID and Secret from a json file that doesn't get committed to the repository
with open("credentials.json", "r") as file:
    credentials = json.load(file)
    
CLIENT_ID = credentials["CLIENT_ID"]
CLIENT_SECRET = credentials["CLIENT_SECRET"]

print("Client ID:", client_id)
print("Client Secret:", client_secret) 

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
# -

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"

catalog = SentinelHubCatalog(config=config)

# +
aoi_coords_wgs84 = [4.20762, 50.764694, 4.487708, 50.916455]
time_interval = '2022-07-01', '2022-07-20'

resolution = 10
aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {aoi_size} pixels')
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
# -

from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt


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







def get_access_token():
    """Obtain an access token from Copernicus Data Space Ecosystem."""
    response = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(CLIENT_ID, CLIENT_SECRET)
    )
    return response.json().get("access_token")


def define_query(lat=-34.389042, lon=148.469499, buffer=0.005, start_year=2020, end_year=2021):
    """Define the bounding box and time range for the Sentinel-2 request."""
    lat_range = (lat-buffer, lat+buffer)
    lon_range = (lon-buffer, lon+buffer)
    time_range = (f"{start_year}-01-01/{end_year}-12-31")

    return {
        "bbox": f"{lon_range[0]},{lat_range[0]},{lon_range[1]},{lat_range[1]}",
        "datetime": time_range,
        "collections": ["sentinel-2-l2a"],
        "band_names": ["B04", "B03", "B02", "B08", "B11", "B12"],  # Red, Green, Blue, NIR, SWIR2, SWIR3
    }


lat = -34.389042
lon = 148.469499
buffer = 0.005    # 0.01 is 1km in each direction to 2kmx2km total     
start_year = 2010  # This automatically gets the earlist timepoint (late 2015)
end_year = 2030    # This automatically gets the most recent timepoint
query = define_query(lat, lon, buffer, start_year, end_year)
query













access_token = get_access_token()
headers = {"Authorization": f"Bearer {access_token}"}



# +
# Define the bounding box and time range
bbox = [13.822174072265625, 45.85080395917834, 14.55963134765625, 46.29191774991382]
start_date = "2021-01-01"
end_date = "2021-12-31"

# Request Sentinel-2 imagery with a time range
response = requests.post(
    "https://sh.dataspace.copernicus.eu/api/v1/process",
    headers={"Authorization": f"Bearer {access_token}"},
    json={
        "input": {
            "bounds": {"bbox": bbox,
                        "properties": {
                            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                          }
                      },
            "data": [
                  {
                    "type": "sentinel-2-l2a",
                    "id": "string",
                    "dataFilter": {
                      "timeRange": {
                        "from": "2018-10-01T00:00:00.000Z",
                        "to": "2018-11-01T00:00:00.000Z"
                      },
                  }
                  }
                ]
        },
        "output": {
            "resx": 10,
            "resy": 10
        },
        "evalscript": """
        //VERSION=3
        function setup() {
          return {
            input: ["B02", "B03", "B04"],
            output: { bands: 3 }
          };
        }

        function evaluatePixel(sample) {
          return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
        """
    },
)
response
# -

response.content

with open("sentinel_output.tif", "wb") as f:
    f.write(response.content)




