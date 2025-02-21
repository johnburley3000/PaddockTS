# + endofcell="--"
# # +
# Need to create a Copernicus CLIENT_ID and CLIENT_SECRET by following these instructions
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html

# Then add these details to credentials.json 
# (this file is ignored by git, so you don't commit your credentials to github)

# # +
# The Copernicus API reference is here: 
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/ApiReference.html
# -

# --

import json
import requests
import xarray as xr
import numpy as np
import rasterio

# +
# # +
# Load your Copernicus Client ID and Secret from a json file that doesn't get committed to the repository
with open("credentials.json", "r") as file:
    credentials = json.load(file)
    
CLIENT_ID = credentials["CLIENT_ID"]
CLIENT_SECRET = credentials["CLIENT_SECRET"]

print("Client ID:", CLIENT_ID)
print("Client Secret:", CLIENT_SECRET) 

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"



# +

def get_access_token():
    """Obtain an access token from Copernicus Data Space Ecosystem."""
    response = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(CLIENT_ID, CLIENT_SECRET)
    )
    return response.json().get("access_token")



# -

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


access_token = get_access_token()


# +
headers = {"Authorization": f"Bearer {access_token}"}

# Copernicus example
response = requests.post('https://sh.dataspace.copernicus.eu/api/v1/process',
  headers={"Authorization" : f"Bearer {access_token}"},
  json={
    "input": {
        "bounds": {
            "bbox": [
                13.822174072265625,
                45.85080395917834,
                14.55963134765625,
                46.29191774991382
            ]
        },
        "data": [{
            "type": "sentinel-2-l2a"
        }]
    },
    "evalscript": """
    //VERSION=3

    function setup() {
      return {
        input: ["B02", "B03", "B04"],
        output: {
          bands: 3
        }
      };
    }

    function evaluatePixel(
      sample,
      scenes,
      inputMetadata,
      customData,
      outputMetadata
    ) {
      return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """
})


with open("sentinel_output.tif", "wb") as f:
    f.write(response.content)

# -

response.headers
