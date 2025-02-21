# +
# # !pip install sentinelhub

# +
# Need to create a Copernicus CLIENT_ID and CLIENT_SECRET by following these instructions
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html

# Then add these details to credentials.json 
# (this file is ignored by git, so you don't commit your credentials to github)

# +
# Useful links:
# Example notebook demo-ing Sentinel Hub: https://github.com/eu-cdse/notebook-samples/blob/c0e0ade601973c5d4e4bf66a13c0b76ebb099805/sentinelhub/migration_from_scihub_guide.ipynb
# Copernicus API reference is here: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/ApiReference.html
# SentinelHub ReadTheDocs is here: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html#Example-8-:-Multiple-timestamps-data
# SentinelHub Usage allowance is here: https://shapps.dataspace.copernicus.eu/dashboard/#/
# Note: Free allowance is 30,000 units = (5km x 5km x 3 bands) x 300 timepoints x 100 locations

# +
import json
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions, 
    SHConfig, 
    SentinelHubCatalog, 
)

# +
# Load your Copernicus Client ID and Secret from a json file (or copy here, but don't commit)
with open("credentials.json", "r") as file:
    credentials = json.load(file)
    
CLIENT_ID = credentials["CLIENT_ID"]
CLIENT_SECRET = credentials["CLIENT_SECRET"]

print("Client ID:", CLIENT_ID)
print("Client Secret:", CLIENT_SECRET) 
# -

# Setup the authentication
config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"

# Define the region of interest and date
aoi_coords_wgs84 = [148.46449900000002, -34.394042000000006, 148.474499, -34.384042]
time_interval = ('2022-07-01', '2022-10-20')
max_cloud_percent = 10
resolution = 10

# Calculate the bounding box and resolution
aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)
print(f'Image shape at {resolution} m resolution: {aoi_size} pixels')

# Search the catalogue for images at this location, timerange and cloud percent 
catalog = SentinelHubCatalog(config=config)
data_collection = DataCollection.SENTINEL2_L2A.define_from(
                name="s2", service_url="https://sh.dataspace.copernicus.eu"
            )
search_iterator = catalog.search(
    data_collection,
    bbox=aoi_bbox,
    time=time_interval,
    filter=f"eo:cloud_cover < {max_cloud_percent}",
    fields={"include": ["id", "properties.datetime"], "exclude": []},
)
results = list(search_iterator)
unique_dates = set(result['properties']['datetime'][:10] for result in results)
unique_dates

# +
# Setup a minimal evaluation script.
# Can add to this script things like masking cloud cover and calculating NDVI

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

def get_true_color_request(date, bbox, size):
    return SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=(date, date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

list_of_requests = [get_true_color_request(date, aoi_bbox, aoi_size) for date in unique_dates]
list_of_requests = [request.download_list[0] for request in list_of_requests]
# -

# Download the bands
data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
len(data)

data[0].shape

# +
# Visualising the data
ncols = 4
nrows = 3
aspect_ratio = aoi_size[0] / aoi_size[1]
subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}

fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows), subplot_kw=subplot_kw)

for idx, image in enumerate(data):
    ax = axs[idx // ncols][idx % ncols]
    ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
    ax.set_title(f"{list(unique_dates)[idx]}", fontsize=10)

plt.tight_layout()

# +
# TODO
# 
# Allow user to add more bands like in this example: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html#Example-3:-All-Sentinel-2's-raw-band-values
# Mask out clouds like in this example: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/examples/#true-color-cloudy-pixels-masked-out
# Combine into xarray with coordinates (not sure if we can download images with metadata directly or not)
# Compare with imagery from DEA
# -




