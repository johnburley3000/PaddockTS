# +
# Following this example notebook
# https://github.com/eu-cdse/notebook-samples/blob/main/stac/stac_ndvi.ipynb

# +
# # !pip install stackstac && pip install pystac_client && pip install geogif
# -

import pystac_client
import geogif
import stackstac
import os

# +
# Load your Copernicus s3 keys from a json file (or copy here, but don't commit)
import json
with open("credentials-s3.json", "r") as file:
    credentials = json.load(file)
    
ACCESS_KEY = credentials["ACCESS_KEY"]
SECRET_KEY = credentials["SECRET_KEY"]

print("ACCESS_KEY:", ACCESS_KEY)
print("SECRET_KEY:", SECRET_KEY) 
# -

URL = "https://stac.dataspace.copernicus.eu/v1"
cat = pystac_client.Client.open(URL)
cat.add_conforms_to("ITEM_SEARCH")

geom = {
    "type": "Polygon",
    "coordinates": [
        [
            [14.254, 50.014],
            [14.587, 50.014],
            [14.587, 50.133],
            [14.254, 50.133],
            [14.254, 50.014],
        ]
    ],
}

params = {
    "max_items": 100,
    "collections": "sentinel-2-l2a",
    "datetime": "2024-01-01/2024-12-01",
    "intersects": geom,
    "filter": {
    "op": "<",
    "args": [
        {
            "property": "eo:cloud_cover"
        },
        10
    ]
},
    "sortby": "properties.eo:cloud_cover",
    "fields": {"exclude": ["geometry"]}
}

# %%time
items = list(cat.search(**params).items_as_dicts())

# %%time
stack = stackstac.stack(
    items=items,
    resolution=(20, 20),
    bounds_latlon=(14.254, 50.014, 14.587, 50.133),
    chunksize=98304,
    epsg=32634, 
    gdal_env=stackstac.DEFAULT_GDAL_ENV.updated({
        'GDAL_NUM_THREADS': -1,
        'GDAL_HTTP_UNSAFESSL': 'YES',
        'GDAL_HTTP_TCP_KEEPALIVE': 'YES',
        'AWS_VIRTUAL_HOSTING': 'FALSE',
        'AWS_HTTPS': 'YES',
    })
)

ds = stack.to_dataset(dim="band")
ds = ds.reset_coords(drop=True).set_coords(["x", "y", "time"])

ds
