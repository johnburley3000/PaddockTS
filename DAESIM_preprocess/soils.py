# +
# Catalog is here: https://www.asris.csiro.au/arcgis/rest/services/TERN
# -


import pickle
import xarray as xr
import pandas as pd
from owslib.wcs import WebCoverageService
from pyproj import Proj, Transformer
import time

# Taken from GeoDataHarvester: https://github.com/Sydney-Informatics-Hub/geodata-harvester/blob/main/src/geodata_harvester/getdata_slga.py
asris_urls = {
    "Clay": "https://www.asris.csiro.au/arcgis/services/TERN/CLY_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Silt": "https://www.asris.csiro.au/ArcGIS/services/TERN/SLT_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Sand": "https://www.asris.csiro.au/ArcGIS/services/TERN/SND_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "pH_CaCl2": "https://www.asris.csiro.au/ArcGIS/services/TERN/PHC_ACLEP_AU_NAT_C/MapServer/WCSServer"
}
identifiers = {
    "0-5cm": '0',
    "5-15cm": '4',
    "15-30cm":'8',
    "30-60cm":'12',
    "60-100cm":'16',
    "100-200cm":'20'
}


def create_bbox(lat=-35.274603, lon=149.098498, buffer=0.005):
    """Generates a bbox in the order [West, North, East, South] that's required for most APIs"""
    # 0.01 degrees is about 1km in each direction, so 2kmx2km total
    # From my experimentation, the asris.csiro API allows a maximum bbox of about 40km (0.2 degrees in each direction)

    left, top, right, bottom = lon - buffer, lat - buffer, lon + buffer, lat + buffer 
    bbox = [left, top, right, bottom] 
    return bbox


def download_tif(bbox=[148.46449900000002, -34.3940427, 148.474499, -34.384042699999995], 
                  url="https://www.asris.csiro.au/arcgis/services/TERN/CLY_ACLEP_AU_NAT_C/MapServer/WCSServer", 
                  identifier='4', 
                  filename="output.tif"):
    
    # Request the data using WebCoverageService
    wcs = WebCoverageService(url, version='1.0.0')    
    crs = 'EPSG:4326'
    resolution = 1
    response = wcs.getCoverage(
        identifier=identifier,
        bbox=bbox,
        crs=crs,
        format='GeoTIFF',
        resx=resolution,
        resy=resolution
    )
    
    # Save the data to a tif file
    with open(filename, 'wb') as file:
        file.write(response.read())

    # Make sure to time.sleep(1) if running this multiple times to avoid throttling


if __name__ == '__main__':
    
    lat, lon = -34.3890427, 148.469499
    buffer = 0.005  
    
    bbox = create_bbox(lat, lon, buffer)
    identifier = identifiers["5-15cm"]
    
    for variable in asris_urls:
        filename = f"{variable}.tif"
        url = asris_urls[layer]
        download_tif(bbox, url, identifier, filename)
        time.sleep(1)
        print(f"Downloaded {filename}")
