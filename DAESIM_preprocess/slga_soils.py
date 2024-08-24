# +
# Catalog is here: https://www.asris.csiro.au/arcgis/rest/services/TERN
# -

# Standard Libraries
import os
import pickle
import time

# Dependencies
import xarray as xr
import pandas as pd
from owslib.wcs import WebCoverageService
from pyproj import Proj, Transformer

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import create_bbox, scratch_dir

# Taken from GeoDataHarvester: https://github.com/Sydney-Informatics-Hub/geodata-harvester/blob/main/src/geodata_harvester/getdata_slga.py
asris_urls = {
    "Clay": "https://www.asris.csiro.au/arcgis/services/TERN/CLY_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Silt": "https://www.asris.csiro.au/arcgis/services/TERN/SLT_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Sand": "https://www.asris.csiro.au/arcgis/services/TERN/SND_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "pH_CaCl2": "https://www.asris.csiro.au/arcgis/services/TERN/PHC_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Bulk_Density": "https://www.asris.csiro.au/arcgis/services/TERN/BDW_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Available_Water_Capacity": "https://www.asris.csiro.au/arcgis/services/TERN/AWC_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Effective_Cation_Exchange_Capacity": "https://www.asris.csiro.au/arcgis/services/TERN/ECE_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Total_Nitrogen": "https://www.asris.csiro.au/arcgis/services/TERN/NTO_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Total_Phosphorus": "https://www.asris.csiro.au/arcgis/services/TERN/PTO_ACLEP_AU_NAT_C/MapServer/WCSServer"
}
identifiers = {
    "0-5cm": '0',
    "5-15cm": '4',
    "15-30cm":'8',
    "30-60cm":'12',
    "60-100cm":'16',
    "100-200cm":'20'
}

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

def slga_soils(variables=["Clay", "Sand", "Silt", "pH_CaCl2"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=scratch_dir, stub="Test"):
    """Download soil variables from CSIRO"""
    bbox = create_bbox(lat, lon, buffer)
    identifier = identifiers["5-15cm"]
    for variable in variables:
        filename = os.path.join(outdir, f"{stub}_{variable}.tif")
        url = asris_urls[variable]
        
        # The SLGA server is a bit temperamental, so sometimes you have to try again
        attempt = 0
        delay = 5
        max_retries = 3
        while attempt < max_retries:
            time.sleep(delay)
            try:
                download_tif(bbox, url, identifier, filename)
                print(f"Downloaded {filename}")
                attempt = max_retries
            except:
                print(f"Failed to download {variable}")
                attempt+=1
        

if __name__ == '__main__':
    slga_soils()


