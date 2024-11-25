# +
# Catalog is here: https://www.asris.csiro.au/arcgis/rest/services/TERN
# -

# Standard Libraries
import os
import pickle
import time

# Dependencies
import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
from owslib.wcs import WebCoverageService
from pyproj import Proj, Transformer
import matplotlib.pyplot as plt

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import create_bbox, scratch_dir, plot_categorical

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
    "5-15cm": '4',
    "15-30cm":'8',
    "30-60cm":'12',
    "60-100cm":'16',
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

def slga_soils(variables=["Clay", "Sand", "Silt", "pH_CaCl2"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=scratch_dir, stub="Test",  depths=["5-15cm"]):
    """Download soil variables from CSIRO"""
    bbox = create_bbox(lat, lon, buffer)
    for depth in depths:
        identifier = identifiers[depth]
        for variable in variables:
            filename = os.path.join(outdir, f"{stub}_{variable}_{depth}.tif")
            url = asris_urls[variable]
            
            # The SLGA server is a bit temperamental, so sometimes you have to try again
            attempt = 0
            delay = 10
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


def visualise_soil_texture(outdir, visuals_dir=scratch_dir, stub="Test"):
    """Convert from sand silt and clay percent to the 12 categories in the soil texture triangle"""

    # Load the sand, silt and clay layers
    filename_sand = os.path.join(outdir, f"{stub}_Sand.tif")
    filename_silt = os.path.join(outdir, f"{stub}_Silt.tif")
    filename_clay = os.path.join(outdir, f"{stub}_Clay.tif")
                            
    ds_sand = rxr.open_rasterio(filename_sand)
    ds_silt = rxr.open_rasterio(filename_silt)
    ds_clay = rxr.open_rasterio(filename_clay)
    
    sand_array = ds_sand.isel(band=0).values
    silt_array = ds_silt.isel(band=0).values
    clay_array = ds_clay.isel(band=0).values
    
    # Rescale values to add up to 100 (range was 74%-108% in Mullon example)
    total_percent = sand_array + silt_array + clay_array
    sand_percent = (sand_array / total_percent) * 100
    silt_percent = (silt_array / total_percent) * 100
    clay_percent = (clay_array / total_percent) * 100

    # Assign soil texture categories
    soil_texture = np.empty(sand_array.shape, dtype=object)
    
    # Fudged the boundaries between sand, loamy sand, and sandy loam a little, but the rest of these values should match the soil texture triangle exactly
    soil_texture[(clay_array < 20) & (silt_array < 50)] = 'Sandy Loam'      # Sandy Loam needs to come before Loam
    soil_texture[(sand_array >= 70) & (clay_array < 15)] = 'Loamy Sand'     # Loamy Sand needs to come from Sand
    soil_texture[(sand_array >= 85) & (clay_array < 10)] = 'Sand'
    soil_texture[(clay_array < 30) & (silt_array >= 50)] = 'Silt Loam'     # Silt Loam needs to come before Silt
    soil_texture[(clay_array < 15) & (silt_array >= 80)] = 'Silt'
    soil_texture[(clay_array >= 27) & (clay_array < 40) & (sand_array < 20)] = 'Silty Clay Loam'
    soil_texture[(clay_array >= 40) & (silt_array >= 40)] = 'Silty Clay'
    soil_texture[(clay_array >= 40) & (silt_array < 40) & (sand_array < 45)] = 'Clay'
    soil_texture[(clay_array >= 35) & (sand_array >= 45)] = 'Sandy Clay'
    soil_texture[(clay_array >= 27) & (clay_array < 40) & (sand_array >= 20) & (sand_array < 45) ] = 'Clay Loam'
    soil_texture[(clay_array >= 20) & (clay_array < 35) & (sand_array >= 45) & (silt_array < 28)] = 'Sandy Clay Loam'
    soil_texture[(clay_array >= 15) & (clay_array < 27) & (silt_array >= 28) & (silt_array < 50) & (sand_array < 53)] = 'Loam'

    colour_dict = {
        'Sandy Loam': "violet",
        'Loamy Sand': "lightpink",
        'Sand': "orange",
        'Silt Loam': "yellowgreen",
        'Silt': "limegreen",
        'Silty Clay Loam': "lightseagreen",
        'Silty Clay': "turquoise",
        'Clay': "gold",
        'Sandy Clay': "red",
        'Clay Loam': "greenyellow",
        'Sandy Clay Loam': "salmon",
        'Loam':"chocolate"
    }
    colour_dict = {key: value for key, value in colour_dict.items() if key in np.unique(soil_texture)}
    filename = os.path.join(visuals_dir, f"{stub}_soil_texture.png")
    plot_categorical(soil_texture, colour_dict, "Soil Texture", filename)

def visualise_soil_pH(outdir, visuals_dir=scratch_dir, stub="Test"):
    fig, ax = plt.subplots(figsize=(8, 6))
    filename = os.path.join(outdir, f"{stub}_pH_CaCl2.tif")
    ds_pH = rxr.open_rasterio(filename)
    ds_pH_scaled = ds_pH[0].values
    plt.imshow(ds_pH_scaled, cmap='RdYlGn')
    plt.colorbar()
    plt.title("Soil pH")
    plt.tight_layout()
    filename = os.path.join(visuals_dir, f"{stub}_pH.png")
    plt.savefig(filename)
    print("Saved", filename)
    plt.show()

if __name__ == '__main__':

    # Download all the soil layers for a single location
    stub = "Harden"
    variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
    depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    buffer=0.0003
    latitude = -34.52194
    longitude=148.30472
    slga_soils(variables=variables, latitude=latitude, longitude=longitude, buffer=buffer, stub=stub, depths=depths)
    
    # Load the tiff files we just downloaded (each should just have a single pixel)
    values = []
    stub = "Harden"
    depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    for variable in variables:
        for depth in depths:
            filename = os.path.join(scratch_dir, f"{stub}_{variable}_{depth}.tif")
            ds = rxr.open_rasterio(filename)
            value = float(ds.isel(band=0, x=0, y=0).values)
            values.append({
                "variable":variable,
                "depth":depth,
                "value":value
            })
    
    # Pivot
    df = pd.DataFrame(values)
    pivot_df = df.pivot(index='depth', columns='variable', values='value')
    pivot_df = pivot_df.reset_index() 
    
    # Sort by depth
    df = pivot_df
    depth_order = ['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    df['depth'] = pd.Categorical(df['depth'], categories=depth_order, ordered=True)
    sorted_df = df.sort_values(by='depth')
    
    # Save
    sorted_df.to_csv("Harden_Soils_sorted.csv", index=False)
