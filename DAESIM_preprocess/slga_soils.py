# +
# Catalog is here: https://www.asris.csiro.au/arcgis/rest/services/TERN

# Standard Libraries
import os
import time
import argparse

# Dependencies
import numpy as np
import rioxarray as rxr
from owslib.wcs import WebCoverageService

# +
# Taken from GeoDataHarvester: https://github.com/Sydney-Informatics-Hub/geodata-harvester/blob/main/src/geodata_harvester/getdata_slga.py
slga_soils_abbrevations = {
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


def soil_texture(outdir=".", stub="TEST", depth="5-15cm"):
    """Convert from sand, silt and clay percent to the 12 categories in the soil texture triangle"""

    # Load the sand, silt and clay layers
    filename_sand = os.path.join(outdir, f"{stub}_Sand_{depth}.tif")
    filename_silt = os.path.join(outdir, f"{stub}_Silt_{depth}.tif")
    filename_clay = os.path.join(outdir, f"{stub}_Clay_{depth}.tif")
                            
    ds_sand = rxr.open_rasterio(filename_sand)
    ds_silt = rxr.open_rasterio(filename_silt)
    ds_clay = rxr.open_rasterio(filename_clay)
    
    sand_array = ds_sand.isel(band=0).values
    silt_array = ds_silt.isel(band=0).values
    clay_array = ds_clay.isel(band=0).values
    
    # The sand, silt and clay percent don't necessarily add up to 100% originally, because they get predicted by separate models
    total_percent = sand_array + silt_array + clay_array
    sand_percent = (sand_array / total_percent) * 100
    silt_percent = (silt_array / total_percent) * 100
    clay_percent = (clay_array / total_percent) * 100

    # Assign soil texture categories
    soil_texture = np.empty(sand_array.shape, dtype=object)
    
    # I simplified the boundaries between sand, loamy sand, and sandy loam a little, but the rest of these values should match the soil texture triangle exactly
    soil_texture[(clay_percent < 20)  & (silt_percent < 50)] = 'Sandy Loam'      # Sandy Loam needs to come before Loam
    soil_texture[(sand_percent >= 70) & (clay_percent < 15)] = 'Loamy Sand'     # Loamy Sand needs to come from Sand
    soil_texture[(sand_percent >= 85) & (clay_percent < 10)] = 'Sand'
    soil_texture[(clay_percent < 30)  & (silt_percent >= 50)] = 'Silt Loam'     # Silt Loam needs to come before Silt
    soil_texture[(clay_percent < 15)  & (silt_percent >= 80)] = 'Silt'
    soil_texture[(clay_percent >= 27) & (clay_percent < 40) & (sand_array < 20)] = 'Silty Clay Loam'
    soil_texture[(clay_percent >= 40) & (silt_percent >= 40)] = 'Silty Clay'
    soil_texture[(clay_percent >= 40) & (silt_percent < 40) & (sand_array < 45)] = 'Clay'
    soil_texture[(clay_percent >= 35) & (sand_percent >= 45)] = 'Sandy Clay'
    soil_texture[(clay_percent >= 27) & (clay_percent < 40) & (sand_array >= 20) & (sand_array < 45) ] = 'Clay Loam'
    soil_texture[(clay_percent >= 20) & (clay_percent < 35) & (sand_array >= 45) & (silt_array < 28)] = 'Sandy Clay Loam'
    soil_texture[(clay_percent >= 15) & (clay_percent < 27) & (silt_array >= 28) & (silt_array < 50) & (sand_array < 53)] = 'Loam'

    return soil_texture

def slga_soils(variables=["Clay", "Sand", "Silt"], lat=-34.3890427, lon=148.469499, buffer=0.005, outdir="", stub="TEST",  depths=["5-15cm"]):
    """Download soil variables from CSIRO at 90m resolution for region of interest
    
    Parameters
    ----------
        variables: See slga_soils_abbrevations at the top of this file for a complete list
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area
        outdir: The directory that the tiff files get saved. I recommend using the 'tmpdir' from the climate downloads to avoid having so many files in the outdir
        stub: The name to be prepended to each file download
        depths: See 'identifiers' at the top of this file for a complete list
    
    Downloads
    ---------
        A Tiff file for each variable/depth specified
    
    """
    print("Starting slga_soils")
    buffer = max(0.00001, buffer)
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]     # From my experimentation, the asris.csiro API allows a maximum bbox of about 40km (0.2 degrees in each direction)
    for depth in depths:
        identifier = identifiers[depth]
        for variable in variables:
            filename = os.path.join(outdir, f"{stub}_{variable}_{depth}.tif")
            url = slga_soils_abbrevations[variable]

            # The SLGA server is a bit temperamental, so sometimes you have to try again
            attempt = 0
            base_delay = 5  
            max_retries = 3
            
            while attempt < max_retries:
                try:
                    download_tif(bbox, url, identifier, filename)
                    print(f"Downloaded {filename}")
                    break
                except Exception as e:
                    print(f"Failed to download {variable} {depth}, attempt {attempt + 1} of {max_retries}", e)
                    attempt += 1
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) # Exponential backoff
                        print(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description="""Download variables from the Soils and Landscapes Grid of Australia (SLGA) at 90m resolution for the region of interest""")
    
    parser.add_argument('--variable', default="Clay", help=f"Default is 'Clay', and options are: {list(slga_soils_abbrevations.keys())}")
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--outdir', default='.', help='Directory for the output NetCDF file (default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--depth', default="5-15cm", help=f"Default is '5-15cm', and options are: {list(identifiers.keys())}")

    return parser.parse_args()
# -

if __name__ == '__main__':
    
    args = parse_arguments()
    
    variable = args.variable
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    outdir = args.outdir
    stub = args.stub
    depth = args.depth
    
    slga_soils([variable], lat, lon, buffer, outdir, stub, [depth])
