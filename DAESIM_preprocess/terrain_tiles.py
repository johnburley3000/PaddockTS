# Terrain Tiles documentation is here: https://github.com/tilezen/joerd/blob/master/docs/data-sources.md

# +
# # NCI ARE Setup
# Modules: gdal/3.6.4  
# Environment base: /g/data/xe2/John/geospatenv

# Change directory so that 'DAESIM_preprocess/terrain_tiles.xml' is in the pythonpath
import os, sys
repo_name = "PaddockTS"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)

# Standard library
import subprocess
import argparse

# Dependencies
import numpy as np
import rasterio
from scipy.interpolate import griddata
from pyproj import Transformer
import xarray as xr
import rioxarray as rxr
from affine import Affine

# +
def transform_bbox(bbox=[148.464499, -34.394042, 148.474499, -34.384042], inputEPSG="EPSG:4326", outputEPSG="EPSG:3857"):
    transformer = Transformer.from_crs(inputEPSG, outputEPSG)
    x1,y1 = transformer.transform(bbox[1], bbox[0])
    x2,y2 = transformer.transform(bbox[3], bbox[2])
    return (x1, y1, x2, y2)

def generate_xml(tile_level=14, filename="terrain_tiles.xml"):
    xml_string = f"""<GDAL_WMS>
  <Service name="TMS">
    <ServerUrl>https://s3.amazonaws.com/elevation-tiles-prod/geotiff/${{z}}/${{x}}/${{y}}.tif</ServerUrl>
  </Service>
  <DataWindow>
    <UpperLeftX>-20037508.34</UpperLeftX>
    <UpperLeftY>20037508.34</UpperLeftY>
    <LowerRightX>20037508.34</LowerRightX>
    <LowerRightY>-20037508.34</LowerRightY>
    <TileLevel>{tile_level}</TileLevel>
    <TileCountX>1</TileCountX>
    <TileCountY>1</TileCountY>
    <YOrigin>top</YOrigin>
  </DataWindow>
  <Projection>EPSG:3857</Projection>
  <BlockSizeX>512</BlockSizeX>
  <BlockSizeY>512</BlockSizeY>
  <BandsCount>1</BandsCount>
  <DataType>Int16</DataType>
  <ZeroBlockHttpCodes>403,404</ZeroBlockHttpCodes>
  <DataValues>
    <NoData>-32768</NoData>
  </DataValues>
  <Cache/>
</GDAL_WMS>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(xml_string)


def run_gdalwarp(bbox=[148.464499, -34.394042, 148.474499, -34.3840426], filename="output.tif", tile_level=14, debug=False):
    """Use gdalwarp to download a tif from terrain tiles"""

    if os.path.exists(filename):
        os.remove(filename)
    
    xml_path=os.path.join("DAESIM_preprocess/terrain_tiles.xml")
    generate_xml(tile_level, xml_path)      # tile_level=14 means 10m tiles. 
        
    bbox_3857 = transform_bbox(bbox)
    min_x, min_y, max_x, max_y = bbox_3857
    command = [
        "gdalwarp",
        "-of", "GTiff",
        "-te", str(min_x), str(min_y), str(max_x), str(max_y),
        xml_path, filename
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if debug:
        print("Terrain Tiles STDOUT:", result.stdout, flush=True)  # Debugging if something isn't working
        print("Terrain Tiles STDERR:", result.stderr, flush=True)
    print(f"Downloaded {filename}")

def interpolate_nan(filename="output.tif"):
    """Fix bad measurements in terrain tiles dem"""

    # Load the tiff into a numpy array rasterio
    with rasterio.open(filename) as dataset:
        dem = dataset.read(1) 
        meta = dataset.meta.copy()
    
    # There are some clearly bad measurements in terrain tiles and this attempts to assign them np.nan.
    threshold = 10
    heights = sorted(set(dem.flatten()))
    if len(heights) <= 1:
        return dem, meta
    lowest_correct_height = min(heights)
    for i in range(len(heights)//2 - 1, -1, -1):
        if heights[i + 1] - heights[i] > threshold:
            lowest_correct_height = heights[i + 1] 
            break
    Z = np.where(dem < lowest_correct_height, np.nan, dem)
    
    # Extract into lists for interpolating
    x_coords, y_coords = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = Z.flatten()
    
    # Remove NaN values before interpolating
    mask = ~np.isnan(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]
    xy_coords = np.vstack((x_flat, y_flat), dtype=float).T
    
    # Replace bad/nan/missing values with the nearest neighbour
    X, Y = np.meshgrid(np.linspace(0, Z.shape[1] - 1, Z.shape[1]),
                np.linspace(0, Z.shape[0] - 1, Z.shape[0]))
    nearest = griddata(xy_coords, z_flat, (X, Y), method='nearest')

    return nearest, meta

def download_dem(dem, meta, filename="terrain_tiles.tif"):
    meta.update({
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,  # Number of bands
        "dtype": dem.dtype
    })
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(dem, 1)
    print(f"Saved {filename}")

def create_xarray(dem, meta):
    """Convert the cleaned dem into an xarray, without re-writing and re-reading to file"""
    transform = meta['transform']
    height = meta['height']
    width = meta['width']
    crs = meta['crs']

    x_coords = transform.c + np.arange(width) * transform.a
    y_coords = transform.f + np.arange(height) * transform.e 

    dem_da = xr.DataArray(
        dem,
        dims=("y", "x"),
        coords={"x": x_coords, "y": y_coords},
        attrs={
            "crs": crs.to_string(),
            "transform": transform,  
            "nodata": meta["nodata"]
        },
        name="terrain"
    )
    dem_ds = dem_da.to_dataset()
    return dem_ds
    
def terrain_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=".", stub="TEST", tmpdir=".", tile_level=14, interpolate=True):
    """Download 10m resolution elevation from terrain_tiles
    
    Parameters
    ----------
        lat, lon: Coordinates in WGS 84 (EPSG:4326).
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        outdir: The directory to save the final cleaned tiff file.
        stub: The name to be prepended to each file download.
        tmpdir: The directory to save the raw uncleaned tiff file.
        tile_level: The zoom level to determine the pixel size in the resulting tif. See documentation link at the top of this file for more info. 
        interpolate: Boolean flag to decide whether to try to fix bad values or not. 
    
    Downloads
    ---------
        A Tiff file of elevation with severe outlier pixels replaced by the nearest neighbour

    """
    print(f"Starting terrain_tiles.py")
    buffer = max(0.00002, buffer) # Make sure we download at least 1 pixel
    # Perhaps we should also force a maximum buffer, but then this would have to be dependent on the tile_level.
    
    # Download the raw data from terrain tiles
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    filename = os.path.join(tmpdir, f"{stub}_terrain_original.tif")
    run_gdalwarp(bbox, filename, tile_level)

    if interpolate:
        # Fix bad measurements
        dem, meta = interpolate_nan(filename)        
        filename = os.path.join(outdir, f"{stub}_terrain.tif")
        download_dem(dem, meta, filename)
        ds = create_xarray(dem, meta)
    else:
        # We could use rxr.open_rasterio() but the purpose of the interpolate flag is to reduce computational overhead, so I think it's better not to reload the tif here.  
        ds = None
        
    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description="""Download elevation data from the MapZen Terrain Tiles API.""")
    
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--outdir', default='.', help='The directory to save the final cleaned tiff file.')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--tmpdir', default='.', help='The directory to save the raw uncleaned tiff file.')
    parser.add_argument('--tile_level', default='14', help='The zoom level described by terrain tiles documentation (default is 14, which means 10m pixels): https://github.com/tilezen/joerd/blob/master/docs/data-sources.md')
    parser.add_argument('--just_download', default=False, action="store_true", help='Boolean flag to skip the nearest neighbour interpolation for bad values.')

    return parser.parse_args()
# -

if __name__ == '__main__':
    
    args = parse_arguments()
    
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    outdir = args.outdir
    stub = args.stub
    tmpdir = args.tmpdir
    tile_level = args.tile_level
    interpolate = not args.just_download
    
    terrain_tiles(lat, lon, buffer, outdir, stub, tmpdir, tile_level, interpolate)
