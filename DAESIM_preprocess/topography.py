# +
#  Pysheds documentation is here: https://mattbartos.com/pysheds/
import os
import argparse

# Dependencies
import numpy as np
from pysheds.grid import Grid
import rasterio
from rasterio.enums import Resampling
import xarray as xr
import rioxarray as rxr
from scipy.ndimage import gaussian_filter

topographic_variables = ['accumulation', 'aspect', 'slope', 'twi']

# +
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
def pysheds_accumulation(terrain_tif):
    """Read in the grid and dem and calculate the water flow direction and accumulation"""
    
    # Load both the dem (basically a numpy array), and the grid (all the metadata like the extent)
    grid = Grid.from_raster(terrain_tif)
    dem = grid.read_raster(terrain_tif)

    # Hydrologically enforce the DEM so water can flow downhill to the edge and not get stuck
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Calculate the aspect (fdir) and accumulation of water (acc)
    fdir = grid.flowdir(inflated_dem)
    acc = grid.accumulation(fdir)

    return grid, dem, fdir, acc


def calculate_slope(terrain_tif):
    """Calculate the slope of a DEM"""
    with rasterio.open(terrain_tif) as src:
        dem = src.read(1)  
        transform = src.transform 
    gradient_y, gradient_x = np.gradient(dem, transform[4], transform[0])
    slope = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2)) * (180 / np.pi)
    return slope


def calculate_accumulation(terrain_tif):
    """Calculate the upstream area of each pixel using pysheds"""
    _, _, _, acc = pysheds_accumulation(terrain_tif)
    return acc


def calculate_TWI(acc, slope):
    """Calculate the topographic wetness index based on upstream area and local slope
        TWI = ln( accumulation / tan(slope) )
    """
    slope_radians = np.radians(slope)
    slope_radians[slope_radians == 0] = np.finfo(float).eps     # Avoid division by 0
    twi = np.log(acc / np.tan(slope_radians))
    return twi


def add_numpy_band(ds, variable, array, affine, resampling_method):
    """Add a new band to the xarray from a numpy array and affine using the given resampling method"""
    da = xr.DataArray(
        array, 
        dims=["y", "x"], 
        attrs={
            "transform": affine,
            "crs": "EPSG:3857"
        }
    )
    da.rio.write_crs("EPSG:3857", inplace=True)
    reprojected = da.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected
    return ds


def topography(outdir=".", stub="TEST", smooth=True, sigma=5):
    """Derive topographic variables from the elevation. 
    This function assumes there already exists a file named (outdir)/(stub)_terrain.tif"
    
    Parameters
    ----------
        outdir: The directory to save the topographic variables.
        stub: The name to be prepended to each file download.
        smooth: boolean to determine whether to apply a gaussian filter to the elevation before deriving topographic variables. 
                This is necessary when using terrain tiles to remove artifacts from the elevation being stored as ints.
        sigma: smoothing parameter to use for the gaussian filter. Not used if smooth=False. 
    
    Downloads
    ---------
        Tiff files of aspect, slope, accumulation and TWI.

    """
    print(f"Starting topography.py")

    terrain_tif = os.path.join(outdir, f"{stub}_terrain.tif")
    if not os.path.exists(terrain_tif):
        raise Exception("{terrain_tif} does not exist. Please run terrain_tiles.py first.")

    ds = rxr.open_rasterio(terrain_tif)

    if smooth:
        print("Smoothing the terrain using a gaussian filter")
        terrain_tif = os.path.join(outdir, f"{stub}_terrain_smoothed.tif")
        sigma = int(sigma)
        dem = ds.isel(band=0).drop_vars('band').values
        dem_smooth = gaussian_filter(dem.astype(float), sigma=sigma)
        ds = add_numpy_band(ds, "dem_smooth", dem_smooth, ds.rio.transform(), Resampling.average)
        ds["dem_smooth"].rio.to_raster(terrain_tif)

    print("Calculating accumulation")
    grid, dem, fdir, acc = pysheds_accumulation(terrain_tif)
    aspect = fdir.astype('uint8')

    print("Calculating slope and TWI")
    slope = calculate_slope(terrain_tif)
    twi = calculate_TWI(acc, slope)

    print("Saving the tif files")
    ds = add_numpy_band(ds, "accumulation", acc, grid.affine, Resampling.max)
    ds = add_numpy_band(ds, "aspect", aspect, grid.affine, Resampling.nearest)
    ds = add_numpy_band(ds, "slope", slope, grid.affine, Resampling.average)
    ds = add_numpy_band(ds, "twi", twi, grid.affine, Resampling.max)
    for topographic_variable in topographic_variables:
        filepath = os.path.join(outdir, f"{stub}_{topographic_variable}.tif")
        ds[topographic_variable].rio.to_raster(filepath)
        print("Saved:", filepath)


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description=f"""Derive topographic variables from the elevation ({topographic_variables}). 
                                     Note: this function assumes there already exists a file named (outdir)/(stub)_terrain.tif""")
    
    parser.add_argument('--outdir', default='.', help='The directory to save the topographic variables. (default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--smooth', default=False, action="store_true", help='boolean to determine whether to apply a gaussian filter to the elevation before deriving topographic variables. (default: False)')
    parser.add_argument('--sigma', default='5', help='Smoothing parameter to use for the gaussian filter. Not used if smooth=False (default: 5)')
    
    return parser.parse_args()
# -

if __name__ == '__main__':
    
    args = parse_arguments()
    
    outdir = args.outdir
    stub = args.stub
    smooth = args.smooth
    sigma = args.sigma

    topography(outdir, stub, smooth, sigma)

# +
# # Change directory to the PaddockTS repo
# import os, sys
# if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
#     paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/PaddockTS")
# elif os.path.basename(os.getcwd()) != "PaddockTS":
#     paddockTS_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
# else:  # Already running locally from PaddockTS root
#     paddockTS_dir = os.getcwd()
# os.chdir(paddockTS_dir)
# sys.path.append(paddockTS_dir)

# outdir = "."
# stub="TEST"
# terrain_tif = os.path.join(outdir, f"{stub}_terrain.tif")
# ds = rxr.open_rasterio(terrain_tif)

# -


