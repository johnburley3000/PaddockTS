# +
# Idea of this notebook is assign productivity and shelter scores and plot them against each other

# +
# Standard Libraries
import pickle
import os

# Dependencies
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from dea_tools.plotting import rgb
from shapely.geometry import box, Polygon

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir, transform_bbox
# -



# filename = "/g/data/xe2/John/Data/PadSeg/MILGA_ds2.pkl"
filename = "/g/data/xe2/John/Data/PadSeg/MILG_b033_2023_ds2.pkl"
with open(filename, 'rb') as file:
    ds = pickle.load(file)

# Need to delete the grid mapping attribute for the to_raster function to work
if 'grid_mapping' in ds.attrs:
    del ds.attrs['grid_mapping']
image = ds.isel(time=0)[['nbart_red', 'nbart_green', 'nbart_blue']]
filename = os.path.join(gdata_dir, "sentinel_2020-01-03.tif")
image.rio.to_raster(filename)
print("Saved:", filename)

filename = os.path.join(gdata_dir, "MILG14km_canopy_height.tif")
canopy_height = rxr.open_rasterio(filename)

canopy_height

min_lat = ds.y.min().item()
max_lat = ds.y.max().item()
min_lon = ds.x.min().item()
max_lon = ds.x.max().item()
bbox = [min_lat, min_lon, max_lat, max_lon]
bbox_3857 = transform_bbox(bbox, inputEPSG="EPSG:6933", outputEPSG="EPSG:3857")
roi_coords_3857 = box(*bbox_3857)
roi_polygon_3857 = Polygon(roi_coords_3857)
roi_polygon_3857

cropped_canopy_height = canopy_height.rio.clip([roi_polygon_3857])


filename = os.path.join(gdata_dir, "MILG14km_cropped_canopy_height.tif")
cropped_canopy_height.rio.to_raster(filename)
print("Saved:", filename)

canopy_height_reprojected = cropped_canopy_height.rio.reproject_match(ds, resampling=Resampling.max)

filename = os.path.join(gdata_dir, "MILG14km_canopy_height_max.tif")
canopy_height_reprojected.rio.to_raster(filename)
filename

canopy_height_band = canopy_height_reprojected.isel(band=0)

ds['canopy_height'] = canopy_height_band

ds
