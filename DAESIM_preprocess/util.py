# Standatd libraries
import os

# Dependencies
import numpy as np
import rasterio
import rioxarray as rxr
import matplotlib.pyplot as plt
from pyproj import Transformer

home_dir = os.path.expanduser('~')
username = os.path.basename(home_dir)
gdata_dir = os.path.join("/g/data/xe2", username)
scratch_dir = os.path.join('/scratch/xe2', username)
paddockTS_dir = os.path.join(home_dir, "Projects/PaddockTS")

def create_bbox(lat=-35.274603, lon=149.098498, buffer=0.005):
    """Generates a bbox in the order [West, North, East, South] that's required for most APIs"""
    # 0.01 degrees is about 1km in each direction, so 2kmx2km total
    # From my experimentation, the asris.csiro API allows a maximum bbox of about 40km (0.2 degrees in each direction)

    left, top, right, bottom = lon - buffer, lat - buffer, lon + buffer, lat + buffer 
    bbox = [left, top, right, bottom] 
    return bbox

def transform_bbox(bbox=[148.464499, -34.394042, 148.474499, -34.384042], inputEPSG="EPSG:4326", outputEPSG="EPSG:3857"):
    transformer = Transformer.from_crs(inputEPSG, outputEPSG)
    x1,y1 = transformer.transform(bbox[1], bbox[0])
    x2,y2 = transformer.transform(bbox[3], bbox[2])
    return (x1, y1, x2, y2)

def visualise_tif_rasterio(filename="output.tif", title=""):
    with rasterio.open(filename) as src:
        data = src.read(1)  
        
        # Flip the image to match the orientation in QGIS
        flipped_data = np.flip(data, axis=0)

        plt.figure(figsize=(8, 6))
        img = plt.imshow(flipped_data, cmap='viridis', extent=(
            src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        cbar = plt.colorbar(img, ax=plt.gca())
        plt.show()

def visualise_tif_rioxarray(filename="output.tif", title=""):
    ds = rxr.open_rasterio(filename)
    band = ds.sel(band=1)
    band.plot()
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_time_series(ds, variable="Ssoil", lat=-34.3890427, lon=148.46949):
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    ds_var = ds_point[variable]
    ds_var.plot.line()
    plt.title(f"Latitude: {lat}, Longitude: {lon}")
    plt.show()

def plot_time_point(ds, variable="Ssoil", timepoint='2020-03-13'):
    data = ds[variable].sel(time=timepoint, method='nearest')
    data.plot()
    plt.show()

if __name__ == '__main__':
    print("username:", username)
    print("home_dir:", home_dir)
    print("gdata_dir:", gdata_dir)
    print("scratch_dir:", scratch_dir)
