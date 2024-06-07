'''
Something in this script causes imagery to be offset along the X-axis. See example outputs June 7 2024.
'''

import os
import json
import pickle
import numpy as np
import xarray as xr
import rasterio
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Load 8 band Planetscope data into an xarray named {stub}_xarray.pkl
        
Example usage:
python3 Code/___.py --indir /g/data/xe2/datasets/Planet/Farms/MULL/ --orderid 79f404e3-6b72-43fa-ac13-1b33d0afa755 --outpath /g/data/xe2/chris/Data/MULL_xarray_8b.pkl
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--indir", type=str, required=True, help="Input directory containing planet orders")
    parser.add_argument("--orderid", type=str, required=True, help="the orderid of the folder")
    parser.add_argument("--outpath", type=str, required=True, help="Output path including filename for the xarray .pkl")
    return parser.parse_args()

def find_timestamps(directory):
    """
    Returns
    -------
    timestamps: list of str
        A sorted list of strings taken as the first 23 characters in each filename (except PSScene_collection.json)
    """
    timestamps = set()
    for s in os.listdir(directory):
        timestamp = s[:23]
        timestamps.add(timestamp)
    if 'PSScene_collection.json' in timestamps:
        timestamps.remove('PSScene_collection.json')
    timestamps = sorted(timestamps)
    return timestamps

def load_images(directory, timestamps, tiff_prefix='_3B_AnalyticMS_SR_8b_clip' , udm_suffix="_3B_udm2_clip.tif"):
    """Load images and mask with udm-2 using rasterio

    Info on PlanetScope bands: https://developers.planet.com/docs/apis/data/sensors/
    Info on udm2 bands: https://developers.planet.com/docs/data/udm-2/
    
    Returns
    -------
    images: list of 3d-array
        list of masked 8 dimensional arrays (band, x, y)
    """
    images = []
    for timestamp in timestamps:
        filename = directory + timestamp + tiff_prefix + ".tif"
        with rasterio.open(filename) as src:
            image = src.read()  # Reads all bands; shape (band_count, height, width)

        filemask = directory + timestamp + udm_suffix
        with rasterio.open(filemask) as src_mask:
            udm2 = src_mask.read(1)  # Read the first band for the mask "Regions of a scene that are free of cloud, haze, cloud shadow and/or snow"

        clear_pixels = udm2.astype('float')
        clear_pixels[clear_pixels == 0] = np.nan
        masked_image = image * clear_pixels

        images.append(masked_image)
    return images


def create_datetimes(timestamps):
    """
    Parameters
    ----------
    timestamps: list of str

    Returns
    -------
    datetimestamps: list of DateTime

    """
    datetimestamps = []
    for timestamp in timestamps:
        year = timestamp[0:4]
        month = timestamp[4:6]
        day = timestamp[6:8]
        hour = timestamp[9:11]
        minute = timestamp[11:13]
        second = timestamp[13:15]
        datetimestamp = datetime(int(year), int(month), int(day), hour=int(hour), minute=int(minute), second=int(second))
        datetimestamps.append(datetimestamp)
    return datetimestamps

def find_bbox(directory, timestamp, tiff_prefix='_3B_AnalyticMS_SR_8b_clip'):
    """ Finds the bounding box from the metadata file"""
    filename = directory + timestamp + ".json"
    file = open(filename)
    metadata = json.loads(file.read())
    bbox = metadata['assets'][timestamp + tiff_prefix + "_tif"]['proj:bbox']  # epsg32755 (Local projection)
    # bbox = metadata['bbox']  # epsg6933 (Global projection)
    return bbox

def create_lat_lon(bbox, shape):
    """ Create the latitudes and longitudes

    Parameters
    ----------
    bbox: list of float
    shape: list of int
    
    Returns
    -------
    x: 1d-array
    y: 1d-array
    """
    pixel_size = (bbox[2] - bbox[0])/shape[0], (bbox[3] - bbox[1])/shape[1]
    y = np.arange(bbox[0], bbox[2], pixel_size[0])
    x = np.arange(bbox[1], bbox[3], pixel_size[1])
    return x, y

def load_single_order_8band(base_directory, order_id):
    """Loads all the images from a 8band planetscope folder into an xarray"""
    directory = base_directory + "/" + order_id + "/PSScene/"
    timestamps = find_timestamps(directory)
    print(f"Number of timestamps in order {order_id}: {len(timestamps)}")
    bboxs = [find_bbox(directory, timestamp) for timestamp in timestamps]
    shapes = [(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxs]
    good_timestamps = [timestamp for timestamp, shape in zip(timestamps, shapes) if shape == max(shapes)]
    print(f"Number of timestamps in order {order_id} with full bounding box: {len(good_timestamps)}")
    images = load_images(directory, good_timestamps)
    image_array = np.array(images)
    datetimestamps = create_datetimes(good_timestamps)
    transposed_images = image_array.transpose(1,0,2,3)
    y, x = create_lat_lon(bboxs[0], (transposed_images.shape[3], transposed_images.shape[2]))
    ds_planetscope = xr.Dataset(
        {
        "Coastal Blue":(["time", "y", "x"], transposed_images[0]),
        "nbart_blue":(["time", "y", "x"], transposed_images[1]),
        "Green I":(["time", "y", "x"], transposed_images[2]),
        "nbart_green":(["time", "y", "x"], transposed_images[3]),
        "Yellow":(["time", "y", "x"], transposed_images[4]),
        "nbart_red":(["time", "y", "x"], transposed_images[5]),
        "Red Edge":(["time", "y", "x"], transposed_images[6]),
        "nbart_nir_1":(["time", "y", "x"], transposed_images[7]),
        }, coords={
            "time": datetimestamps,
            "y": ("y", y),
            "x": ("x", x),
        },
    )
    return ds_planetscope

if __name__ == '__main__':
    args = parse_arguments()

    base_directory = args.indir
    order_id = args.orderid
    outpath = args.outpath

    print(f"{datetime.now()} Starting planet_xarray_8b in {base_directory} for {order_id}")

    xarray = load_single_order_8band(base_directory, order_id)

    with open(outpath, 'wb') as handle:
        pickle.dump(xarray, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{datetime.now()} Finished planet_xarray_8b and exported to {outpath}")
