import os
import json
import pickle
import numpy as np
import xarray as xr
import rasterio
from datetime import datetime

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

def load_images(directory, timestamps, tiff_prefix="_3B_Visual_clip"):
    """Load all images without any preprocessing
    
    Returns
    -------
    images:list of 3d-array
        list of 3 dimensional arrays (band, x, y)
        There are 4 bands (red, green, blue, alpha)
    """
    images = []
    for timestamp in timestamps:
        filename = directory + timestamp + tiff_prefix + ".tif"
        src = rasterio.open(filename)
        image = src.read()
        images.append(image)
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

def find_bbox(directory, timestamp, tiff_prefix='_3B_Visual_clip'):
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

def load_single_order_3band(base_directory, order_id):
    """Loads all the images from a 3band planetscope folder into an xarray"""
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
    x, y = create_lat_lon(bboxs[0], (image_array.shape[1], image_array.shape[2]))
    transposed_images = image_array.transpose(3,0,1,2)
    ds_planetscope = xr.Dataset(
        {
            "nbart_red":(["time", "y", "x"], transposed_images[0]),
            "nbart_green":(["time", "y", "x"], transposed_images[1]),
            "nbart_blue":(["time", "y", "x"], transposed_images[2]),
        }, coords={
            "time": datetimestamps,
            "y": ("y", y),
            "x": ("x", x),
        },
    )
    return ds_planetscope

def normalized(band):
    return band / np.sqrt(np.sum(band**2))

def merge_two_xarrays(ds1, ds2):
    """ 
    Combines two xarrays
    Assumes both datasets have the same bands each with dimensions (time, y, x)
    Sorts by timestamp for compatibility with dea_tools.plotting.xr_animation

    Parameters
    ----------
    ds1: xarray.Dataset
    ds2: xarray.Dataset

    Returns
    ----------
    ds_merged: xarray.Dataset
    """
    bands = list(ds1.keys())
    x = ds1.x.values
    y = ds1.y.values

    ds1_normalized = [normalized(ds1[band]) for band in bands]
    ds2_normalized = [normalized(ds2[band]) for band in bands]

    ds1_bands = [[ds1["time"][i]] + [band[i] for band in ds1_normalized] for i in range(len(ds1["time"]))]
    ds2_bands = [[ds2["time"][i]] + [band[i] for band in ds2_normalized] for i in range(len(ds2["time"]))]

    ds_merged_bands = ds1_bands + ds2_bands
    ds_merged_bands.sort(key=lambda x:x[0])
    
    time_merged = [t[0].values for t in ds_merged_bands]
    band_dict = {bands[i]: (["time",  "y", "x"], [t[i+1] for t in ds_merged_bands]) for i in range(len(bands))}

    ds_merged = xr.Dataset(
        band_dict,
        coords={
            "time": time_merged,
            "y": ("y", y),
            "x": ("x", x),
        },
    )
    return ds_merged

def merge_all_xarrays(xarrays):
    """Combines all the xarrays"""
    fully_merged = xarrays[0]
    for xarray in xarrays[1:]:
        fully_merged = merge_two_xarrays(fully_merged, xarray)
    return fully_merged

if __name__ == '__main__':
    print("hello world")

    base_directory = "../SPRV"
    order_ids = ["5fb7ea5a-05ec-4784-a2c1-e4a59540f914", "5fb7ea5a-05ec-4784-a2c1-e4a59540f914"]
    xarray_outpath = 'testing.pickle'

    xarrays = [load_single_order_3band(order_id) for order_id in order_ids]
    merged_xarray = merge_all_xarrays(xarrays)
    with open(xarray_outpath, 'wb') as handle:
        pickle.dump(merged_xarray, handle, protocol=pickle.HIGHEST_PROTOCOL)