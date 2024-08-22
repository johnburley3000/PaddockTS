# +
# Planet CLI refernce is here: https://planet-sdk-for-python.readthedocs.io/en/latest/cli/reference.html
# Planet band info is here: https://developers.planet.com/docs/apis/data/sensors/

# +
# Assumes tiff files have already been downloaded using John's bash script

# +
# Standard library
import os
import sys
import json
import pickle
from datetime import datetime

# Dependencies
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr


# -


sys.path.insert(1, '../Tools/')
from dea_tools.plotting import rgb

# +
orderid = ""
outpath = ""
stub = "MILG_6km"

base_directory = "/g/data/xe2/datasets/Planet/Farms/MILG"
# -

order_ids = "5534c295-c405-4e10-8d51-92c7c7c02427", "570f6f33-9471-4dcb-a553-2c97928c53aa"
order_id = order_ids[0]
order_id


# +
def find_timestamps(directory_order_id):
    """Find all the timestamps in a planetscope order folder"""
    timestamps = set()
    for s in os.listdir(directory_order_id):
        timestamp = s[:23]
        timestamps.add(timestamp)
    if 'PSScene_collection.json' in timestamps:
        timestamps.remove('PSScene_collection.json')
    timestamps = sorted(timestamps)
    return timestamps

directory_order_id = os.path.join(base_directory, order_id, "PSScene")
timestamps = find_timestamps(directory_order_id)
timestamps[:5]


# +
def load_image(directory_order_id, timestamp, bands_tiff_prefix="_3B_Visual_clip"):
    """Load a single planetscope image into an xarray"""
    filename = os.path.join(directory_order_id, f"{timestamp}{bands_tiff_prefix}.tif")
    da = rxr.open_rasterio(filename)
    
    # In the 3band tiff, there is actually 4 bands, with the 4th being a cloud mask. Here we apply the cloud mask to the other bands and then drop it.
    if bands_tiff_prefix == "_3B_Visual_clip":
        cloud_mask = da.sel(band=4)
        da_masked = da.where(cloud_mask != 0, other=np.nan)
        da_3band = da_masked.sel(band=slice(1, 3))

    # Extract the bands into their own variables to match sentinel
    ds = da_3band.to_dataset(dim='band')
    ds_named = ds.rename({1: 'nbart_red', 2: 'nbart_blue', 3: 'nbart_green'})


    return ds_named

ds = load_image(directory_order_id, timestamps[0])
ds
# -

ds1 = load_image(directory_order_id, timestamps[0])
ds2 = load_image(directory_order_id, timestamps[1])
ds2

time_value = pd.to_datetime(ds.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
ds = ds.expand_dims(time=[time_value])
ds

rgb(ds1)

rgb(ds2)

time1 = pd.to_datetime(ds1.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
time2 = pd.to_datetime(ds1.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
time1

    # Add the time dimension
time = pd.to_datetime(da.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
da_timed = da.expand_dims({'time': [time]})


# +
def load_images(directory_order_id, timestamps, bands3_tiff_prefix="_3B_Visual_clip"):
    """Load all the in a single planetscope order folder into a single xarray"""
    
    dss = []
    for timestamp in timestamps:
        ds = load_image(directory_order_id, timestamp, bands3_tiff_prefix)
        time = pd.to_datetime(ds.attrs['TIFFTAG_DATETIME'], format='%Y:%m:%d %H:%M:%S')
        ds_timed = ds.expand_dims(time=[time])
        dss.append(ds_timed)
        
    combined_ds = xr.concat(dss, dim='time')
    return combined_ds

ds = load_images(directory_order_id, timestamps[:4])
ds
# -
rgb(ds, col="time", col_wrap=4)


directory_order_id

# +







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
    transposed_images = image_array.transpose(1,0,2,3)
    y, x = create_lat_lon(bboxs[0], (transposed_images.shape[3], transposed_images.shape[2]))
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

if __name__ == '__main__':
    args = parse_arguments()

    base_directory = args.indir
    order_id = args.orderid
    outpath = args.outpath

    print(f"{datetime.now()} Starting planet_xarray_3b in {base_directory} for {order_id}")

    xarray = load_single_order_3band(base_directory, order_id)

    with open(outpath, 'wb') as handle:
        pickle.dump(xarray, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{datetime.now()} Finished planet_xarray_3b and exported to {outpath}")

