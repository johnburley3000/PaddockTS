import sys
import os
import pickle
from pystac_client import Client
from odc.stac import load
from planetary_computer import sign_url
from shapely.geometry import box
import numpy as np
from scipy.ndimage import uniform_filter, variance
from skimage.filters import threshold_minimum
import xarray as xr

# Define a function to apply lee filtering on S1 image 
def lee_filter(da, size):
    """
    Apply lee filter of specified window size.
    Adapted from https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python

    """
    da_notime = da.squeeze()
    img = da_notime.values
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)

    # Convert numpy array back to xarray, flipping the Y axis
    output = xr.DataArray(img_output, dims=da_notime.dims, coords=da_notime.coords)
    
    return output


def main(stub, out_dir):
    """
    Main function to download Sentinel-1 data from the Microsoft Planetary Computer using parameters
    from a specified stub and output directory. The function loads query parameters from a pickle file,
    constructs a bounding box, searches for data, and saves the results to a new pickle file.

    Parameters:
    - stub (str): Identifier for the input pickle file (e.g., MILG_b033_2017-24).
    - out_dir (str): Output directory where data is saved.

    Outputs:
    - A pickle file containing the loaded data is saved in the specified output directory.
    """

    # Microsoft Planetary Computer STAC Catalog URL
    catalog = "https://planetarycomputer.microsoft.com/api/stac/v1"

    # Create a STAC Client
    client = Client.open(catalog)

    # Load query parameters from the pickle file
    with open(os.path.join(out_dir, f"{stub}_ds2_query.pkl"), 'rb') as handle:
        ds2_query = pickle.load(handle)

    # Set up date range for search
    date_range = ds2_query['time']
    date_range = f"{date_range[0]}/{date_range[1]}"
    print(f"Will request data from date range: {date_range}")

    # Extract bounding box coordinates and create a shapely geometry
    x_min, x_max = ds2_query['x']
    y_min, y_max = ds2_query['y']
    geom = box(x_min, y_min, x_max, y_max)

    # Print information about the bounding box
    print("Created a bounding box to download data based on the following lat lon:")
    print(ds2_query['x'], ds2_query['y'])

    # Search for data items in the Planetary Computer STAC catalog
    items = client.search(
        collections=["sentinel-1-rtc"],
        intersects=geom,
        datetime=date_range,
    ).item_collection()

    print(f"Found {len(items)} items")

    # Load the data using the ODC-STAC loader
    data = load(
        items,
        geopolygon=geom,
        measurements=["vv", "vh"],
        groupby="solar_day",
        patch_url=sign_url,
        chunks={"x": 2048, "y": 2048},
        output_crs="epsg:6933"
    ).compute()

    # Print the dimensions of the loaded data
    print(data.dims)

    # Preprocess the data
    # Ideally, we would use the local incidence angle to pre-process S1 data, but this is not available from MPC. 
    # Probably best to do some spatial smoothing (lee speckle filter is common), then maybe filter out anomolous values in the time series?. 
    # process the data (followoing: https://github.com/auspatious/coastal-applications-workshop/blob/main/notebooks/examples/03_Sentinel-1_WaterDetection.ipynb)
  
    # The lee filter above doesn't handle null values
    # We therefore set null values to 0 before applying the filter
    valid = np.isfinite(data)
    masked = data.where(valid, 0)

    # Create a new entry in dataset corresponding to filtered VV and VH data
    data["filtered_vv"] = masked.vv.groupby("time").map(lee_filter, size=5)
    data["filtered_vh"] = masked.vh.groupby("time").map(lee_filter, size=5)

    # Null pixels should remain null
    data['filtered_vv'] = data.filtered_vv.where(valid.vv)
    data['filtered_vh'] = data.filtered_vh.where(valid.vh)

    # convert digital number to dB (backscatter is provided as linear intensity, it is often useful to convert the backscatter to decible (dB) for analysis. Backscatter in dB unit has a more symmetric noise profile and less skewed value distribution for easier statistical evaluation.)
    data['filtered_vv'] = 10 * np.log10(data.filtered_vv)
    data['filtered_vh'] = 10 * np.log10(data.filtered_vh)

    # Compute a global reference mean for VV (you could do the same for VH separately)
    global_ref_vv = data['filtered_vv'].mean(dim=["x", "y"]).median(dim="time").values

    # Create new DataArrays for normalized data (copy to preserve original)
    vv_normalized = data['filtered_vv'].copy()
    vh_normalized = data['filtered_vh'].copy()

    # Loop over each time slice and adjust by the offset
    for t in data.time:
        # Select the current time slice
        vv_slice = data['filtered_vv'].sel(time=t)
        vh_slice = data['filtered_vh'].sel(time=t)
        
        # Compute the mean backscatter for the slice
        slice_mean_vv = vv_slice.mean()
        slice_mean_vh = vh_slice.mean()
        
        # Compute the offsets needed to match the global reference
        vv_offset = global_ref_vv - slice_mean_vv
        # For VH you could choose a separate global reference or use the same approach:
        # Here we use the VV global reference for demonstration.
        vh_offset = global_ref_vv - slice_mean_vh
        
        # Apply the offset correction
        vv_normalized.loc[dict(time=t)] = vv_slice + vv_offset
        vh_normalized.loc[dict(time=t)] = vh_slice + vh_offset

    # Add the normalized data back into the dataset
    data['vv_filt_normalized'] = vv_normalized
    data['vh_filt_normalized'] = vh_normalized


    # Save the data as a pickle file
    output_path = os.path.join(out_dir, f"{stub}_ds1.pkl")
    with open(output_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Get command-line arguments
    stub = sys.argv[1]
    out_dir = sys.argv[2]

    # Run the main function
    main(stub, out_dir)

