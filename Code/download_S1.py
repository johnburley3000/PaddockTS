import sys
import os
import pickle
from pystac_client import Client
from odc.stac import load
from planetary_computer import sign_url
from shapely.geometry import box

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

