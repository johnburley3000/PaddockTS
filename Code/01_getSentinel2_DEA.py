"""
Description:
This script just downloads SENTINEL2 data (using DEA) as an xarray dataset and saves it as a pickle. Only variables necessary for the 05_shelter.py are included.

Usage:
The script is designed to be executed from the command line, where the user can specify the stub name for file naming and the directory for output files, and other parameters:

Requirements:
- Runs using a python environment designed for DEA Sandbox use on NCI. 
module use /g/data/v10/public/modules/modulefiles
module load dea/20231204
- A fair amount of memory for downloading large regions of data. 

Inputs:
- stub name
- coordinates
- buffer (degrees)
- start/end date

Outputs:
- A pickle dump of the xarray (ds2) containing 4 bands of sentinel data (RGB & NIR), and metadata, for the ROI and TOI
- a pickle containg dict of the query used to generate ds2
"""
import argparse
import os
import sys
import logging
import pickle
import numpy as np
import xarray as xr
import rioxarray
import datacube
from dea_tools.temporal import xr_phenology, temporal_statistics
from dea_tools.datahandling import load_ard
from dea_tools.bandindices import calculate_indices
from dea_tools.plotting import display_map, rgb
from dea_tools.dask import create_local_dask_cluster
import hdstats

# Adjust logging configuration for the script
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Download and save Sentinel data, prepare input image for SAMGeo.
        
Example usage:
python3 Code/00_sentinel.py --stub test --outdir /g/data/xe2/cb8590/Data/shelter --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31'""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stub", type=str, required=True, help="Stub name for file naming")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the center of the area of interest")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the center of the area of interest")
    parser.add_argument("--buffer", type=float, required=True, help="Buffer in degrees to define the area around the center point")
    parser.add_argument("--start_time", type=str, required=True, help="Start time for the data query (YYYY-MM-DD)")
    parser.add_argument("--end_time", type=str, required=True, help="End time for the data query (YYYY-MM-DD)")
    return parser.parse_args()

def define_query(lat, lon, buffer, time_range):
    lat_range = (lat-buffer, lat+buffer)
    lon_range = (lon-buffer, lon+buffer)
    query = {
        'centre': (lat, lon),
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'output_crs': 'epsg:6933',
        'group_by': 'solar_day'
    }
    # note that centre is not recognized as query option in load_arc, but we want to output it as a record.
    return query

def load_and_process_data(dc, query):
    query = {k: v for k, v in query.items() if k != 'centre'} # this takes centre out of the query	
    ds = load_ard(
        dc=dc,
        products=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        cloud_mask='s2cloudless',
        min_gooddata=0.9,
        measurements=['nbart_blue', 'nbart_green', 'nbart_red', 
                      'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
                      'nbart_nir_1', 'nbart_nir_2',
                      'nbart_swir_2', 'nbart_swir_3'],
        **query
    )
    return ds
    

def main(args):
    client = create_local_dask_cluster(return_client=True)
    dc = datacube.Datacube(app='Shelter')
    query = define_query(args.lat, args.lon, args.buffer, (args.start_time, args.end_time))
    ds = load_and_process_data(dc, query)

    # save ds for later
    with open(os.path.join(args.outdir, args.stub + '_ds2.pkl'), 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Data saved successfully to {args.outdir}")

    # save query for record keeping
    with open(os.path.join(args.outdir, args.stub + '_ds2_query.pkl'), 'wb') as f:
        pickle.dump(query, f)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)