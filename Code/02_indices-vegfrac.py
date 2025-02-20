# 02_indices_vegfrac.py
# Feb 19 2025

'''Description:
This script will load time series Sentinel-2 data for a region/timeframe of interest.
It will calculate a bunch of default indices for the automated analysis of paddock time series data. 
It will also calculate the vegetation fractional cover using Scarth et al method. 
There will also be an option to turn off the veg frac cover calculation. This will save timem, but the results won't be compatible with later scripts. 
There will also be an option to specify additional indices to calculate. 

Inputs:

Outputs:
- A pickle dump of the xarray (ds) containing the indices and fractional cover, as well as the original data. This will replace the <stub>_ds.pkl file.
'''

import numpy as np
import pickle
#import xarray as xr
import rioxarray  # activate the rio accessor
import rasterio
import os
import argparse
import logging

from indices_etc import *

# Setting up logging
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Download and save Sentinel data, prepare input image for SAMGeo.
        
Example usage:
python3 Code/02_indices-vegfrac.py --stub TEST5 --outdir /g/data/xe2/John/Data/PadSeg --calc_veg_frac --indices_file indices.txt""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stub", type=str, required=True, help="Stub name for file naming")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    parser.add_argument("--calc_veg_frac", action='store_true', default=True, help="Flag to calculate vegetation fractional cover (default: yes)")
    parser.add_argument("--indices_file", type=str, help="Path to a txt file containing line-separated index names to calculate")
    return parser.parse_args()


def main():
    args = parse_arguments()
    stub = args.stub
    outdir = args.outdir

    # load the data
    with open(f'{outdir}/{stub}_ds2.pkl', 'rb') as f:
        ds = pickle.load(f)

    if args.calc_veg_frac:
        ## Add veg fractions to ds
        band_names = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3']
        i = 4  # or whichever model index you want to use
        fractions = calculate_fractional_cover(ds, band_names, i, correction=False)
        ds = add_fractional_cover_to_ds(ds, fractions)
    
    if args.indices_file:
        with open(args.indices_file, 'r') as f:
            indices = {line.strip(): globals()[f'calculate_{line.strip().lower()}'] for line in f if line.strip() in globals()}
    else:
        indices = {
            'NDVI': calculate_ndvi,
            'CFI': calculate_cfi,
            'NIRv': calculate_nirv
        }

    # Check if the elements of 'indices' are defined as functions in indices_etc
    for index_name, func in indices.items():
        if not callable(func):
            raise ValueError(f"The function for index '{index_name}' is not defined in indices_etc or is not callable.")
        
    ds = calculate_indices(ds, indices)

    #  save ds
    out_name = os.path.join(outdir, stub + '_ds2.pkl')
    with open(out_name, 'wb') as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Data saved successfully to {out_name}")

if __name__ == "__main__":
    main()

