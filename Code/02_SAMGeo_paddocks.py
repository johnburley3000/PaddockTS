#SAMGeo_paddocks.py
# Apr 23 2024

"""
Paddock Segmentation Script

Description:
This script processes satellite imagery to segment paddocks using the SamGeo segmentation model. It takes a 3-band geotiff image, representing the Fourier Transform of multi-year time series imagery from Sentinel, as input and performs segmentation to identify potential paddock areas. The segmented areas are then converted to polygons, which are filtered based on specified minimum and maximum area thresholds, as well as a maximum perimeter-to-area ratio. The final filtered polygons are saved as a GeoPackage.

Usage:
The script is designed to be executed from the command line, where the user can specify the stub name for file naming and the base directory for input/output files, along with optional parameters for the filtering criteria.

Requirements:
- The script utilizes the 'samgeo' library for image segmentation, 'rasterio' for handling raster data, and 'geopandas' for geospatial data manipulation.
- Load the venv called geospatenv

Outputs:
- A GeoPackage containing the filtered polygons representing the paddocks, saved in the specified output directory.

Next:
- I'm not sure that it's properly using the GPU.. Check this (could be to do with PBS script)
"""

import argparse
import os
import sys
import logging
import rasterio as rio
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from samgeo import SamGeo, sam_model_registry

# Setting up logging
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="""Process satellite imagery to segment paddocks.
Example usage:
python3 Code/SAMGeo_paddocks.py test1 /g/data/xe2/John/Data/PadSeg/ --min_area_ha 10 --max_area_ha 1500 --max_perim_area_ratio 30""")
    parser.add_argument('stub', type=str, help='Stub name for file naming.')
    parser.add_argument('base_directory', type=str, help='Base directory for input/output files.')
    parser.add_argument('--min_area_ha', type=float, default=10, help='Minimum area in hectares for paddock to be kept.')
    parser.add_argument('--max_area_ha', type=float, default=1500, help='Maximum area in hectares for paddock to be kept.')
    parser.add_argument('--max_perim_area_ratio', type=float, default=30, help='Maximum perimeter to area ratio.')
    return parser.parse_args()

def setup_sam_model(base_directory, stub):
    checkpoint_path = os.path.join(base_directory, 'sam_vit_h_4b8939.pth')
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint file found: {checkpoint_path}")
    else:
        print(f"Checkpoint file not found at: {checkpoint_path}. It will be downloaded when SamGeo() is run.")
    return SamGeo(
        model_type='vit_h',
        checkpoint=checkpoint_path,
        sam_kwargs=None,
    )

def segment_image(image_path, output_mask_path, sam_model):
    sam_model.generate(
        image_path, output_mask_path, batch=True, foreground=True, erosion_kernel=(3, 3), mask_multiplier=255
    )
    return output_mask_path

def filter_polygons(vector, min_area_ha, max_area_ha, max_perim_area_ratio):
    pol = gpd.read_file(vector).drop(labels = 'value', axis = 1)
    pol['area_ha'] = pol.area/1000
    pol['log_area_ha'] = np.log10(pol['area_ha'])
    pol['perim-area'] = pol.length/pol['area_ha']
    pol_filt = pol[
        (pol['area_ha'] >= min_area_ha) &
        (pol['area_ha'] <= max_area_ha) &
        (pol['perim-area'] <= max_perim_area_ratio)
    ]
    return pol_filt

def main():
    args = parse_arguments()
    base_directory = args.base_directory
    stub = args.stub
    image_path = os.path.join(base_directory, stub + '.tif')
    output_mask_path = os.path.join(base_directory, stub + '_segment.tif')
    output_vector_path = os.path.join(base_directory, stub + '_segment.gpkg')

    sam_model = setup_sam_model(base_directory, stub)
    segment_image(image_path, output_mask_path, sam_model) # takes about 20min on CPU, use GPU for much faster.. 
    sam_model.tiff_to_gpkg(output_mask_path, output_vector_path, simplify_tolerance=None) # polygonise raster
    filtered_gdf = filter_polygons(output_vector_path, args.min_area_ha, args.max_area_ha, args.max_perim_area_ratio)
    filtered_gdf.to_file(os.path.join(base_directory, stub + '_filt.gpkg'), driver='GPKG') # can take 10min+ on CPU, not sure if GPU will help this...

if __name__ == '__main__':
    main()
