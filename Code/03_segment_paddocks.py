# 03_segment_paddocks.py
# Feb 19 2025

"""
Paddock Segmentation Script

Description:
This script processes satellite imagery to segment paddocks using the SamGeo segmentation model. It first generates a 3-band geotiff image, representing the Fourier Transform of multi-year time series imagery from Sentinel, as input and performs segmentation to identify potential paddock areas. The segmented areas are then converted to polygons, which are filtered based on specified minimum and maximum area thresholds, as well as a maximum perimeter-to-area ratio. The final filtered polygons are saved as a GeoPackage.

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
import hdstats
import xarray as xr
import rioxarray
import pickle

# Setting up logging
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="""Process satellite imagery to segment paddocks.
Example usage:
python3 Code/02_segment_paddocks.py test1 /g/data/xe2/John/Data/PadSeg/ --min_area_ha 10 --max_area_ha 1500 --max_perim_area_ratio 30""")
    parser.add_argument('stub', type=str, help='Stub name for file naming.')
    parser.add_argument('base_directory', type=str, help='Base directory for input/output files.')
    parser.add_argument('--min_area_ha', type=float, default=10, help='Minimum area in hectares for paddock to be kept.')
    parser.add_argument('--max_area_ha', type=float, default=1500, help='Maximum area in hectares for paddock to be kept.')
    parser.add_argument('--max_perim_area_ratio', type=float, default=30, help='Maximum perimeter to area ratio.')
    parser.add_argument('--model', type=str, help='full path to the pre-trained SAMgeo model file. eg. sam_vit_h_4b8939.pth')
    return parser.parse_args()

def transform(ds):
	keep_vars = ['nbart_red','nbart_green','nbart_blue','nbart_nir_1']
	data = ds[keep_vars].to_array().transpose('y', 'x','variable', 'time').values.astype(np.float32)
	data[data == 0] = np.nan
	data /= 10000.
	ndwi_obs = (data[:,:,1,:]-data[:,:,3,:])/(data[:,:,1,:]+data[:,:,3,:]) # w = water. (g-nir)/(g+nir)
	ndwi = hdstats.completion(ndwi_obs)
	f2 = hdstats.fourier_mean(ndwi)
	return f2
	
def rescale(im):
    '''rescale raster (im) to between 0 and 255.
    Attempts to rescale each band separately, then join them back together to achieve exact same shape as input.
    Note. Assumes multiple bands, otherwise breaks'''
    n_bands = im.shape[2]
    _im = np.empty(im.shape)
    for n in range(0,n_bands):
        matrix = im[:,:,n]
        scaled_matrix = (255*(matrix - np.min(matrix))/np.ptp(matrix)).astype(int)
        _im[:,:,n] = scaled_matrix
    print('output shape equals input:', im.shape == im.shape)
    return(_im)

def export_for_segmentation(ds, inp, out_stub):
    '''prepares a 3-band image for SAMgeo. 
    First rescale bands in the image. Then convert to xarray with original geo info. Then save geotif'''
    if inp.shape[2] == 3:
        image = rescale(inp) # 3d array 
        lat = list(ds.y.values) # latitude is the same size as the first axis
        lon = list(ds.x.values) # longitude is the same size as second axis
        bands = list(range(1,image.shape[2]+1)) # band is the 3rd axis
        crs = ds.rio.crs
        # create xarray object
        data_xr = xr.DataArray(image, 
                       coords={'y': lat,'x': lon,'band': bands}, 
                       dims=["y", "x", "band"])
        data_xr.rio.write_crs(crs, inplace=True)
        # save as geotif:
        data_xr.transpose('band', 'y', 'x').rio.to_raster(out_stub + '.tif')
    else:
        print("Input image is wrong shape! No action taken")

def setup_sam_model(checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint file found: {checkpoint_path}")
    else:
        print(f"Checkpoint file not found at: {checkpoint_path}. It should have been downloaded during setup. Now SAMGeo might try to download it, but this might not work...")
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

    with open(base_directory+stub+'_ds2.pkl', 'rb') as handle:
        ds = pickle.load(handle)

    f2 = transform(ds)
    im = rescale(f2)
    export_for_segmentation(ds, im, base_directory+stub)

    sam_model = setup_sam_model(args.model)
    segment_image(image_path, output_mask_path, sam_model) # takes about 20min on CPU, use GPU for much faster.. 
    sam_model.tiff_to_gpkg(output_mask_path, output_vector_path, simplify_tolerance=None) # polygonise raster
    filtered_gdf = filter_polygons(output_vector_path, args.min_area_ha, args.max_area_ha, args.max_perim_area_ratio)
    filtered_gdf.to_file(os.path.join(base_directory, stub + '_filt.gpkg'), driver='GPKG') # can take 10min+ on CPU, not sure if GPU will help this...

if __name__ == '__main__':
    main()
