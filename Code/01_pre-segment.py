"""
Description:
This script downloads SENTINEL2 data (using DEA) as an xarray dataset, saves it as a pickle, and also prepares a 3-band Fourier Transform of the NDWI time series that can be used as input in SAMGeo for paddock segmentation. 

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
- A pickle dump of the xarray (ds2) containing sentinel data and MANY band incides, and metadata, for the ROI and TOI
- a tif image of the Fourier Transform reprojection of the time-stack into 3 axes. This is a 3-band image normalised between 1-255 to be compatible with SAM model.
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
python3 Code/01_pre-segment.py --stub test1 --outdir /g/data/xe2/John/Data/PadSeg/ --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31'""",
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
        **query
    )
    ds = calculate_indices(ds, 
                           index=['NDVI', 'kNDVI', 'EVI', 'LAI', 'SAVI', 'MSAVI', 'NDMI', 'NDWI', 'MNDWI', 'NBR', 'NDCI', 'NDTI', 'BSI'], 
                           collection='ga_s2_3')
    return ds
    
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
        
def main(args):
    client = create_local_dask_cluster(return_client=True)
    dc = datacube.Datacube(app='Vegetation_phenology')
    query = define_query(args.lat, args.lon, args.buffer, (args.start_time, args.end_time))
    ds = load_and_process_data(dc, query)
    client.close()
    f2 = transform(ds)
    im = rescale(f2)
    export_for_segmentation(ds, im, args.outdir+args.stub)
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
