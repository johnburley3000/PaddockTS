import argparse
import logging
import pickle
import xarray as xr
import geopandas as gpd
from dea_tools.plotting import rgb, xr_animation
import plotting_functions as pf
import os

# Setting up logging
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="""Generate a series of plots to check out the data and segmentation results.
Example usage:
python3 Code/checkpoint_plots.py TEST5 /g/data/xe2/John/Data/PadSeg/ """)
    parser.add_argument('stub', type=str, help='Stub name for file naming.')
    parser.add_argument('base_directory', type=str, help='Base directory for input/output files.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    stub = args.stub
    out_dir = args.base_directory

    ## Open the satellite data stack
    with open(out_dir+stub+'_ds2i.pkl', 'rb') as handle:
        ds = pickle.load(handle)

    # read in the polygons and plot:
    pol = gpd.read_file(out_dir+stub+'_filt.gpkg')
    pol['paddock'] = range(1,len(pol)+1)
    pol['paddock'] = pol.paddock.astype('category')
    pol['color'] = 'None' # needs to be set in the gpd to achieve no colour polygon fill. 

    # Load the Fourier Transform image
    raster_path = out_dir+stub+'.tif'

    # load the silo data
    silo = xr.open_dataset(out_dir+stub+'_silo_daily.nc')

    # load the oswald data
    ozwald = xr.open_dataset(out_dir+stub+'_ozwald_8day.nc')
    #print(ozwald)

    # Average Ssoil over latitude and longitude
    Ssoil = ozwald['Ssoil'].mean(dim=['latitude', 'longitude'])
    #print(Ssoil)

    # Run the plotting functions:
    #pf.plot_indices_timeseries(ds, out_dir, stub)
    pf.plot_paddock_map_auto_rgb(ds, pol, out_dir, stub)
    pf.plot_paddock_map_auto_fourier(raster_path, pol, out_dir, stub)
    pf.plot_silo_daily(silo, ds, out_dir, stub)
    pf.plot_env_ts(silo, ds, Ssoil, out_dir, stub)

    # Save the RGB image as a TIFF file
    output_name_rgb = os.path.join(out_dir, f'{stub}_thumbs_rgb.tif')
    rgb(ds, 
        bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
        col="time", 
        col_wrap=len(ds.time),
        savefig_path=output_name_rgb)
    
    # Save the veg fraction image as a TIFF file
    output_name_vegfrac = os.path.join(out_dir, f'{stub}_thumbs_vegfrac.tif')
    rgb(ds, 
        bands=['bg', 'pv', 'npv'],
        col="time", 
        col_wrap=len(ds.time),
        savefig_path=output_name_vegfrac)

    # Save the time lapses of RGB and veg fract with paddocks overlaid
    output_path = out_dir+stub+'_manpad_RGB.mp4'
    xr_animation(ds, 
                bands = ['nbart_red', 'nbart_green', 'nbart_blue'], 
                output_path = output_path, 
                show_gdf = pol, 
                gdf_kwargs={"edgecolor": 'white'})

    output_path = out_dir+stub+'_manpad_vegfrac.mp4'
    xr_animation(ds, 
                bands = ['bg', 'pv', 'npv'], 
                output_path = output_path, 
                show_gdf = pol, 
                gdf_kwargs={"edgecolor": 'white'})

if __name__ == '__main__':
    main()