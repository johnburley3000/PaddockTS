# +
# Download data for lots of lat lon coordinates and add to an existing tsv file
# -

# Change directory to this repo - this should work on gadi or locally via python or jupyter.
# Unfortunately, this needs to be in all files that can be run directly & use local imports.
import os, sys
repo_name = "PaddockTS"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)

import pandas as pd
import xarray as xr
import argparse
import pickle


from DAESIM_preprocess.ozwald_daily import ozwald_daily
from DAESIM_preprocess.ozwald_8day import ozwald_8day
from DAESIM_preprocess.silo_daily import silo_daily


def dataset_to_series(ds):
    return pd.Series(ds.to_array().values.flatten(), index=ds.time.values)


def multipoints(df, func, variable="Tmin", start_year="2020", end_year="2021", outdir=".", stub="Test", tmp_dir="."):
    """Extract data for each lat lon in the dataframe. 
    Assumes the dataframe has at least columns 'X' and 'Y' corresponding to lon and lat"""
    
    dss = []
    sample_info = [] 
    for i, row in df.iterrows():
        lon, lat = row['X'], row['Y']
        sample_name = row['sample_name']
        ds = func([variable], lat, lon, 0, start_year, end_year, outdir, stub, tmp_dir, thredds=False, save_netcdf=False)
        print(f'time: {len(ds.time)}, lat: {len(ds.lat)}, lon: {len(ds.lon)}, for {sample_name}, {lat}, {lon}')
        dss.append(ds)
        sample_info.append(row.to_dict())
    
    # Save the dss as a pickle for debugging
    filename = f'/scratch/xe2/cb8590/{variable}_{start_year}_{end_year}_{stub}_dss.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(dss, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved', filename)
    
    # Create DataFrame with sample info and time series data
    das = [dataset_to_series(ds) for ds in dss]
    result_data = []
    for i, (info, da) in enumerate(zip(sample_info, das)):
        
        # Add date columns to the original columns
        row_data = info.copy()
        for date, value in da.items():
            date_string = str(date)[:10]
            row_data[date_string] = value
            
        result_data.append(row_data)
    
    result_df = pd.DataFrame(result_data)
    return result_df


funcs = {
    'ozwald_daily':ozwald_daily,
    'silo_daily':silo_daily,
    'ozwald_8day':ozwald_8day
}


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description='Extract climate data for multiple points')
    
    parser.add_argument('--func', default='ozwald_daily', help='Function to use for data extraction (default: ozwald_daily)')
    parser.add_argument('--variable', default='Tmin', help='Climate variable to extract (default: Tmin)')
    parser.add_argument('--start_year', default='1880', help='Start year for data extraction (default: 1880)')
    parser.add_argument('--end_year', default='2030', help='End year for data extraction (default: 2030)')
    parser.add_argument('--stub', default='EUC', help='Stub name for output files (default: EUC)')
    parser.add_argument('--outdir', default='/scratch/xe2/cb8590/Eucalypts', help='Output directory (default: /scratch/xe2/cb8590/Eucalypts)')
    parser.add_argument('--tmpdir', default='/scratch/xe2/cb8590/Eucalypts', help='Output directory (default: /scratch/xe2/cb8590/Eucalypts)')
    parser.add_argument('--filename_latlon', default='/g/data/xe2/cb8590/Eucalypts/all_euc_sample_metadata_20250525_geo_bioclim.tsv', help='Path to input file with lat/lon coordinates')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process (default: None)')
    

    return parser.parse_args()

if __name__ == '__main__':
    
    # Parse command line arguments
    args = parse_arguments()
    
    func = args.func
    variable = args.variable
    start_year = args.start_year
    end_year = args.end_year
    stub = args.stub
    outdir = args.outdir
    tmpdir = args.tmpdir
    filename_latlon = args.filename_latlon
    max_samples = args.max_samples
    
    # Read and process data
    print(f"Reading data from: {filename_latlon}")
    df = pd.read_csv(filename_latlon, sep='\t')
    
    if max_samples:
        df = df[:max_samples]
    print(f"Processing {len(df)} samples")
    
    # Extract climate data
    print(f"Extracting {variable} data from {start_year} to {end_year}")
    df = multipoints(df, funcs[func], variable, start_year, end_year, outdir, stub, tmpdir)
    
    # Save results
    filename_out = os.path.join(outdir, f"{stub}_{func}_{variable}_{start_year}_{end_year}.tsv")
    df.to_csv(filename_out, index=False, sep='\t')
    
    print(f"Saved: {filename_out}")
    print(f"Output contains {len(df)} rows and {len(df.columns)} columns")
