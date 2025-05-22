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

from DAESIM_preprocess.ozwald_daily import ozwald_daily
from DAESIM_preprocess.ozwald_8day import ozwald_8day
from DAESIM_preprocess.silo_daily import silo_daily


def dataset_to_series(ds):
    return pd.Series(ds.to_array().values.flatten(), index=ds.time.values)


# +
# def multipoints(df, func, variables=["VPeff", "Uavg"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2021", outdir=".", stub="Test", tmp_dir=".", thredds=True):
#     """Extract data for each lat lon in the dataframe. 
#     Assumes the dataframe has at least columns 'X' and 'Y' corresponding to lon and lat"""

#     dss = []
#     sample_info = [] 
#     for i, row in df[:2].iterrows():
#         lon, lat = row['X'], row['Y']
#         sample_name = row['sample_name']
#         ds = ozwald_daily([variable], lat, lon, 0, start_year, end_year, outdir, "EUC", outdir, False)
#         dss.append(ds)
#         sample_info.append({'sample_name': sample_name, 'lat': lat, 'lon': lon})

#     # Create the time series data
#     das = [dataset_to_series(ds) for ds in dss]

#     # Create DataFrame with sample info and time series data
#     result_data = []
#     for i, (info, da) in enumerate(zip(sample_info, das)):
#         row_data = {
#             'sample_name': info['sample_name'],
#             'lat': info['lat'],
#             'lon': info['lon']
#         }
#         # Add all the date columns
#         for date, value in da.items():
#             date_string = str(date)[:10]
#             row_data[date_string] = value
#         result_data.append(row_data)

#     df = pd.DataFrame(result_data)
#     return df
# -

def multipoints(df, func, variable="Tmin", start_year="2020", end_year="2021", outdir=".", stub="Test", tmp_dir="."):
    """Extract data for each lat lon in the dataframe. 
    Assumes the dataframe has at least columns 'X' and 'Y' corresponding to lon and lat"""
    
    dss = []
    sample_info = [] 
    for i, row in df.iterrows():
        lon, lat = row['X'], row['Y']
        sample_name = row['sample_name']
        ds = func([variable], lat, lon, 0, start_year, end_year, outdir, stub, outdir, thredds=False, save_netcdf=False)
        dss.append(ds)
        sample_info.append(row.to_dict())
    
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


# %%time
if __name__ == '__main__':
    filename = "/g/data/xe2/cb8590/Eucalypts/all_euc_sample_metadata_20250525_geo_bioclim.tsv"
    outdir = '/scratch/xe2/cb8590/Eucalypts'
    start_year = "2020"
    end_year = "2030"
    variable = "Tmin"
    stub = "EUC"

    df_original = pd.read_csv(filename, sep='\t')
    df = df_original[['sample_name', 'X', 'Y']]
    
    df = df[:4]
    
    df = multipoints(df, ozwald_daily, variable, start_year, end_year, outdir, stub, outdir)

    filename_out = f"{stub}_{variable}_{start_year}_{end_year}.tsv"
    df.to_csv(filename_out, index=False, sep='\t')
    print("Saved", filename_out)

df


