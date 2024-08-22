# Comparing pickle vs netcdf
# Conclusions were inconsistent. Both lazy load, so it's hard to evaluate. I think .nc is more common though.

import os
import datacube
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import time

# Check memory usage
import resource
def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return f"Memory usage: {usage.ru_maxrss / 1024} MB"
print(memory_usage())

# Comparing scratch and gdata. Scratch is about 3x faster to read and write from.
stubs = ['MILGA2_ds2', 'fm_ndwi_5_ds2', 'ADAMO_ds2']
for stub in stubs:
    scratch_netcdf = f'/scratch/xe2/cb8590/tmp/{stub}.nc'
    gdata_netcdf = f'/g/data/xe2/Chris/Data/PadSeg/{stub}.nc'

    start = time.time()
    print(f"Loading from scratch: {stub}.nc")
    ds_netcdf = xr.open_dataset(scratch_netcdf)
    ds_netcdf.load()
    print(memory_usage())
    print("Time: ", time.time()-start, "seconds")
    print(ds_netcdf['nbart_blue'].shape)
    print()
    
    start = time.time()
    print(f"Loading from g/data: {stub}.nc")
    ds_netcdf = xr.open_dataset(gdata_netcdf)
    ds_netcdf.load()
    print(memory_usage())
    print("Time: ", time.time()-start, "seconds")
    print(ds_netcdf['nbart_blue'].shape)
    print()

# Filenames
stub = 'ADAMO_ds2'
john_padseg = '/g/data/xe2/John/Data/PadSeg/'
original_filepath = john_padseg + stub + '.pkl'
chris_temp = '/scratch/xe2/cb8590/tmp/'
outpath_pickle = f'/scratch/xe2/cb8590/tmp/{stub}.pkl'
outpath_netcdf = f'/scratch/xe2/cb8590/tmp/{stub}.nc'


%%time
# Create two new files from existing for consistent testing 
# Comment out this code after running it once. Then kernel restart and run run all
with open(original_filepath, 'rb') as handle:
    ds = pickle.load(handle)
with open(outpath_pickle, 'wb') as handle:
    pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Sometimes need to remove these attributes for the netcdf save to work
ds['time'].attrs.pop('units', None)
if 'flags_definition' in ds.attrs:
    ds.attrs.pop('flags_definition')
for var in ds.variables:
    if 'flags_definition' in ds[var].attrs:
        ds[var].attrs.pop('flags_definition')
ds.to_netcdf(outpath_netcdf)

# Checking starting memory usage
print(memory_usage())

%%time
# NetCDF load
ds_netcdf = xr.open_dataset(outpath_netcdf)
print(memory_usage())

%%time
# Pickle load
with open(outpath_pickle, 'rb') as handle:
    ds_pickle = pickle.load(handle)
print(memory_usage())

# Need to remove an existing .nc file before you can make a new one
if os.path.exists(outpath_netcdf):
    os.remove(outpath_netcdf)

%%time
# NetCDF save
ds_netcdf.to_netcdf(outpath_netcdf)
print(memory_usage())

%%time
# Pickle save
with open(outpath_pickle, 'wb') as handle:
    pickle.dump(ds_netcdf, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(memory_usage())

# Filesizes are about the same
!du -sh {outpath_pickle}
!du -sh {outpath_netcdf}