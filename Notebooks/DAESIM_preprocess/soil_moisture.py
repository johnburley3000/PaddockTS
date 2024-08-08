os.chdir('/home/147/cb8590/Projects/PaddockTS/Notebooks/DAESIM_preprocess')

import requests
import os
import xarray as xr
import pandas as pd

# +
# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 0.0000000001    # This buffer is less than the grid size of 500m, so you get a single point

north = lat + buffer 
south = lat - buffer 
west = lon - buffer
east = lon + buffer

var = "Ssoil"
time_start = "2023-01-01"
time_end = "2023-03-31"

url = f'https://thredds.nci.org.au/thredds/ncss/grid/ub8/au/OzWALD/8day/Ssoil/OzWALD.Ssoil.2023.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
url
# -

# %%time
response = requests.get(url)
with open('data/soil_moisture.nc', 'wb') as f:
    f.write(response.content)
ds = xr.open_dataset('data/soil_moisture.nc')
df = ds.to_dataframe().reset_index()
df


