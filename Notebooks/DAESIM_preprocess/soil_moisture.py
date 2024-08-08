import requests
import os
import xarray as xr
import pandas as pd

# # TODO
# Remove the os.chdir
# Add a cleanup option to remove leftover files
# Add an "abbreviations" dictionary

os.chdir('/home/147/cb8590/Projects/PaddockTS/Notebooks/DAESIM_preprocess')

# +
# %%time

# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 0.0000000001    # This buffer is less than the grid size of 500m (~0.005 degrees), so you get a single point

north = lat + buffer 
south = lat - buffer 
west = lon - buffer
east = lon + buffer

var = "Ssoil"
time_start = "2023-01-01"
time_end = "2023-03-31"

url = f'https://thredds.nci.org.au/thredds/ncss/grid/ub8/au/OzWALD/8day/Ssoil/OzWALD.Ssoil.2023.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 

response = requests.get(url)
with open('data/soil_moisture.nc', 'wb') as f:
    f.write(response.content)
ds = xr.open_dataset('data/soil_moisture.nc')
df = ds.to_dataframe().reset_index()
df


# -

def ozwald_singleyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, year="2021"):
    
    buffer = 0.0000000001    # This buffer is less than the grid size of 500m (0.005 degrees), so you get a single point
    
    north = lat + buffer 
    south = lat - buffer 
    west = lon - buffer
    east = lon + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-03-31"
    
    url = f'https://thredds.nci.org.au/thredds/ncss/grid/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 

    response = requests.get(url)
    filename = f'{var}_{year}.nc' 
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    ds = xr.open_dataset(filename)
    df = ds.to_dataframe().reset_index()
    df_indexed = df.set_index('time')
    df_dropped = df_indexed.drop(columns=['latitude', 'longitude'])
    
    # os.remove(filename)

    return df_dropped


def ozwald_multiyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"]):
    dfs = []
    for year in years:
        df_year = ozwald_singleyear(var, latitude, longitude, year)
        dfs.append(df_year)
    df_concat = pd.concat(dfs)
    return df_concat


def ozwald_multivariable(variables=["Ssoil", "Qtot"], latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"]):
    dfs = []
    for variable in variables:
        df_variable = ozwald_multiyear(variable, latitude, longitude, years)
        dfs.append(df_variable)
    df_concat = pd.concat(dfs, axis=1)
    return df_concat


# %%time
df = ozwald_multivariable()
df


