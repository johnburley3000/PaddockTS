# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/daily/meteo/catalog.html
# -

import requests
import os
import xarray as xr
import pandas as pd
import glob


def ozwald_daily_singleyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, year="2021"):
    
    buffer = 0.0000000001    # This buffer is less than the grid size of 500m (0.005 degrees), so you get a single point
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    # base_url = "https://thredds.nci.org.au"  # This is the new url (dapds00 is supposedly deprecated), but LAI only works with the old url
    base_url = "https://dapds00.nci.org.au"
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/daily/meteo/{var}/OzWALD.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
    print(url)
    
    response = requests.get(url)
    filename = f'{var}_{year}.nc' 
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    ds = xr.open_dataset(filename)
    df = ds.to_dataframe().reset_index()
    df_indexed = df.set_index('time')
    df_dropped = df_indexed.drop(columns=['latitude', 'longitude'])

    return df_dropped


def ozwald_daily_multiyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"]):
    dfs = []
    for year in years:
        df_year = ozwald_daily_singleyear(var, latitude, longitude, year)
        dfs.append(df_year)
    df_concat = pd.concat(dfs)
    return df_concat


def ozwald_daily_multivariable(variables=["VPeff", "Uavg"], latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"], cleanup=True):
    dfs = []
    for variable in variables:
        df_variable = ozwald_daily_multiyear(variable, latitude, longitude, years)
        dfs.append(df_variable)
    df_concat = pd.concat(dfs, axis=1)
    
    if cleanup:
        nc_files = glob.glob(os.path.join(os.getcwd(), '*.nc'))
        for file in nc_files:
            os.remove(file)
            
    return df_concat


# %%time
if __name__ == '__main__':
    df = ozwald_daily_multivariable()  # Took 2 seconds (0.5 seconds per variable per year)
    abbreviations = {
        "Ssoil":"Soil moisture",
        "Qtot":"Runoff",
        "LAI":"Vegetation leaf area",
        "GPP":"Vegetation growth"
    }
    df.rename(columns=abbreviations, inplace=True)
    df.rename_axis("date", inplace=True)
    df.to_csv("ozwald_daily.csv")
    print(df.head())


