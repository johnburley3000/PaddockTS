# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/gh70/ANUClimate/v2-0/stable/day/catalog.html
# -

import requests
import os
import xarray as xr
import pandas as pd
import glob


# +
def anuclimate_singlemonth(var="rain", latitude=-34.3890427, longitude=148.469499, year="2021", month="01"):
    
    buffer = 0.0000000001    # This buffer is less than the grid size of 1km (0.01 degrees), so you get a single point
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    base_url = "https://dapds00.nci.org.au"
    url = f'{base_url}/thredds/ncss/gh70/ANUClimate/v2-0/stable/day/{var}/{year}/ANUClimate_v2-0_{var}_daily_{year}{month}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
    if month == "01":
        print(url)
    
    response = requests.get(url)
    filename = f'{var}_{year}_{month}.nc' 
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    ds = xr.open_dataset(filename)
    df = ds.to_dataframe().reset_index()
    df_indexed = df.set_index('time')
    df_dropped = df_indexed.drop(columns=['lat', 'lon', 'crs'])

    return df_dropped

# df = anuclimate_singlemonth()  # Took 2 secs
# df.head()


# -

def anuclimate_singleyear(var="rain", latitude=-34.3890427, longitude=148.469499, year="2021"):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    dfs = []
    for month in months:
        df_month = anuclimate_singlemonth(var, latitude, longitude, year, month)
        dfs.append(df_month)
    df_concat = pd.concat(dfs)
    return df_concat


def anuclimate_multiyear(var="rain", latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"]):
    dfs = []
    for year in years:
        df_year = anuclimate_singleyear(var, latitude, longitude, year)
        dfs.append(df_year)
    df_concat = pd.concat(dfs)
    return df_concat


def anuclimate_multivariable(variables=["rain", "tmin", "tmax", "srad"], latitude=-34.3890427, longitude=148.469499, years=["2020", "2021"], cleanup=True):
    dfs = []
    for variable in variables:
        df_variable = anuclimate_multiyear(variable, latitude, longitude, years)
        dfs.append(df_variable)
    df_concat = pd.concat(dfs, axis=1)
    
    if cleanup:
        nc_files = glob.glob(os.path.join(os.getcwd(), '*.nc'))
        for file in nc_files:
            os.remove(file)
            
    return df_concat


if __name__ == '__main__':
    df = anuclimate_multivariable(["rain", "tmin"])  # Took 2 mins (30 secs per variable per year)
    abbreviations = {
        "rain":"Precipitation",
        "tmin":"Minimum temperature",
        "tmax":"Maximum temperature",
        "srad":"SRAD"
    }
    df.rename(columns=abbreviations, inplace=True)
    df.rename_axis("date", inplace=True)
    df.to_csv("anuclimate_daily.csv")
    print(df.head())