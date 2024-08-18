# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/8day/catalog.html
# -

import requests
import os
import xarray as xr
import pandas as pd
import glob
import matplotlib.pyplot as plt

ozwald_abbreviations = {
    "Ssoil":"Soil moisture",
    "Qtot":"Runoff",
    "LAI":"Vegetation leaf area",
    "GPP":"Vegetation growth"
}


def ozwald_8day_singleyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2021", stub="test", tmp_dir=""):
    
    # buffer = 0.0000000001    # Using a buffer less than the grid size of 500m (0.005 degrees) gives you a single point
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    # base_url = "https://thredds.nci.org.au"  # This is the new url (dapds00 is supposedly deprecated), but LAI only works with the old url
    base_url = "https://dapds00.nci.org.au"
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
    # print(url)
    
    response = requests.get(url)
    filename = os.path.join(tmp_dir, f"{stub}_{var}_{year}.nc")
    with open(filename, 'wb') as f:
        f.write(response.content)
        
    print("Downloaded", filename)

    ds = xr.open_dataset(filename)
    return ds


# +
def ozwald_8day_multiyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, years=["2020", "2021"], stub="test", tmp_dir=""):
    dss = []
    for year in years:
        ds_year = ozwald_8day_singleyear(var, latitude, longitude, buffer, year, stub, tmp_dir)
        dss.append(df_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat

ds_years = ozwald_8day_multiyear()


# +
def ozwald_8day(variables=["Ssoil", "GPP"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year=2020, end_year=2021, outdir="", stub="test", tmp_dir="", cleanup=True):
    """Download 8day variables from OzWald"""
    dss = []
    years = [str(year) for year in list(range(start_year, end_year + 1))]
    for variable in variables:
        ds_variable = ozwald_8day_multiyear(variable, lat, lon, buffer, years, stub, tmp_dir)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)
    
    if cleanup:
        nc_files = glob.glob(os.path.join(tmp_dir, '*.nc'))
        for file in nc_files:
            os.remove(file)

    filename = os.path.join(outdir, f'{stub}_ozwald_8day.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat

ds_vars = ozwald_8day()
# -

# %%time
if __name__ == '__main__':

    # ds = ozwald_8day()
    
    filename = 'test_ozwald_8day.nc'
    
    ds = xr.open_dataset()
    ds_point = ds.sel(latitude=-34.3890427, longitude=148.469499, method='nearest')
    df = ds_point.to_dataframe().reset_index()
    df_dropped = df.drop(columns=['latitude', 'longitude'])
    df_indexed = df_dropped.set_index('time')
    
    # Will want to do some aggregation when visualising the daily variables
    # aggregation = {
    #     "Rainfall": 'sum',
    #     "Soil moisture": 'mean',
    #     "Radiation": 'mean'
    # }
    # weekly_df = df_indexed.resample('W').agg(aggregation)
    # weekly_df = weekly_df.interpolate('linear')
    
    weekly_df = df_indexed
    plt.figure(figsize=(50, 10))
    
    # Plot the data
    temp_plot = plt.plot(weekly_df.index, weekly_df['GPP'], color='orange')
    moisture_plot = plt.plot(weekly_df.index, weekly_df['Ssoil']/10, color='blue')
    
    # Adjust the size of the tick labels on the x-axis and y-axis
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    
    # Reorder the legend items
    handles = [temp_plot[0], moisture_plot[0]]
    labels = ['GPP', 'Soil Moisture (mm/10)']
    plt.legend(handles=handles, labels=labels, fontsize=30, loc='upper left')
    plt.title(f'OzWald 8day', fontsize=30)
    
    plt.tight_layout()
    plt.savefig(f'OzWald_8day.png')



