# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/daily/meteo/catalog.html

# +
# Standard Libraries
import os
import glob

# Dependencies
import requests
import xarray as xr
import pandas as pd
# -

ozwald_daily_abbreviations = {
    "Pg" : "Gross precipitation",
    "Tmax" : "Maximum temperature",
    "Tmin" : "Minimum temperature",
    "Uavg" : "Average 24h windspeed",
    "Ueff" : "Effective daytime windspeed",
    "VPeff" : "Volume of effective rainfall",
    "kTavg" : "Coefficient to calculate mean screen level temperature",
    "kTeff" : "Coefficient to calculate effective screen level temperature"
}


def ozwald_daily_singleyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021", stub="", tmp_dir=""):
    
    # buffer = 0.0000000001    # Using a buffer less than the grid size of 5km (0.05 degrees), so you get a single point
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    # base_url = "https://thredds.nci.org.au"  # This is the new url (dapds00 is supposedly deprecated), but LAI only works with the old url
    base_url = "https://dapds00.nci.org.au"
    prefix = ".daily" if var == "Pg" else ""
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/daily/meteo/{var}/OzWALD{prefix}.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 
    
    response = requests.get(url)
    filename = os.path.join(tmp_dir, f"{stub}_{var}_{year}.nc")
    with open(filename, 'wb') as f:
        f.write(response.content)
    print("Downloaded", filename)
        
    ds = xr.open_dataset(filename)
    
    return ds


ozwald_daily_singleyear(buffer=0.00001)


def ozwald_daily_multiyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], stub="", tmp_dir=""):
    dss = []
    for year in years:
        ds_year = ozwald_daily_singleyear(var, latitude, longitude, buffer, year, stub, tmp_dir)
        dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def ozwald_daily(variables=["VPeff", "Uavg"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year=2020, end_year=2021, outdir="", stub="", tmp_dir="", cleanup=True):
    dss = []
    years = [str(year) for year in list(range(start_year, end_year + 1))]
    for variable in variables:
        ds_variable = ozwald_daily_multiyear(variable, lat, lon, buffer, years, stub, tmp_dir)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)
    
    if cleanup:
        nc_files = glob.glob(os.path.join(os.getcwd(), '*.nc'))
        for file in nc_files:
            os.remove(file)

    filename = os.path.join(outdir, f'{stub}_ozwald_daily.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat


# %%time
if __name__ == '__main__':
    ds = ozwald_daily()  # Took 2 seconds (0.5 seconds per variable per year)
    print(ds)

    os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
    from DAESIM_preprocess.util import plot_time_series, plot_time_point
    
    plot_time_series(ds, "VPeff",  lat=-34.3890427, lon=148.469499)
    plot_time_point(ds, "VPeff", "2020-03-11")
