# +
# Catalog is here: https://dapds00.nci.org.au/thredds/catalog/ub8/au/OzWALD/daily/meteo/catalog.html

# +
# Standard Libraries
import os

# Dependencies
import requests
import xarray as xr

# +
ozwald_daily_abbreviations = {
    "Pg" : "Gross precipitation",  # 4km grid
    "Tmax" : "Maximum temperature",  # 250m grid
    "Tmin" : "Minimum temperature",  # 250m grid
    "Uavg" : "Average 24h windspeed",  # 5km grid
    "Ueff" : "Effective daytime windspeed",  # 5km grid
    "VPeff" : "Volume of effective rainfall",  # 5km grid
    "kTavg" : "Coefficient to calculate mean screen level temperature",  # 5km grid
    "kTeff" : "Coefficient to calculate effective screen level temperature"  # 5km grid
}


def ozwald_daily_singleyear_thredds(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021", stub="Test", tmp_dir="scratch_dir"):

    buffer = max(0.1, buffer)  # Minimum buffer for OzWald rainfall is 0.1 (~1km)
    
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


def ozwald_daily_singleyear_gdata(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021"):
    
    buffer = max(0.1, buffer)  # Minimum buffer for OzWald rainfall is 0.1 (~1km)

    prefix = ".daily" if var == "Pg" else ""
    filename = os.path.join(f"/g/data/ub8/au/OzWALD/daily/meteo/{var}/OzWALD{prefix}.{var}.{year}.nc")

    # OzWald doesn't have 2024 data in this folder yet.
    if not os.path.exists(filename):
        return None
        
    ds = xr.open_dataset(filename)
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))
    
    # If the region is too small, then just find a single point
    if ds_region[var].shape[1] == 0:
        ds_region = ds.sel(latitude=latitude, longitude=longitude, method="nearest")
    
    return ds_region


def ozwald_daily_multiyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], stub="Test", tmp_dir=".", thredds=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = ozwald_daily_singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmp_dir)
        else:
            ds_year = ozwald_daily_singleyear_gdata(var, latitude, longitude, buffer, year)
        dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def ozwald_daily(variables=["VPeff", "Uavg"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2021", outdir=".", stub="Test", tmp_dir=".", thredds=True):
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = ozwald_daily_multiyear(variable, lat, lon, buffer, years, stub, tmp_dir, thredds)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)

    # Appending the first variable to the filename, so you can download temperature, rainfall, and wind/humidity separately since they use different grids.
    filename = os.path.join(outdir, f'{stub}_ozwald_daily_{variables[0]}.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat


# -

# %%time
if __name__ == '__main__':
    ozwald_daily()