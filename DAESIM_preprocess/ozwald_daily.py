# +
# Catalog is here: https://thredds.nci.org.au/thredds/catalog/ub8/au/OzWALD/daily/meteo/catalog.html

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
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    base_url = "https://thredds.nci.org.au"
    prefix = ".daily" if var == "Pg" else ""
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/daily/meteo/{var}/OzWALD{prefix}.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 

    # Check the file exists before downloading it
    head_response = requests.head(url)
    if head_response.status_code == 200:
        response = requests.get(url)
        filename = os.path.join(tmp_dir, f"{stub}_{var}_{year}.nc")
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("Downloaded", filename)
        ds = xr.open_dataset(filename)
    else:
        return None

    return ds


def ozwald_daily_singleyear_gdata(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021"):
    
    prefix = ".daily" if var == "Pg" else ""
    filename = os.path.join(f"/g/data/ub8/au/OzWALD/daily/meteo/{var}/OzWALD{prefix}.{var}.{year}.nc")

    # OzWald doesn't have 2024 data in this folder yet.
    print(filename)
    if not os.path.exists(filename):
        return None
        
    ds = xr.open_dataset(filename)
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))
    
    # If the buffer was smaller than the pixel size, than just assign a single lat and lon coordinate
    if len(ds_region.lat) == 0:
        ds_region = ds_region.drop_dims('lat').expand_dims(lat=1).assign_coords(lat=[latitude])
    if len(ds_region.lon) == 0:
        ds_region = ds_region.drop_dims('lon').expand_dims(lon=1).assign_coords(lon=[longitude])
        
    return ds_region


def ozwald_daily_multiyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], stub="Test", tmp_dir=".", thredds=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = ozwald_daily_singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmp_dir)
        else:
            ds_year = ozwald_daily_singleyear_gdata(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def ozwald_daily(variables=["VPeff", "Uavg"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2021", outdir=".", stub="Test", tmp_dir=".", thredds=True):
    """Download daily variables from OzWald at varying resolutions for the region/time of interest

    Parameters
    ----------
        variables: See ozwald_daily_abbreviations at the top of this file for a complete list & resolutions
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved. The filename includes the first variable in the csv.
        stub: The name to be prepended to each file download.
        tmp_dir: The directory that the temporary NetCDFs get saved if downloading from Thredds. This does not get used if Thredds=False.
        thredds: A boolean flag to choose between using the public facing API (slower but works locally), or running directly on NCI (requires access to the ub8 project)
    
    Returns
    -------
        ds_concat: an xarray containing the requested variables in the region of interest for the time period specified
        A NetCDF file of this xarray gets downloaded to outdir/(stub)_ozwald_daily_(first_variable).nc'
    """

    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = ozwald_daily_multiyear(variable, lat, lon, buffer, years, stub, tmp_dir, thredds)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)

    # Appending the first variable to the filename, so you can download temperature, rainfall, and wind/humidity separately since they use different grids.
    filename = os.path.join(tmp_dir, f'{stub}_ozwald_daily_{variables[0]}.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat


# -

# %%time
if __name__ == '__main__':
    ds = ozwald_daily()
