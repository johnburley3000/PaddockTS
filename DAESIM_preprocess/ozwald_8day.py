# +
# Catalog is here: https://thredds.nci.org.au/thredds/catalog/ub8/au/OzWALD/8day/catalog.html

# Standard Libraries
import os

# Dependencies
import xarray as xr

# +
ozwald_8day_abbreviations = {
    "Alb": "Albedo",
    "BS": "Bare Surface",
    "EVI": "Enhanced Vegetation Index",
    "FMC": "Fuel Moisture Content",
    "GPP": "Gross Primary Productivity",
    "LAI": "Leaf Area Index",
    "NDVI": "Normalised Difference Vegetation Index",
    "NPV": "Non Photosynthetic Vegetation",
    "OW": "Open Water",
    "PV": "Photosynthetic Vegetation",
    "Qtot": "Streamflow",
    "SN": "Snow",
    "Ssoil": "Soil profile moisture change"
}


# This function uses the public facing Thredds API, so does not need to be run on NCI
# However it doesn't work in a PBS script from my tests
def ozwald_8day_singleyear_thredds(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2021", stub="Test", tmp_dir="."):

    buffer = max(0.003, buffer)  # If you specify an area that's too small then no data gets returned from thredds
    
    url = f"https://thredds.nci.org.au/thredds/dodsC/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc"

    try:
        ds = xr.open_dataset(url, engine="netcdf4")
    except Exception as e:
        # Likely no data for the specified year
        return None

    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))

    # If the buffer was smaller than the pixel size, than just assign a single lat and lon coordinate
    if len(ds_region.latitude) == 0:
        ds_region = ds_region.drop_dims('latitude').expand_dims(latitude=1).assign_coords(latitude=[latitude])
    if len(ds_region.longitude) == 0:
        ds_region = ds_region.drop_dims('longitude').expand_dims(longitude=1).assign_coords(longitude=[longitude])
        
    return ds_region


# This function accesses files directly, so is much faster but needs to be run on NCI with access to the ub8 project
def ozwald_8day_singleyear_gdata(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021"):
    """Select the region of interest from the Australia wide NetCDF file"""
    filename = os.path.join("/g/data/ub8/au/OzWALD/8day", var, f"OzWALD.{var}.{year}.nc")

    # OzWald doesn't have 2024 data in this folder yet.
    if not os.path.exists(filename):
        return None
    
    ds = xr.open_dataset(filename)
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))

    # If the buffer was smaller than the pixel size, than just assign a single lat and lon coordinate
    if len(ds_region.lat) == 0:
        ds_region = ds_region.drop_dims('latitude').expand_dims(latitude=1).assign_coords(latitude=[latitude])
    if len(ds_region.lon) == 0:
        ds_region = ds_region.drop_dims('longitude').expand_dims(longitude=1).assign_coords(longitude=[longitude])
        
    return ds_region


def ozwald_8day_multiyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, years=["2020", "2021"], stub="Test", tmp_dir=".", thredds=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = ozwald_8day_singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmp_dir)
        else:
            ds_year = ozwald_8day_singleyear_gdata(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def ozwald_8day(variables=["Ssoil", "GPP"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir=".", stub="Test", tmp_dir=".", thredds=True):
    """Download 8day variables from OzWald at 500m resolution for the region/time of interest

    Parameters
    ----------
        variables: See ozwald_8day_abbreviations at the top of this file for a complete list
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved.
        stub: The name to be prepended to each file download.
        tmp_dir: The directory that the temporary NetCDFs get saved if downloading from Thredds. This does not get used if Thredds=False.
        thredds: A boolean flag to choose between using the public facing API (slower but works locally), or running directly on NCI (requires access to the ub8 project)
    
    Returns
    -------
        ds_concat: an xarray containing the requested variables in the region of interest for the time period specified
        A NetCDF file of this xarray gets downloaded to outdir/(stub)_ozwald_8day.nc'
    """
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = ozwald_8day_multiyear(variable, lat, lon, buffer, years, stub, tmp_dir, thredds=thredds)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)
    
    filename = os.path.join(outdir, f'{stub}_ozwald_8day.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat


# -

# %%time
if __name__ == '__main__':  
    ds = ozwald_8day()
