# +
# Catalog is here: https://thredds.nci.org.au/thredds/catalog/ub8/au/OzWALD/8day/catalog.html

# Standard Libraries
import os
import argparse
from contextlib import redirect_stderr, redirect_stdout

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

# Ignore messages like "Error:curl error: SSL connect error" when searching for a year that doesn't exist
def open_dataset_silently(url):
    try:
        with open(os.devnull, 'w') as devnull, \
             redirect_stdout(devnull), \
             redirect_stderr(devnull):
            ds = xr.open_dataset(url, engine="netcdf4")
        return ds
    except Exception:
        return None

# This function uses the public facing Thredds API, so does not need to be run on NCI
# However it doesn't work in a PBS script from my tests
def ozwald_8day_singleyear_thredds(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2021"):

    buffer = max(0.003, buffer)  # If you specify an area that's too small then no data gets returned from thredds
    
    url = f"https://thredds.nci.org.au/thredds/dodsC/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc"
    ds = open_dataset_silently(url)

    if not ds:
        return None

    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))

    if buffer < 0.03:
        # Find a single point but keep the lat and lon dimensions for consistency
        ds_region = ds.sel(latitude=[latitude], longitude=[longitude], method='nearest')
    
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

    if buffer < 0.03:
        # Find a single point but keep the lat and lon dimensions for consistency
        ds_region = ds.sel(latitude=[latitude], longitude=[longitude], method='nearest')
    
    return ds_region


def ozwald_8day_multiyear(var="Ssoil", latitude=-34.3890427, longitude=148.469499, buffer=0.01, years=["2020", "2021"], thredds=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = ozwald_8day_singleyear_thredds(var, latitude, longitude, buffer, year)
        else:
            ds_year = ozwald_8day_singleyear_gdata(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def ozwald_8day(variables=["Ssoil", "GPP"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir=".", stub="TEST", tmpdir=None, thredds=True, save_netcdf=True, plot=True):
    """Download 8day variables from OzWald at 500m resolution for the region/time of interest

    Parameters
    ----------
        variables: See ozwald_8day_abbreviations at the top of this file for a complete list
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved.
        stub: The name to be prepended to each file download.
        tmpdir: Unused - just an input for consistency with silo_daily.
        thredds: A boolean flag to choose between using the public facing API (slower but works locally), or running directly on NCI (requires access to the ub8 project)
    
    Returns
    -------
        ds_concat: an xarray containing the requested variables in the region of interest for the time period specified
        A NetCDF file of this xarray gets downloaded to outdir/(stub)_ozwald_8day.nc'
    """
    print(f"Starting ozwald_8day")

    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = ozwald_8day_multiyear(variable, lat, lon, buffer, years, thredds)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)
    
    if save_netcdf:
        filename = os.path.join(outdir, f'{stub}_ozwald_8day.nc')
        ds_concat.to_netcdf(filename)
        print("Saved:", filename)

    if plot:
        import matplotlib.pyplot as plt
        variables = list(ds_concat.data_vars)
        figsize = (10, 2 * len(variables))
        ds_point = ds_concat.median(dim=['latitude', 'longitude'])
        fig, axes = plt.subplots(nrows=len(variables), figsize=figsize, sharex=True)
        if len(variables) == 1: 
            axes = [axes]
        for ax, var in zip(axes, variables):
            ds_point[var].plot(ax=ax, add_legend=False)
            ax.set_xlabel("")
        filename = os.path.join(outdir, f'{stub}_ozwald_8day.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print("Saved:", filename)

    return ds_concat

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description="""Download 8 day variables from OzWald at 500m resolution for the region/time of interest""")
    
    parser.add_argument('--variable', default="Ssoil", help=f"Default is 'Ssoil', and options are: {list(ozwald_8day_abbreviations.keys())}")
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.01', help='Buffer in each direction in degrees (default is 0.01, or about 2kmx2km)')
    parser.add_argument('--start_year', default='2020', help='Inclusive, and the minimum start year is 2000. Setting the start and end year to the same value will get all data for that year.')
    parser.add_argument('--end_year', default='2021', help='Specifying a larger end_year than available will automatically give data up to the most recent date (currently 2024)')
    parser.add_argument('--outdir', default='.', help='Directory for the output NetCDF file (default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--thredds', default=True, help='A boolean flag to choose between using the public facing Thredds API (slower but works locally), or running directly on NCI (requires access to the ub8 project). (default: True)')
    parser.add_argument('--plot', default=False, action="store_true", help='Boolean flag to generate a time series plot of the downloaded variables (default: False)')

    return parser.parse_args()
# -

# +
if __name__ == '__main__':
    
    args = parse_arguments()
    
    variable = args.variable
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    start_year = args.start_year
    end_year = args.end_year
    outdir = args.outdir
    stub = args.stub
    thredds=args.thredds
    plot = args.plot

    ozwald_8day([variable], lat, lon, buffer, start_year, end_year, outdir, stub, thredds=thredds, plot=plot)
    