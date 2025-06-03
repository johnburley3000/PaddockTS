# +
# Documentation for the SILO variables is here: https://www.longpaddock.qld.gov.au/silo/gridded-data

# +
# Standard Libraries
import os
import shutil
import argparse

# Dependencies
import requests
import xarray as xr

# +
# Taken from https://github.com/Sydney-Informatics-Hub/geodata-harvester/blob/main/src/geodata_harvester/getdata_silo.py
silo_abbreviations = {
        "daily_rain": "Daily rainfall, mm",
        "monthly_rain": "Monthly rainfall, mm",
        "max_temp": "Maximum temperature, degrees Celsius",
        "min_temp": "Minimum temperature, degrees Celsius",
        "vp": "Vapour pressure, hPa",
        "vp_deficit": "Vapour pressure deficit, hPa",
        "evap_pan": "Class A pan evaporation, mm",
        "evap_syn": "Synthetic estimate, mm",
        "evap_morton_lake": "Morton's shallow lake evaporation, mm",
        "radiation": "Solar radiation: Solar exposure, consisting of both direct and diffuse components, MJ/m2",
        "rh_tmax": "Relative humidity:	Relative humidity at the time of maximum temperature, %",
        "rh_tmin": "Relative humidity at the time of minimum temperature, %",
        "et_short_crop": "Evapotranspiration FAO564 short crop, mm",
        "et_tall_crop": "ASCE5 tall crop6, mm",
        "et_morton_actual": "Morton's areal actual evapotranspiration, mm",
        "et_morton_potential": "Morton's point potential evapotranspiration, mm",
        "et_morton_wet": "Morton's wet-environment areal potential evapotranspiration over land, mm",
        "mslp": "Mean sea level pressure Mean sea level pressure, hPa",
    }


def download_from_SILO(var="radiation", year="2020", silo_folder="."):
    """Download a NetCDF for the whole of Australia, for a given year and variable"""
    # I haven't found a way to download only the region of interest from SILO, hence we are downloading all of Australia
    silo_baseurl = "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/"
    url = silo_baseurl + var + "/" + str(year) + "." + var + ".nc"
    filename = os.path.join(silo_folder, f"{year}.{var}.nc")

    # Check the file exists before attempting to download it
    response = requests.head(url)
    if response.status_code == 200:
        print(f"Downloading from SILO: {var} {year} ~400MB")
        with requests.get(url, stream=True) as stream:
            with open(filename, "wb") as file:
                shutil.copyfileobj(stream.raw, file)
        print(f"Downloaded {filename}")


def silo_daily_singleyear(var="radiation", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2020", silo_folder="."):
    """Select the region of interest from the Australia wide NetCDF file"""
    filename = os.path.join(silo_folder, f"{year}.{var}.nc")
    
    if not os.path.exists(filename):
        download_from_SILO(var, year, silo_folder)

    try:
        ds = xr.open_dataset(filename)
    except Exception as e:
        # Likely no data for the specified year
        return None
    
    if 'crs' in list(ds.data_vars):
        ds = ds.drop_vars('crs')
    
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

    min_buffer_size = 0.03
    if buffer < min_buffer_size:
        # Find a single point but keep the lat and lon dimensions for consistency
        ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')
    
    return ds_region


def silo_daily_multiyear(var="radiation", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], silo_folder="."):
    dss = []
    for year in years:
        ds = silo_daily_singleyear(var, latitude, longitude, buffer, year, silo_folder)
        if ds:
            dss.append(ds)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def silo_daily(variables=["radiation"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2020", outdir=".", stub="TEST", tmpdir=".", thredds=None, save_netcdf=True, plot=True):
    """Download daily variables from SILO at 5km resolution for the region/time of interest

    Parameters
    ----------
        variables: See silo_abbreviations at the top of this file for a complete list
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved.
        stub: The name to be prepended to each file download.
        tmpdir: The directory that Australia wide SILO data gets downloaded. Each variable per year is ~400MB, so this can take a while to download. Use "/g/data/xe2/datasets/Climate_SILO" when running on NCI gadi under the xe2 project
        thredds: Unused - just an input for consistency with ozwald_daily.
        save_netcdf: Whether to save the xarray to file or not. This gets downloaded to 'outdir/(stub)_silo_daily.nc'.
        
    Returns
    -------
        ds_concat: an xarray containing the requested variables in the region of interest for the time period specified
    """
    print(f"Starting silo_daily for stub {stub}")
    
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds = silo_daily_multiyear(variable, lat, lon, buffer, years, tmpdir)
        dss.append(ds)
    ds_concat = xr.merge(dss)
    
    if save_netcdf:
        filename = os.path.join(outdir, f'{stub}_silo_daily.nc')
        ds_concat.to_netcdf(filename)
        print("Saved:", filename)

    if plot:
        # Copy pasting this between ozwald_daily, ozwald_8day and silo_daily. Not sure if it's worth creating an import, because small changes between the 3 API's keep cropping up.
        import matplotlib.pyplot as plt
        variables = list(ds.data_vars)
        figsize = (10, 2 * len(variables))
        ds_point = ds.median(dim=['lat', 'lon'])
        fig, axes = plt.subplots(nrows=len(variables), figsize=figsize, sharex=True)
        if len(variables) == 1: 
            axes = [axes]
        for ax, var in zip(axes, variables):
            ds_point[var].plot(ax=ax, add_legend=False)
            ax.set_xlabel("")
        filename = os.path.join(outdir, f'{stub}_silo_daily.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print("Saved:", filename)

    return ds_concat


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description="""Download daily variables from SILO at 5km resolution for the region/time of interest
    Note: This will take ~5 mins and 400MB per variable year if downloading for the first time.""")
    
    parser.add_argument('--variable', default="radiation", help=f"Default is 'radiation', and options are: {list(silo_abbreviations.keys())}")
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--start_year', default='2020', help='Inclusive, and the minimum start year is 1889. Setting the start and end year to the same value will get all data for that year.')
    parser.add_argument('--end_year', default='2021', help='Specifying a larger end_year than available will automatically give data up to the most recent date (currently 2025)')
    parser.add_argument('--outdir', default='.', help='Directory for the output NetCDF file (default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--tmpdir', default='.', help='Directory for copying files from the SILO AWS folder for the whole of Australia (default is the current directory). Use "/g/data/xe2/datasets/Climate_SILO" when running on NCI gadi under the xe2 project.')
    parser.add_argument('--plot', default=False, action="store_true", help='Boolean flag to generate a time series plot of the downloaded variables (default: False)')
  
    return parser.parse_args()
# -

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
    tmpdir = args.tmpdir
    plot = args.plot
    
    silo_daily([variable], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, plot)
    
