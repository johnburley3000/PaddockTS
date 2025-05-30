# +
# This script merges SILO and OzWald data into a csv for DAESim

# +
# I'm using python 3.9
# # !pip install jupyter jupytext xarray pandas scipy cftime
# -

# Standard Libraries
import os

# Dependencies
import pandas as pd
import xarray as xr
import rioxarray as rxr


abbreviations = {
    "daily_rain" : "Precipitation (SILO)",  # SILO
    "max_temp" : "Maximum temperature (SILO)",  # SILO
    "min_temp" : "Minimum temperature (SILO)",  # SILO
    "vp":"VPeff (SILO)",  # SILO
    "radiation":"SRAD",  # SILO
    "Pg":"Precipitation",  # OzWald
    "Tmax":"Maximum temperature",  # OzWald
    "Tmin":"Minimum temperature",  # OzWald
    "VPeff":"VPeff",  # OzWald
    "Uavg":"Uavg",  # OzWald
    "Ssoil":"Soil moisture",  # OzWald
    "Qtot":"Runoff",  # OzWald
    "LAI":"Vegetation leaf area",  # OzWald
    "GPP":"Vegetation growth"  # OzWald
    }


# This function is much simpler now that we enforce every ds to have a latitude, longitude and time dimension
def aggregate_pixels(ds):
    ds = ds.median(dim=["latitude", "longitude"])
    return ds


def daesim_forcing(outdir=".", stub="TEST"):
    """Merge the ozwald and silo netcdf's into a dataframe for input into DAESim
    
    Parameters
    ----------
        outdir: The directory containing input NetCDFs, and for saving the output csv
        stub: The name that is prepended to each file.
    
    Requirements: This function expects 5 NetCDF files to be predownloaded
    ------------
        outdir/(stub)_silo_daily.nc: For radiation
        outdir/(stub)_ozwald_8day.nc: For soil moisture, runoff, LAI, GPP
        outdir/(stub)_ozwald_daily_Pg.nc: For rainfall 
        outdir/(stub)_ozwald_daily_Tmax.nc: For maximum and minimum temperatures 
        outdir/(stub)_ozwald_daily_Uavg: For wind and vapour pressure

    Returns
    -------
        df_ordered: A dataframe with all the variables required for input into DAESim
        A csv file of this dataframe gets downloaded to outdir/(stub)_DAESim_forcing.csv'
    """

    # Open the pre-downloaded netcdf files
    ds_silo_daily = xr.open_dataset(os.path.join(outdir, stub+'_silo_daily.nc'))
    ds_ozwald_8day = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_8day.nc'))
    ds_ozwald_daily_Pg = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_daily_Pg.nc'))
    ds_ozwald_daily_Tmax = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_daily_Tmax.nc'))
    ds_ozwald_daily_Uavg = xr.open_dataset(os.path.join(outdir, stub+'_ozwald_daily_Uavg.nc'))

    # Make silo coordinates match ozwald 
    if 'crs' in ds_silo_daily.data_vars:
        ds_silo_daily = ds_silo_daily.drop_vars(['crs'])
    ds_silo_daily = ds_silo_daily.rename({"lat": "latitude", "lon": "longitude"})

    # Combine pixels into a single value per timepoint
    ds_silo_daily = aggregate_pixels(ds_silo_daily)
    ds_ozwald_8day = aggregate_pixels(ds_ozwald_8day)
    ds_ozwald_daily_Pg = aggregate_pixels(ds_ozwald_daily_Pg)
    ds_ozwald_daily_Tmax = aggregate_pixels(ds_ozwald_daily_Tmax)
    ds_ozwald_daily_Uavg = aggregate_pixels(ds_ozwald_daily_Uavg)

    # Even though they have overlapping variables it's fine to merge ozwald and SILO, because the overlapping variables all have different names (e.g. 'Pg' and 'daily_rain')
    ds_merged = xr.merge([ds_silo_daily, ds_ozwald_8day, ds_ozwald_daily_Pg, ds_ozwald_daily_Tmax, ds_ozwald_daily_Uavg])

    # Create a DataFrame
    df = ds_merged.to_dataframe().reset_index()
    df = df.set_index('time')
    df.rename(columns=abbreviations, inplace=True)
    df.rename_axis("date", inplace=True)

    # Reorder to match the original DAESim_forcing.csv
    daesim_ordering = ["Precipitation", "Runoff", "Minimum temperature", "Maximum temperature", "Soil moisture", "Vegetation growth", "Vegetation leaf area", "VPeff",	"Uavg", "SRAD"]
    df_ordered = df[daesim_ordering] 

    # Save
    filepath = os.path.join(outdir, stub + "_DAESim_forcing.csv")
    df_ordered.to_csv(filepath)
    print("Saved", filepath)

    return df_ordered


def daesim_soils(outdir=".", stub="Test", tmpdir="."):
    """Merge the soil tiffs into a csv required for DAESim
    
    Parameters
    ----------
        outdir: The directory for saving the output csv
        stub: The name that is prepended to each file.
        tmpdir: The directory containing the input tiff files
    
    Requirements: This function expects 9 soil variables x 4 depths = 36 tiff files to be predownloaded
    ------------

    Returns
    -------
        sorted_df: A dataframe with all the variables required for input into DAESim
        A csv file of this dataframe gets downloaded to outdir/(stub)_DAESim_Soils.csv'
    
    """
    variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
    depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    values = []
    for variable in variables:
        for depth in depths:
            filename = os.path.join(tmpdir, f"{stub}_{variable}_{depth}.tif")
            ds = rxr.open_rasterio(filename)
            value = float(ds.isel(band=0, x=0, y=0).values)  # Assumes a single point was downloaded
            values.append({
                "variable":variable,
                "depth":depth,
                "value":value
            })
    
    # Pivot
    df = pd.DataFrame(values)
    pivot_df = df.pivot(index='depth', columns='variable', values='value')
    pivot_df = pivot_df.reset_index() 
    
    # Sort by depth
    df = pivot_df
    depth_order = ['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    df['depth'] = pd.Categorical(df['depth'], categories=depth_order, ordered=True)
    sorted_df = df.sort_values(by='depth')
    
    # Save
    filepath = os.path.join(outdir, stub + "_DAESim_Soils.csv")
    sorted_df.to_csv(filepath, index=False)
    print("Saved", filepath)

    return sorted_df

if __name__ == '__main__':
    outdir = "."
    stub = "Test"
    daesim_forcing(outdir, stub)
    daesim_soils(outdir, stub)
