import pandas as pd
import xarray as xr

# List all the files
# .tsv files contain time series data for each of the 3888 samples
# .nc files contain time series data for a ~50km buffer around Berridale
# .tif files contain static data for a ~50km buffer around Berridale
# !du -sh /g/data/xe2/cb8590/Eucalypts/*

# Inspect one of the ozwald tsv files
filename = '/g/data/xe2/cb8590/Eucalypts/EUC_ozwald_daily_Tmin_2020_2024.tsv'
df = pd.read_csv(filename, sep='\t', low_memory=False)
bio_columns = [column for column in df.columns if 'bio' in column]
df = df.drop(columns=bio_columns)
print(f"Number of samples: {df.shape[0]}")
print(f"Number of days: {df.shape[1] - 6}")
df.head()

# Inspect one of the silo tsv files
filename = '/g/data/xe2/cb8590/Eucalypts/EUC_silo_daily_min_temp_2020_2025.tsv'
df = pd.read_csv(filename, sep='\t', low_memory=False)
bio_columns = [column for column in df.columns if 'bio' in column]
df = df.drop(columns=bio_columns)
print(f"Number of samples: {df.shape[0]}")
print(f"Number of days: {df.shape[1] - 6}")
df.head()

# Create a geopackage to visualise the tsv results in QGIS
print()


# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_min_temp__1889_2025_silo_daily.nc'
ds = xr.load_dataset(filename)
ds.rio.write_crs("EPSG:4326", inplace=True)

# Visualise the number of days below negative 1 degrees
days_below_neg1 = (ds['min_temp'] < -1).sum(dim="time").astype(float)
filename = '/scratch/xe2/cb8590/tmp/silo_below_neg1.tif'
days_below_neg1.rio.to_raster(filename)
print(filename)

days_below_neg5 = (ds['min_temp'] < -5).sum(dim="time").astype(float)
filename = '/scratch/xe2/cb8590/tmp/silo_below_neg5.tif'
days_below_neg5.rio.to_raster(filename)
print(filename)

days_below_neg10 = (ds['min_temp'] < -10).sum(dim="time").astype(float)
filename = '/scratch/xe2/cb8590/tmp/silo_below_neg10.tif'
days_below_neg10.rio.to_raster(filename)
print(filename)

# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_Tmin__2000_2024_ozwald_daily_Tmin.nc'
ds = xr.load_dataset(filename)
ds.rio.write_crs("EPSG:4326", inplace=True)

# Visualise the number of days below negative 1 degrees
days_below_neg1 = (ds['Tmin'] < -1).sum(dim="time").astype(float)
filename = '/scratch/xe2/cb8590/tmp/ozwald_below_neg1.tif'
days_below_neg1.rio.to_raster(filename)
print(filename)

# Visualise the number of days below negative 1 degrees
days_below_neg5 = (ds['Tmin'] < -5).sum(dim="time").astype(float)
filename = '/scratch/xe2/cb8590/tmp/ozwald_below_neg5.tif'
days_below_neg5.rio.to_raster(filename)
print(filename)

# Visualise the number of days below negative 1 degrees
days_below_neg10 = (ds['Tmin'] < -10).sum(dim="time").astype(float)
filename = '/scratch/xe2/cb8590/tmp/ozwald_below_neg10.tif'
days_below_neg10.rio.to_raster(filename)
print(filename)
